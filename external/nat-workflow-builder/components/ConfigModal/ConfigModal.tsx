import React, { useState, useEffect } from 'react';
import { X, ChevronDown, Save, AlertCircle, Link2 } from 'lucide-react';
import { PlacedComponent, NAT_COMPONENTS, NATComponentType } from '@/types';
import { useRegistry } from '@/contexts/RegistryContext';
import { useWorkflow } from '@/contexts/WorkflowContext';
import { RegisteredTypeInfo, ComponentCategory, REF_TYPE_LABELS, REF_TYPE_COLORS } from '@/types/registry';
import { ComponentIcon } from '@/components/NATComponents/ComponentIcon';
import { SchemaFormField } from './SchemaFormField';
import { formatDisplayName } from '@/lib/format';

// Map UI component types to registry categories
const TYPE_TO_CATEGORY: Partial<Record<NATComponentType, ComponentCategory>> = {
  llm: 'llm',
  embedder: 'embedder',
  agent: 'agent',
  function: 'function',
  function_group: 'function_group',
  retriever: 'retriever',
  memory: 'memory',
  object_store: 'object_store',
  authentication: 'authentication',
  middleware: 'middleware',
};

interface ConfigModalProps {
  component: PlacedComponent;
  onClose: () => void;
  onSave: (config: Record<string, unknown>) => void;
}

export function ConfigModal({ component, onClose, onSave }: ConfigModalProps) {
  const { getTypesForCategory, loading, error, connected } = useRegistry();
  const { updateComponentRegisteredType } = useWorkflow();
  const [selectedType, setSelectedType] = useState<RegisteredTypeInfo | null>(null);
  const [formData, setFormData] = useState<Record<string, unknown>>(component.config || {});
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const config = NAT_COMPONENTS[component.type];
  const category = TYPE_TO_CATEGORY[component.type];
  const availableTypes = category ? getTypesForCategory(category) : [];

  // Initialize selected type from existing config
  useEffect(() => {
    if (component.config?._selected_type && availableTypes.length > 0) {
      const existing = availableTypes.find(
        (t) => t.full_type === component.config._selected_type
      );
      if (existing) {
        setSelectedType(existing);
      }
    }
  }, [component.config, availableTypes]);

  const handleFieldChange = (fieldName: string, value: unknown) => {
    setFormData((prev) => ({ ...prev, [fieldName]: value }));
  };

  const handleSave = () => {
    if (selectedType) {
      const saveData = {
        ...formData,
        _selected_type: selectedType.full_type,
      };
      // Use the new method that also updates input ports
      updateComponentRegisteredType(component.id, selectedType, saveData);
    } else {
      onSave(formData);
    }
    onClose();
  };

  const handleTypeSelect = (type: RegisteredTypeInfo) => {
    setSelectedType(type);
    setDropdownOpen(false);
    // Reset form data when type changes, keeping type field defaults
    const defaults: Record<string, unknown> = {};
    type.fields.forEach((field) => {
      if (field.default !== null && field.default !== undefined) {
        defaults[field.name] = field.default;
      }
    });
    setFormData(defaults);
  };

  const colorClasses: Record<string, { border: string; bg: string; icon: string }> = {
    'nat-agent': { border: 'border-purple-500', bg: 'bg-purple-500/10', icon: 'text-purple-400' },
    'nat-llm': { border: 'border-amber-500', bg: 'bg-amber-500/10', icon: 'text-amber-400' },
    'nat-embedder': { border: 'border-emerald-500', bg: 'bg-emerald-500/10', icon: 'text-emerald-400' },
    'nat-function': { border: 'border-blue-500', bg: 'bg-blue-500/10', icon: 'text-blue-400' },
    'nat-function-group': { border: 'border-indigo-500', bg: 'bg-indigo-500/10', icon: 'text-indigo-400' },
    'nat-retriever': { border: 'border-teal-500', bg: 'bg-teal-500/10', icon: 'text-teal-400' },
    'nat-memory': { border: 'border-pink-500', bg: 'bg-pink-500/10', icon: 'text-pink-400' },
    'nat-object-store': { border: 'border-violet-500', bg: 'bg-violet-500/10', icon: 'text-violet-400' },
    'nat-auth': { border: 'border-red-500', bg: 'bg-red-500/10', icon: 'text-red-400' },
    'nat-middleware': { border: 'border-cyan-500', bg: 'bg-cyan-500/10', icon: 'text-cyan-400' },
  };

  const colors = colorClasses[config.color] || { border: 'border-gray-500', bg: 'bg-gray-500/10', icon: 'text-gray-400' };

  // Get fields that are NOT component refs (connection ports)
  const nonRefFields = selectedType?.fields.filter(
    (field) => !field.name.startsWith('_') && field.name !== 'type' && !field.is_component_ref
  ) || [];

  // Get connection ports for the selected type
  const connectionPorts = selectedType?.input_ports || [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-3xl max-h-[90vh] overflow-hidden rounded-2xl bg-canvas-light border border-gray-700 shadow-2xl animate-drop">
        {/* Header */}
        <div className={`flex items-center gap-4 p-6 border-b border-gray-700 ${colors.bg}`}>
          <div className={`p-3 rounded-xl ${colors.bg} ${colors.border} border-2`}>
            <ComponentIcon icon={config.icon} size={24} className={colors.icon} />
          </div>
          <div className="flex-1">
            <h2 className="text-xl font-display font-bold text-white">
              Configure {component.name}
            </h2>
            <p className="text-sm text-gray-400 mt-1">{config.description}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-700/50 text-gray-400 hover:text-white transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content - expands when dropdown is open to show options */}
        <div className={`p-6 overflow-y-auto max-h-[70vh] transition-all duration-200 ${
          dropdownOpen ? 'min-h-[450px]' : ''
        }`}>
          {!connected ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <AlertCircle className="text-yellow-500 mb-4" size={48} />
              <h3 className="text-lg font-semibold text-gray-300 mb-2">
                NAT API Not Connected
              </h3>
              <p className="text-sm text-gray-500 max-w-md">
                Start the NAT Workflow Builder API to load available component types.
                <br />
                <code className="mt-2 inline-block px-2 py-1 bg-gray-800 rounded text-xs">
                  python -m nat.workflow_builder_api.server
                </code>
              </p>
            </div>
          ) : loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-2 border-accent border-t-transparent" />
            </div>
          ) : error ? (
            <div className="text-red-400 p-4 bg-red-500/10 rounded-lg">
              {error}
            </div>
          ) : (
            <div className="space-y-6">
              {/* Type Selector Dropdown */}
              {availableTypes.length > 0 && (
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Implementation Type
                  </label>
                  <div className="relative">
                    <button
                      onClick={() => setDropdownOpen(!dropdownOpen)}
                      className={`w-full flex items-center justify-between px-4 py-3 rounded-lg border-2 transition-colors ${
                        selectedType
                          ? `${colors.border} ${colors.bg}`
                          : 'border-gray-600 bg-gray-800 hover:border-gray-500'
                      }`}
                    >
                      <span className="text-left">
                        {selectedType ? (
                          <span className="flex flex-col">
                            <span className="font-medium text-white">
                              {formatDisplayName(selectedType.local_name)}
                            </span>
                            <span className="text-xs text-gray-400">
                              {selectedType.module_name}
                            </span>
                          </span>
                        ) : (
                          <span className="text-gray-400">Select a type...</span>
                        )}
                      </span>
                      <ChevronDown
                        size={20}
                        className={`text-gray-400 transition-transform ${
                          dropdownOpen ? 'rotate-180' : ''
                        }`}
                      />
                    </button>

                    {dropdownOpen && (
                      <div className="absolute z-10 w-full mt-2 py-2 bg-gray-800 border border-gray-700 rounded-lg shadow-xl max-h-[400px] overflow-y-auto">
                        {availableTypes.map((type) => (
                          <button
                            key={type.full_type}
                            onClick={() => handleTypeSelect(type)}
                            className={`w-full px-4 py-3 text-left hover:bg-gray-700 transition-colors ${
                              selectedType?.full_type === type.full_type
                                ? 'bg-gray-700/50'
                                : ''
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <div className="font-medium text-white">
                                {formatDisplayName(type.local_name)}
                              </div>
                              {type.input_ports.length > 0 && (
                                <div className="flex items-center gap-1 text-xs text-gray-500">
                                  <Link2 size={12} />
                                  {type.input_ports.length}
                                </div>
                              )}
                            </div>
                            <div className="text-xs text-gray-400">
                              {type.module_name}
                            </div>
                            {type.description && (
                              <div className="text-xs text-gray-500 mt-1 line-clamp-2">
                                {type.description}
                              </div>
                            )}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Connection Ports Info */}
              {selectedType && connectionPorts.length > 0 && (
                <div className="space-y-3">
                  <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                    <Link2 size={14} />
                    Connections Required
                  </h3>
                  <p className="text-xs text-gray-500">
                    These fields require connections to other components. Save this configuration, 
                    then drag from the output port of a component to connect.
                  </p>
                  <div className="space-y-2">
                    {connectionPorts.map((port) => (
                      <div
                        key={port.field_name}
                        className="flex items-center gap-3 p-3 rounded-lg bg-gray-800/50 border border-gray-700"
                      >
                        <div
                          className="w-3 h-3 rounded-full border-2"
                          style={{
                            borderColor: REF_TYPE_COLORS[port.ref_type],
                            backgroundColor: 'transparent',
                          }}
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-gray-300">
                              {port.title || port.field_name}
                            </span>
                            {port.required ? (
                              <span className="text-[10px] px-1.5 py-0.5 bg-red-500/20 text-red-400 rounded">
                                required
                              </span>
                            ) : (
                              <span className="text-[10px] px-1.5 py-0.5 bg-gray-700 text-gray-400 rounded">
                                optional
                              </span>
                            )}
                            {port.is_list && (
                              <span className="text-[10px] px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded">
                                multiple
                              </span>
                            )}
                          </div>
                          {port.description && (
                            <p className="text-xs text-gray-500 mt-0.5 truncate">
                              {port.description}
                            </p>
                          )}
                        </div>
                        <span
                          className="text-xs font-medium"
                          style={{ color: REF_TYPE_COLORS[port.ref_type] }}
                        >
                          {REF_TYPE_LABELS[port.ref_type]}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Configuration Fields (non-connection fields only) */}
              {selectedType && nonRefFields.length > 0 && (
                <div className="space-y-4">
                  <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
                    Configuration
                  </h3>
                  <div className="space-y-4">
                    {nonRefFields.map((field) => (
                      <SchemaFormField
                        key={field.name}
                        field={field}
                        value={formData[field.name]}
                        onChange={(value) => handleFieldChange(field.name, value)}
                      />
                    ))}
                  </div>
                </div>
              )}

              {availableTypes.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  <p>No registered implementations available for this component type.</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-6 border-t border-gray-700 bg-gray-800/30">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg border border-gray-600 text-gray-300 hover:bg-gray-700 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={!selectedType && availableTypes.length > 0}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent text-black font-medium hover:bg-accent-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Save size={16} />
            Save Configuration
          </button>
        </div>
      </div>
    </div>
  );
}
