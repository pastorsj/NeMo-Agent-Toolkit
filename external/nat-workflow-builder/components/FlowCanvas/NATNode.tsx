import React, { memo, useMemo, useState, useCallback, useRef, useEffect } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { PlacedComponent, NAT_COMPONENTS } from '@/types';
import { REF_TYPE_COLORS, REF_TYPE_LABELS } from '@/types/registry';
import { ProviderIcon } from '@/components/ProviderIcon';
import { formatDisplayName } from '@/lib/format';
import { Settings, Check, Trash2, X, Link2, Pencil } from 'lucide-react';

interface ConnectionInfo {
  connectionId: string;
  fieldName: string;
  sourceComponentId: string;
  sourceComponentName: string;
}

interface NATNodeData {
  component: PlacedComponent;
  label: string;
  connections: ConnectionInfo[];
  iconUrl?: string | null;
  onConfigure: () => void;
  onDelete: () => void;
  onDeleteConnection: (connectionId: string) => void;
  onNameChange: (newName: string) => void;
}

const colorClasses: Record<string, { border: string; bg: string; icon: string; accent: string }> = {
  'nat-agent': {
    border: 'border-purple-500/50',
    bg: 'from-purple-500/20 to-purple-600/5',
    icon: 'text-purple-400',
    accent: '#a855f7',
  },
  'nat-embedder': {
    border: 'border-emerald-500/50',
    bg: 'from-emerald-500/20 to-emerald-600/5',
    icon: 'text-emerald-400',
    accent: '#10b981',
  },
  'nat-function': {
    border: 'border-blue-500/50',
    bg: 'from-blue-500/20 to-blue-600/5',
    icon: 'text-blue-400',
    accent: '#3b82f6',
  },
  'nat-function-group': {
    border: 'border-indigo-500/50',
    bg: 'from-indigo-500/20 to-indigo-600/5',
    icon: 'text-indigo-400',
    accent: '#6366f1',
  },
  'nat-llm': {
    border: 'border-amber-500/50',
    bg: 'from-amber-500/20 to-amber-600/5',
    icon: 'text-amber-400',
    accent: '#f59e0b',
  },
  'nat-memory': {
    border: 'border-pink-500/50',
    bg: 'from-pink-500/20 to-pink-600/5',
    icon: 'text-pink-400',
    accent: '#ec4899',
  },
  'nat-object-store': {
    border: 'border-violet-500/50',
    bg: 'from-violet-500/20 to-violet-600/5',
    icon: 'text-violet-400',
    accent: '#8b5cf6',
  },
  'nat-retriever': {
    border: 'border-teal-500/50',
    bg: 'from-teal-500/20 to-teal-600/5',
    icon: 'text-teal-400',
    accent: '#14b8a6',
  },
  'nat-auth': {
    border: 'border-red-500/50',
    bg: 'from-red-500/20 to-red-600/5',
    icon: 'text-red-400',
    accent: '#ef4444',
  },
  'nat-middleware': {
    border: 'border-cyan-500/50',
    bg: 'from-cyan-500/20 to-cyan-600/5',
    icon: 'text-cyan-400',
    accent: '#06b6d4',
  },
  // Front-end and observability
  'nat-frontend': {
    border: 'border-emerald-500/50',
    bg: 'from-emerald-500/20 to-emerald-600/5',
    icon: 'text-emerald-400',
    accent: '#10b981',
  },
  'nat-logger': {
    border: 'border-slate-500/50',
    bg: 'from-slate-500/20 to-slate-600/5',
    icon: 'text-slate-400',
    accent: '#64748b',
  },
  'nat-telemetry': {
    border: 'border-amber-500/50',
    bg: 'from-amber-500/20 to-amber-600/5',
    icon: 'text-amber-400',
    accent: '#f59e0b',
  },
  // Evaluation
  'nat-evaluator': {
    border: 'border-sky-500/50',
    bg: 'from-sky-500/20 to-sky-600/5',
    icon: 'text-sky-400',
    accent: '#0ea5e9',
  },
  // Finetuning components
  'nat-trainer': {
    border: 'border-lime-500/50',
    bg: 'from-lime-500/20 to-lime-600/5',
    icon: 'text-lime-400',
    accent: '#84cc16',
  },
  'nat-trajectory': {
    border: 'border-purple-500/50',
    bg: 'from-purple-500/20 to-purple-600/5',
    icon: 'text-purple-400',
    accent: '#a855f7',
  },
  'nat-adapter': {
    border: 'border-stone-500/50',
    bg: 'from-stone-500/20 to-stone-600/5',
    icon: 'text-stone-400',
    accent: '#78716c',
  },
  // Test-Time Compute strategies
  'nat-ttc': {
    border: 'border-yellow-400/50',
    bg: 'from-yellow-400/20 to-yellow-500/5',
    icon: 'text-yellow-400',
    accent: '#facc15',
  },
  // Workflow-level configuration containers
  'nat-workflow': {
    border: 'border-lime-500/50',
    bg: 'from-lime-500/20 to-lime-600/5',
    icon: 'text-lime-400',
    accent: '#76b900',
  },
  'nat-config': {
    border: 'border-gray-500/50',
    bg: 'from-gray-500/20 to-gray-600/5',
    icon: 'text-gray-400',
    accent: '#6b7280',
  },
  'nat-optimizer': {
    border: 'border-orange-500/50',
    bg: 'from-orange-500/20 to-orange-600/5',
    icon: 'text-orange-400',
    accent: '#e25a1c',
  },
  'nat-finetuner': {
    border: 'border-red-500/50',
    bg: 'from-red-500/20 to-red-600/5',
    icon: 'text-red-400',
    accent: '#ee4c2c',
  },
};

function NATNodeComponent({ data, selected }: NodeProps) {
  const nodeData = data as unknown as NATNodeData;
  const { component, connections, iconUrl, onConfigure, onDelete, onDeleteConnection, onNameChange } = nodeData;
  const config = NAT_COMPONENTS[component.type];
  const colors = colorClasses[config.color] || {
    border: 'border-gray-500/50',
    bg: 'from-gray-500/20 to-gray-600/5',
    icon: 'text-gray-400',
    accent: '#6b7280',
  };

  // Component is configured if it has a selected type (from UI) or registeredType (from import)
  const isConfigured = Boolean(component.config?._selected_type || component.registeredType);
  // Use display_name from docstring if available, otherwise format local_name
  const selectedTypeName = component.registeredType
    ? component.registeredType.display_name || formatDisplayName(component.registeredType.local_name)
    : component.config?._selected_type
    ? formatDisplayName(String(component.config._selected_type).split('/').pop() || '')
    : null;

  // Name editing state
  const [isEditingName, setIsEditingName] = useState(false);
  const [editedName, setEditedName] = useState(component.name);
  const nameInputRef = useRef<HTMLInputElement>(null);

  // Update local state when component name changes externally
  useEffect(() => {
    setEditedName(component.name);
  }, [component.name]);

  // Focus input when editing starts
  useEffect(() => {
    if (isEditingName && nameInputRef.current) {
      nameInputRef.current.focus();
      nameInputRef.current.select();
    }
  }, [isEditingName]);

  const handleNameSubmit = useCallback(() => {
    const trimmedName = editedName.trim();
    if (trimmedName && trimmedName !== component.name) {
      onNameChange?.(trimmedName);
    } else {
      setEditedName(component.name);
    }
    setIsEditingName(false);
  }, [editedName, component.name, onNameChange]);

  const handleNameKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleNameSubmit();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setEditedName(component.name);
      setIsEditingName(false);
    }
  }, [handleNameSubmit, component.name]);

  // Get input ports for handles
  const inputPorts = useMemo(() => {
    if (!isConfigured) return [];
    return component.inputPorts || [];
  }, [component.inputPorts, isConfigured]);

  // Calculate handle positions for input ports
  const inputHandlePositions = useMemo(() => {
    if (inputPorts.length === 0) return [];
    const spacing = 100 / (inputPorts.length + 1);
    return inputPorts.map((port, i) => ({
      ...port,
      position: spacing * (i + 1),
    }));
  }, [inputPorts]);

  return (
    <div
      className={`
        relative min-w-[200px] max-w-[280px] rounded-xl border-2 
        bg-gradient-to-b backdrop-blur-sm shadow-xl
        transition-all duration-200
        ${colors.border} ${colors.bg}
        ${selected ? 'ring-2 ring-accent ring-offset-2 ring-offset-gray-900' : ''}
      `}
    >
      {/* Output Handle (left side) - only if configured */}
      {isConfigured && component.outputRefType && (
        <Handle
          type="source"
          position={Position.Left}
          id="output"
          className="!w-3 !h-3 !border-2 !rounded-full"
          style={{
            backgroundColor: colors.accent,
            borderColor: colors.accent,
            left: -6,
          }}
        />
      )}

      {/* Input Handles (right side) - only if configured */}
      {inputHandlePositions.map((port) => {
        const acceptedTypes = port.accepts_ref_types?.length 
          ? port.accepts_ref_types 
          : [port.ref_type];
        const typesLabel = acceptedTypes.map(t => REF_TYPE_LABELS[t] || t).join(' | ');
        const listLabel = port.is_list ? ' (multiple)' : '';
        
        return (
          <Handle
            key={port.field_name}
            type="target"
            position={Position.Right}
            id={port.field_name}
            className="!w-3 !h-3 !border-2 !rounded-full"
            style={{
              backgroundColor: REF_TYPE_COLORS[port.ref_type] || '#6b7280',
              borderColor: REF_TYPE_COLORS[port.ref_type] || '#6b7280',
              right: -6,
              top: `${port.position}%`,
            }}
            title={`${port.title || port.field_name}: ${typesLabel}${listLabel}`}
          />
        );
      })}

      {/* Header */}
      <div className="flex items-center gap-3 px-3 py-2.5 border-b border-gray-700/30">
        <div className={`w-8 h-8 rounded-lg bg-gray-800/50 flex items-center justify-center ${iconUrl ? '' : colors.icon}`}>
          <ProviderIcon
            iconUrl={iconUrl}
            fallbackIcon={config.icon}
            size={18}
            className={iconUrl ? '' : colors.icon}
          />
        </div>
        <div className="flex-1 min-w-0">
          {isEditingName ? (
            <input
              ref={nameInputRef}
              type="text"
              value={editedName}
              onChange={(e) => setEditedName(e.target.value)}
              onBlur={handleNameSubmit}
              onKeyDown={handleNameKeyDown}
              onClick={(e) => e.stopPropagation()}
              className="w-full px-1.5 py-0.5 -ml-1.5 bg-gray-800 border border-gray-600 rounded text-sm text-white font-medium focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent"
              style={{ fontFamily: 'inherit' }}
            />
          ) : (
            <div className="group flex items-center gap-1.5">
              <h3 
                className="font-medium text-sm text-white truncate cursor-pointer hover:text-accent transition-colors"
                onClick={(e) => {
                  e.stopPropagation();
                  setIsEditingName(true);
                }}
                title="Click to rename"
              >
                {component.name}
              </h3>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setIsEditingName(true);
                }}
                className="opacity-0 group-hover:opacity-100 w-4 h-4 flex items-center justify-center text-gray-500 hover:text-accent transition-all"
                title="Rename"
              >
                <Pencil size={10} />
              </button>
            </div>
          )}
          <p className="text-[10px] text-gray-500 uppercase tracking-wider">{config.label}</p>
        </div>
        {/* Delete Button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete?.();
          }}
          className="w-6 h-6 rounded-lg flex items-center justify-center text-gray-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
          title="Delete component"
        >
          <Trash2 size={14} />
        </button>
      </div>

      {/* Body */}
      <div className="px-3 py-2.5">
        {isConfigured ? (
          <div className="flex items-center gap-2">
            <Check size={14} className="text-accent flex-shrink-0" />
            <span className="text-xs text-accent font-medium truncate">{selectedTypeName}</span>
          </div>
        ) : (
          <p className="text-xs text-gray-400 line-clamp-2">{config.description}</p>
        )}
      </div>

      {/* Configure Button */}
      <div className="px-3 pb-2.5">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onConfigure?.();
          }}
          className={`
            w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
            transition-all duration-200
            ${isConfigured 
              ? 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 hover:text-gray-300' 
              : 'bg-amber-500/20 text-amber-400 hover:bg-amber-500/30'
            }
          `}
        >
          <Settings size={12} />
          {isConfigured ? 'Edit Config' : 'Configure'}
        </button>
      </div>

      {/* Port Labels with Connection Info (when configured) */}
      {isConfigured && inputPorts.length > 0 && (
        <div className="px-3 pb-2.5 border-t border-gray-700/30 pt-2">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5">Inputs</div>
          <div className="space-y-2">
            {inputPorts.map((port) => {
              // Get ALL connections for this port (for list fields)
              const portConnections = connections?.filter((c) => c.fieldName === port.field_name) || [];
              const hasConnections = portConnections.length > 0;
              
              // Get accepted types label
              const acceptedTypes = port.accepts_ref_types?.length 
                ? port.accepts_ref_types 
                : [port.ref_type];
              const acceptsLabel = acceptedTypes.map(t => REF_TYPE_LABELS[t] || t).join(' | ');
              
              return (
                <div key={port.field_name} className="flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ backgroundColor: REF_TYPE_COLORS[port.ref_type] }}
                    />
                    <div className="flex-1 min-w-0">
                      <span className="text-[10px] text-gray-400">
                        {port.title || formatDisplayName(port.field_name)}
                        {port.required && <span className="text-red-400 ml-0.5">*</span>}
                        {port.is_list && <span className="text-gray-500 ml-0.5">[]</span>}
                      </span>
                      {acceptedTypes.length > 1 && (
                        <span className="text-[8px] text-gray-600 ml-1">
                          ({acceptsLabel})
                        </span>
                      )}
                    </div>
                  </div>
                  {/* Show all connections for this port */}
                  {hasConnections && (
                    <div className="ml-4 space-y-0.5">
                      {portConnections.map((conn) => (
                        <div key={conn.connectionId} className="flex items-center gap-1">
                          <Link2 size={8} className="text-green-400 flex-shrink-0" />
                          <span 
                            className="text-[9px] text-green-400 truncate flex-1" 
                            title={conn.sourceComponentName}
                          >
                            {conn.sourceComponentName}
                          </span>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onDeleteConnection?.(conn.connectionId);
                            }}
                            className="w-3.5 h-3.5 rounded flex items-center justify-center text-gray-500 hover:text-red-400 hover:bg-red-500/20 transition-colors flex-shrink-0"
                            title="Remove connection"
                          >
                            <X size={8} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Output Label (when configured) */}
      {isConfigured && component.outputRefType && (
        <div className="px-3 pb-2.5 border-t border-gray-700/30 pt-2">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5">Output</div>
          <div className="flex items-center gap-2">
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: colors.accent }}
            />
            <span className="text-[10px] text-gray-400">
              {REF_TYPE_LABELS[component.outputRefType]}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export const NATNode = memo(NATNodeComponent);

