import React, { useState, useRef } from 'react';
import { AlertTriangle, Eye, EyeOff, Check, X, Lock, Key, FileText, Shield, List } from 'lucide-react';
import { useWorkflow, EnvironmentVariable } from '@/contexts/WorkflowContext';

interface EnvVarPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

/**
 * Chip input for list-type environment variables.
 */
function ChipListInput({
  value,
  onChange,
  placeholder,
}: {
  value: string | null;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Parse the stored value as an array (stored as JSON or space-separated)
  const parseItems = (v: string | null): string[] => {
    if (!v) return [];
    try {
      const parsed = JSON.parse(v);
      if (Array.isArray(parsed)) return parsed;
    } catch {
      // Try space-separated
      return v.split(/\s+/).filter(Boolean);
    }
    return [v];
  };
  
  const items = parseItems(value);

  const handleAddItem = () => {
    const trimmed = inputValue.trim();
    if (trimmed) {
      const newItems = [...items, trimmed];
      // Store as JSON array
      onChange(JSON.stringify(newItems));
      setInputValue('');
      inputRef.current?.focus();
    }
  };

  const handleRemoveItem = (index: number) => {
    const newItems = items.filter((_, i) => i !== index);
    onChange(newItems.length > 0 ? JSON.stringify(newItems) : '');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddItem();
    } else if (e.key === 'Backspace' && inputValue === '' && items.length > 0) {
      handleRemoveItem(items.length - 1);
    }
  };

  return (
    <div 
      className="min-h-[32px] p-1.5 rounded border border-gray-600 bg-gray-800 focus-within:border-accent focus-within:ring-1 focus-within:ring-accent transition-colors"
      onClick={() => inputRef.current?.focus()}
    >
      <div className="flex flex-wrap gap-1">
        {items.map((item, index) => (
          <div
            key={index}
            className="flex items-center gap-0.5 px-1.5 py-0.5 bg-gray-700 hover:bg-gray-600 rounded-full text-[11px] text-white group transition-colors"
          >
            <span className="max-w-[120px] truncate">{item}</span>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                handleRemoveItem(index);
              }}
              className="p-0.5 rounded-full hover:bg-gray-500 text-gray-400 hover:text-white transition-colors"
            >
              <X size={10} />
            </button>
          </div>
        ))}
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={() => {
            if (inputValue.trim()) {
              handleAddItem();
            }
          }}
          placeholder={items.length === 0 ? placeholder || 'Type + Enter...' : ''}
          className="flex-1 min-w-[60px] bg-transparent border-none outline-none text-[11px] text-white placeholder-gray-500 py-0.5"
        />
      </div>
    </div>
  );
}

/**
 * Environment Variables Panel (Right Side Panel)
 *
 * Displays all detected environment variables from an imported config
 * and allows users to enter values for them.
 *
 * Features:
 * - Shows all env vars with their locations
 * - Sensitive fields are masked by default
 * - Users can enter values that will be used when running the workflow
 * - Shows which env vars are still unresolved
 * - Displays export type indicator (Secure Text vs Plain Text)
 *   Note: Export type is determined by field type (SecretStr) and can only
 *   be changed in the component config modal, not in this panel.
 * - List fields use a chip-based input for multiple values
 */
export function EnvVarPanel({ isOpen, onClose }: EnvVarPanelProps) {
  const { environmentVariables, unresolvedEnvVars, setEnvVarValue } = useWorkflow();
  const [showSensitive, setShowSensitive] = useState<Record<string, boolean>>({});
  const [editingValues, setEditingValues] = useState<Record<string, string>>({});

  if (!isOpen || environmentVariables.length === 0) {
    return null;
  }

  const handleToggleSensitive = (name: string) => {
    setShowSensitive((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  const handleValueChange = (name: string, value: string) => {
    setEditingValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSaveValue = (name: string) => {
    const value = editingValues[name];
    if (value !== undefined && value !== '') {
      setEnvVarValue(name, value);
      // Clear editing state
      setEditingValues((prev) => {
        const { [name]: _, ...rest } = prev;
        return rest;
      });
    }
  };

  const handleClearValue = (name: string) => {
    setEditingValues((prev) => {
      const { [name]: _, ...rest } = prev;
      return rest;
    });
  };

  const getStatusIcon = (envVar: EnvironmentVariable) => {
    if (envVar.value) {
      return <Check size={14} className="text-green-400" />;
    }
    return <AlertTriangle size={14} className="text-yellow-400" />;
  };

  const isEditing = (name: string) => editingValues[name] !== undefined;

  return (
    <div className="w-80 bg-canvas-light border-l border-gray-700 flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <Key size={18} className="text-accent" />
          <h2 className="text-white font-medium text-sm">Environment Variables</h2>
        </div>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-white transition-colors p-1 rounded hover:bg-gray-700"
        >
          <X size={16} />
        </button>
      </div>

      {/* Status bar */}
      <div className="px-4 py-2 border-b border-gray-700/50 bg-canvas">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400">Total: {environmentVariables.length}</span>
          {unresolvedEnvVars.length > 0 ? (
            <span className="bg-yellow-500/20 text-yellow-400 px-2 py-0.5 rounded">
              {unresolvedEnvVars.length} unresolved
            </span>
          ) : (
            <span className="bg-green-500/20 text-green-400 px-2 py-0.5 rounded">
              All resolved
            </span>
          )}
        </div>
      </div>

      {/* Variable list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {environmentVariables.map((envVar) => (
          <div
            key={envVar.name}
            className="bg-canvas rounded-lg p-3 border border-gray-600/50"
          >
            {/* Variable name row */}
            <div className="flex items-center gap-2 mb-2">
              {getStatusIcon(envVar)}
              <code className="text-accent text-xs font-mono flex-1 truncate" title={`\${${envVar.name}}`}>
                ${envVar.name}
              </code>
              {envVar.is_sensitive && (
                <Lock size={12} className="text-gray-400 flex-shrink-0" title="Sensitive field" />
              )}
            </div>

            {/* Value input - different UI for list vs scalar fields */}
            {envVar.is_list_field ? (
              /* Chip-based input for list fields */
              <div>
                <div className="flex items-center gap-1 mb-1">
                  <List size={10} className="text-blue-400" />
                  <span className="text-[10px] text-blue-400">List field</span>
                </div>
                <ChipListInput
                  value={envVar.value}
                  onChange={(newValue) => setEnvVarValue(envVar.name, newValue)}
                  placeholder="Type + Enter to add..."
                />
                <p className="text-[9px] text-gray-500 mt-1">Press Enter to add, Backspace to remove</p>
              </div>
            ) : (
              /* Standard input for scalar fields */
              <div className="flex items-center gap-1.5">
                <div className="flex-1 relative">
                  <input
                    type={envVar.is_sensitive && !showSensitive[envVar.name] ? 'password' : 'text'}
                    value={
                      isEditing(envVar.name)
                        ? editingValues[envVar.name]
                        : envVar.value || ''
                    }
                    onChange={(e) => handleValueChange(envVar.name, e.target.value)}
                    placeholder={envVar.value ? '••••••' : 'Enter value...'}
                    className="w-full bg-gray-800 border border-gray-600 rounded px-2.5 py-1.5 text-xs text-white placeholder-gray-500 focus:border-accent focus:outline-none pr-7"
                  />
                  {envVar.is_sensitive && (
                    <button
                      onClick={() => handleToggleSensitive(envVar.name)}
                      className="absolute right-1.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white p-0.5"
                      title={showSensitive[envVar.name] ? 'Hide value' : 'Show value'}
                    >
                      {showSensitive[envVar.name] ? <EyeOff size={12} /> : <Eye size={12} />}
                    </button>
                  )}
                </div>
                {isEditing(envVar.name) && (
                  <>
                    <button
                      onClick={() => handleSaveValue(envVar.name)}
                      className="p-1 bg-green-600 hover:bg-green-500 rounded text-white transition-colors flex-shrink-0"
                      title="Save value"
                    >
                      <Check size={12} />
                    </button>
                    <button
                      onClick={() => handleClearValue(envVar.name)}
                      className="p-1 bg-gray-600 hover:bg-gray-500 rounded text-white transition-colors flex-shrink-0"
                      title="Cancel"
                    >
                      <X size={12} />
                    </button>
                  </>
                )}
              </div>
            )}

            {/* Export mode indicator (read-only - change in component config) */}
            <div className="mt-2 flex items-center justify-between">
              <span className="text-[10px] text-gray-500">Export type:</span>
              <div
                className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] ${
                  envVar.has_secret_field_usage
                    ? 'bg-accent/20 text-accent border border-accent/30'
                    : 'bg-gray-600/50 text-gray-400 border border-gray-600'
                }`}
                title={envVar.has_secret_field_usage 
                  ? 'This is a secure field (SecretStr type)' 
                  : 'This is a plain text field'}
              >
                {envVar.has_secret_field_usage ? (
                  <>
                    <Shield size={10} />
                    <span>Secure Text</span>
                  </>
                ) : (
                  <>
                    <FileText size={10} />
                    <span>Plain Text</span>
                  </>
                )}
              </div>
            </div>

            {/* Location info */}
            {envVar.locations.length > 0 && (
              <div className="mt-2 text-[10px] text-gray-500 leading-relaxed">
                <span className="text-gray-600">Used in: </span>
                {envVar.locations.map((loc, i) => (
                  <span key={loc.path}>
                    {i > 0 && ', '}
                    <span className="text-gray-400">{loc.field_name}</span>
                  </span>
                ))}
              </div>
            )}

            {/* Warning indicator */}
            {envVar.warning && (
              <div className="mt-2 flex items-start gap-1.5 text-[10px] text-yellow-500/80">
                <AlertTriangle size={10} className="flex-shrink-0 mt-0.5" />
                <span>Non-secret field - value may be exported as plain text</span>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Footer help text */}
      <div className="px-4 py-2 border-t border-gray-700/50 bg-canvas">
        <p className="text-[10px] text-gray-500 leading-relaxed">
          Enter values for environment variables detected in the config. 
          Click the check mark to save each value.
        </p>
      </div>
    </div>
  );
}

/**
 * Compact badge that shows env var status in the header
 */
export function EnvVarBadge({ onClick }: { onClick: () => void }) {
  const { environmentVariables, unresolvedEnvVars, hasUnresolvedEnvVars } = useWorkflow();

  if (environmentVariables.length === 0) {
    return null;
  }

  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${
        hasUnresolvedEnvVars
          ? 'bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30 border border-yellow-500/30'
          : 'bg-green-500/20 text-green-400 hover:bg-green-500/30 border border-green-500/30'
      }`}
      title={
        hasUnresolvedEnvVars
          ? `${unresolvedEnvVars.length} environment variable(s) need values`
          : 'All environment variables resolved'
      }
    >
      {hasUnresolvedEnvVars ? (
        <>
          <AlertTriangle size={14} />
          <span>{unresolvedEnvVars.length} env vars</span>
        </>
      ) : (
        <>
          <Check size={14} />
          <span>{environmentVariables.length} env vars</span>
        </>
      )}
    </button>
  );
}

