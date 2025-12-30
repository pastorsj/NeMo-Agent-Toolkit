import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown, Search, Eye, EyeOff, Lock, Variable, X, Plus } from 'lucide-react';
import { FieldInfo } from '@/types/registry';
import { isEnvVarPlaceholder, getEnvVarNameFromPlaceholder, placeholderToDisplayFormat } from '@/lib/envVarUtils';

interface SecretFieldExportConfig {
  exportMode: 'plain' | 'env_var';
  envVarName: string;
}

interface SchemaFormFieldProps {
  field: FieldInfo;
  value: unknown;
  onChange: (value: unknown) => void;
  /** For secret fields: current export configuration */
  secretConfig?: SecretFieldExportConfig;
  /** For secret fields: callback when export config changes */
  onSecretConfigChange?: (config: SecretFieldExportConfig) => void;
  /** Component ID for generating default env var names */
  componentId?: string;
  /** Map of env var names to their resolved values */
  envVarValues?: Map<string, string | null>;
  /** Callback when an env var value is updated */
  onEnvVarChange?: (varName: string, value: string) => void;
  /** If this field is associated with an env var (for syncing) */
  associatedEnvVar?: string;
}

/**
 * Combobox component for fields with options - allows both selection and free text input.
 */
function ComboboxField({
  field,
  value,
  onChange,
  inputClasses,
  labelClasses,
  descriptionClasses,
}: {
  field: FieldInfo;
  value: unknown;
  onChange: (value: unknown) => void;
  inputClasses: string;
  labelClasses: string;
  descriptionClasses: string;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const options = field.options || [];
  const currentValue = (value as string) || '';

  // Filter options based on search query
  const filteredOptions = options.filter((option) =>
    option.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (option: string) => {
    onChange(option);
    setSearchQuery('');
    setIsOpen(false);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    onChange(newValue || null);
    setSearchQuery(newValue);
    setIsOpen(true);
  };

  return (
    <div ref={containerRef}>
      <label className={labelClasses}>
        {field.title || field.name}
        {field.required && <span className="text-red-400 ml-1">*</span>}
      </label>
      <div className="relative">
        <div className="relative">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
          <input
            ref={inputRef}
            type="text"
            value={currentValue}
            onChange={handleInputChange}
            onFocus={() => setIsOpen(true)}
            placeholder={field.default ? `Default: ${field.default}` : 'Type or select...'}
            className={`${inputClasses} pl-10 pr-10`}
          />
          <button
            type="button"
            onClick={() => setIsOpen(!isOpen)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
          >
            <ChevronDown size={16} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
          </button>
        </div>

        {isOpen && filteredOptions.length > 0 && (
          <div className="absolute z-20 w-full mt-1 bg-gray-800 border border-gray-600 rounded-lg shadow-xl max-h-[250px] overflow-y-auto">
            {filteredOptions.map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => handleSelect(option)}
                className={`w-full px-4 py-2.5 text-left text-sm hover:bg-gray-700 transition-colors ${
                  option === currentValue ? 'bg-gray-700/50 text-accent' : 'text-white'
                }`}
              >
                {option}
              </button>
            ))}
          </div>
        )}
      </div>
      {field.description && <p className={descriptionClasses}>{field.description}</p>}
    </div>
  );
}

/**
 * Special field component for environment variable placeholders.
 * Shows the variable name and allows users to enter/update the value.
 */
function EnvVarField({
  field,
  envVarName,
  resolvedValue,
  onValueChange,
  inputClasses,
  labelClasses,
  descriptionClasses,
}: {
  field: FieldInfo;
  envVarName: string;
  resolvedValue: string | null | undefined;
  onValueChange: (value: string) => void;
  inputClasses: string;
  labelClasses: string;
  descriptionClasses: string;
}) {
  const [localValue, setLocalValue] = useState(resolvedValue || '');
  const [showValue, setShowValue] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  
  // Determine if this is likely a sensitive field
  const isSensitive = field.name.toLowerCase().includes('secret') || 
                      field.name.toLowerCase().includes('password') ||
                      field.name.toLowerCase().includes('key') ||
                      field.name.toLowerCase().includes('token');

  // Sync with resolved value when it changes externally
  useEffect(() => {
    if (resolvedValue !== null && resolvedValue !== undefined) {
      setLocalValue(resolvedValue);
    }
  }, [resolvedValue]);

  const handleSave = () => {
    if (localValue) {
      onValueChange(localValue);
    }
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSave();
    } else if (e.key === 'Escape') {
      setLocalValue(resolvedValue || '');
      setIsEditing(false);
    }
  };

  return (
    <div>
      {/* Label with env var indicator */}
      <label className={labelClasses}>
        <span className="flex items-center gap-2">
          <Variable size={14} className="text-accent" />
          {field.title || field.name}
          {field.required && <span className="text-red-400 ml-1">*</span>}
        </span>
      </label>

      {/* Env var reference display */}
      <div className="mb-2 flex items-center gap-2 text-xs">
        <span className="text-gray-500">Variable:</span>
        <code className="px-2 py-0.5 bg-accent/20 text-accent rounded font-mono">
          {`\${${envVarName}}`}
        </code>
      </div>

      {/* Value input */}
      <div className="relative">
        <input
          type={isSensitive && !showValue ? 'password' : 'text'}
          value={localValue}
          onChange={(e) => {
            setLocalValue(e.target.value);
            setIsEditing(true);
          }}
          onBlur={handleSave}
          onKeyDown={handleKeyDown}
          placeholder={resolvedValue ? '••••••' : 'Enter value...'}
          className={`${inputClasses} ${isSensitive ? 'pr-12' : ''} ${
            isEditing ? 'border-accent ring-1 ring-accent' : ''
          }`}
        />
        {isSensitive && (
          <button
            type="button"
            onClick={() => setShowValue(!showValue)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
          >
            {showValue ? <EyeOff size={18} /> : <Eye size={18} />}
          </button>
        )}
      </div>

      {/* Status indicator */}
      <div className="mt-1.5 flex items-center gap-2 text-xs">
        {resolvedValue ? (
          <span className="text-green-400">✓ Value set</span>
        ) : (
          <span className="text-yellow-400">⚠ Value required</span>
        )}
        {isEditing && (
          <span className="text-gray-500">(press Enter to save)</span>
        )}
      </div>

      {field.description && <p className={descriptionClasses}>{field.description}</p>}
    </div>
  );
}

/**
 * Chip-based array field component.
 * Displays array items as dismissable chips with an input to add new items.
 * Supports syncing with environment variables when envVarName is provided.
 */
function ChipArrayField({
  field,
  value,
  onChange,
  inputClasses,
  labelClasses,
  descriptionClasses,
  envVarName,
  onEnvVarChange,
}: {
  field: FieldInfo;
  value: unknown;
  onChange: (value: unknown) => void;
  inputClasses: string;
  labelClasses: string;
  descriptionClasses: string;
  /** If this field is backed by an env var, the variable name */
  envVarName?: string;
  /** Callback to update the env var value */
  onEnvVarChange?: (varName: string, value: string) => void;
}) {
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  
  const arrayValue = Array.isArray(value) ? value : [];

  // Helper to sync changes with env var if applicable
  const syncWithEnvVar = (newArray: unknown[]) => {
    if (envVarName && onEnvVarChange) {
      // Store as JSON array string for the env var
      onEnvVarChange(envVarName, JSON.stringify(newArray));
    }
  };

  const handleAddItem = () => {
    const trimmed = inputValue.trim();
    if (trimmed) {
      const newArray = [...arrayValue, trimmed];
      onChange(newArray);
      syncWithEnvVar(newArray);
      setInputValue('');
      inputRef.current?.focus();
    }
  };

  const handleRemoveItem = (index: number) => {
    const newArray = arrayValue.filter((_, i) => i !== index);
    onChange(newArray.length > 0 ? newArray : null);
    syncWithEnvVar(newArray);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddItem();
    } else if (e.key === 'Backspace' && inputValue === '' && arrayValue.length > 0) {
      // Remove last item on backspace when input is empty
      handleRemoveItem(arrayValue.length - 1);
    }
  };

  return (
    <div>
      <label className={labelClasses}>
        <span className="flex items-center gap-2">
          {envVarName && <Variable size={14} className="text-accent" />}
          {field.title || field.name}
          {field.required && <span className="text-red-400 ml-1">*</span>}
        </span>
      </label>
      
      {/* Env var reference display (if synced) */}
      {envVarName && (
        <div className="mb-2 flex items-center gap-2 text-xs">
          <span className="text-gray-500">Synced with:</span>
          <code className="px-2 py-0.5 bg-accent/20 text-accent rounded font-mono">
            {`\${${envVarName}}`}
          </code>
        </div>
      )}
      
      {/* Chips container */}
      <div 
        className={`min-h-[42px] p-2 rounded-lg border bg-gray-800 focus-within:border-accent focus-within:ring-1 focus-within:ring-accent transition-colors ${
          envVarName ? 'border-accent/30' : 'border-gray-600'
        }`}
        onClick={() => inputRef.current?.focus()}
      >
        <div className="flex flex-wrap gap-1.5">
          {/* Existing chips */}
          {arrayValue.map((item, index) => (
            <div
              key={index}
              className="flex items-center gap-1 px-2.5 py-1 bg-gray-700 hover:bg-gray-600 rounded-full text-sm text-white group transition-colors"
            >
              <span className="max-w-[200px] truncate">{String(item)}</span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemoveItem(index);
                }}
                className="p-0.5 rounded-full hover:bg-gray-500 text-gray-400 hover:text-white transition-colors"
                title="Remove item"
              >
                <X size={12} />
              </button>
            </div>
          ))}
          
          {/* Input for new items */}
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={() => {
              // Add item on blur if there's input
              if (inputValue.trim()) {
                handleAddItem();
              }
            }}
            placeholder={arrayValue.length === 0 ? 'Type and press Enter to add...' : 'Add more...'}
            className="flex-1 min-w-[120px] bg-transparent border-none outline-none text-sm text-white placeholder-gray-500 py-1"
          />
        </div>
      </div>
      
      {/* Helper text */}
      <div className="flex items-center justify-between mt-1.5">
        {field.description ? (
          <p className={descriptionClasses}>{field.description}</p>
        ) : (
          <p className="text-xs text-gray-500">Press Enter to add, Backspace to remove last</p>
        )}
        {arrayValue.length > 0 && (
          <span className="text-xs text-gray-500">{arrayValue.length} item{arrayValue.length !== 1 ? 's' : ''}</span>
        )}
      </div>
    </div>
  );
}

/**
 * Special field component for secret fields (SerializableSecretStr/OptionalSecretStr).
 * Allows the user to:
 * 1. Enter the secret value (masked by default)
 * 2. Choose whether to export as plain text or as an environment variable
 * 3. Specify the env var name if exporting as env var
 */
function SecretField({
  field,
  value,
  onChange,
  secretConfig,
  onSecretConfigChange,
  componentId,
}: {
  field: FieldInfo;
  value: unknown;
  onChange: (value: unknown) => void;
  secretConfig?: SecretFieldExportConfig;
  onSecretConfigChange?: (config: SecretFieldExportConfig) => void;
  componentId?: string;
}) {
  const [showValue, setShowValue] = useState(false);
  const [isEditingEnvVar, setIsEditingEnvVar] = useState(false);
  
  // Generate default env var name from component ID and field name
  const defaultEnvVarName = React.useMemo(() => {
    const prefix = componentId ? componentId.toUpperCase().replace(/-/g, '_') : 'SECRET';
    return `${prefix}_${field.name.toUpperCase()}`;
  }, [componentId, field.name]);
  
  const currentConfig = secretConfig || {
    exportMode: 'env_var' as const,
    envVarName: defaultEnvVarName,
  };

  const inputClasses = `
    w-full px-4 py-2.5 rounded-lg border border-gray-600 bg-gray-800 text-white
    focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent
    placeholder:text-gray-500 transition-colors
  `;
  const labelClasses = 'block text-sm font-medium text-gray-300 mb-1.5';
  const descriptionClasses = 'text-xs text-gray-500 mt-1';

  const handleExportModeChange = (mode: 'plain' | 'env_var') => {
    onSecretConfigChange?.({
      ...currentConfig,
      exportMode: mode,
    });
  };

  const handleEnvVarNameChange = (name: string) => {
    onSecretConfigChange?.({
      ...currentConfig,
      envVarName: name,
    });
  };

  return (
    <div className="space-y-3">
      {/* Field Label */}
      <label className={labelClasses}>
        <span className="flex items-center gap-2">
          <Lock size={14} className="text-amber-400" />
          {field.title || field.name}
          {field.required && <span className="text-red-400 ml-1">*</span>}
          <span className="text-xs text-amber-400 font-normal">(Secret)</span>
        </span>
      </label>

      {/* Value Input */}
      <div className="relative">
        <input
          type={showValue ? 'text' : 'password'}
          value={(value as string) || ''}
          onChange={(e) => onChange(e.target.value || null)}
          placeholder={field.default !== null ? `Default: ${field.default}` : 'Enter secret value...'}
          className={`${inputClasses} pr-12`}
        />
        <button
          type="button"
          onClick={() => setShowValue(!showValue)}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
        >
          {showValue ? <EyeOff size={18} /> : <Eye size={18} />}
        </button>
      </div>

      {field.description && <p className={descriptionClasses}>{field.description}</p>}

      {/* Export Mode Toggle */}
      <div className="mt-3 p-3 bg-gray-900/50 rounded-lg border border-gray-700">
        <p className="text-xs text-gray-400 mb-2">Export Options:</p>
        <div className="space-y-2">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name={`export-mode-${field.name}`}
              checked={currentConfig.exportMode === 'env_var'}
              onChange={() => handleExportModeChange('env_var')}
              className="w-4 h-4 text-accent bg-gray-700 border-gray-600 focus:ring-accent"
            />
            <span className="text-sm text-gray-300">Export as environment variable</span>
          </label>
          
          {currentConfig.exportMode === 'env_var' && (
            <div className="ml-6 flex items-center gap-2">
              <span className="text-sm text-gray-500">$&#123;</span>
              {isEditingEnvVar ? (
                <input
                  type="text"
                  value={currentConfig.envVarName}
                  onChange={(e) => handleEnvVarNameChange(e.target.value.toUpperCase().replace(/[^A-Z0-9_]/g, '_'))}
                  onBlur={() => setIsEditingEnvVar(false)}
                  onKeyDown={(e) => e.key === 'Enter' && setIsEditingEnvVar(false)}
                  autoFocus
                  className="px-2 py-1 text-sm bg-gray-800 border border-gray-600 rounded text-accent font-mono"
                  placeholder="VAR_NAME"
                />
              ) : (
                <button
                  type="button"
                  onClick={() => setIsEditingEnvVar(true)}
                  className="px-2 py-1 text-sm bg-gray-800 border border-gray-600 rounded text-accent font-mono hover:border-accent"
                >
                  {currentConfig.envVarName || defaultEnvVarName}
                </button>
              )}
              <span className="text-sm text-gray-500">&#125;</span>
            </div>
          )}

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name={`export-mode-${field.name}`}
              checked={currentConfig.exportMode === 'plain'}
              onChange={() => handleExportModeChange('plain')}
              className="w-4 h-4 text-accent bg-gray-700 border-gray-600 focus:ring-accent"
            />
            <span className="text-sm text-gray-300">Export as plain text</span>
            <span className="text-xs text-amber-400">(⚠️ Value visible in YAML)</span>
          </label>
        </div>
      </div>
    </div>
  );
}

export function SchemaFormField({ 
  field, 
  onChange, 
  value, 
  secretConfig, 
  onSecretConfigChange, 
  componentId,
  envVarValues,
  onEnvVarChange,
  associatedEnvVar,
}: SchemaFormFieldProps) {
  const inputClasses = `
    w-full px-4 py-2.5 rounded-lg border border-gray-600 bg-gray-800 text-white
    focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent
    placeholder:text-gray-500 transition-colors
  `;

  const labelClasses = 'block text-sm font-medium text-gray-300 mb-1.5';
  const descriptionClasses = 'text-xs text-gray-500 mt-1';

  // Check if the current value is an env var placeholder
  const envVarName = typeof value === 'string' ? getEnvVarNameFromPlaceholder(value) : null;
  const isEnvVar = envVarName !== null;
  const resolvedEnvValue = envVarName ? envVarValues?.get(envVarName) : null;

  // If this is an env var field, render special env var UI
  if (isEnvVar) {
    return (
      <EnvVarField
        field={field}
        envVarName={envVarName}
        resolvedValue={resolvedEnvValue}
        onValueChange={(newValue) => {
          onEnvVarChange?.(envVarName, newValue);
        }}
        inputClasses={inputClasses}
        labelClasses={labelClasses}
        descriptionClasses={descriptionClasses}
      />
    );
  }

  // Render secret fields with special handling
  if (field.is_secret) {
    return (
      <SecretField
        field={field}
        value={value}
        onChange={onChange}
        secretConfig={secretConfig}
        onSecretConfigChange={onSecretConfigChange}
        componentId={componentId}
      />
    );
  }

  // Render enum as strict select (no free text)
  if (field.enum && field.enum.length > 0) {
    return (
      <div>
        <label className={labelClasses}>
          {field.title || field.name}
          {field.required && <span className="text-red-400 ml-1">*</span>}
        </label>
        <select
          value={(value as string) || ''}
          onChange={(e) => onChange(e.target.value || null)}
          className={inputClasses}
        >
          <option value="">Select...</option>
          {field.enum.map((option) => (
            <option key={String(option)} value={String(option)}>
              {String(option)}
            </option>
          ))}
        </select>
        {field.description && <p className={descriptionClasses}>{field.description}</p>}
      </div>
    );
  }

  // Render options as combobox (searchable with free text allowed)
  if (field.options && field.options.length > 0) {
    return (
      <ComboboxField
        field={field}
        value={value}
        onChange={onChange}
        inputClasses={inputClasses}
        labelClasses={labelClasses}
        descriptionClasses={descriptionClasses}
      />
    );
  }

  // Render based on type
  switch (field.type) {
    case 'boolean':
      return (
        <div>
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={Boolean(value)}
              onChange={(e) => onChange(e.target.checked)}
              className="w-5 h-5 rounded border-gray-600 bg-gray-800 text-accent focus:ring-accent focus:ring-offset-0"
            />
            <span className="text-sm font-medium text-gray-300">
              {field.title || field.name}
              {field.required && <span className="text-red-400 ml-1">*</span>}
            </span>
          </label>
          {field.description && <p className={`${descriptionClasses} ml-8`}>{field.description}</p>}
        </div>
      );

    case 'integer':
    case 'number':
      return (
        <div>
          <label className={labelClasses}>
            {field.title || field.name}
            {field.required && <span className="text-red-400 ml-1">*</span>}
          </label>
          <input
            type="number"
            value={value !== undefined && value !== null ? Number(value) : ''}
            onChange={(e) =>
              onChange(e.target.value ? (field.type === 'integer' ? parseInt(e.target.value) : parseFloat(e.target.value)) : null)
            }
            min={field.minimum !== null ? field.minimum : undefined}
            max={field.maximum !== null ? field.maximum : undefined}
            step={field.type === 'integer' ? 1 : 'any'}
            placeholder={field.default !== null ? `Default: ${field.default}` : undefined}
            className={inputClasses}
          />
          {field.description && <p className={descriptionClasses}>{field.description}</p>}
        </div>
      );

    case 'array': {
      // Check if array contains env var placeholders
      const arrayValue = Array.isArray(value) ? value : [];
      const hasEnvVars = arrayValue.some((item) => isEnvVarPlaceholder(item));
      
      // If the array has env var placeholders, filter them out for editing
      // The actual values will be synced via the associatedEnvVar
      const editableValue = hasEnvVars 
        ? arrayValue.filter((item) => !isEnvVarPlaceholder(item))
        : arrayValue;
      
      return (
        <ChipArrayField
          field={field}
          value={editableValue}
          onChange={onChange}
          inputClasses={inputClasses}
          labelClasses={labelClasses}
          descriptionClasses={descriptionClasses}
          envVarName={associatedEnvVar}
          onEnvVarChange={onEnvVarChange}
        />
      );
    }

    case 'object':
      return (
        <div>
          <label className={labelClasses}>
            {field.title || field.name}
            {field.required && <span className="text-red-400 ml-1">*</span>}
          </label>
          <textarea
            value={value && typeof value === 'object' ? JSON.stringify(value, null, 2) : ''}
            onChange={(e) => {
              try {
                const parsed = JSON.parse(e.target.value);
                onChange(parsed);
              } catch {
                // Keep the raw value for editing
              }
            }}
            placeholder='{"key": "value"}'
            rows={4}
            className={`${inputClasses} font-mono text-sm`}
          />
          {field.description && <p className={descriptionClasses}>{field.description}</p>}
          <p className="text-xs text-gray-600 mt-1">Enter as JSON object</p>
        </div>
      );

    case 'string':
    default:
      // Check if it might be a longer text field
      const isLongText =
        field.name.includes('description') ||
        field.name.includes('prompt') ||
        field.name.includes('content');

      if (isLongText) {
        return (
          <div>
            <label className={labelClasses}>
              {field.title || field.name}
              {field.required && <span className="text-red-400 ml-1">*</span>}
            </label>
            <textarea
              value={(value as string) || ''}
              onChange={(e) => onChange(e.target.value || null)}
              placeholder={field.default !== null ? `Default: ${field.default}` : undefined}
              rows={3}
              className={inputClasses}
            />
            {field.description && <p className={descriptionClasses}>{field.description}</p>}
          </div>
        );
      }

      return (
        <div>
          <label className={labelClasses}>
            {field.title || field.name}
            {field.required && <span className="text-red-400 ml-1">*</span>}
          </label>
          <input
            type={field.name.includes('password') || field.name.includes('secret') || field.name.includes('key') ? 'password' : 'text'}
            value={(value as string) || ''}
            onChange={(e) => onChange(e.target.value || null)}
            placeholder={field.default !== null ? `Default: ${field.default}` : undefined}
            pattern={field.pattern || undefined}
            className={inputClasses}
          />
          {field.description && <p className={descriptionClasses}>{field.description}</p>}
        </div>
      );
  }
}

