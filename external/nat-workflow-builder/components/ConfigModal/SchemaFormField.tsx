import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown, Search } from 'lucide-react';
import { FieldInfo } from '@/types/registry';

interface SchemaFormFieldProps {
  field: FieldInfo;
  value: unknown;
  onChange: (value: unknown) => void;
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

export function SchemaFormField({ field, onChange, value }: SchemaFormFieldProps) {
  const inputClasses = `
    w-full px-4 py-2.5 rounded-lg border border-gray-600 bg-gray-800 text-white
    focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent
    placeholder:text-gray-500 transition-colors
  `;

  const labelClasses = 'block text-sm font-medium text-gray-300 mb-1.5';
  const descriptionClasses = 'text-xs text-gray-500 mt-1';

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

    case 'array':
      return (
        <div>
          <label className={labelClasses}>
            {field.title || field.name}
            {field.required && <span className="text-red-400 ml-1">*</span>}
          </label>
          <textarea
            value={Array.isArray(value) ? JSON.stringify(value, null, 2) : ''}
            onChange={(e) => {
              try {
                const parsed = JSON.parse(e.target.value);
                onChange(parsed);
              } catch {
                // Keep the raw value for editing
              }
            }}
            placeholder='["item1", "item2"]'
            rows={3}
            className={`${inputClasses} font-mono text-sm`}
          />
          {field.description && <p className={descriptionClasses}>{field.description}</p>}
          <p className="text-xs text-gray-600 mt-1">Enter as JSON array</p>
        </div>
      );

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

