import React from 'react';
import { FieldInfo } from '@/types/registry';

interface SchemaFormFieldProps {
  field: FieldInfo;
  value: unknown;
  onChange: (value: unknown) => void;
}

export function SchemaFormField({ field, value, onChange }: SchemaFormFieldProps) {
  const inputClasses = `
    w-full px-4 py-2.5 rounded-lg border border-gray-600 bg-gray-800 text-white
    focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent
    placeholder:text-gray-500 transition-colors
  `;

  const labelClasses = 'block text-sm font-medium text-gray-300 mb-1.5';
  const descriptionClasses = 'text-xs text-gray-500 mt-1';

  // Render enum as select
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

