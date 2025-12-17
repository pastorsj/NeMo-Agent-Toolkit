import React from 'react';
import { Layers } from 'lucide-react';

/**
 * FunctionGroupComponent - Represents a function group in NAT
 * 
 * Function groups organize related functions together for better
 * management and agent tool selection.
 * 
 * Corresponds to: FunctionGroupRef in component_ref.py
 * Component Group: FUNCTION_GROUPS
 */

interface FunctionGroupComponentProps {
  name: string;
  config?: {
    functions?: string[];
    description?: string;
  };
  isSelected?: boolean;
}

export function FunctionGroupComponent({ name, config, isSelected }: FunctionGroupComponentProps) {
  return (
    <div
      className={`
        nat-component border-indigo-500 bg-gradient-to-br from-indigo-500/20 to-indigo-600/10
        ${isSelected ? 'ring-2 ring-indigo-400' : ''}
      `}
    >
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-indigo-500/20">
          <Layers className="text-indigo-400" size={20} />
        </div>
        <div>
          <h4 className="component-title text-indigo-300">{name}</h4>
          <p className="component-subtitle">Function Group</p>
        </div>
      </div>
      {config?.functions && (
        <div className="mt-3 text-xs text-gray-400">
          {config.functions.length} function(s)
        </div>
      )}
    </div>
  );
}

