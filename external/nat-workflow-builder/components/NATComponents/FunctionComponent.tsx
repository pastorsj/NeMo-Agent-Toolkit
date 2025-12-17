import React from 'react';
import { Code2 } from 'lucide-react';

/**
 * FunctionComponent - Represents a function in NAT
 * 
 * Functions are custom tools that agents can call to perform actions.
 * They are the primary building blocks for agent capabilities.
 * 
 * Corresponds to: FunctionRef in component_ref.py
 * Component Group: FUNCTIONS
 */

interface FunctionComponentProps {
  name: string;
  config?: {
    description?: string;
    parameters?: Record<string, unknown>;
  };
  isSelected?: boolean;
}

export function FunctionComponent({ name, config, isSelected }: FunctionComponentProps) {
  return (
    <div
      className={`
        nat-component border-blue-500 bg-gradient-to-br from-blue-500/20 to-blue-600/10
        ${isSelected ? 'ring-2 ring-blue-400' : ''}
      `}
    >
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-blue-500/20">
          <Code2 className="text-blue-400" size={20} />
        </div>
        <div>
          <h4 className="component-title text-blue-300">{name}</h4>
          <p className="component-subtitle">Function</p>
        </div>
      </div>
      {config?.description && (
        <div className="mt-3 text-xs text-gray-400">
          {config.description}
        </div>
      )}
    </div>
  );
}

