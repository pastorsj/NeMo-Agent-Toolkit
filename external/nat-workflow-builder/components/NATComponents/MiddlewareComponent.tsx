import React from 'react';
import { Workflow } from 'lucide-react';

/**
 * MiddlewareComponent - Represents middleware in NAT
 * 
 * Middleware components handle request/response processing,
 * transformations, and cross-cutting concerns.
 * 
 * Corresponds to: MiddlewareRef in component_ref.py
 * Component Group: MIDDLEWARE
 */

interface MiddlewareComponentProps {
  name: string;
  config?: {
    type?: string;
    priority?: number;
  };
  isSelected?: boolean;
}

export function MiddlewareComponent({ name, config, isSelected }: MiddlewareComponentProps) {
  return (
    <div
      className={`
        nat-component border-cyan-500 bg-gradient-to-br from-cyan-500/20 to-cyan-600/10
        ${isSelected ? 'ring-2 ring-cyan-400' : ''}
      `}
    >
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-cyan-500/20">
          <Workflow className="text-cyan-400" size={20} />
        </div>
        <div>
          <h4 className="component-title text-cyan-300">{name}</h4>
          <p className="component-subtitle">Middleware</p>
        </div>
      </div>
      {config && (
        <div className="mt-3 text-xs text-gray-400 space-y-1">
          {config.type && <div>Type: {config.type}</div>}
          {config.priority !== undefined && <div>Priority: {config.priority}</div>}
        </div>
      )}
    </div>
  );
}

