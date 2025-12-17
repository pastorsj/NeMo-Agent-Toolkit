import React from 'react';
import { Shield } from 'lucide-react';

/**
 * AuthenticationComponent - Represents an authentication provider in NAT
 * 
 * Authentication components handle API authentication for secure
 * access to external services and APIs.
 * 
 * Corresponds to: AuthenticationRef in component_ref.py
 * Component Group: AUTHENTICATION
 */

interface AuthenticationComponentProps {
  name: string;
  config?: {
    type?: string;
    provider?: string;
  };
  isSelected?: boolean;
}

export function AuthenticationComponent({ name, config, isSelected }: AuthenticationComponentProps) {
  return (
    <div
      className={`
        nat-component border-red-500 bg-gradient-to-br from-red-500/20 to-red-600/10
        ${isSelected ? 'ring-2 ring-red-400' : ''}
      `}
    >
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-red-500/20">
          <Shield className="text-red-400" size={20} />
        </div>
        <div>
          <h4 className="component-title text-red-300">{name}</h4>
          <p className="component-subtitle">Authentication</p>
        </div>
      </div>
      {config?.type && (
        <div className="mt-3 text-xs text-gray-400">
          Type: {config.type}
        </div>
      )}
    </div>
  );
}

