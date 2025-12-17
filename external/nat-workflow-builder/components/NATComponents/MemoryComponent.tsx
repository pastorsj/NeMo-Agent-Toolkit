import React from 'react';
import { HardDrive } from 'lucide-react';

/**
 * MemoryComponent - Represents memory storage in NAT
 * 
 * Memory components provide persistent storage for agent context,
 * conversation history, and other stateful data.
 * 
 * Corresponds to: MemoryRef in component_ref.py
 * Component Group: MEMORY
 */

interface MemoryComponentProps {
  name: string;
  config?: {
    type?: string;
    capacity?: number;
  };
  isSelected?: boolean;
}

export function MemoryComponent({ name, config, isSelected }: MemoryComponentProps) {
  return (
    <div
      className={`
        nat-component border-pink-500 bg-gradient-to-br from-pink-500/20 to-pink-600/10
        ${isSelected ? 'ring-2 ring-pink-400' : ''}
      `}
    >
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-pink-500/20">
          <HardDrive className="text-pink-400" size={20} />
        </div>
        <div>
          <h4 className="component-title text-pink-300">{name}</h4>
          <p className="component-subtitle">Memory</p>
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

