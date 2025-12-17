import React from 'react';
import { Brain } from 'lucide-react';

/**
 * LLMComponent - Represents a Large Language Model in NAT
 * 
 * LLMs are the reasoning engines for agents. They process input,
 * make decisions, and generate responses.
 * 
 * Corresponds to: LLMRef in component_ref.py
 * Component Group: LLMS
 */

interface LLMComponentProps {
  name: string;
  config?: {
    model?: string;
    provider?: string;
    temperature?: number;
    maxTokens?: number;
  };
  isSelected?: boolean;
}

export function LLMComponent({ name, config, isSelected }: LLMComponentProps) {
  return (
    <div
      className={`
        nat-component border-amber-500 bg-gradient-to-br from-amber-500/20 to-amber-600/10
        ${isSelected ? 'ring-2 ring-amber-400' : ''}
      `}
    >
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-amber-500/20">
          <Brain className="text-amber-400" size={20} />
        </div>
        <div>
          <h4 className="component-title text-amber-300">{name}</h4>
          <p className="component-subtitle">LLM</p>
        </div>
      </div>
      {config && (
        <div className="mt-3 text-xs text-gray-400 space-y-1">
          {config.model && <div>Model: {config.model}</div>}
          {config.provider && <div>Provider: {config.provider}</div>}
          {config.temperature !== undefined && <div>Temperature: {config.temperature}</div>}
        </div>
      )}
    </div>
  );
}

