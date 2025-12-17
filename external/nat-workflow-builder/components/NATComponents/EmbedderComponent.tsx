import React from 'react';
import { Sparkles } from 'lucide-react';

/**
 * EmbedderComponent - Represents an embedder in NAT
 * 
 * Embedders are used to generate text embeddings for semantic search
 * and vector-based retrieval operations.
 * 
 * Corresponds to: EmbedderRef in component_ref.py
 * Component Group: EMBEDDERS
 */

interface EmbedderComponentProps {
  name: string;
  config?: {
    model?: string;
    dimensions?: number;
  };
  isSelected?: boolean;
}

export function EmbedderComponent({ name, config, isSelected }: EmbedderComponentProps) {
  return (
    <div
      className={`
        nat-component border-emerald-500 bg-gradient-to-br from-emerald-500/20 to-emerald-600/10
        ${isSelected ? 'ring-2 ring-emerald-400' : ''}
      `}
    >
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-emerald-500/20">
          <Sparkles className="text-emerald-400" size={20} />
        </div>
        <div>
          <h4 className="component-title text-emerald-300">{name}</h4>
          <p className="component-subtitle">Embedder</p>
        </div>
      </div>
      {config && (
        <div className="mt-3 text-xs text-gray-400 space-y-1">
          {config.model && <div>Model: {config.model}</div>}
          {config.dimensions && <div>Dimensions: {config.dimensions}</div>}
        </div>
      )}
    </div>
  );
}

