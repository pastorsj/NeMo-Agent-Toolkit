import React from 'react';
import { Search } from 'lucide-react';

/**
 * RetrieverComponent - Represents a retriever in NAT
 * 
 * Retrievers are used for RAG (Retrieval-Augmented Generation) to
 * fetch relevant documents based on semantic similarity.
 * 
 * Corresponds to: RetrieverRef in component_ref.py
 * Component Group: RETRIEVERS
 */

interface RetrieverComponentProps {
  name: string;
  config?: {
    topK?: number;
    embedder?: string;
    vectorStore?: string;
  };
  isSelected?: boolean;
}

export function RetrieverComponent({ name, config, isSelected }: RetrieverComponentProps) {
  return (
    <div
      className={`
        nat-component border-teal-500 bg-gradient-to-br from-teal-500/20 to-teal-600/10
        ${isSelected ? 'ring-2 ring-teal-400' : ''}
      `}
    >
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-teal-500/20">
          <Search className="text-teal-400" size={20} />
        </div>
        <div>
          <h4 className="component-title text-teal-300">{name}</h4>
          <p className="component-subtitle">Retriever</p>
        </div>
      </div>
      {config && (
        <div className="mt-3 text-xs text-gray-400 space-y-1">
          {config.topK && <div>Top K: {config.topK}</div>}
          {config.embedder && <div>Embedder: {config.embedder}</div>}
        </div>
      )}
    </div>
  );
}

