import React from 'react';
import { Database } from 'lucide-react';

/**
 * ObjectStoreComponent - Represents an object store in NAT
 * 
 * Object stores (like S3) provide storage for files, documents,
 * and other binary data that agents may need to access.
 * 
 * Corresponds to: ObjectStoreRef in component_ref.py
 * Component Group: OBJECT_STORES
 */

interface ObjectStoreComponentProps {
  name: string;
  config?: {
    provider?: string;
    bucket?: string;
  };
  isSelected?: boolean;
}

export function ObjectStoreComponent({ name, config, isSelected }: ObjectStoreComponentProps) {
  return (
    <div
      className={`
        nat-component border-violet-500 bg-gradient-to-br from-violet-500/20 to-violet-600/10
        ${isSelected ? 'ring-2 ring-violet-400' : ''}
      `}
    >
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-violet-500/20">
          <Database className="text-violet-400" size={20} />
        </div>
        <div>
          <h4 className="component-title text-violet-300">{name}</h4>
          <p className="component-subtitle">Object Store</p>
        </div>
      </div>
      {config && (
        <div className="mt-3 text-xs text-gray-400 space-y-1">
          {config.provider && <div>Provider: {config.provider}</div>}
          {config.bucket && <div>Bucket: {config.bucket}</div>}
        </div>
      )}
    </div>
  );
}

