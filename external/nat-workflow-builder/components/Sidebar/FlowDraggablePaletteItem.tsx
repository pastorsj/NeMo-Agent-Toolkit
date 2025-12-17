import React from 'react';
import { NATComponentType, NATComponentConfig } from '@/types';
import { ComponentIcon } from '@/components/NATComponents/ComponentIcon';

interface FlowDraggablePaletteItemProps {
  componentType: NATComponentType;
  config: NATComponentConfig;
}

export function FlowDraggablePaletteItem({
  componentType,
  config,
}: FlowDraggablePaletteItemProps) {
  const colorClasses: Record<string, string> = {
    'nat-agent': 'border-purple-500/50 bg-purple-500/10 hover:border-purple-400 hover:bg-purple-500/20',
    'nat-embedder': 'border-emerald-500/50 bg-emerald-500/10 hover:border-emerald-400 hover:bg-emerald-500/20',
    'nat-function': 'border-blue-500/50 bg-blue-500/10 hover:border-blue-400 hover:bg-blue-500/20',
    'nat-function-group': 'border-indigo-500/50 bg-indigo-500/10 hover:border-indigo-400 hover:bg-indigo-500/20',
    'nat-llm': 'border-amber-500/50 bg-amber-500/10 hover:border-amber-400 hover:bg-amber-500/20',
    'nat-memory': 'border-pink-500/50 bg-pink-500/10 hover:border-pink-400 hover:bg-pink-500/20',
    'nat-object-store': 'border-violet-500/50 bg-violet-500/10 hover:border-violet-400 hover:bg-violet-500/20',
    'nat-retriever': 'border-teal-500/50 bg-teal-500/10 hover:border-teal-400 hover:bg-teal-500/20',
    'nat-auth': 'border-red-500/50 bg-red-500/10 hover:border-red-400 hover:bg-red-500/20',
    'nat-middleware': 'border-cyan-500/50 bg-cyan-500/10 hover:border-cyan-400 hover:bg-cyan-500/20',
  };

  const iconColorClasses: Record<string, string> = {
    'nat-agent': 'text-purple-400',
    'nat-embedder': 'text-emerald-400',
    'nat-function': 'text-blue-400',
    'nat-function-group': 'text-indigo-400',
    'nat-llm': 'text-amber-400',
    'nat-memory': 'text-pink-400',
    'nat-object-store': 'text-violet-400',
    'nat-retriever': 'text-teal-400',
    'nat-auth': 'text-red-400',
    'nat-middleware': 'text-cyan-400',
  };

  const onDragStart = (event: React.DragEvent) => {
    event.dataTransfer.setData('application/natcomponent', componentType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      draggable
      onDragStart={onDragStart}
      className={`
        flex items-center gap-3 rounded-lg border-2 p-3 cursor-grab 
        transition-all duration-200 select-none
        ${colorClasses[config.color] || 'border-gray-500/50 bg-gray-500/10 hover:border-gray-400'}
        active:cursor-grabbing active:scale-95
      `}
    >
      <div className={`flex-shrink-0 ${iconColorClasses[config.color] || 'text-gray-400'}`}>
        <ComponentIcon icon={config.icon} size={20} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="font-display text-sm font-semibold text-gray-200 truncate">
          {config.label}
        </div>
        <div className="text-xs text-gray-400 truncate">{config.description}</div>
      </div>
    </div>
  );
}

