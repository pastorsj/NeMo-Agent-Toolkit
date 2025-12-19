import React from 'react';
import { NATComponentType, NATComponentConfig } from '@/types';
import { ComponentIcon } from '@/components/NATComponents/ComponentIcon';

interface FlowDraggablePaletteItemProps {
  componentType: NATComponentType;
  config: NATComponentConfig;
  disabled?: boolean;
}

export function FlowDraggablePaletteItem({
  componentType,
  config,
  disabled = false,
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
    // Front-end and observability
    'nat-frontend': 'border-emerald-500/50 bg-emerald-500/10 hover:border-emerald-400 hover:bg-emerald-500/20',
    'nat-logger': 'border-slate-500/50 bg-slate-500/10 hover:border-slate-400 hover:bg-slate-500/20',
    'nat-telemetry': 'border-amber-500/50 bg-amber-500/10 hover:border-amber-400 hover:bg-amber-500/20',
    // Evaluation
    'nat-evaluator': 'border-sky-500/50 bg-sky-500/10 hover:border-sky-400 hover:bg-sky-500/20',
    // Finetuning components
    'nat-trainer': 'border-lime-500/50 bg-lime-500/10 hover:border-lime-400 hover:bg-lime-500/20',
    'nat-trajectory': 'border-purple-500/50 bg-purple-500/10 hover:border-purple-400 hover:bg-purple-500/20',
    'nat-adapter': 'border-stone-500/50 bg-stone-500/10 hover:border-stone-400 hover:bg-stone-500/20',
    // Workflow-level configuration containers
    'nat-workflow': 'border-lime-500/50 bg-lime-500/10 hover:border-lime-400 hover:bg-lime-500/20',
    'nat-config': 'border-gray-500/50 bg-gray-500/10 hover:border-gray-400 hover:bg-gray-500/20',
    'nat-optimizer': 'border-orange-500/50 bg-orange-500/10 hover:border-orange-400 hover:bg-orange-500/20',
    'nat-finetuner': 'border-red-500/50 bg-red-500/10 hover:border-red-400 hover:bg-red-500/20',
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
    // Front-end and observability
    'nat-frontend': 'text-emerald-400',
    'nat-logger': 'text-slate-400',
    'nat-telemetry': 'text-amber-400',
    // Evaluation
    'nat-evaluator': 'text-sky-400',
    // Finetuning components
    'nat-trainer': 'text-lime-400',
    'nat-trajectory': 'text-purple-400',
    'nat-adapter': 'text-stone-400',
    // Workflow-level configuration containers
    'nat-workflow': 'text-lime-400',
    'nat-config': 'text-gray-400',
    'nat-optimizer': 'text-orange-400',
    'nat-finetuner': 'text-red-400',
  };

  const onDragStart = (event: React.DragEvent) => {
    if (disabled) {
      event.preventDefault();
      return;
    }
    event.dataTransfer.setData('application/natcomponent', componentType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const disabledClasses = disabled
    ? 'opacity-40 cursor-not-allowed border-gray-600/30 bg-gray-600/5'
    : '';

  return (
    <div
      draggable={!disabled}
      onDragStart={onDragStart}
      className={`
        flex items-center gap-3 rounded-lg border-2 p-3 
        transition-all duration-200 select-none
        ${disabled ? disabledClasses : `cursor-grab ${colorClasses[config.color] || 'border-gray-500/50 bg-gray-500/10 hover:border-gray-400'}`}
        ${disabled ? '' : 'active:cursor-grabbing active:scale-95'}
      `}
      title={disabled ? 'Already on canvas (only one allowed)' : undefined}
    >
      <div className={`flex-shrink-0 ${disabled ? 'text-gray-600' : iconColorClasses[config.color] || 'text-gray-400'}`}>
        <ComponentIcon icon={config.icon} size={20} />
      </div>
      <div className="flex-1 min-w-0">
        <div className={`font-display text-sm font-semibold truncate ${disabled ? 'text-gray-500' : 'text-gray-200'}`}>
          {config.label}
        </div>
        <div className={`text-xs truncate ${disabled ? 'text-gray-600' : 'text-gray-400'}`}>
          {disabled ? 'Already on canvas' : config.description}
        </div>
      </div>
    </div>
  );
}

