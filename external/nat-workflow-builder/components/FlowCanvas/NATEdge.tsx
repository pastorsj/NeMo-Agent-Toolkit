import React from 'react';
import {
  BaseEdge,
  EdgeProps,
  getBezierPath,
  EdgeLabelRenderer,
  useReactFlow,
} from '@xyflow/react';
import { RefType, REF_TYPE_COLORS, REF_TYPE_LABELS } from '@/types/registry';
import { X } from 'lucide-react';

interface NATEdgeData {
  refType: RefType;
}

export function NATEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  selected,
  markerEnd,
}: EdgeProps) {
  const { setEdges } = useReactFlow();
  const edgeData = data as NATEdgeData | undefined;
  const refType = edgeData?.refType;
  const color = refType ? REF_TYPE_COLORS[refType] : '#6b7280';

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    curvature: 0.25,
  });

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEdges((edges) => edges.filter((edge) => edge.id !== id));
  };

  return (
    <>
      {/* Shadow/glow effect */}
      <path
        d={edgePath}
        fill="none"
        stroke={color}
        strokeWidth={selected ? 6 : 4}
        strokeOpacity={0.15}
        filter="blur(4px)"
      />

      {/* Main edge */}
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          stroke: color,
          strokeWidth: selected ? 3 : 2,
          strokeOpacity: 0.9,
        }}
      />

      {/* Animated flow particles */}
      <circle r={3} fill={color}>
        <animateMotion dur="2s" repeatCount="indefinite" path={edgePath} />
      </circle>

      {/* Label and delete button on select */}
      {selected && (
        <EdgeLabelRenderer>
          <div
            className="flex items-center gap-2 px-2 py-1 rounded-md shadow-lg"
            style={{
              position: 'absolute',
              backgroundColor: '#1f2937',
              border: `1px solid ${color}`,
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: 'all',
            }}
          >
            {refType && (
              <span className="text-[10px] font-medium" style={{ color }}>
                {REF_TYPE_LABELS[refType]}
              </span>
            )}
            <button
              onClick={handleDelete}
              className="w-4 h-4 rounded flex items-center justify-center text-gray-400 hover:text-red-400 hover:bg-red-500/20 transition-colors"
              title="Delete connection"
            >
              <X size={12} />
            </button>
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

