'use client';

import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  Controls,
  Background,
  BackgroundVariant,
  MiniMap,
  Connection,
  Edge,
  Node,
  NodeTypes,
  EdgeTypes,
  ConnectionMode,
  useReactFlow,
  Panel,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { useWorkflow } from '@/contexts/WorkflowContext';
import { useRegistry } from '@/contexts/RegistryContext';
import { NATComponentType, NAT_COMPONENTS } from '@/types';
import { RefType, REF_TYPE_COLORS } from '@/types/registry';
import { NATNode } from './NATNode';
import { NATEdge } from './NATEdge';
import { ConfigModal } from '@/components/ConfigModal';
import { PlacedComponent } from '@/types';

// Custom node types
const nodeTypes: NodeTypes = {
  natNode: NATNode,
};

// Custom edge types
const edgeTypes: EdgeTypes = {
  natEdge: NATEdge,
};

// Connection info for display in nodes
interface ConnectionInfo {
  connectionId: string;
  fieldName: string;
  sourceComponentId: string;
  sourceComponentName: string;
}

// Convert workflow state to React Flow nodes
function workflowToNodes(
  components: PlacedComponent[],
  connections: { id: string; sourceId: string; targetId: string; targetField: string; refType: RefType }[],
  onConfigure: (_component: PlacedComponent) => void,
  onDelete: (_componentId: string) => void,
  onDeleteConnection: (_connectionId: string) => void,
  onNameChange: (_componentId: string, _newName: string) => void
): Node[] {
  // Build a map of component IDs to names for quick lookup
  const componentNameMap = new Map(components.map((c) => [c.id, c.name]));

  return components.map((comp) => {
    // Get connections where this component is the target
    const incomingConnections: ConnectionInfo[] = connections
      .filter((conn) => conn.targetId === comp.id)
      .map((conn) => ({
        connectionId: conn.id,
        fieldName: conn.targetField,
        sourceComponentId: conn.sourceId,
        sourceComponentName: componentNameMap.get(conn.sourceId) || 'Unknown',
      }));

    return {
      id: comp.id,
      type: 'natNode',
      position: comp.position,
      data: {
        component: comp,
        label: comp.name,
        connections: incomingConnections,
        onConfigure: () => onConfigure(comp),
        onDelete: () => onDelete(comp.id),
        onDeleteConnection: (connectionId: string) => onDeleteConnection(connectionId),
        onNameChange: (newName: string) => onNameChange(comp.id, newName),
      },
    };
  });
}

// Convert workflow connections to React Flow edges
function workflowToEdges(
  connections: { id: string; sourceId: string; targetId: string; targetField: string; refType: RefType }[]
): Edge[] {
  return connections.map((conn) => ({
    id: conn.id,
    source: conn.sourceId,
    target: conn.targetId,
    targetHandle: conn.targetField,
    type: 'natEdge',
    data: {
      refType: conn.refType,
    },
    style: {
      stroke: REF_TYPE_COLORS[conn.refType] || '#6b7280',
      strokeWidth: 2,
    },
    animated: true,
  }));
}

function FlowCanvasInner() {
  const {
    state,
    addComponent,
    removeComponent,
    updateComponentPosition,
    updateComponentName,
    addConnection,
    removeConnection,
    getAndClearLastAddedComponent,
  } = useWorkflow();
  const { connected } = useRegistry();

  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();
  const [configComponent, setConfigComponent] = useState<PlacedComponent | null>(null);

  // Handler to open config modal
  const handleOpenConfig = useCallback((component: PlacedComponent) => {
    setConfigComponent(component);
  }, []);

  // Handler to delete a component
  const handleDeleteComponent = useCallback((componentId: string) => {
    removeComponent(componentId);
  }, [removeComponent]);

  // Handler to delete a connection
  const handleDeleteConnection = useCallback((connectionId: string) => {
    removeConnection(connectionId);
  }, [removeConnection]);

  // Handler to rename a component
  const handleNameChange = useCallback((componentId: string, newName: string) => {
    updateComponentName(componentId, newName);
  }, [updateComponentName]);

  // Convert workflow state to React Flow format
  const nodes = useMemo(
    () => workflowToNodes(
      state.components,
      state.connections,
      handleOpenConfig,
      handleDeleteComponent,
      handleDeleteConnection,
      handleNameChange
    ),
    [state.components, state.connections, handleOpenConfig, handleDeleteComponent, handleDeleteConnection, handleNameChange]
  );
  const edges = useMemo(() => workflowToEdges(state.connections), [state.connections]);

  // Handle node position changes
  const onNodesChange = useCallback(
    (changes: any[]) => {
      changes.forEach((change) => {
        if (change.type === 'position' && change.position) {
          updateComponentPosition(change.id, change.position);
        }
      });
    },
    [updateComponentPosition]
  );

  // Handle edge changes
  const onEdgesChange = useCallback(
    (changes: any[]) => {
      changes.forEach((change) => {
        if (change.type === 'remove') {
          removeConnection(change.id);
        }
      });
    },
    [removeConnection]
  );

  // Handle new connections
  const onConnect = useCallback(
    (connection: Connection) => {
      if (!connection.source || !connection.target) return;

      // Find the source component to get its output ref type
      const sourceComp = state.components.find((c) => c.id === connection.source);
      if (!sourceComp?.outputRefType) return;

      // Find the target component and port
      const targetComp = state.components.find((c) => c.id === connection.target);
      const targetPort = targetComp?.inputPorts.find(
        (p) => p.field_name === connection.targetHandle
      );
      if (!targetPort) return;

      // Validate connection type - check if source's refType is in accepted types
      const acceptedTypes = targetPort.accepts_ref_types?.length 
        ? targetPort.accepts_ref_types 
        : [targetPort.ref_type];
      
      if (!acceptedTypes.includes(sourceComp.outputRefType)) {
        console.warn(`Incompatible connection types: ${sourceComp.outputRefType} not in ${acceptedTypes.join(', ')}`);
        return;
      }

      addConnection(
        connection.source,
        connection.target,
        connection.targetHandle || '',
        sourceComp.outputRefType // Use the actual source's refType
      );
    },
    [state.components, addConnection]
  );

  // Handle dropping new nodes
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/natcomponent') as NATComponentType;
      if (!type || !NAT_COMPONENTS[type]) return;

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      addComponent(type, position);
    },
    [screenToFlowPosition, addComponent]
  );

  // Handle node double-click to configure
  const onNodeDoubleClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      const component = state.components.find((c) => c.id === node.id);
      if (component) {
        setConfigComponent(component);
      }
    },
    [state.components]
  );

  // Handle node deletion
  const onNodesDelete = useCallback(
    (nodesToDelete: Node[]) => {
      nodesToDelete.forEach((node) => {
        removeComponent(node.id);
      });
    },
    [removeComponent]
  );

  // Auto-open config modal for newly added components
  useEffect(() => {
    const newComponent = getAndClearLastAddedComponent();
    if (newComponent) {
      setTimeout(() => {
        setConfigComponent(newComponent);
      }, 100);
    }
  }, [state.components.length, getAndClearLastAddedComponent]);

  // Validate connection before allowing it
  const isValidConnection = useCallback(
    (connection: Edge | Connection) => {
      const source = connection.source;
      const target = connection.target;
      const targetHandle = 'targetHandle' in connection ? connection.targetHandle : null;
      
      if (!source || !target) return false;
      if (source === target) return false;

      const sourceComp = state.components.find((c) => c.id === source);
      const targetComp = state.components.find((c) => c.id === target);

      if (!sourceComp?.outputRefType) return false;
      if (!targetComp?.config?._selected_type) return false; // Target must be configured

      const targetPort = targetComp.inputPorts.find(
        (p) => p.field_name === targetHandle
      );
      if (!targetPort) return false;

      // Check if source's refType is in the port's accepted types
      const acceptedTypes = targetPort.accepts_ref_types?.length 
        ? targetPort.accepts_ref_types 
        : [targetPort.ref_type];
      
      if (!acceptedTypes.includes(sourceComp.outputRefType)) {
        return false;
      }

      // Check existing connections for this port
      const existingConnections = state.connections.filter(
        (c) => c.targetId === target && c.targetField === targetHandle
      );

      // For non-list fields, only allow one connection
      if (!targetPort.is_list && existingConnections.length > 0) {
        return false;
      }

      // Prevent duplicate connection from the same source
      const duplicateFromSource = existingConnections.some((c) => c.sourceId === source);
      if (duplicateFromSource) {
        return false;
      }

      return true;
    },
    [state.components, state.connections]
  );

  const handleSaveConfig = () => {
    // Config is saved via the modal's internal logic
  };

  return (
    <div className="flex-1 h-full" ref={reactFlowWrapper}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeDoubleClick={onNodeDoubleClick}
        onNodesDelete={onNodesDelete}
        onDragOver={onDragOver}
        onDrop={onDrop}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        connectionMode={ConnectionMode.Loose}
        isValidConnection={isValidConnection}
        defaultViewport={{ x: 100, y: 50, zoom: 0.9 }}
        snapToGrid
        snapGrid={[16, 16]}
        deleteKeyCode={['Backspace', 'Delete']}
        defaultEdgeOptions={{
          type: 'natEdge',
          animated: true,
        }}
        proOptions={{ hideAttribution: true }}
        style={{
          backgroundColor: '#0f0f0f',
        }}
      >
        <Controls
          className="!bg-gray-800 !border-gray-700 !rounded-lg !shadow-lg"
          showZoom
          showFitView
          showInteractive
        />
        <MiniMap
          className="!bg-gray-800 !border-gray-700 !rounded-lg"
          nodeColor={(node) => {
            const comp = node.data?.component as PlacedComponent;
            if (!comp) return '#6b7280';
            const config = NAT_COMPONENTS[comp.type];
            const colorMap: Record<string, string> = {
              'nat-agent': '#a855f7',
              'nat-llm': '#f59e0b',
              'nat-function': '#3b82f6',
              'nat-function-group': '#6366f1',
              'nat-embedder': '#10b981',
              'nat-memory': '#ec4899',
              'nat-retriever': '#14b8a6',
              'nat-object-store': '#8b5cf6',
              'nat-auth': '#ef4444',
              'nat-middleware': '#06b6d4',
            };
            return colorMap[config.color] || '#6b7280';
          }}
          maskColor="rgba(0, 0, 0, 0.8)"
        />
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="#2a2a2a"
        />

        {/* Connection status panel */}
        <Panel position="top-right" className="!m-4">
          <div className="flex items-center gap-2 px-3 py-2 bg-gray-800/90 border border-gray-700 rounded-lg backdrop-blur-sm">
            <div
              className={`w-2 h-2 rounded-full ${
                connected ? 'bg-green-500' : 'bg-red-500'
              }`}
            />
            <span className="text-xs text-gray-400">
              {connected ? 'API Connected' : 'API Disconnected'}
            </span>
          </div>
        </Panel>

        {/* Instructions panel */}
        {state.components.length === 0 && (
          <Panel position="top-center" className="!mt-20">
            <div className="text-center p-6 bg-gray-800/80 border border-gray-700 rounded-xl backdrop-blur-sm max-w-md">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gray-700/50 flex items-center justify-center">
                <svg
                  className="w-8 h-8 text-gray-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                  />
                </svg>
              </div>
              <h3 className="font-display text-lg font-semibold text-gray-300 mb-2">
                Start Building Your Agent
              </h3>
              <p className="text-sm text-gray-500">
                Drag components from the sidebar and drop them here.
                Connect them by dragging from output ports to input ports.
              </p>
            </div>
          </Panel>
        )}
      </ReactFlow>

      {/* Configuration Modal */}
      {configComponent && (
        <ConfigModal
          component={configComponent}
          onClose={() => setConfigComponent(null)}
          onSave={handleSaveConfig}
        />
      )}
    </div>
  );
}

export function FlowCanvas() {
  return (
    <ReactFlowProvider>
      <FlowCanvasInner />
    </ReactFlowProvider>
  );
}

