import React, { createContext, useContext, useReducer, useCallback, ReactNode } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  PlacedComponent,
  ComponentConnection,
  WorkflowState,
  NATComponentType,
  NAT_COMPONENTS,
  COMPONENT_TO_REF_TYPE,
  SINGLE_INSTANCE_TYPES,
} from '@/types';
import { ConnectionPort, RefType, RegisteredTypeInfo } from '@/types/registry';

// Action types
type WorkflowAction =
  | { type: 'ADD_COMPONENT'; payload: { componentType: NATComponentType; position: { x: number; y: number }; id?: string } }
  | { type: 'REMOVE_COMPONENT'; payload: { id: string } }
  | { type: 'UPDATE_COMPONENT_POSITION'; payload: { id: string; position: { x: number; y: number } } }
  | { type: 'UPDATE_COMPONENT_CONFIG'; payload: { id: string; config: Record<string, unknown> } }
  | { type: 'UPDATE_COMPONENT_NAME'; payload: { id: string; name: string } }
  | {
      type: 'UPDATE_COMPONENT_REGISTERED_TYPE';
      payload: { id: string; registeredType: RegisteredTypeInfo; config: Record<string, unknown> };
    }
  | { type: 'ADD_CONNECTION'; payload: { sourceId: string; targetId: string; targetField: string; refType: RefType } }
  | { type: 'REMOVE_CONNECTION'; payload: { id: string } }
  | { type: 'REMOVE_CONNECTIONS_FOR_FIELD'; payload: { componentId: string; fieldName: string } }
  | { type: 'CLEAR_WORKFLOW'; payload?: undefined };

// Helper function to convert PascalCase/camelCase/snake_case to human-readable format
function toHumanReadable(name: string): string {
  // Remove common suffixes like "Config", "Workflow", etc. for cleaner names
  let cleanName = name
    .replace(/_?config$/i, '')
    .replace(/_?workflow$/i, '')
    .replace(/_?workflow_?config$/i, '');
  
  // Check if it's snake_case (contains underscores)
  if (cleanName.includes('_')) {
    // Convert snake_case to Title Case
    cleanName = cleanName
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ')
      .trim();
  } else {
    // Handle PascalCase/camelCase
    // Insert spaces before capital letters (handle acronyms like "LLM", "API", "ReAct")
    cleanName = cleanName
      // Handle transitions from lowercase to uppercase
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      // Handle transitions from uppercase acronym to uppercase start of new word
      .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
      .trim();
  }
  
  return cleanName || name; // Fallback to original if result is empty
}

// Helper function to generate a unique name based on type
function generateUniqueName(baseName: string, existingNames: string[]): string {
  // Convert to human-readable format
  const humanReadable = toHumanReadable(baseName);
  
  // Check if base name is already taken
  if (!existingNames.includes(humanReadable)) {
    return humanReadable;
  }
  
  // Find the next available number
  let counter = 2;
  while (existingNames.includes(`${humanReadable} (${counter})`)) {
    counter++;
  }
  
  return `${humanReadable} (${counter})`;
}

// Initial state
const initialState: WorkflowState = {
  components: [],
  connections: [],
};

// Track the last added component ID for auto-configuration
let lastAddedComponentId: string | null = null;

// Reducer
function workflowReducer(state: WorkflowState, action: WorkflowAction): WorkflowState {
  switch (action.type) {
    case 'ADD_COMPONENT': {
      const { componentType, position, id } = action.payload;
      const config = NAT_COMPONENTS[componentType];
      const newComponent: PlacedComponent = {
        id: id || uuidv4(),
        type: componentType,
        position,
        config: {},
        name: `${config.label} ${state.components.filter((c) => c.type === componentType).length + 1}`,
        inputPorts: [],
        outputRefType: COMPONENT_TO_REF_TYPE[componentType],
      };
      return {
        ...state,
        components: [...state.components, newComponent],
      };
    }

    case 'REMOVE_COMPONENT': {
      const { id } = action.payload;
      return {
        ...state,
        components: state.components.filter((c) => c.id !== id),
        connections: state.connections.filter((conn) => conn.sourceId !== id && conn.targetId !== id),
      };
    }

    case 'UPDATE_COMPONENT_POSITION': {
      const { id, position } = action.payload;
      return {
        ...state,
        components: state.components.map((c) => (c.id === id ? { ...c, position } : c)),
      };
    }

    case 'UPDATE_COMPONENT_CONFIG': {
      const { id, config } = action.payload;
      return {
        ...state,
        components: state.components.map((c) => (c.id === id ? { ...c, config: { ...c.config, ...config } } : c)),
      };
    }

    case 'UPDATE_COMPONENT_NAME': {
      const { id, name } = action.payload;
      // Get existing names excluding the current component
      const existingNames = state.components
        .filter((c) => c.id !== id)
        .map((c) => c.name);
      
      // If the new name is already taken, generate a unique one
      const uniqueName = existingNames.includes(name)
        ? generateUniqueName(name, existingNames)
        : name;
      
      return {
        ...state,
        components: state.components.map((c) => (c.id === id ? { ...c, name: uniqueName } : c)),
      };
    }

    case 'UPDATE_COMPONENT_REGISTERED_TYPE': {
      const { id, registeredType, config } = action.payload;
      
      // Generate a unique name based on the type's local_name
      const existingNames = state.components
        .filter((c) => c.id !== id)
        .map((c) => c.name);
      const baseName = registeredType.local_name;
      const newName = generateUniqueName(baseName, existingNames);
      
      return {
        ...state,
        components: state.components.map((c) =>
          c.id === id
            ? {
                ...c,
                name: newName,
                registeredType: {
                  full_type: registeredType.full_type,
                  local_name: registeredType.local_name,
                  icon_url: registeredType.icon_url,
                },
                inputPorts: registeredType.input_ports,
                config,
              }
            : c
        ),
        // Clear any existing connections for this component's input ports when type changes
        connections: state.connections.filter((conn) => conn.targetId !== id),
      };
    }

    case 'ADD_CONNECTION': {
      const { sourceId, targetId, targetField, refType } = action.payload;

      // Find target component and its port
      const targetComponent = state.components.find((c) => c.id === targetId);
      const targetPort = targetComponent?.inputPorts.find((p) => p.field_name === targetField);

      // For list fields, allow multiple connections; for single fields, prevent duplicates
      const existingConnections = state.connections.filter(
        (c) => c.targetId === targetId && c.targetField === targetField
      );
      
      if (!targetPort?.is_list && existingConnections.length > 0) {
        // Single field already has a connection
        return state;
      }

      // Prevent duplicate connection from same source to same field
      const duplicateFromSource = existingConnections.some((c) => c.sourceId === sourceId);
      if (duplicateFromSource) return state;

      // Validate that source provides an acceptable refType
      const sourceComponent = state.components.find((c) => c.id === sourceId);
      if (!sourceComponent?.outputRefType) {
        console.warn('Invalid connection: source does not provide a refType');
        return state;
      }

      // Check if the source's refType is accepted by the target port
      const acceptedTypes = targetPort?.accepts_ref_types || [targetPort?.ref_type].filter(Boolean);
      if (!acceptedTypes.includes(sourceComponent.outputRefType)) {
        console.warn(`Invalid connection: source provides ${sourceComponent.outputRefType} but port accepts ${acceptedTypes.join(', ')}`);
        return state;
      }

      const newConnection: ComponentConnection = {
        id: uuidv4(),
        sourceId,
        targetId,
        targetField,
        refType: sourceComponent.outputRefType, // Use the actual source's refType
      };
      return {
        ...state,
        connections: [...state.connections, newConnection],
      };
    }

    case 'REMOVE_CONNECTION': {
      const { id } = action.payload;
      return {
        ...state,
        connections: state.connections.filter((c) => c.id !== id),
      };
    }

    case 'REMOVE_CONNECTIONS_FOR_FIELD': {
      const { componentId, fieldName } = action.payload;
      return {
        ...state,
        connections: state.connections.filter((c) => !(c.targetId === componentId && c.targetField === fieldName)),
      };
    }

    case 'CLEAR_WORKFLOW': {
      return initialState;
    }

    default:
      return state;
  }
}

// Context type
interface WorkflowContextType {
  state: WorkflowState;
  addComponent: (componentType: NATComponentType, position: { x: number; y: number }) => void;
  addComponentWithId: (componentType: NATComponentType, position: { x: number; y: number }) => string;
  removeComponent: (id: string) => void;
  updateComponentPosition: (id: string, position: { x: number; y: number }) => void;
  updateComponentConfig: (id: string, config: Record<string, unknown>) => void;
  updateComponentName: (id: string, name: string) => void;
  updateComponentRegisteredType: (id: string, registeredType: RegisteredTypeInfo, config: Record<string, unknown>) => void;
  addConnection: (sourceId: string, targetId: string, targetField: string, refType: RefType) => void;
  removeConnection: (id: string) => void;
  removeConnectionsForField: (componentId: string, fieldName: string) => void;
  clearWorkflow: () => void;
  getConnectionsForComponent: (componentId: string) => ComponentConnection[];
  getConnectionForPort: (componentId: string, fieldName: string) => ComponentConnection | undefined;
  getComponentsOfType: (refType: RefType) => PlacedComponent[];
  getAndClearLastAddedComponent: () => PlacedComponent | undefined;
  getPlacedSingleInstanceTypes: () => Set<NATComponentType>;
  isTypeDisabled: (componentType: NATComponentType) => boolean;
}

const WorkflowContext = createContext<WorkflowContextType | undefined>(undefined);

// Provider component
export function WorkflowProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(workflowReducer, initialState);

  const addComponent = useCallback((componentType: NATComponentType, position: { x: number; y: number }) => {
    const newId = uuidv4();
    lastAddedComponentId = newId;
    dispatch({ type: 'ADD_COMPONENT', payload: { componentType, position, id: newId } });
  }, []);

  // Add a component and return its ID (useful for immediately connecting)
  const addComponentWithId = useCallback((componentType: NATComponentType, position: { x: number; y: number }): string => {
    const newId = uuidv4();
    lastAddedComponentId = newId;
    dispatch({ type: 'ADD_COMPONENT', payload: { componentType, position, id: newId } });
    return newId;
  }, []);

  const removeComponent = useCallback((id: string) => {
    dispatch({ type: 'REMOVE_COMPONENT', payload: { id } });
  }, []);

  const updateComponentPosition = useCallback((id: string, position: { x: number; y: number }) => {
    dispatch({ type: 'UPDATE_COMPONENT_POSITION', payload: { id, position } });
  }, []);

  const updateComponentConfig = useCallback((id: string, config: Record<string, unknown>) => {
    dispatch({ type: 'UPDATE_COMPONENT_CONFIG', payload: { id, config } });
  }, []);

  const updateComponentName = useCallback((id: string, name: string) => {
    dispatch({ type: 'UPDATE_COMPONENT_NAME', payload: { id, name } });
  }, []);

  const updateComponentRegisteredType = useCallback(
    (id: string, registeredType: RegisteredTypeInfo, config: Record<string, unknown>) => {
      dispatch({ type: 'UPDATE_COMPONENT_REGISTERED_TYPE', payload: { id, registeredType, config } });
    },
    []
  );

  const addConnection = useCallback((sourceId: string, targetId: string, targetField: string, refType: RefType) => {
    dispatch({ type: 'ADD_CONNECTION', payload: { sourceId, targetId, targetField, refType } });
  }, []);

  const removeConnection = useCallback((id: string) => {
    dispatch({ type: 'REMOVE_CONNECTION', payload: { id } });
  }, []);

  const removeConnectionsForField = useCallback((componentId: string, fieldName: string) => {
    dispatch({ type: 'REMOVE_CONNECTIONS_FOR_FIELD', payload: { componentId, fieldName } });
  }, []);

  const clearWorkflow = useCallback(() => {
    dispatch({ type: 'CLEAR_WORKFLOW' });
  }, []);

  // Helper to get all connections for a component
  const getConnectionsForComponent = useCallback(
    (componentId: string) => {
      return state.connections.filter((c) => c.sourceId === componentId || c.targetId === componentId);
    },
    [state.connections]
  );

  // Helper to get connection for a specific input port
  const getConnectionForPort = useCallback(
    (componentId: string, fieldName: string) => {
      return state.connections.find((c) => c.targetId === componentId && c.targetField === fieldName);
    },
    [state.connections]
  );

  // Helper to get all components that provide a specific RefType
  const getComponentsOfType = useCallback(
    (refType: RefType) => {
      return state.components.filter((c) => c.outputRefType === refType);
    },
    [state.components]
  );

  // Get and clear the last added component (for auto-opening config modal)
  const getAndClearLastAddedComponent = useCallback(() => {
    if (lastAddedComponentId) {
      const component = state.components.find((c) => c.id === lastAddedComponentId);
      lastAddedComponentId = null;
      return component;
    }
    return undefined;
  }, [state.components]);

  // Get all single-instance types that are currently on the canvas
  const getPlacedSingleInstanceTypes = useCallback(() => {
    const placedTypes = new Set<NATComponentType>();
    state.components.forEach((component) => {
      if (SINGLE_INSTANCE_TYPES.has(component.type)) {
        placedTypes.add(component.type);
      }
    });
    return placedTypes;
  }, [state.components]);

  // Check if a component type should be disabled in the sidebar
  const isTypeDisabled = useCallback(
    (componentType: NATComponentType) => {
      if (!SINGLE_INSTANCE_TYPES.has(componentType)) {
        return false;
      }
      return state.components.some((c) => c.type === componentType);
    },
    [state.components]
  );

  return (
    <WorkflowContext.Provider
      value={{
        state,
        addComponent,
        addComponentWithId,
        removeComponent,
        updateComponentPosition,
        updateComponentConfig,
        updateComponentName,
        updateComponentRegisteredType,
        addConnection,
        removeConnection,
        removeConnectionsForField,
        clearWorkflow,
        getConnectionsForComponent,
        getConnectionForPort,
        getComponentsOfType,
        getAndClearLastAddedComponent,
        getPlacedSingleInstanceTypes,
        isTypeDisabled,
      }}
    >
      {children}
    </WorkflowContext.Provider>
  );
}

// Hook to use the workflow context
export function useWorkflow() {
  const context = useContext(WorkflowContext);
  if (context === undefined) {
    throw new Error('useWorkflow must be used within a WorkflowProvider');
  }
  return context;
}
