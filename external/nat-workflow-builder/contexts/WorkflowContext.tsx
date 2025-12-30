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
import { ConnectionPort, RefType, RegisteredTypeInfo, FieldInfo } from '@/types/registry';

// Imported workflow state from backend
export interface ImportedConnectionPort {
  field_name: string;
  ref_type: string;
  accepts_ref_types: string[];
  required: boolean;
  is_list: boolean;
  description: string | null;
  title: string | null;
}

export interface ImportedFieldInfo {
  name: string;
  type: string;
  title: string | null;
  description: string | null;
  required: boolean;
  default: unknown;
  enum: unknown[] | null;
  minimum: number | null;
  maximum: number | null;
  pattern: string | null;
  items: Record<string, unknown> | null;
  properties: Record<string, unknown> | null;
  is_component_ref: boolean;
  ref_type: string | null;
  is_ref_list: boolean;
  options: string[] | null;
}

export interface ImportedComponent {
  id: string;
  component_type: string;
  name: string;
  position: { x: number; y: number };
  full_type: string;
  config: Record<string, unknown>;
  input_ports: ImportedConnectionPort[];
  fields: ImportedFieldInfo[];
  icon_url: string | null;
  display_name: string | null;
}

export interface ImportedConnection {
  id: string;
  source_id: string;
  target_id: string;
  target_field: string;
  ref_type: string;
}

// Environment variable types
export interface EnvVarLocation {
  path: string;
  component_type: string | null;
  component_id: string | null;
  field_name: string;
  is_list_field?: boolean;
}

export interface EnvironmentVariable {
  name: string;
  locations: EnvVarLocation[];
  value: string | null;
  is_sensitive: boolean;
  export_as_variable: boolean;
  original_reference: string;
  warning?: string | null;
  has_secret_field_usage?: boolean;
  has_non_secret_field_usage?: boolean;
  is_list_field?: boolean;
}

export interface ImportedWorkflowState {
  components: ImportedComponent[];
  connections: ImportedConnection[];
  environment_variables?: EnvironmentVariable[];
}

// Extended workflow state that includes environment variables
interface ExtendedWorkflowState extends WorkflowState {
  environmentVariables: EnvironmentVariable[];
}

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
  | { type: 'CLEAR_WORKFLOW'; payload?: undefined }
  | { type: 'LOAD_IMPORTED_STATE'; payload: { importedState: ImportedWorkflowState } }
  | { type: 'SET_ENV_VAR_VALUE'; payload: { name: string; value: string } }
  | { type: 'SET_ENV_VAR_EXPORT_MODE'; payload: { name: string; exportAsVariable: boolean } }
  | { type: 'SET_ENV_VARS'; payload: { envVars: EnvironmentVariable[] } };

// Helper function to convert PascalCase/camelCase/snake_case to human-readable format
function toHumanReadable(name: string): string {
  // Remove common suffixes like "Config", "Workflow", etc. for cleaner names
  let cleanName = name
    .replace(/_?config$/i, '')
    .replace(/_?workflow$/i, '')
    .replace(/_?workflow_?config$/i, '');
  
  // If the name already contains spaces, it's already human-readable (e.g., "ReAct Agent")
  // Just return it without applying PascalCase/camelCase conversion
  if (cleanName.includes(' ')) {
    return cleanName.trim() || name;
  }
  
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
const initialState: ExtendedWorkflowState = {
  components: [],
  connections: [],
  environmentVariables: [],
};

// Track the last added component ID for auto-configuration
let lastAddedComponentId: string | null = null;

// Reducer
function workflowReducer(state: ExtendedWorkflowState, action: WorkflowAction): ExtendedWorkflowState {
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
      
      // Generate a unique name based on the type's display_name (or local_name as fallback)
      const existingNames = state.components
        .filter((c) => c.id !== id)
        .map((c) => c.name);
      const baseName = registeredType.display_name || registeredType.local_name;
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
                  display_name: registeredType.display_name,
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

    case 'SET_ENV_VAR_VALUE': {
      const { name, value } = action.payload;
      const placeholder = `__ENV_VAR__${name}__`;

      // Find the env var to get its locations
      const envVar = state.environmentVariables.find((v) => v.name === name);

      // Try to parse value as JSON array for list fields
      let parsedValue: unknown = value;
      try {
        const parsed = JSON.parse(value);
        if (Array.isArray(parsed)) {
          parsedValue = parsed;
        }
      } catch {
        // Not JSON, use as-is
      }

      // Build a map of component_id -> field_name -> value for direct updates
      const directUpdates: Map<string, Map<string, unknown>> = new Map();
      if (envVar) {
        for (const loc of envVar.locations) {
          if (loc.component_id && loc.field_name) {
            if (!directUpdates.has(loc.component_id)) {
              directUpdates.set(loc.component_id, new Map());
            }
            // Use parsed array for list fields, string otherwise
            const valueToSet = loc.is_list_field && Array.isArray(parsedValue) ? parsedValue : parsedValue;
            directUpdates.get(loc.component_id)!.set(loc.field_name, valueToSet);
          }
        }
      }

      // Helper to recursively replace placeholder values OR apply direct updates
      const replaceInObject = (
        obj: Record<string, unknown>,
        fieldUpdates?: Map<string, unknown>
      ): Record<string, unknown> => {
        const result: Record<string, unknown> = {};
        for (const [key, val] of Object.entries(obj)) {
          // Check if this field should be directly updated
          if (fieldUpdates?.has(key)) {
            result[key] = fieldUpdates.get(key);
          } else if (typeof val === 'string') {
            // Replace the placeholder with the actual value
            result[key] = val === placeholder ? parsedValue : val;
          } else if (typeof val === 'object' && val !== null && !Array.isArray(val)) {
            result[key] = replaceInObject(val as Record<string, unknown>);
          } else if (Array.isArray(val)) {
            // Check if the array contains just the placeholder (for list-type env vars)
            if (val.length === 1 && val[0] === placeholder && Array.isArray(parsedValue)) {
              // Replace the entire array with the parsed value
              result[key] = parsedValue;
            } else {
              // Replace individual items
              result[key] = val.map((item) => {
                if (typeof item === 'string') {
                  return item === placeholder ? parsedValue : item;
                }
                if (typeof item === 'object' && item !== null) {
                  return replaceInObject(item as Record<string, unknown>);
                }
                return item;
              });
            }
          } else {
            result[key] = val;
          }
        }
        return result;
      };

      // Update environment variables
      const updatedEnvVars = state.environmentVariables.map((v) =>
        v.name === name ? { ...v, value } : v
      );

      // Update component configs - use direct field updates if available
      const updatedComponents = state.components.map((comp) => {
        const fieldUpdates = directUpdates.get(comp.id);
        return {
          ...comp,
          config: replaceInObject(comp.config, fieldUpdates),
        };
      });

      return {
        ...state,
        environmentVariables: updatedEnvVars,
        components: updatedComponents,
      };
    }

    case 'SET_ENV_VAR_EXPORT_MODE': {
      const { name, exportAsVariable } = action.payload;
      return {
        ...state,
        environmentVariables: state.environmentVariables.map((v) =>
          v.name === name ? { ...v, export_as_variable: exportAsVariable } : v
        ),
      };
    }

    case 'SET_ENV_VARS': {
      return {
        ...state,
        environmentVariables: action.payload.envVars,
      };
    }

    case 'LOAD_IMPORTED_STATE': {
      const { importedState } = action.payload;

      // Track used names to ensure uniqueness
      const usedNames: string[] = [];

      // Convert imported components to PlacedComponents
      const components: PlacedComponent[] = importedState.components.map((ic) => {
        const componentType = ic.component_type as NATComponentType;

        // Convert imported input ports to ConnectionPort format
        const inputPorts: ConnectionPort[] = ic.input_ports.map((port) => ({
          field_name: port.field_name,
          ref_type: port.ref_type as RefType,
          accepts_ref_types: port.accepts_ref_types as RefType[],
          required: port.required,
          is_list: port.is_list,
          description: port.description,
          title: port.title,
        }));

        // Convert imported fields to FieldInfo format
        const fields: FieldInfo[] = (ic.fields || []).map((field) => ({
          name: field.name,
          type: field.type,
          title: field.title,
          description: field.description,
          required: field.required,
          default: field.default,
          enum: field.enum,
          minimum: field.minimum,
          maximum: field.maximum,
          pattern: field.pattern,
          items: field.items,
          properties: field.properties,
          is_component_ref: field.is_component_ref,
          ref_type: field.ref_type as RefType | null,
          is_ref_list: field.is_ref_list,
          options: field.options,
        }));

        // Generate unique name to avoid duplicates (e.g., two "Milvus Retriever" components)
        const baseName = ic.name;
        const uniqueName = generateUniqueName(baseName, usedNames);
        usedNames.push(uniqueName);

        return {
          id: ic.id,
          type: componentType,
          position: { x: ic.position.x, y: ic.position.y },
          config: ic.config,
          name: uniqueName,
          // Populate registeredType with info from the import
          registeredType: ic.full_type
            ? {
                full_type: ic.full_type,
                local_name: ic.full_type.split('/').pop() || ic.full_type,
                display_name: ic.display_name,
                icon_url: ic.icon_url,
              }
            : undefined,
          inputPorts,
          outputRefType: COMPONENT_TO_REF_TYPE[componentType],
          fields,
        };
      });

      // Convert imported connections to ComponentConnections
      const connections: ComponentConnection[] = importedState.connections.map((ic) => ({
        id: ic.id,
        sourceId: ic.source_id,
        targetId: ic.target_id,
        targetField: ic.target_field,
        refType: ic.ref_type as RefType,
      }));

      // Load environment variables if present
      const environmentVariables = importedState.environment_variables || [];

      return {
        components,
        connections,
        environmentVariables,
      };
    }

    default:
      return state;
  }
}

// Context type
interface WorkflowContextType {
  state: ExtendedWorkflowState;
  // Direct access to state for export
  components: PlacedComponent[];
  connections: ComponentConnection[];
  environmentVariables: EnvironmentVariable[];
  // Computed properties for env vars
  unresolvedEnvVars: EnvironmentVariable[];
  hasUnresolvedEnvVars: boolean;
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
  loadImportedState: (importedState: ImportedWorkflowState) => void;
  getConnectionsForComponent: (componentId: string) => ComponentConnection[];
  getConnectionForPort: (componentId: string, fieldName: string) => ComponentConnection | undefined;
  getComponentsOfType: (refType: RefType) => PlacedComponent[];
  getAndClearLastAddedComponent: () => PlacedComponent | undefined;
  getPlacedSingleInstanceTypes: () => Set<NATComponentType>;
  isTypeDisabled: (componentType: NATComponentType) => boolean;
  // Environment variable functions
  setEnvVarValue: (name: string, value: string) => void;
  setEnvVarExportMode: (name: string, exportAsVariable: boolean) => void;
  setEnvVars: (envVars: EnvironmentVariable[]) => void;
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

  const loadImportedState = useCallback((importedState: ImportedWorkflowState) => {
    dispatch({ type: 'LOAD_IMPORTED_STATE', payload: { importedState } });
  }, []);

  // Environment variable functions
  const setEnvVarValue = useCallback((name: string, value: string) => {
    dispatch({ type: 'SET_ENV_VAR_VALUE', payload: { name, value } });
  }, []);

  const setEnvVarExportMode = useCallback((name: string, exportAsVariable: boolean) => {
    dispatch({ type: 'SET_ENV_VAR_EXPORT_MODE', payload: { name, exportAsVariable } });
  }, []);

  const setEnvVars = useCallback((envVars: EnvironmentVariable[]) => {
    dispatch({ type: 'SET_ENV_VARS', payload: { envVars } });
  }, []);

  // Computed values for environment variables
  const unresolvedEnvVars = state.environmentVariables.filter((v) => v.value === null);
  const hasUnresolvedEnvVars = unresolvedEnvVars.length > 0;

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
        // Direct access to state for export
        components: state.components,
        connections: state.connections,
        environmentVariables: state.environmentVariables,
        unresolvedEnvVars,
        hasUnresolvedEnvVars,
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
        loadImportedState,
        getConnectionsForComponent,
        getConnectionForPort,
        getComponentsOfType,
        getAndClearLastAddedComponent,
        getPlacedSingleInstanceTypes,
        isTypeDisabled,
        setEnvVarValue,
        setEnvVarExportMode,
        setEnvVars,
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
