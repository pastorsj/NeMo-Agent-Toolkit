import { ConnectionPort, RefType } from './registry';

// NAT Component Types - Derived from component_ref.py
// Focused on core agent building components
export type NATComponentType =
  | 'agent'
  | 'embedder'
  | 'function'
  | 'function_group'
  | 'llm'
  | 'memory'
  | 'object_store'
  | 'retriever'
  | 'authentication'
  | 'middleware';

// Component configuration for each type
export interface NATComponentConfig {
  type: NATComponentType;
  label: string;
  description: string;
  color: string;
  icon: string;
  category: 'core' | 'data' | 'ai' | 'infrastructure';
}

// A placed component instance on the canvas
export interface PlacedComponent {
  id: string;
  type: NATComponentType;
  position: { x: number; y: number };
  config: Record<string, unknown>;
  name: string;
  // Selected registered type from the registry
  registeredType?: {
    full_type: string;
    local_name: string;
  };
  // Input ports derived from the selected registered type
  inputPorts: ConnectionPort[];
  // Output port type this component provides
  outputRefType?: RefType;
}

// Connection between components
export interface ComponentConnection {
  id: string;
  // Source component provides the value
  sourceId: string;
  // Target component receives the value
  targetId: string;
  // The field on the target that receives the connection
  targetField: string;
  // The RefType of the connection
  refType: RefType;
}

// Workflow state
export interface WorkflowState {
  components: PlacedComponent[];
  connections: ComponentConnection[];
}

// Component palette item (for sidebar)
export interface PaletteItem {
  type: NATComponentType;
  config: NATComponentConfig;
}

// All NAT component configurations
export const NAT_COMPONENTS: Record<NATComponentType, NATComponentConfig> = {
  agent: {
    type: 'agent',
    label: 'Agent',
    description: 'AI agent that reasons, uses tools, and completes tasks',
    color: 'nat-agent',
    icon: 'Bot',
    category: 'core',
  },
  embedder: {
    type: 'embedder',
    label: 'Embedder',
    description: 'Generate text embeddings for semantic search',
    color: 'nat-embedder',
    icon: 'Sparkles',
    category: 'ai',
  },
  function: {
    type: 'function',
    label: 'Function',
    description: 'Custom function or tool for agent actions',
    color: 'nat-function',
    icon: 'Code2',
    category: 'core',
  },
  function_group: {
    type: 'function_group',
    label: 'Function Group',
    description: 'Group of related functions',
    color: 'nat-function-group',
    icon: 'Layers',
    category: 'core',
  },
  llm: {
    type: 'llm',
    label: 'LLM',
    description: 'Large Language Model for reasoning',
    color: 'nat-llm',
    icon: 'Brain',
    category: 'ai',
  },
  memory: {
    type: 'memory',
    label: 'Memory',
    description: 'Persistent memory storage for context',
    color: 'nat-memory',
    icon: 'HardDrive',
    category: 'data',
  },
  object_store: {
    type: 'object_store',
    label: 'Object Store',
    description: 'Store and retrieve objects (S3, etc.)',
    color: 'nat-object-store',
    icon: 'Database',
    category: 'data',
  },
  retriever: {
    type: 'retriever',
    label: 'Retriever',
    description: 'Retrieve relevant documents for RAG',
    color: 'nat-retriever',
    icon: 'Search',
    category: 'ai',
  },
  authentication: {
    type: 'authentication',
    label: 'Authentication',
    description: 'API authentication provider',
    color: 'nat-auth',
    icon: 'Shield',
    category: 'infrastructure',
  },
  middleware: {
    type: 'middleware',
    label: 'Middleware',
    description: 'Request/response middleware',
    color: 'nat-middleware',
    icon: 'Workflow',
    category: 'infrastructure',
  },
};

// Map component type to the RefType it provides (for output connections)
// Note: Agents provide 'function' RefType since they can be used as tools
export const COMPONENT_TO_REF_TYPE: Record<NATComponentType, RefType> = {
  llm: 'llm',
  embedder: 'embedder',
  function: 'function',
  function_group: 'function_group',
  agent: 'function', // Agents can be used as functions/tools
  retriever: 'retriever',
  memory: 'memory',
  object_store: 'object_store',
  authentication: 'authentication',
  middleware: 'middleware',
};

// Group components by category
export const COMPONENT_CATEGORIES = {
  core: {
    label: 'Core',
    description: 'Essential workflow components',
    types: ['agent', 'function', 'function_group'] as NATComponentType[],
  },
  ai: {
    label: 'AI & ML',
    description: 'AI and machine learning components',
    types: ['llm', 'embedder', 'retriever'] as NATComponentType[],
  },
  data: {
    label: 'Data',
    description: 'Data storage and management',
    types: ['memory', 'object_store'] as NATComponentType[],
  },
  infrastructure: {
    label: 'Infrastructure',
    description: 'System infrastructure components',
    types: ['authentication', 'middleware'] as NATComponentType[],
  },
};
