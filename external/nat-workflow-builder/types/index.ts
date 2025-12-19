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
  | 'middleware'
  // Front-end and observability
  | 'front_end'
  | 'logger'
  | 'telemetry_exporter'
  // Evaluation
  | 'evaluator'
  // Finetuning components
  | 'trainer'
  | 'trajectory_builder'
  | 'trainer_adapter'
  // Workflow-level configuration containers (single-type, no dropdown)
  | 'nat_workflow'
  | 'general_config'
  | 'evaluation_config'
  | 'optimizer_config'
  | 'finetuner_config';

// Component configuration for each type
export interface NATComponentConfig {
  type: NATComponentType;
  label: string;
  description: string;
  color: string;
  icon: string;
  category:
    | 'workflow'
    | 'core'
    | 'rag'
    | 'storage'
    | 'infrastructure'
    | 'observability'
    | 'evaluation'
    | 'finetuning'
    | 'improvement';
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
    icon_url?: string | null;
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
  // Workflow section (top)
  nat_workflow: {
    type: 'nat_workflow',
    label: 'Workflow',
    description: 'Main workflow entry point',
    color: 'nat-workflow',
    icon: 'Boxes',
    category: 'workflow',
  },
  general_config: {
    type: 'general_config',
    label: 'General Config',
    description: 'Loggers, telemetry, and front-end configuration',
    color: 'nat-config',
    icon: 'Settings',
    category: 'workflow',
  },
  // Core components
  agent: {
    type: 'agent',
    label: 'Agent',
    description: 'AI agent that reasons, uses tools, and completes tasks',
    color: 'nat-agent',
    icon: 'Bot',
    category: 'core',
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
    category: 'core',
  },
  // RAG (Embedders & Retrievers)
  embedder: {
    type: 'embedder',
    label: 'Embedder',
    description: 'Generate text embeddings for semantic search',
    color: 'nat-embedder',
    icon: 'Sparkles',
    category: 'rag',
  },
  retriever: {
    type: 'retriever',
    label: 'Retriever',
    description: 'Retrieve relevant documents for RAG',
    color: 'nat-retriever',
    icon: 'Search',
    category: 'rag',
  },
  // Storage
  memory: {
    type: 'memory',
    label: 'Memory',
    description: 'Persistent memory storage for context',
    color: 'nat-memory',
    icon: 'HardDrive',
    category: 'storage',
  },
  object_store: {
    type: 'object_store',
    label: 'Object Store',
    description: 'Store and retrieve objects (S3, etc.)',
    color: 'nat-object-store',
    icon: 'Database',
    category: 'storage',
  },
  // Infrastructure
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
  // Observability (front-end and logging)
  front_end: {
    type: 'front_end',
    label: 'Front End',
    description: 'Deployment front end (FastAPI, Console, MCP)',
    color: 'nat-frontend',
    icon: 'Globe',
    category: 'observability',
  },
  logger: {
    type: 'logger',
    label: 'Logger',
    description: 'Logging configuration for runtime observability',
    color: 'nat-logger',
    icon: 'FileText',
    category: 'observability',
  },
  telemetry_exporter: {
    type: 'telemetry_exporter',
    label: 'Telemetry',
    description: 'Telemetry exporter for tracing and metrics',
    color: 'nat-telemetry',
    icon: 'Activity',
    category: 'observability',
  },
  // Evaluation components
  evaluator: {
    type: 'evaluator',
    label: 'Evaluator',
    description: 'Evaluation metric for testing workflow quality',
    color: 'nat-evaluator',
    icon: 'CheckCircle',
    category: 'evaluation',
  },
  // Finetuning components
  trainer: {
    type: 'trainer',
    label: 'Trainer',
    description: 'Training loop orchestrator for finetuning',
    color: 'nat-trainer',
    icon: 'Dumbbell',
    category: 'finetuning',
  },
  trajectory_builder: {
    type: 'trajectory_builder',
    label: 'Trajectory Builder',
    description: 'Training data collector',
    color: 'nat-trajectory',
    icon: 'Route',
    category: 'finetuning',
  },
  trainer_adapter: {
    type: 'trainer_adapter',
    label: 'Trainer Adapter',
    description: 'Training backend adapter',
    color: 'nat-adapter',
    icon: 'Plug',
    category: 'finetuning',
  },
  // Improvement section (evaluation, optimization, finetuning configs)
  evaluation_config: {
    type: 'evaluation_config',
    label: 'Evaluation Config',
    description: 'Evaluation configuration with evaluators',
    color: 'nat-evaluator',
    icon: 'ClipboardCheck',
    category: 'improvement',
  },
  optimizer_config: {
    type: 'optimizer_config',
    label: 'Optimizer Config',
    description: 'Hyperparameter optimization configuration',
    color: 'nat-optimizer',
    icon: 'Zap',
    category: 'improvement',
  },
  finetuner_config: {
    type: 'finetuner_config',
    label: 'Finetuner Config',
    description: 'Model finetuning configuration',
    color: 'nat-finetuner',
    icon: 'Wrench',
    category: 'improvement',
  },
};

// Component types that can only have ONE instance on the canvas
// Once placed, they should be disabled in the sidebar until deleted
export const SINGLE_INSTANCE_TYPES: Set<NATComponentType> = new Set<NATComponentType>([
  'nat_workflow',
  'general_config',
  'evaluation_config',
  'optimizer_config',
  'finetuner_config',
  'trainer',
  'trajectory_builder',
  'trainer_adapter',
]);

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
  // Front-end and observability
  front_end: 'front_end',
  logger: 'logger',
  telemetry_exporter: 'telemetry_exporter',
  // Evaluation
  evaluator: 'evaluator',
  // Finetuning components
  trainer: 'trainer',
  trajectory_builder: 'trajectory_builder',
  trainer_adapter: 'trainer_adapter',
  // Workflow-level configuration containers
  nat_workflow: 'nat_workflow',
  general_config: 'general_config',
  evaluation_config: 'evaluation_config',
  optimizer_config: 'optimizer_config',
  finetuner_config: 'finetuner_config',
};

// Group components by category (order matters for sidebar display)
export const COMPONENT_CATEGORIES = {
  workflow: {
    label: 'Workflow',
    description: 'Workflow entry point and configuration',
    types: ['nat_workflow', 'general_config'] as NATComponentType[],
  },
  core: {
    label: 'Core',
    description: 'Essential workflow components',
    types: ['agent', 'function', 'function_group', 'llm'] as NATComponentType[],
  },
  rag: {
    label: 'Embedders & Retrievers',
    description: 'RAG components for embeddings and retrieval',
    types: ['embedder', 'retriever'] as NATComponentType[],
  },
  storage: {
    label: 'Storage',
    description: 'Data storage and management',
    types: ['memory', 'object_store'] as NATComponentType[],
  },
  infrastructure: {
    label: 'Infrastructure',
    description: 'System infrastructure components',
    types: ['authentication', 'middleware'] as NATComponentType[],
  },
  observability: {
    label: 'Observability',
    description: 'Front-ends, logging, and telemetry',
    types: ['front_end', 'logger', 'telemetry_exporter'] as NATComponentType[],
  },
  evaluation: {
    label: 'Evaluation',
    description: 'Evaluation and testing components',
    types: ['evaluator'] as NATComponentType[],
  },
  finetuning: {
    label: 'Finetuning',
    description: 'Model finetuning components',
    types: ['trainer', 'trajectory_builder', 'trainer_adapter'] as NATComponentType[],
  },
  improvement: {
    label: 'Improvement',
    description: 'Workflow evaluation, optimization, and finetuning configs',
    types: ['evaluation_config', 'optimizer_config', 'finetuner_config'] as NATComponentType[],
  },
};
