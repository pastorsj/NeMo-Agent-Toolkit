// Types for the NAT Registry API responses

// Reference types that can be connected between components
export type RefType =
  | 'llm'
  | 'embedder'
  | 'function'
  | 'function_group'
  | 'retriever'
  | 'memory'
  | 'object_store'
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
  // Workflow-level configuration containers
  | 'nat_workflow'
  | 'general_config'
  | 'evaluation_config'
  | 'optimizer_config'
  | 'finetuner_config';

// A connection port on a component - represents a field that accepts references
export interface ConnectionPort {
  field_name: string;
  ref_type: RefType;
  // All ref types this port can accept (for union types like FunctionRef | FunctionGroupRef)
  accepts_ref_types: RefType[];
  required: boolean;
  is_list: boolean;
  description: string | null;
  title: string | null;
}

export interface FieldInfo {
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
  // Component reference information
  is_component_ref: boolean;
  ref_type: RefType | null;
  is_ref_list: boolean;
  // Suggested options for dropdown (not strict like enum - allows free text)
  options: string[] | null;
}

export interface RegisteredTypeInfo {
  full_type: string;
  module_name: string;
  local_name: string;
  description: string | null;
  json_schema: Record<string, unknown>;
  fields: FieldInfo[];
  is_per_user: boolean;
  // Connection ports - fields that accept references to other components
  input_ports: ConnectionPort[];
  // Custom icon URL for the component (e.g., provider logo)
  icon_url: string | null;
}

// Component categories supported by the workflow builder
export type ComponentCategory =
  | 'llm'
  | 'embedder'
  | 'function'
  | 'function_group'
  | 'agent'
  | 'retriever'
  | 'memory'
  | 'object_store'
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
  // Workflow-level configuration containers
  | 'nat_workflow'
  | 'general_config'
  | 'evaluation_config'
  | 'optimizer_config'
  | 'finetuner_config';

export interface ComponentTypeInfo {
  category: ComponentCategory;
  display_name: string;
  description: string;
  registered_types: RegisteredTypeInfo[];
  // The RefType this category provides (for output connections)
  provides_ref_type: RefType | null;
}

export interface HealthResponse {
  status: string;
  registry_loaded: boolean;
}

// Categories we want to show in the UI (core agent building components only)
export const WORKFLOW_CATEGORIES: ComponentCategory[] = [
  'llm',
  'embedder',
  'agent',
  'function',
  'function_group',
  'retriever',
  'memory',
  'object_store',
  'authentication',
  'middleware',
  // Front-end and observability
  'front_end',
  'logger',
  'telemetry_exporter',
  // Evaluation
  'evaluator',
  // Finetuning components
  'trainer',
  'trajectory_builder',
  'trainer_adapter',
  // Workflow-level configuration containers
  'nat_workflow',
  'general_config',
  'evaluation_config',
  'optimizer_config',
  'finetuner_config',
];

// Human-readable labels for RefTypes
export const REF_TYPE_LABELS: Record<RefType, string> = {
  llm: 'LLM',
  embedder: 'Embedder',
  function: 'Function',
  function_group: 'Function Group',
  retriever: 'Retriever',
  memory: 'Memory',
  object_store: 'Object Store',
  authentication: 'Authentication',
  middleware: 'Middleware',
  // Front-end and observability
  front_end: 'Front End',
  logger: 'Logger',
  telemetry_exporter: 'Telemetry',
  // Evaluation
  evaluator: 'Evaluator',
  // Finetuning components
  trainer: 'Trainer',
  trajectory_builder: 'Trajectory Builder',
  trainer_adapter: 'Trainer Adapter',
  // Workflow-level configuration containers
  nat_workflow: 'Workflow',
  general_config: 'General Config',
  evaluation_config: 'Evaluation',
  optimizer_config: 'Optimizer',
  finetuner_config: 'Finetuner',
};

// Colors for each RefType (for connection lines and ports)
export const REF_TYPE_COLORS: Record<RefType, string> = {
  llm: '#22c55e', // green-500
  embedder: '#3b82f6', // blue-500
  function: '#a855f7', // purple-500
  function_group: '#8b5cf6', // violet-500
  retriever: '#06b6d4', // cyan-500
  memory: '#f97316', // orange-500
  object_store: '#eab308', // yellow-500
  authentication: '#ef4444', // red-500
  middleware: '#ec4899', // pink-500
  // Front-end and observability
  front_end: '#10b981', // emerald-500
  logger: '#64748b', // slate-500
  telemetry_exporter: '#f59e0b', // amber-500
  // Evaluation
  evaluator: '#0ea5e9', // sky-500
  // Finetuning components
  trainer: '#84cc16', // lime-500
  trajectory_builder: '#a855f7', // purple-500
  trainer_adapter: '#78716c', // stone-500
  // Workflow-level configuration containers
  nat_workflow: '#76b900', // NVIDIA green
  general_config: '#6b7280', // gray-500
  evaluation_config: '#0ea5e9', // sky-500
  optimizer_config: '#e25a1c', // spark orange
  finetuner_config: '#ee4c2c', // pytorch red
};
