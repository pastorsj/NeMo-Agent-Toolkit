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
  | 'middleware';

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
}

// All possible categories from the API
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
  | 'ttc_strategy'
  | 'trainer'
  | 'trainer_adapter'
  | 'trajectory_builder'
  | 'front_end'
  | 'evaluator'
  | 'telemetry_exporter'
  | 'logging'
  | 'registry_handler';

export interface ComponentTypeInfo {
  category: ComponentCategory;
  display_name: string;
  description: string;
  registered_types: RegisteredTypeInfo[];
  // The RefType this category provides (for output connections)
  provides_ref_type: RefType | null;
}

export interface RegistryResponse {
  components: ComponentTypeInfo[];
  total_types: number;
}

export interface HealthResponse {
  status: string;
  registry_loaded: boolean;
}

// Map NAT categories to our UI component types
export const CATEGORY_TO_UI_TYPE: Record<ComponentCategory, string> = {
  llm: 'llm',
  embedder: 'embedder',
  function: 'function',
  function_group: 'function_group',
  agent: 'agent',
  retriever: 'retriever',
  memory: 'memory',
  object_store: 'object_store',
  authentication: 'authentication',
  middleware: 'middleware',
  ttc_strategy: 'ttc_strategy',
  trainer: 'trainer',
  trainer_adapter: 'trainer_adapter',
  trajectory_builder: 'trajectory_builder',
  front_end: 'front_end',
  evaluator: 'evaluator',
  telemetry_exporter: 'telemetry_exporter',
  logging: 'logging',
  registry_handler: 'registry_handler',
};

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
};
