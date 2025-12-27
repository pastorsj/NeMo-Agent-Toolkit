/**
 * Chat-related type definitions for the Workflow Builder.
 * Modeled after nat-ui types for consistency.
 */

// =============================================================================
// Message Types
// =============================================================================

export type MessageRole = 'user' | 'assistant' | 'system';

export interface IntermediateStep {
  id?: string;
  parent_id?: string;
  index?: number;
  content?: {
    id?: string;
    parent_id?: string;
    type?: string;
    name?: string;
    payload?: string;
    data?: string;
  };
  [key: string]: unknown;
}

export interface ErrorMessage {
  type: 'error';
  content?: {
    error?: string;
    text?: string;
  };
}

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
  parentId?: string;
  intermediateSteps?: IntermediateStep[];
  errorMessages?: ErrorMessage[];
}

export interface Conversation {
  id: string;
  name: string;
  messages: Message[];
  createdAt: number;
}

// =============================================================================
// WebSocket Message Types
// =============================================================================

export type WebSocketMessageType =
  | 'user_message'
  | 'system_response_message'
  | 'system_intermediate_message'
  | 'system_interaction_message'
  | 'error';

export type WebSocketMessageStatus = 'in_progress' | 'complete';

export interface WebSocketMessageBase {
  id?: string;
  conversation_id?: string;
  parent_id?: string;
  timestamp?: string;
  status?: WebSocketMessageStatus;
}

export interface SystemResponseMessage extends WebSocketMessageBase {
  type: 'system_response_message';
  status: WebSocketMessageStatus;
  content?: {
    text?: string;
  };
}

export interface SystemIntermediateMessage extends WebSocketMessageBase {
  type: 'system_intermediate_message';
  content?: {
    name?: string;
    payload?: string;
    data?: string;
  };
}

export interface SystemInteractionMessage extends WebSocketMessageBase {
  type: 'system_interaction_message';
  content?: {
    input_type?: string;
    text?: string;
  };
}

export interface WebSocketErrorMessage extends WebSocketMessageBase {
  type: 'error';
  content?: {
    error?: string;
    text?: string;
  };
}

export type WebSocketInboundMessage =
  | SystemResponseMessage
  | SystemIntermediateMessage
  | SystemInteractionMessage
  | WebSocketErrorMessage;

// =============================================================================
// Outbound Message Types
// =============================================================================

export interface UserMessageContent {
  role: 'user';
  content: Array<{ type: 'text'; text: string }>;
}

export interface WebSocketUserMessage {
  type: 'user_message';
  id: string;
  conversation_id: string;
  schema_type: 'chat_stream';
  content: {
    messages: UserMessageContent[];
  };
  timestamp: string;
}

// =============================================================================
// Session Types
// =============================================================================

export interface CreateSessionRequest {
  components: Array<{
    id: string;
    component_type: string;
    full_type: string;
    config: Record<string, unknown>;
  }>;
  connections: Array<{
    id: string;
    source_id: string;
    target_id: string;
    target_field: string;
  }>;
}

export interface CreateSessionResponse {
  session_id: string;
  websocket_path: string;
}

export interface ValidateWorkflowRequest {
  components: Array<{
    id: string;
    component_type: string;
    full_type: string;
    config: Record<string, unknown>;
  }>;
  connections: Array<{
    id: string;
    source_id: string;
    target_id: string;
    target_field: string;
  }>;
}

export interface ValidateWorkflowResponse {
  valid: boolean;
  errors: string[];
}

// =============================================================================
// Type Guards
// =============================================================================

export function isSystemResponseMessage(
  message: WebSocketInboundMessage
): message is SystemResponseMessage {
  return message.type === 'system_response_message';
}

export function isSystemResponseComplete(
  message: WebSocketInboundMessage
): message is SystemResponseMessage {
  return (
    isSystemResponseMessage(message) && message.status === 'complete'
  );
}

export function isSystemIntermediateMessage(
  message: WebSocketInboundMessage
): message is SystemIntermediateMessage {
  return message.type === 'system_intermediate_message';
}

export function isSystemInteractionMessage(
  message: WebSocketInboundMessage
): message is SystemInteractionMessage {
  return message.type === 'system_interaction_message';
}

export function isErrorMessage(
  message: WebSocketInboundMessage
): message is WebSocketErrorMessage {
  return message.type === 'error';
}

