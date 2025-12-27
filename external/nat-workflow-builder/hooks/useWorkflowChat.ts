/**
 * Custom hook for managing WebSocket-based chat with a workflow session.
 * Handles connection, message sending, and response streaming.
 */

import { useCallback, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { registryAPI, CreateSessionRequest } from '@/lib/api';
import {
  Message,
  Conversation,
  WebSocketInboundMessage,
  WebSocketUserMessage,
  isSystemResponseMessage,
  isSystemResponseComplete,
  isSystemIntermediateMessage,
  isErrorMessage,
} from '@/types/chat';

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

interface UseWorkflowChatReturn {
  // State
  conversation: Conversation | null;
  connectionStatus: ConnectionStatus;
  isStreaming: boolean;
  error: string | null;
  
  // Actions
  createSession: (request: CreateSessionRequest) => Promise<boolean>;
  sendMessage: (content: string) => void;
  clearConversation: () => void;
  disconnect: () => void;
}

export function useWorkflowChat(): UseWorkflowChatReturn {
  // State
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const activeMessageIdRef = useRef<string | null>(null);

  /**
   * Create a new session and connect to it
   */
  const createSession = useCallback(async (request: CreateSessionRequest): Promise<boolean> => {
    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionStatus('connecting');
    setError(null);

    try {
      // Create the session
      const response = await registryAPI.createSession(request);
      sessionIdRef.current = response.session_id;

      // Initialize conversation
      const conversationId = uuidv4();
      setConversation({
        id: conversationId,
        name: 'Workflow Chat',
        messages: [],
        createdAt: Date.now(),
      });

      // Connect to WebSocket
      const wsUrl = registryAPI.getWebSocketUrl(response.session_id);
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('[WorkflowChat] WebSocket connected');
        setConnectionStatus('connected');
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketInboundMessage = JSON.parse(event.data);
          handleIncomingMessage(message);
        } catch (e) {
          console.error('[WorkflowChat] Failed to parse message:', e);
        }
      };

      ws.onerror = (event) => {
        console.error('[WorkflowChat] WebSocket error:', event);
        setConnectionStatus('error');
        setError('WebSocket connection error');
      };

      ws.onclose = (event) => {
        console.log('[WorkflowChat] WebSocket closed:', event.code, event.reason);
        setConnectionStatus('disconnected');
        wsRef.current = null;
        
        if (event.code !== 1000 && event.code !== 1001) {
          setError(`Connection closed: ${event.reason || 'Unknown reason'}`);
        }
      };

      wsRef.current = ws;
      return true;
    } catch (e) {
      console.error('[WorkflowChat] Session creation failed:', e);
      setConnectionStatus('error');
      setError(e instanceof Error ? e.message : 'Failed to create session');
      return false;
    }
  }, []);

  /**
   * Handle incoming WebSocket messages
   */
  const handleIncomingMessage = useCallback((message: WebSocketInboundMessage) => {
    // Ignore messages if we've stopped or for different conversations
    if (activeMessageIdRef.current === null) {
      return;
    }

    if (isSystemResponseComplete(message)) {
      // Response is complete
      setIsStreaming(false);
      activeMessageIdRef.current = null;
      return;
    }

    if (isSystemResponseMessage(message)) {
      // Streaming response text
      const text = message.content?.text || '';
      if (text) {
        setConversation((prev) => {
          if (!prev) return prev;

          const messages = [...prev.messages];
          const lastMessage = messages[messages.length - 1];

          if (lastMessage && lastMessage.role === 'assistant') {
            // Append to existing assistant message
            messages[messages.length - 1] = {
              ...lastMessage,
              content: lastMessage.content + text,
              timestamp: Date.now(),
            };
          } else {
            // Create new assistant message
            messages.push({
              id: message.id || uuidv4(),
              role: 'assistant',
              content: text,
              timestamp: Date.now(),
              parentId: message.parent_id,
            });
          }

          return { ...prev, messages };
        });
      }
    } else if (isSystemIntermediateMessage(message)) {
      // Intermediate step - add to last assistant message
      setConversation((prev) => {
        if (!prev) return prev;

        const messages = [...prev.messages];
        const lastMessage = messages[messages.length - 1];

        if (lastMessage && lastMessage.role === 'assistant') {
          const steps = lastMessage.intermediateSteps || [];
          messages[messages.length - 1] = {
            ...lastMessage,
            intermediateSteps: [...steps, message],
            timestamp: Date.now(),
          };
        } else {
          // Create new assistant message with step
          messages.push({
            id: message.id || uuidv4(),
            role: 'assistant',
            content: '',
            timestamp: Date.now(),
            parentId: message.parent_id,
            intermediateSteps: [message],
          });
        }

        return { ...prev, messages };
      });
    } else if (isErrorMessage(message)) {
      // Error message
      const errorText = message.content?.text || message.content?.error || 'Unknown error';
      setError(errorText);
      setIsStreaming(false);

      setConversation((prev) => {
        if (!prev) return prev;

        const messages = [...prev.messages];
        const lastMessage = messages[messages.length - 1];

        if (lastMessage && lastMessage.role === 'assistant') {
          const errors = lastMessage.errorMessages || [];
          messages[messages.length - 1] = {
            ...lastMessage,
            errorMessages: [...errors, message],
            timestamp: Date.now(),
          };
        }

        return { ...prev, messages };
      });
    }
  }, []);

  /**
   * Send a message to the workflow
   */
  const sendMessage = useCallback((content: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected to workflow');
      return;
    }

    if (!conversation) {
      setError('No active conversation');
      return;
    }

    const trimmedContent = content.trim();
    if (!trimmedContent) {
      return;
    }

    // Create user message
    const userMessageId = uuidv4();
    const userMessage: Message = {
      id: userMessageId,
      role: 'user',
      content: trimmedContent,
      timestamp: Date.now(),
    };

    // Add user message to conversation
    setConversation((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        messages: [...prev.messages, userMessage],
      };
    });

    // Build chat history for the request
    const chatMessages = [...(conversation.messages || []), userMessage].map((msg) => ({
      role: msg.role,
      content: [{ type: 'text' as const, text: msg.content }],
    }));

    // Build WebSocket message
    const wsMessage: WebSocketUserMessage = {
      type: 'user_message',
      id: userMessageId,
      conversation_id: conversation.id,
      schema_type: 'chat_stream',
      content: {
        messages: chatMessages,
      },
      timestamp: new Date().toISOString(),
    };

    // Track this message as active
    activeMessageIdRef.current = userMessageId;
    setIsStreaming(true);
    setError(null);

    // Send the message
    wsRef.current.send(JSON.stringify(wsMessage));
  }, [conversation]);

  /**
   * Clear the conversation (start fresh)
   */
  const clearConversation = useCallback(() => {
    setConversation((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        messages: [],
        createdAt: Date.now(),
      };
    });
    setError(null);
  }, []);

  /**
   * Disconnect from the session
   */
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }

    if (sessionIdRef.current) {
      // Fire and forget - clean up session on server
      registryAPI.destroySession(sessionIdRef.current).catch(() => {});
      sessionIdRef.current = null;
    }

    setConnectionStatus('disconnected');
    setIsStreaming(false);
    activeMessageIdRef.current = null;
  }, []);

  return {
    conversation,
    connectionStatus,
    isStreaming,
    error,
    createSession,
    sendMessage,
    clearConversation,
    disconnect,
  };
}

export default useWorkflowChat;

