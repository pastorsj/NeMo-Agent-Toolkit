/**
 * Slide-over panel for the workflow chat interface.
 * Slides in from the right side of the screen when running a workflow.
 */

import { FC, useEffect, useCallback } from 'react';
import { useWorkflowChat } from '@/hooks/useWorkflowChat';
import { CreateSessionRequest } from '@/lib/api';
import { ChatPanel } from './ChatPanel';

interface ChatSlideOverProps {
  isOpen: boolean;
  onClose: () => void;
  sessionRequest: CreateSessionRequest | null;
}

export const ChatSlideOver: FC<ChatSlideOverProps> = ({
  isOpen,
  onClose,
  sessionRequest,
}) => {
  const {
    conversation,
    connectionStatus,
    isStreaming,
    error,
    createSession,
    sendMessage,
    clearConversation,
    disconnect,
  } = useWorkflowChat();

  // Create session when opened with a new request
  useEffect(() => {
    if (isOpen && sessionRequest && connectionStatus === 'disconnected') {
      createSession(sessionRequest);
    }
  }, [isOpen, sessionRequest, connectionStatus, createSession]);

  // Cleanup on close
  const handleClose = useCallback(() => {
    disconnect();
    onClose();
  }, [disconnect, onClose]);

  // Reconnect handler
  const handleReconnect = useCallback(() => {
    if (sessionRequest) {
      disconnect();
      createSession(sessionRequest);
    }
  }, [sessionRequest, disconnect, createSession]);

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        handleClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, handleClose]);

  // Prevent body scroll when open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }

    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40 transition-opacity"
        onClick={handleClose}
        aria-hidden="true"
      />

      {/* Slide-over panel */}
      <div
        className={`
          fixed inset-y-0 right-0 z-50 w-full max-w-lg
          transform transition-transform duration-300 ease-in-out
          ${isOpen ? 'translate-x-0' : 'translate-x-full'}
        `}
      >
        <div className="h-full shadow-2xl">
          <ChatPanel
            conversation={conversation}
            connectionStatus={connectionStatus}
            isStreaming={isStreaming}
            error={error}
            onSendMessage={sendMessage}
            onClearConversation={clearConversation}
            onReconnect={handleReconnect}
            onClose={handleClose}
          />
        </div>
      </div>
    </>
  );
};

export default ChatSlideOver;

