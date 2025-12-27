/**
 * Main chat panel component that combines all chat elements.
 * Used inside the slide-over panel for the workflow runner.
 */

import { FC, useCallback } from 'react';
import { X, Wifi, WifiOff, AlertCircle, RefreshCw, Trash2 } from 'lucide-react';
import { ConnectionStatus } from '@/hooks/useWorkflowChat';
import { Conversation } from '@/types/chat';
import { ChatMessages } from './ChatMessages';
import { ChatInput } from './ChatInput';

interface ChatPanelProps {
  conversation: Conversation | null;
  connectionStatus: ConnectionStatus;
  isStreaming: boolean;
  error: string | null;
  onSendMessage: (content: string) => void;
  onClearConversation: () => void;
  onReconnect: () => void;
  onClose: () => void;
}

/**
 * Connection status indicator component
 */
const ConnectionIndicator: FC<{ status: ConnectionStatus }> = ({ status }) => {
  const statusConfig = {
    connected: {
      icon: Wifi,
      color: 'text-green-400',
      bgColor: 'bg-green-400/10',
      label: 'Connected',
    },
    connecting: {
      icon: Wifi,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-400/10',
      label: 'Connecting...',
    },
    disconnected: {
      icon: WifiOff,
      color: 'text-gray-400',
      bgColor: 'bg-gray-400/10',
      label: 'Disconnected',
    },
    error: {
      icon: AlertCircle,
      color: 'text-red-400',
      bgColor: 'bg-red-400/10',
      label: 'Error',
    },
  };

  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <div
      className={`flex items-center gap-1.5 px-2 py-1 rounded-md ${config.bgColor}`}
      title={config.label}
    >
      <Icon size={14} className={`${config.color} ${status === 'connecting' ? 'animate-pulse' : ''}`} />
      <span className={`text-xs font-medium ${config.color}`}>{config.label}</span>
    </div>
  );
};

export const ChatPanel: FC<ChatPanelProps> = ({
  conversation,
  connectionStatus,
  isStreaming,
  error,
  onSendMessage,
  onClearConversation,
  onReconnect,
  onClose,
}) => {
  const messages = conversation?.messages || [];
  const isConnected = connectionStatus === 'connected';
  const isLoading = isStreaming && messages.length > 0 && messages[messages.length - 1].role === 'user';

  return (
    <div className="flex flex-col h-full bg-canvas">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700 bg-canvas-light">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-white">Workflow Chat</h2>
          <ConnectionIndicator status={connectionStatus} />
        </div>

        <div className="flex items-center gap-2">
          {/* Clear conversation button */}
          {messages.length > 0 && (
            <button
              onClick={onClearConversation}
              className="p-2 text-gray-400 hover:text-gray-200 hover:bg-gray-700 rounded-lg transition-colors"
              title="Clear conversation"
            >
              <Trash2 size={18} />
            </button>
          )}

          {/* Reconnect button */}
          {(connectionStatus === 'error' || connectionStatus === 'disconnected') && (
            <button
              onClick={onReconnect}
              className="p-2 text-gray-400 hover:text-accent hover:bg-gray-700 rounded-lg transition-colors"
              title="Reconnect"
            >
              <RefreshCw size={18} />
            </button>
          )}

          {/* Close button */}
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-200 hover:bg-gray-700 rounded-lg transition-colors"
            title="Close chat"
          >
            <X size={18} />
          </button>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="px-4 py-3 bg-red-500/10 border-b border-red-500/30 flex items-center gap-3">
          <AlertCircle size={18} className="text-red-400 flex-shrink-0" />
          <p className="text-sm text-red-400 flex-1">{error}</p>
          <button
            onClick={onReconnect}
            className="text-xs text-red-400 hover:text-red-300 underline flex-shrink-0"
          >
            Retry
          </button>
        </div>
      )}

      {/* Messages area */}
      <ChatMessages
        messages={messages}
        isLoading={isLoading}
        isStreaming={isStreaming}
      />

      {/* Input area */}
      <ChatInput
        onSend={onSendMessage}
        isStreaming={isStreaming}
        isConnected={isConnected}
        placeholder="Ask your agent anything..."
      />
    </div>
  );
};

export default ChatPanel;

