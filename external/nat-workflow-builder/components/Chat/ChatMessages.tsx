/**
 * Container component for the list of chat messages.
 * Handles scrolling, auto-scroll, and empty state.
 */

import { FC, useRef, useEffect, useCallback, useState } from 'react';
import { Message } from '@/types/chat';
import { ChatMessage } from './ChatMessage';
import { ChatLoader } from './ChatLoader';
import { MessageSquare, ArrowDown } from 'lucide-react';

interface ChatMessagesProps {
  messages: Message[];
  isLoading: boolean;
  isStreaming: boolean;
}

export const ChatMessages: FC<ChatMessagesProps> = ({
  messages,
  isLoading,
  isStreaming,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [autoScrollEnabled, setAutoScrollEnabled] = useState(true);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (autoScrollEnabled && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading, autoScrollEnabled]);

  // Handle scroll to detect if user scrolled up
  const handleScroll = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    const { scrollTop, scrollHeight, clientHeight } = container;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;

    setShowScrollButton(!isNearBottom);

    // Re-enable auto-scroll if near bottom
    if (isNearBottom && !autoScrollEnabled) {
      setAutoScrollEnabled(true);
    } else if (!isNearBottom && autoScrollEnabled && isStreaming) {
      // Disable auto-scroll if user scrolled up during streaming
      setAutoScrollEnabled(false);
    }
  }, [autoScrollEnabled, isStreaming]);

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    setAutoScrollEnabled(true);
    setShowScrollButton(false);
  }, []);

  // Empty state
  if (messages.length === 0 && !isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center max-w-md">
          <div className="w-16 h-16 rounded-2xl bg-accent/10 flex items-center justify-center mx-auto mb-4">
            <MessageSquare size={32} className="text-accent" />
          </div>
          <h3 className="text-lg font-semibold text-white mb-2">
            Start a Conversation
          </h3>
          <p className="text-gray-400 text-sm">
            Type a message below to start chatting with your workflow. 
            Your agent is ready to help!
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex-1 flex flex-col">
      {/* Messages container */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto"
      >
        {messages.map((message, index) => {
          const isLastMessage = index === messages.length - 1;
          const showStreaming = isStreaming && isLastMessage && message.role === 'assistant';

          return (
            <ChatMessage
              key={message.id}
              message={message}
              isStreaming={showStreaming}
            />
          );
        })}

        {/* Loading indicator */}
        {isLoading && !isStreaming && (
          <ChatLoader text="Thinking..." />
        )}

        {/* Scroll anchor */}
        <div ref={bottomRef} className="h-4" />
      </div>

      {/* Scroll to bottom button */}
      {showScrollButton && (
        <button
          onClick={scrollToBottom}
          className="absolute bottom-20 right-4 p-2 bg-gray-700 hover:bg-gray-600 rounded-full shadow-lg transition-all"
          title="Scroll to bottom"
        >
          <ArrowDown size={20} className="text-gray-300" />
        </button>
      )}
    </div>
  );
};

export default ChatMessages;

