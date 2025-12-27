/**
 * Chat input component for sending messages.
 * Includes textarea, send button, and stop button during streaming.
 */

import { FC, useCallback, useRef, useState, useEffect, KeyboardEvent } from 'react';
import { Send, Square, Loader2 } from 'lucide-react';

interface ChatInputProps {
  onSend: (content: string) => void;
  onStop?: () => void;
  isStreaming: boolean;
  isConnected: boolean;
  placeholder?: string;
}

export const ChatInput: FC<ChatInputProps> = ({
  onSend,
  onStop,
  isStreaming,
  isConnected,
  placeholder = 'Type a message...',
}) => {
  const [content, setContent] = useState('');
  const [isComposing, setIsComposing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'inherit';
      const scrollHeight = textarea.scrollHeight;
      textarea.style.height = `${Math.min(scrollHeight, 200)}px`;
    }
  }, [content]);

  // Focus textarea on mount
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const handleSend = useCallback(() => {
    const trimmed = content.trim();
    if (!trimmed || isStreaming || !isConnected) return;

    onSend(trimmed);
    setContent('');

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'inherit';
    }
  }, [content, isStreaming, isConnected, onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      // Send on Enter (without Shift)
      if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend, isComposing]
  );

  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setContent(e.target.value);
  }, []);

  return (
    <div className="border-t border-gray-700 bg-canvas-light p-4">
      {/* Stop button when streaming */}
      {isStreaming && onStop && (
        <div className="flex justify-center mb-3">
          <button
            onClick={onStop}
            className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-gray-300 text-sm font-medium transition-colors"
          >
            <Square size={14} fill="currentColor" />
            Stop Generating
          </button>
        </div>
      )}

      <div className="relative flex items-end gap-2">
        {/* Textarea */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={content}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            onCompositionStart={() => setIsComposing(true)}
            onCompositionEnd={() => setIsComposing(false)}
            placeholder={isConnected ? placeholder : 'Connecting...'}
            disabled={!isConnected}
            rows={1}
            className={`
              w-full resize-none rounded-xl border bg-gray-800 px-4 py-3 pr-12
              text-gray-200 placeholder-gray-500
              focus:outline-none focus:ring-2 focus:ring-accent/50
              disabled:opacity-50 disabled:cursor-not-allowed
              ${isConnected ? 'border-gray-600' : 'border-gray-700'}
            `}
            style={{
              minHeight: '48px',
              maxHeight: '200px',
            }}
          />

          {/* Send button (inside textarea) */}
          <button
            onClick={handleSend}
            disabled={!content.trim() || isStreaming || !isConnected}
            className={`
              absolute right-2 bottom-2 p-2 rounded-lg transition-all
              ${
                content.trim() && !isStreaming && isConnected
                  ? 'bg-accent text-black hover:bg-accent/90'
                  : 'bg-gray-700 text-gray-500 cursor-not-allowed'
              }
            `}
            title="Send message (Enter)"
          >
            {isStreaming ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Send size={18} />
            )}
          </button>
        </div>
      </div>

      {/* Helper text */}
      <p className="text-xs text-gray-500 mt-2 text-center">
        Press <kbd className="px-1.5 py-0.5 bg-gray-700 rounded text-gray-400">Enter</kbd> to send,{' '}
        <kbd className="px-1.5 py-0.5 bg-gray-700 rounded text-gray-400">Shift+Enter</kbd> for new line
      </p>
    </div>
  );
};

export default ChatInput;

