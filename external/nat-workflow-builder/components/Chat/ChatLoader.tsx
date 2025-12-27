/**
 * Loading indicator component for when the assistant is thinking.
 */

import { FC } from 'react';
import { Bot } from 'lucide-react';

interface ChatLoaderProps {
  text?: string;
}

export const ChatLoader: FC<ChatLoaderProps> = ({ text = 'Thinking...' }) => {
  return (
    <div className="px-4 py-6 bg-canvas border-y border-gray-700/50">
      <div className="max-w-3xl mx-auto flex gap-4">
        {/* Avatar */}
        <div className="flex-shrink-0">
          <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center">
            <Bot size={18} className="text-accent" />
          </div>
        </div>

        {/* Loading indicator */}
        <div className="flex-1">
          <div className="mb-1">
            <span className="text-sm font-medium text-accent">Assistant</span>
          </div>
          <div className="flex items-center gap-2 text-gray-400">
            <span>{text}</span>
            <span className="text-accent animate-pulse">▍</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatLoader;

