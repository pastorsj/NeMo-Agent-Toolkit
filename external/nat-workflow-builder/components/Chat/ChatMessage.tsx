/**
 * Individual chat message component.
 * Renders user or assistant messages with proper styling.
 */

import { FC, memo, useState } from 'react';
import { Copy, Check, User, Bot, ChevronDown, ChevronRight } from 'lucide-react';
import { Message, IntermediateStep } from '@/types/chat';

interface ChatMessageProps {
  message: Message;
  isStreaming?: boolean;
}

/**
 * Cleans up a payload string for better readability
 */
const cleanPayload = (payload: string): string => {
  // Decode HTML entities
  let cleaned = payload
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/\\n/g, '\n');
  
  // Remove verbose Python repr for common types
  cleaned = cleaned
    .replace(/<ChatContentType\.TEXT: 'text'>/g, 'text')
    .replace(/<UserMessageContentRoleType\.USER: 'user'>/g, 'user')
    .replace(/<UserMessageContentRoleType\.ASSISTANT: 'assistant'>/g, 'assistant')
    .replace(/TextContent\(type='?text'?, text=/g, '')
    .replace(/Message\(content=\[/g, '')
    .replace(/\], role='?user'?\)/g, '')
    .replace(/model=None frequency_penalty=[\d.]+ logit_bias=None.*$/gm, '')
    .replace(/\*\*Function Input:\*\*\s*```python\s*/g, '**Input:** ')
    .replace(/\*\*Function Input:\*\*\s*```json\s*/g, '**Input:** ')
    .replace(/\*\*Function Output:\*\*\s*```python\s*/g, '\n**Output:** ')
    .replace(/```\s*$/gm, '');
  
  return cleaned.trim();
};

/**
 * Extracts just the user query from a workflow start step
 */
const extractUserQuery = (payload: string): string | null => {
  // Try to extract the actual user text
  const textMatch = payload.match(/text='([^']+)'/);
  if (textMatch) {
    return textMatch[1];
  }
  return null;
};

/**
 * Renders intermediate steps in a collapsible format
 */
const IntermediateStepsDisplay: FC<{ steps: IntermediateStep[] }> = ({ steps }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!steps || steps.length === 0) return null;

  // Extract step info from content
  const getStepName = (step: IntermediateStep): string => {
    const content = step.content;
    if (!content) return 'Unknown Step';
    const name = content.name || 'Step';
    // Clean up the name
    return name.replace('<workflow>', 'Workflow').replace('_', ' ');
  };

  const getStepPayload = (step: IntermediateStep): string | null => {
    const content = step.content;
    if (!content?.payload) return null;
    return content.payload;
  };

  // Format step for display
  const formatStepContent = (step: IntermediateStep, stepName: string): string | null => {
    const payload = getStepPayload(step);
    if (!payload) return null;
    
    // For workflow start, just show the user query
    if (stepName.includes('Workflow') && stepName.includes('Start')) {
      const query = extractUserQuery(payload);
      return query ? `Query: "${query}"` : null;
    }
    
    return cleanPayload(payload);
  };

  return (
    <div className="mt-3 border border-gray-600/30 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-400 hover:bg-gray-700/30 transition-colors"
      >
        {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span className="font-medium">Agent Steps</span>
        <span className="text-xs text-gray-500">({steps.length})</span>
      </button>
      
      {isExpanded && (
        <div className="max-h-64 overflow-y-auto divide-y divide-gray-700/30">
          {steps.map((step, index) => {
            const stepName = getStepName(step);
            const formattedContent = formatStepContent(step, stepName);
            const isComplete = stepName.includes('Complete');
            const isStart = stepName.includes('Start');
            
            return (
              <div 
                key={step.id || step.content?.id || index} 
                className="px-3 py-2 bg-gray-800/30"
              >
                <div className="flex items-center gap-2">
                  <span className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold ${
                    isComplete ? 'bg-green-500/20 text-green-400' : 
                    isStart ? 'bg-blue-500/20 text-blue-400' : 
                    'bg-accent/20 text-accent'
                  }`}>
                    {index + 1}
                  </span>
                  <span className={`text-xs font-medium ${
                    isComplete ? 'text-green-400' : isStart ? 'text-blue-400' : 'text-gray-300'
                  }`}>
                    {stepName}
                  </span>
                </div>
                {formattedContent && (
                  <div className="mt-1.5 ml-7 text-xs text-gray-400">
                    <div className="whitespace-pre-wrap break-words leading-relaxed">
                      {formattedContent}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export const ChatMessage: FC<ChatMessageProps> = memo(({ message, isStreaming }) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';

  const handleCopy = () => {
    if (!navigator.clipboard) return;
    
    navigator.clipboard.writeText(message.content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  // Don't render empty assistant messages (unless streaming)
  if (isAssistant && !message.content && !message.intermediateSteps?.length && !isStreaming) {
    return null;
  }

  return (
    <div
      className={`group px-4 py-6 ${
        isUser
          ? 'bg-canvas-light'
          : 'bg-canvas border-y border-gray-700/50'
      }`}
    >
      <div className="max-w-3xl mx-auto flex gap-4">
        {/* Avatar */}
        <div className="flex-shrink-0">
          <div
            className={`w-8 h-8 rounded-lg flex items-center justify-center ${
              isUser
                ? 'bg-blue-500/20 text-blue-400'
                : 'bg-accent/20 text-accent'
            }`}
          >
            {isUser ? <User size={18} /> : <Bot size={18} />}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Role label */}
          <div className="mb-1">
            <span className={`text-sm font-medium ${isUser ? 'text-blue-400' : 'text-accent'}`}>
              {isUser ? 'You' : 'Assistant'}
            </span>
          </div>

          {/* Message content */}
          <div className="prose prose-invert prose-sm max-w-none">
            <div className="text-gray-200 whitespace-pre-wrap break-words">
              {message.content}
              {isStreaming && isAssistant && (
                <span className="text-accent animate-pulse ml-1">▍</span>
              )}
            </div>
          </div>

          {/* Intermediate steps */}
          {message.intermediateSteps && message.intermediateSteps.length > 0 && (
            <IntermediateStepsDisplay steps={message.intermediateSteps} />
          )}

          {/* Error messages */}
          {message.errorMessages && message.errorMessages.length > 0 && (
            <div className="mt-2 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
              {message.errorMessages.map((err, index) => (
                <div key={index} className="text-sm text-red-400">
                  {err.content?.text || err.content?.error || 'An error occurred'}
                </div>
              ))}
            </div>
          )}

          {/* Copy button - only for assistant messages */}
          {isAssistant && message.content && !isStreaming && (
            <div className="mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={handleCopy}
                className="flex items-center gap-1 text-xs text-gray-500 hover:text-accent transition-colors"
                title="Copy to clipboard"
              >
                {copied ? (
                  <>
                    <Check size={14} className="text-accent" />
                    <span className="text-accent">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy size={14} />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

ChatMessage.displayName = 'ChatMessage';

export default ChatMessage;

