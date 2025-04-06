import React, { useState, useRef, useEffect } from 'react';
import { FiSend, FiCpu } from 'react-icons/fi';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Message {
  text: string;
  isUser: boolean;
}

const ThinkingAnimation = ({ useReasoning }: { useReasoning: boolean }) => (
  <div className="flex justify-start">
    <div className="bg-secondary text-white rounded-lg p-3 max-w-[80%] flex items-center space-x-2">
      <div className="text-sm">
        {useReasoning 
          ? "Hunting very very hard for the best answer" 
          : "Hunting for the best answer"}
      </div>
      <div className="flex space-x-1">
        <div className="w-1 h-1 bg-accent rounded-full animate-[bounce_1.4s_infinite]" style={{ animationDelay: '0s' }} />
        <div className="w-1 h-1 bg-accent rounded-full animate-[bounce_1.4s_infinite]" style={{ animationDelay: '0.2s' }} />
        <div className="w-1 h-1 bg-accent rounded-full animate-[bounce_1.4s_infinite]" style={{ animationDelay: '0.4s' }} />
      </div>
    </div>
  </div>
);

const SearchBar: React.FC = () => {
  const [query, setQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [useReasoning, setUseReasoning] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isProcessing]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isProcessing) return;

    const userMessage = query.trim();
    setMessages(prev => [...prev, { text: userMessage, isUser: true }]);
    setQuery('');
    
    try {
      setIsProcessing(true);

      const endpoint = useReasoning ? '/api/query/agent' : '/api/query';
      const result = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage,
          system_prompt: "You are a helpful assistant that answers questions based on the provided context."
        }),
      });

      if (!result.ok) {
        throw new Error('Failed to fetch response');
      }

      const reader = result.body?.getReader();
      if (!reader) throw new Error('No reader available');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = new TextDecoder().decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            if (data.status === 'completed') {
              setMessages(prev => [...prev, { text: data.data.response, isUser: false }]);
              break;
            }
          } catch (e) {
            console.error('Error parsing JSON:', e);
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { text: 'An error occurred while processing your request.', isUser: false }]);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-primary rounded-lg shadow-lg overflow-hidden max-w-4xl mx-auto">
      {/* Messages area with padding to prevent content from being hidden behind input */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="p-4 pb-2">
          <div className="space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    message.isUser
                      ? 'bg-accent text-black ml-auto'
                      : 'bg-secondary text-white'
                  }`}
                >
                  {message.isUser ? (
                    <p className="whitespace-pre-wrap text-sm">{message.text}</p>
                  ) : (
                    <div className="prose prose-sm prose-invert">
                      <ReactMarkdown 
                        remarkPlugins={[remarkGfm]}
                        components={{
                          p: ({node, ...props}) => <p className="text-sm my-1" {...props} />,
                          ul: ({node, ...props}) => <ul className="my-2 list-disc list-inside" {...props} />,
                          ol: ({node, ...props}) => <ol className="my-2 list-decimal list-inside" {...props} />,
                          li: ({node, ...props}) => <li className="my-0.5" {...props} />,
                          strong: ({node, ...props}) => <strong className="text-accent font-bold" {...props} />,
                          h1: ({node, ...props}) => <h1 className="text-lg font-bold my-2" {...props} />,
                          h2: ({node, ...props}) => <h2 className="text-base font-bold my-2" {...props} />,
                          h3: ({node, ...props}) => <h3 className="text-sm font-bold my-1" {...props} />,
                          code: ({node, inline, ...props}) => 
                            inline ? 
                              <code className="bg-black/20 rounded px-1 py-0.5 text-xs" {...props} /> :
                              <code className="block bg-black/20 rounded p-2 text-xs my-2 overflow-x-auto" {...props} />,
                        }}
                      >
                        {message.text}
                      </ReactMarkdown>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isProcessing && <ThinkingAnimation useReasoning={useReasoning} />}
            <div ref={messagesEndRef} className="h-1" />
          </div>
        </div>
      </div>

      {/* Input area - fixed at the bottom */}
      <div className="border-t border-gray-700 bg-primary p-3">
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setUseReasoning(!useReasoning)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg ${
              useReasoning 
                ? 'bg-accent text-black' 
                : 'bg-secondary text-white hover:bg-secondary/80'
            } transition-colors`}
            title="Toggle Reasoning Mode"
          >
            <FiCpu size={18} />
            <span className="text-sm font-medium">DeepSupplyThink</span>
          </button>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about suppliers, logistics, or procurement insights..."
            className="flex-1 px-3 py-2 bg-secondary rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-accent text-sm"
            disabled={isProcessing}
          />
          <button
            type="submit"
            className={`p-2 rounded-full text-accent hover:bg-white/10 ${
              isProcessing ? 'opacity-50 cursor-not-allowed' : ''
            }`}
            disabled={isProcessing}
          >
            <FiSend size={18} />
          </button>
        </form>
      </div>
    </div>
  );
};

export default SearchBar; 