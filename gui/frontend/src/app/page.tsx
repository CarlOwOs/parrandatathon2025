'use client';

import { useState } from 'react';
import { Globe, Plus, Mic, MessageSquare, MoreHorizontal } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    
    try {
      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: input }),
      });
      
      const data = await res.json();
      const assistantMessage: Message = { role: 'assistant', content: data.response };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = { role: 'assistant', content: 'Error occurred while fetching response' };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[#1E1E1E] text-white">
      <div className="flex-1 overflow-y-auto p-4">
        <h1 className="text-4xl font-bold text-center mb-8 mt-4">What can I help with?</h1>
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-[80%] rounded-lg p-4 ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-[#2D2D2D] text-gray-100'
                }`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-[#2D2D2D] rounded-lg p-4">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-gray-800 p-4">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="relative flex items-center gap-2 bg-[#2D2D2D] rounded-lg p-2">
            <button
              type="button"
              className="p-2 hover:bg-[#3D3D3D] rounded-lg transition-colors"
            >
              <Plus className="w-5 h-5" />
            </button>
            
            <button
              type="button"
              className="flex items-center gap-2 p-2 hover:bg-[#3D3D3D] rounded-lg transition-colors"
            >
              <Globe className="w-5 h-5" />
              <span>Search</span>
            </button>
            
            <button
              type="button"
              className="flex items-center gap-2 p-2 hover:bg-[#3D3D3D] rounded-lg transition-colors"
            >
              <MessageSquare className="w-5 h-5" />
              <span>Deep research</span>
            </button>
            
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask anything"
              className="flex-1 bg-transparent p-2 focus:outline-none text-white placeholder-gray-400"
              disabled={isLoading}
            />
            
            <button
              type="button"
              className="p-2 hover:bg-[#3D3D3D] rounded-lg transition-colors"
            >
              <MoreHorizontal className="w-5 h-5" />
            </button>
            
            <button
              type="button"
              className="p-2 hover:bg-[#3D3D3D] rounded-lg transition-colors"
            >
              <Mic className="w-5 h-5" />
            </button>
            
            <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center">
              <div className="w-4 h-4 bg-black"></div>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
