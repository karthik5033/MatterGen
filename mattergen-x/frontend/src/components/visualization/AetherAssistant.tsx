
"use client";

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ApiService } from '@/lib/api';

interface AetherAssistantProps {
    onApplySuggestions: (prompt: string, weights: Record<string, number>) => void;
}

export const AetherAssistant = ({ onApplySuggestions }: AetherAssistantProps) => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState<{role: 'user' | 'assistant', content: string, suggestion?: any}[]>([
        { role: 'assistant', content: "Greetings. I am Aether, your Discovery Assistant. How can I help you refine your material search today?" }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const chatEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        if (isOpen) scrollToBottom();
    }, [messages, isOpen]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMsg = input;
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        setIsLoading(true);

        try {
            const result = await ApiService.chatRefine(userMsg);
            setMessages(prev => [...prev, { 
                role: 'assistant', 
                content: result.response,
                suggestion: result.suggested_prompt ? { 
                    prompt: result.suggested_prompt, 
                    weights: result.suggested_weights 
                } : null
            }]);
        } catch (e) {
            setMessages(prev => [...prev, { role: 'assistant', content: "I encountered a synchronization error. Please check your connection." }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="relative z-[60]">
            {/* Toggle Button (Floating) */}
            <AnimatePresence>
                {!isOpen && (
                    <motion.button
                        initial={{ scale: 0, rotate: -180 }}
                        animate={{ scale: 1, rotate: 0 }}
                        exit={{ scale: 0, rotate: 180 }}
                        onClick={() => setIsOpen(true)}
                        className="fixed bottom-6 right-6 w-14 h-14 bg-slate-900 rounded-full shadow-2xl flex items-center justify-center text-white hover:scale-110 active:scale-95 transition-transform z-50 group border border-slate-700"
                    >
                         <div className="absolute inset-0 rounded-full bg-indigo-500/20 animate-ping"></div>
                         <div className="relative">
                            {/* Aether Icon */}
                            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                            </svg>
                         </div>
                    </motion.button>
                )}
            </AnimatePresence>

            {/* Chat Window (Docked Bottom Right) */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 20, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 20, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                        className="fixed bottom-6 right-6 w-[90vw] md:w-[380px] h-[600px] max-h-[80vh] bg-white rounded-3xl shadow-2xl flex flex-col overflow-hidden border border-gray-200 z-50 font-sans"
                    >
                        {/* Header */}
                        <div className="bg-slate-900 px-5 py-4 text-white flex justify-between items-center relative shrink-0">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-[10px] font-black shadow-inner">
                                    AE
                                </div>
                                <div>
                                    <div className="text-[10px] font-bold uppercase tracking-wider text-indigo-400">Aether Assistant</div>
                                    <div className="text-sm font-semibold opacity-90">Discovery Engine Online</div>
                                </div>
                            </div>
                            <button 
                                onClick={() => setIsOpen(false)} 
                                className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-white/10 transition-colors"
                            >
                                <svg className="w-5 h-5 opacity-70" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>

                        {/* Messages Area */}
                        <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar bg-slate-50/50">
                             <style jsx>{`
                                .custom-scrollbar::-webkit-scrollbar { width: 4px; }
                                .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
                                .custom-scrollbar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
                            `}</style>
                            
                            {messages.map((m, i) => (
                                <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
                                    <div className={`max-w-[85%] p-3.5 rounded-2xl text-[13px] leading-relaxed shadow-sm ${
                                        m.role === 'user' 
                                        ? 'bg-slate-900 text-white rounded-tr-sm' 
                                        : 'bg-white text-slate-800 border border-gray-100 rounded-tl-sm'
                                    }`}>
                                        <div className="whitespace-pre-wrap">{m.content}</div>
                                        {m.suggestion && (
                                            <div className="mt-3 pt-3 border-t border-gray-100">
                                                <div className="flex items-center justify-between mb-2">
                                                    <span className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Configuration Ready</span>
                                                </div>
                                                <button 
                                                    onClick={() => {
                                                        onApplySuggestions(m.suggestion.prompt, m.suggestion.weights);
                                                        setIsOpen(false);
                                                    }}
                                                    className="w-full py-2 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-xs font-bold rounded-lg transition-colors flex items-center justify-center gap-2 border border-indigo-200"
                                                >
                                                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                                    </svg>
                                                    Apply Settings
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                            
                            {isLoading && (
                                <div className="flex justify-start">
                                    <div className="bg-white px-4 py-3 rounded-2xl rounded-tl-sm border border-gray-100 shadow-sm flex gap-1.5 items-center">
                                        <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"></span>
                                        <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0.2s]"></span>
                                        <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0.4s]"></span>
                                    </div>
                                </div>
                            )}
                            <div ref={chatEndRef} />
                        </div>

                        {/* Input Area */}
                        <div className="p-3 bg-white border-t border-gray-100">
                            <div className="relative flex items-center gap-2">
                                <input 
                                    type="text"
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                                    placeholder="Type to refine..."
                                    className="flex-1 bg-slate-50 border border-gray-200 rounded-xl px-4 py-2.5 text-xs focus:ring-2 focus:ring-indigo-500/10 focus:border-indigo-500 transition-all outline-none text-slate-900 placeholder:text-slate-400"
                                />
                                <button 
                                    onClick={handleSend}
                                    disabled={!input.trim() || isLoading}
                                    className="w-10 h-10 rounded-xl bg-slate-900 text-white flex items-center justify-center hover:bg-black transition-all disabled:opacity-50 shadow-md active:scale-95"
                                >
                                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
                                    </svg>
                                </button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
