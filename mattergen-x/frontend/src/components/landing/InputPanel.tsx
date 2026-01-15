"use client";

import { useState } from 'react';

export default function InputPanel({ onGenerate }: { onGenerate: (prompt: str, weights: any) => void }) {
  const [prompt, setPrompt] = useState("");
  const [weights, setWeights] = useState({
    stability: 0.5,
    conductivity: 0.5,
    bandGap: 0.5
  });

  const handleWeightChange = (key: string, val: string) => {
    setWeights(prev => ({ ...prev, [key]: parseFloat(val) }));
  };

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 shadow-2xl">
      <h2 className="text-lg font-semibold mb-4 flex items-center space-x-2">
        <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
        <span>Design Specification</span>
      </h2>
      
      <div className="space-y-6">
        <div>
          <label className="block text-xs font-mono text-zinc-500 uppercase mb-2">Natural Language Prompt</label>
          <textarea 
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g., Generate a stable semiconductor with high thermal conductivity..."
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-4 text-sm focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition-all h-32 resize-none"
          />
        </div>

        <div>
          <label className="block text-xs font-mono text-zinc-500 uppercase mb-4">Property Weighting</label>
          <div className="space-y-4">
            {Object.entries(weights).map(([key, val]) => (
              <div key={key} className="flex flex-col space-y-1">
                <div className="flex justify-between items-center px-1">
                  <span className="text-xs capitalize">{key.replace(/([A-Z])/g, ' $1')}</span>
                  <span className="text-xs font-mono text-emerald-500">{val.toFixed(2)}</span>
                </div>
                <input 
                  type="range" 
                  min="0" 
                  max="1" 
                  step="0.05"
                  value={val}
                  onChange={(e) => handleWeightChange(key, e.target.value)}
                  className="w-full accent-emerald-500 h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            ))}
          </div>
        </div>

        <button 
          onClick={() => onGenerate(prompt, weights)}
          className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-3 rounded-lg transition-all transform active:scale-[0.98] shadow-lg shadow-emerald-900/20"
        >
          GENERATE CANDIDATES
        </button>
      </div>
    </div>
  );
}
