import React, { useState } from 'react';
import { WeightSliders } from './WeightSliders';

interface InputPanelProps {
  onGenerate: (prompt: string, weights: Record<string, number>) => Promise<void>;
  isLoading: boolean;
  error: string | null;
}

/**
 * Component for the project specification input panel.
 * Encapsulates the prompt textarea and the submission button.
 */
export default function InputPanel({ onGenerate, isLoading, error }: InputPanelProps) {
  const [prompt, setPrompt] = useState("");
  const [weights, setWeights] = useState({
    density: 0.5,
    stability: 0.5,
    band_gap: 0.5
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onGenerate(prompt, weights);
  };

  const quickPrompts = [
    "High-stability perovskite solar cell",
    "Ultra-hard carbon structure",
    "Transparent conductor"
  ];

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm ring-1 ring-gray-950/5">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <div className="flex justify-between items-center mb-2">
            <label htmlFor="prompt" className="text-sm font-semibold text-gray-900">
              Target Specification
            </label>
            <span className="text-xs text-gray-400">Describe your ideal material</span>
          </div>
          
          <textarea
            id="prompt"
            rows={4}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="E.g., A stable semiconductor with specific band gap..."
            className="w-full bg-gray-50 border border-gray-200 rounded-lg p-3 text-sm text-gray-900 placeholder:text-gray-400 focus:bg-white focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none resize-none transition-all"
          />

          {/* Quick Prompts */}
          <div className="mt-3 flex flex-wrap gap-2">
            {quickPrompts.map((p, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setPrompt(p)}
                className="text-xs px-2.5 py-1 bg-gray-100 hover:bg-indigo-50 text-gray-600 hover:text-indigo-600 rounded-full transition-colors border border-transparent hover:border-indigo-100"
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        {/* Sliders Section */}
        <div className="border-t border-gray-100 pt-6">
           <WeightSliders weights={weights} onChange={setWeights} />
        </div>

        {error && (
          <div className="p-3 bg-red-50 text-red-600 text-xs rounded-lg border border-red-100 flex items-start gap-2">
            <span className="mt-0.5 font-bold">!</span>
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={isLoading || !prompt.trim()}
          className="w-full py-3.5 bg-zinc-900 hover:bg-indigo-600 text-white text-sm font-medium rounded-lg shadow-sm hover:shadow-md hover:scale-[1.01] active:scale-[0.99] transition-all disabled:opacity-50 disabled:pointer-events-none flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Simulating structures...</span>
            </>
          ) : (
            <>
                <span>Generate Candidates</span>
                <svg className="w-4 h-4 opacity-70" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
