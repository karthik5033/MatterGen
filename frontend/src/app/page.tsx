"use client";

import { useState } from 'react';
import InputPanel from '@/components/InputPanel';
import CandidateCard from '@/components/CandidateCard';

export default function Dashboard() {
  const [candidates, setCandidates] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleGenerate = async (prompt: string, weights: any) => {
    setIsLoading(true);
    // Simulate API call
    // In a real app, this would be: 
    // const res = await fetch('http://localhost:8000/api/v1/generate/', { ... })
    
    setTimeout(() => {
      setCandidates([
        {
          id: "mat_001",
          formula: "Fe2O3",
          properties: { band_gap: 2.1, stability: 0.95, conductivity: 0.1 },
          crystal_structure: "..."
        },
        {
          id: "mat_002",
          formula: "SrTiO3",
          properties: { band_gap: 3.2, stability: 0.99, conductivity: 0.05 },
          crystal_structure: "..."
        },
        {
            id: "mat_003",
            formula: "CsPbI3",
            properties: { band_gap: 1.7, stability: 0.4, conductivity: 0.8 },
            crystal_structure: "..."
          }
      ]);
      setIsLoading(false);
    }, 1500);
  };

  return (
    <div className="max-w-6xl mx-auto">
      <header className="mb-10">
        <h1 className="text-4xl font-bold tracking-tight mb-2">Discovery Workflow</h1>
        <p className="text-zinc-400 max-w-2xl text-lg">
          Specify material constraints using natural language and numerical weights. 
          The generative AI will propose candidate structures optimized for your requirements.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-4">
          <InputPanel onGenerate={handleGenerate} />
        </div>

        <div className="lg:col-span-8">
          <div className="bg-zinc-900/30 border border-zinc-800 rounded-xl min-h-[600px] p-6 relative">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-sm font-mono text-zinc-500 uppercase tracking-widest">Candidate Generations</h2>
              <div className="flex space-x-2">
                <span className="text-[10px] text-zinc-500 font-mono">Found: {candidates.length}</span>
              </div>
            </div>

            {isLoading ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4">
                <div className="w-12 h-12 border-2 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin" />
                <p className="text-sm font-mono text-zinc-500">Processing latent space optimization...</p>
              </div>
            ) : candidates.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {candidates.map((c) => (
                  <CandidateCard key={c.id} material={c} />
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-[500px] text-zinc-600 border border-dashed border-zinc-800 rounded-lg">
                <p className="text-sm font-mono">NO ACTIVE GENERATIONS</p>
                <p className="text-xs">Awaiting design specification input</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
