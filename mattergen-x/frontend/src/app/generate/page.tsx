"use client";
import React, { useState } from 'react';
import { ApiService } from '@/lib/api';
import { MaterialCandidate } from '@/types/api';

import MaterialCard from '@/components/MaterialCard';
import InputPanel from '@/components/InputPanel';
import { PeriodicTableHeatmap } from '@/components/visualization/PeriodicTableHeatmap';
import { AetherAssistant } from '@/components/visualization/AetherAssistant';
import { useMaterialGeneration } from '@/hooks/useMaterialGeneration';

import { motion } from 'framer-motion';
import { FADE_IN_ANIMATION } from '@/utils/animations';

export default function Home() {
  const { generate, isLoading, error, candidates } = useMaterialGeneration();
  
  // Lifted State for Input Panel & Aether Assistant
  const [prompt, setPrompt] = useState("");
  const [weights, setWeights] = useState<Record<string, number>>({
    density: 0.5,
    stability: 0.5,
    band_gap: 0.5,
    shear_modulus: 0.5,
    thermal_conductivity: 0.5,
    refractive_index: 0.5
  });

  return (
    <>
      <motion.div 
        className="text-gray-900 selection:bg-gray-100 pb-20 relative"
        {...FADE_IN_ANIMATION}
      >
        <main className="max-w-7xl mx-auto px-6 py-12 lg:py-20">
          
          {/* Header (Full Width) */}
          <div className="relative text-center max-w-4xl mx-auto mb-20">
              <div className="space-y-6 animate-in fade-in slide-in-from-top-4 duration-1000">
                  {/* System Status Line */}
                  <div className="flex items-center justify-center gap-6 text-[10px] font-mono font-medium text-slate-500 uppercase tracking-widest border-b border-slate-100 pb-6 w-fit mx-auto">
                      <div className="flex items-center gap-2">
                          <span className="relative flex h-2 w-2">
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                          </span>
                          <span>System Online</span>
                      </div>
                      <div className="hidden md:block w-px h-3 bg-slate-200"></div>
                      <div className="hidden md:block">Model: CGCNN-v2.1</div>
                      <div className="hidden md:block w-px h-3 bg-slate-200"></div>
                      <div className="hidden md:block">Latency: 45ms</div>
                  </div>

                  <h2 className="text-5xl md:text-7xl font-bold tracking-tighter text-slate-900 leading-none">
                        Discovery <span className="font-light text-slate-400">Console</span>
                  </h2>
                  
                  <p className="text-xl text-slate-500 max-w-2xl mx-auto leading-relaxed font-light">
                        Synthesize novel crystal structures using our state-of-the-art <span className="font-semibold text-slate-900">Generative AI</span>. 
                        Optimize for stability, band gap, and energy in real-time.
                  </p>
              </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">
              
              {/* Left Column: Input Panel (4/12) */}
              <div className="lg:col-span-5 xl:col-span-4 sticky top-10 space-y-8 animate-in fade-in slide-in-from-left-4 duration-700 z-50">
                  <div className="w-full relative group">
                      {/* Glow Effect */}
                      <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl blur opacity-10 group-hover:opacity-20 transition duration-500"></div>
                      
                      <div className="relative">
                          <InputPanel 
                              onGenerate={generate} 
                              isLoading={isLoading} 
                              error={error}
                              prompt={prompt}
                              setPrompt={setPrompt}
                              weights={weights}
                              setWeights={setWeights}
                          />
                      </div>
                  </div>
              </div>

              {/* Right Column: Results (8/12) */}
              <div className="lg:col-span-7 xl:col-span-8 space-y-8">
                  <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-gray-100 pb-6">
                      <div>
                        <h3 className="text-2xl font-bold text-gray-900 tracking-tight flex items-center gap-2">
                              Proposed Candidates
                              <span className="flex h-2 w-2 relative">
                                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                              </span>
                        </h3>
                        <p className="text-sm text-gray-500 mt-1">AI-generated structures matching your criteria.</p>
                      </div>
                      {candidates.length > 0 && (
                          <div className="px-4 py-1.5 bg-gray-900 text-white text-xs font-mono rounded-md shadow-sm">
                              FOUND: {candidates.length}
                          </div>
                      )}
                  </div>

                  {candidates.length > 0 ? (
                      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-8 duration-700">
                        {candidates.map((candidate) => (
                          <MaterialCard key={candidate.id} candidate={candidate} />
                        ))}
                      </div>
                  ) : (
                      <div className="relative rounded-3xl border border-dashed border-gray-200 bg-gray-50/30 p-12 lg:p-24 text-center overflow-hidden h-[600px] flex flex-col items-center justify-center">
                          {/* Placeholder Content */}
                          <div className="absolute inset-0 opacity-[0.02] pointer-events-none" 
                                  style={{ backgroundImage: 'linear-gradient(#000 1px, transparent 1px), linear-gradient(90deg, #000 1px, transparent 1px)', backgroundSize: '48px 48px' }}>
                          </div>
                          
                          <div className="relative z-10 max-w-md mx-auto">
                              <div className="mx-auto w-16 h-16 bg-white rounded-2xl shadow-sm border border-gray-100 flex items-center justify-center mb-6">
                                  <svg className="w-8 h-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                                      </svg>
                              </div>
                              <h3 className="text-xl font-bold text-gray-900 mb-2">Awaiting Input</h3>
                              <p className="text-gray-500">
                                  The generation engine is idle. Configure the parameters on the left to synthesize new materials.
                              </p>
                          </div>
                      </div>
                  )}
              </div>
          </div>

          {/* Material Discovery Map */}
          <section className="mt-32 pt-16 border-t border-gray-100">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-12 gap-6">
               <div className="max-w-2xl">
                  <h2 className="text-3xl font-black tracking-tight mb-4 text-gray-900">Chemical Space Intelligence</h2>
                  <p className="text-gray-500 leading-relaxed">
                    Real-time analysis of the generative model's chemical preferences. 
                    Elements are color-coded by the <span className="font-bold text-emerald-600">mean stability (energy above hull)</span> of candidates containing them.
                  </p>
               </div>
               <div className="hidden md:block">
                   <div className="flex items-center gap-4 text-xs font-mono text-gray-500 bg-white px-4 py-2 rounded-full border border-gray-100 shadow-sm">
                      <div className="flex items-center gap-2"><span className="block w-2 h-2 rounded-full bg-emerald-500"></span> Highly Stable</div>
                      <div className="flex items-center gap-2"><span className="block w-2 h-2 rounded-full bg-amber-400"></span> Metastable</div>
                      <div className="flex items-center gap-2"><span className="block w-2 h-2 rounded-full bg-rose-400"></span> Unstable</div>
                   </div>
               </div>
            </div>
            
            <div className="bg-white rounded-3xl p-8 border border-gray-200 shadow-xl shadow-gray-100/50">
                 <PeriodicTableHeatmap />
            </div>
          </section>
        </main>
      </motion.div>

      <AetherAssistant 
        onApplySuggestions={(p, w) => {
          setPrompt(p);
          if (w) setWeights(prev => ({ ...prev, ...w }));
        }} 
      />
    </>
  );
}
