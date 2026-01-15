"use client";

import { useState } from 'react';
import { ApiService } from '@/lib/api';
import { MaterialCandidate } from '@/types/api';

import MaterialCard from '@/components/MaterialCard';
import InputPanel from '@/components/InputPanel';
import { EmbeddingMap } from '@/components/visualization/EmbeddingMap';
import { useMaterialGeneration } from '@/hooks/useMaterialGeneration';

import { motion } from 'framer-motion';
import { FADE_IN_ANIMATION } from '@/utils/animations';

export default function Home() {
  const { generate, isLoading, error, candidates } = useMaterialGeneration();

  return (
    <motion.div 
      className="text-gray-900 selection:bg-gray-100 pb-20"
      {...FADE_IN_ANIMATION}
    >
      <main className="max-w-7xl mx-auto px-6 py-20 lg:py-32">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-16">
          
          {/* Left Column: Input */}
          <div className="lg:col-span-5">
            <div className="sticky top-32">
              <h2 className="text-4xl lg:text-5xl font-bold tracking-tight mb-4 text-gray-900">
                AI-Driven Material Discovery
              </h2>
              <p className="text-lg text-gray-600 mb-8 leading-relaxed">
                Accelerate research by generating stable, high-performance crystal candidates tailored to your constraints using Generative AI.
              </p>

              <InputPanel 
                onGenerate={generate} 
                isLoading={isLoading} 
                error={error} 
              />
            </div>
          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-7">
            {candidates.length > 0 ? (
              <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
                <div className="flex justify-between items-end px-1">
                  <h3 className="text-xs font-extrabold uppercase tracking-[0.2em] text-gray-400">Proposed Candidates</h3>
                  <span className="text-[10px] font-mono text-gray-300 uppercase">Found: {candidates.length}</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-1 gap-6">
                  {candidates.map((candidate) => (
                    <MaterialCard key={candidate.id} candidate={candidate} />
                  ))}
                </div>
              </div>
            ) : (
              <div className="h-full min-h-[500px] flex flex-col items-center justify-center border border-dashed border-gray-200 rounded-3xl p-12 text-center bg-gray-50/50 relative overflow-hidden">
                {/* Background Decoration */}
                <div className="absolute inset-0 opacity-[0.03] pointer-events-none" 
                     style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, black 1px, transparent 0)', backgroundSize: '32px 32px' }}>
                </div>

                <div className="relative z-10 max-w-md mx-auto">
                    <div className="w-20 h-20 bg-white rounded-2xl shadow-sm border border-gray-100 flex items-center justify-center mx-auto mb-8 rotate-3 transition-transform hover:rotate-6">
                         <svg className="w-8 h-8 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                           <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                         </svg>
                    </div>

                    <h3 className="text-lg font-bold text-gray-900 mb-2">Ready to Discover?</h3>
                    <p className="text-gray-500 mb-8">
                       Define your target properties on the left to begin the generative process.
                    </p>

                    {/* Steps */}
                    <div className="grid grid-cols-3 gap-4 text-left">
                        {[
                            { step: "01", title: "Define", desc: "Set target properties & constraints" },
                            { step: "02", title: "Generate", desc: "AI creates candidate structures" },
                            { step: "03", title: "Evaluate", desc: "Review stability & band gap" }
                        ].map((s, i) => (
                            <div key={i} className="bg-white p-4 rounded-xl border border-gray-100 shadow-sm">
                                <span className="text-xs font-bold text-indigo-400 mb-1 block">{s.step}</span>
                                <h4 className="text-sm font-semibold text-gray-900">{s.title}</h4>
                                <p className="text-[10px] text-gray-400 mt-1 leading-tight">{s.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
              </div>
            )}
          </div>

        </div>

        {/* Material Discovery Map */}
        <section className="mt-24 pt-12 border-t border-gray-100">
          <div className="mb-8">
            <h2 className="text-2xl font-black tracking-tight mb-2">Material Manifold</h2>
            <p className="text-gray-500 max-w-2xl">
              Explore the latent space of the Generative Model. Each point represents a material crystal structure, projected from 64-dimensional feature space. 
              Clusters indicate chemical similarity.
            </p>
          </div>
          <EmbeddingMap height={600} />
        </section>
      </main>
    </motion.div>
  );
}
