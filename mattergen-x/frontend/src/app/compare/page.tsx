"use client";

import { useState } from 'react';
import { motion } from 'framer-motion';
import { FADE_IN_ANIMATION } from '@/utils/animations';
import ComparisonCharts from '@/components/charts/ComparisonCharts';

// Mock Data for Comparison
const MOCK_MATERIALS = [
  { id: 'mat-1', name: 'LiFePO4', density: 3.6, stability: 0.95, band_gap: 3.2, formation_energy: -2.1 },
  { id: 'mat-2', name: 'SiC', density: 3.21, stability: 0.98, band_gap: 2.4, formation_energy: -0.8 },
  { id: 'mat-3', name: 'GaN', density: 6.15, stability: 0.85, band_gap: 3.4, formation_energy: -1.2 },
  { id: 'mat-4', name: 'SrTiO3', density: 5.12, stability: 0.92, band_gap: 3.25, formation_energy: -3.4 },
  { id: 'mat-5', name: 'BaTiO3', density: 6.02, stability: 0.88, band_gap: 3.1, formation_energy: -3.1 },
];

export default function ComparePage() {
  const [selectedIds, setSelectedIds] = useState<string[]>(['mat-1', 'mat-2']);

  const toggleSelection = (id: string) => {
    setSelectedIds(prev => 
      prev.includes(id) 
        ? prev.filter(x => x !== id) 
        : [...prev, id]
    );
  };

  // Filter data for charts
  const chartData = MOCK_MATERIALS.filter(m => selectedIds.includes(m.id));

  return (
    <motion.div 
      className="min-h-screen bg-gray-50 pb-20"
      {...FADE_IN_ANIMATION}
    >
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-12">
           <h1 className="text-3xl font-bold text-gray-900 mb-2">Material Comparison</h1>
           <p className="text-gray-500">Analyze trade-offs between candidate materials properties side-by-side.</p>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Sidebar: Selection */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm sticky top-24">
               <h3 className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-4">Select Candidates</h3>
               <div className="space-y-3">
                 {MOCK_MATERIALS.map(mat => (
                   <label key={mat.id} className="flex items-center gap-3 cursor-pointer group">
                      <input 
                        type="checkbox" 
                        checked={selectedIds.includes(mat.id)}
                        onChange={() => toggleSelection(mat.id)}
                        className="w-4 h-4 rounded border-gray-300 text-black focus:ring-black transition-all"
                      />
                      <div>
                        <div className="font-bold text-gray-700 group-hover:text-black transition-colors">{mat.name}</div>
                        <div className="text-[10px] text-gray-400 font-mono">ID: {mat.id}</div>
                      </div>
                   </label>
                 ))}
               </div>
               
               <div className="mt-8 pt-6 border-t border-gray-100">
                 <div className="text-xs text-gray-400">
                    <span className="font-bold text-black">{selectedIds.length}</span> materials selected
                 </div>
               </div>
            </div>
          </div>

          {/* Main Content: Charts */}
          <div className="lg:col-span-9 space-y-8">
             {selectedIds.length > 0 ? (
                <ComparisonCharts data={chartData} />
             ) : (
               <div className="h-64 flex items-center justify-center border-2 border-dashed border-gray-200 rounded-xl text-gray-400">
                 Select materials to compare
               </div>
             )}

             {/* Detailed Table */}
             {selectedIds.length > 0 && (
                <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                   <table className="w-full text-sm text-left">
                      <thead className="bg-gray-50 text-xs text-gray-500 uppercase font-bold tracking-wider">
                        <tr>
                          <th className="px-6 py-4">Formula</th>
                          <th className="px-6 py-4">Density <span className="text-[9px] lowercase opacity-75">(g/cm³)</span></th>
                          <th className="px-6 py-4">Stability <span className="text-[9px] lowercase opacity-75">(0-1)</span></th>
                          <th className="px-6 py-4">Band Gap <span className="text-[9px] lowercase opacity-75">(eV)</span></th>
                          <th className="px-6 py-4">Form. Energy <span className="text-[9px] lowercase opacity-75">(eV)</span></th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {chartData.map(mat => (
                          <tr key={mat.id} className="hover:bg-gray-50 transition-colors">
                            <td className="px-6 py-4 font-bold text-gray-900">{mat.name}</td>
                            <td className="px-6 py-4 text-gray-600 font-mono">{mat.density.toFixed(2)}</td>
                            <td className="px-6 py-4 text-gray-600 font-mono">{mat.stability.toFixed(2)}</td>
                            <td className="px-6 py-4 text-gray-600 font-mono">{mat.band_gap.toFixed(2)}</td>
                            <td className="px-6 py-4 text-gray-600 font-mono">{mat.formation_energy.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                   </table>
                </div>
             )}
          </div>

        </div>
      </main>
    </motion.div>
  );
}
