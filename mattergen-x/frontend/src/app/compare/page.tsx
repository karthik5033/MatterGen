"use client";

import { useState } from 'react';
import { motion } from 'framer-motion';
import { FADE_IN_ANIMATION } from '@/utils/animations';
import ComparisonCharts from '@/components/charts/ComparisonCharts';
import { Search, Plus, BarChart3, FlaskConical, Atom, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';

// Mock Data for Comparison
const MOCK_MATERIALS = [
  { id: 'mat-1', name: 'LiFePO4', density: 3.6, stability: 0.95, band_gap: 3.2, formation_energy: -2.1, type: 'Cathode' },
  { id: 'mat-2', name: 'SiC', density: 3.21, stability: 0.98, band_gap: 2.4, formation_energy: -0.8, type: 'Semiconductor' },
  { id: 'mat-3', name: 'GaN', density: 6.15, stability: 0.85, band_gap: 3.4, formation_energy: -1.2, type: 'Semiconductor' },
  { id: 'mat-4', name: 'SrTiO3', density: 5.12, stability: 0.92, band_gap: 3.25, formation_energy: -3.4, type: 'Perovskite' },
  { id: 'mat-5', name: 'BaTiO3', density: 6.02, stability: 0.88, band_gap: 3.1, formation_energy: -3.1, type: 'Perovskite' },
];

export default function ComparePage() {
  const [selectedIds, setSelectedIds] = useState<string[]>(['mat-1', 'mat-2', 'mat-4']);

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
      className="min-h-screen bg-[#fafafa] pb-20 pt-24"
      {...FADE_IN_ANIMATION}
    >
      {/* Header Section */}
      <div className="max-w-7xl mx-auto px-6 mb-8 flex items-end justify-between border-b border-gray-200 pb-8">
        <div>
           <div className="flex items-center gap-2 mb-2">
              <div className="p-1.5 bg-zinc-100 rounded-lg border border-zinc-200">
                <BarChart3 className="w-4 h-4 text-zinc-600" />
              </div>
              <span className="text-xs font-bold text-zinc-500 uppercase tracking-widest">Analytics Suite</span>
           </div>
           <h1 className="text-3xl font-bold text-zinc-900 tracking-tight">Comparative Analysis</h1>
           <p className="text-zinc-500 mt-2 max-w-2xl text-sm">
             Evaluate trade-offs between candidate materials across key physical electronic properties.
           </p>
        </div>
        <div className="hidden md:flex gap-3">
             <Button variant="outline" size="sm" className="bg-white hover:bg-zinc-50 text-zinc-700 border-zinc-200 shadow-sm transition-all hover:scale-[1.02]">
                <FlaskConical className="w-3.5 h-3.5 mr-2 text-zinc-500" />
                Run Simulation
             </Button>
             <Button size="sm" className="bg-zinc-900 hover:bg-zinc-800 text-white shadow-md transition-all hover:scale-[1.02]">
                <Download className="w-3.5 h-3.5 mr-2" />
                Export Report
             </Button>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Sidebar: Selection Panel */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-xl border border-zinc-200 shadow-sm sticky top-28 overflow-hidden">
               <div className="p-4 border-b border-zinc-100 bg-zinc-50/50">
                   <h3 className="text-xs font-bold uppercase tracking-widest text-zinc-400 mb-3">Candidate Pool</h3>
                   <div className="relative">
                       <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-400" />
                       <input 
                          type="text" 
                          placeholder="Search ID..." 
                          className="w-full pl-9 pr-3 py-2 text-xs bg-white border border-zinc-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-zinc-400 focus:border-zinc-400 transition-all placeholder:text-zinc-400"
                       />
                   </div>
               </div>
               
               <div className="p-2 max-h-[400px] overflow-y-auto space-y-1">
                 {MOCK_MATERIALS.map(mat => {
                    const isSelected = selectedIds.includes(mat.id);
                    return (
                       <div 
                          key={mat.id} 
                          onClick={() => toggleSelection(mat.id)}
                          className={`
                            group flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all border
                            ${isSelected 
                                ? 'bg-zinc-100 border-zinc-200' 
                                : 'bg-transparent border-transparent hover:bg-zinc-50 hover:border-zinc-100'}
                          `}
                       >
                          <div className={`
                             w-4 h-4 rounded border flex items-center justify-center transition-colors
                             ${isSelected 
                                ? 'bg-zinc-900 border-zinc-900' 
                                : 'bg-white border-zinc-300 group-hover:border-zinc-400'}
                          `}>
                             {isSelected && <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" /></svg>}
                          </div>
                          
                          <div className="flex-1 min-w-0">
                            <div className={`font-medium text-sm truncate ${isSelected ? 'text-zinc-900' : 'text-zinc-600'}`}>
                                {mat.name}
                            </div>
                            <div className="flex items-center justify-between mt-0.5">
                                <span className="text-[10px] text-zinc-400 font-mono uppercase">{mat.id}</span>
                                <span className={`text-[10px] px-1.5 py-0.5 rounded border ${isSelected ? 'bg-white border-zinc-200 text-zinc-700' : 'bg-transparent border-transparent text-zinc-400'}`}>
                                    {mat.type}
                                </span>
                            </div>
                          </div>
                       </div>
                    );
                 })}
               </div>
               
               <div className="p-3 border-t border-zinc-100 bg-zinc-50/30">
                 <button className="w-full py-2 flex items-center justify-center gap-2 text-xs font-medium text-zinc-500 hover:text-zinc-900 hover:bg-white border border-dashed border-zinc-200 hover:border-zinc-300 rounded-lg transition-all">
                    <Plus className="w-3.5 h-3.5" />
                    <span>Add Material</span>
                 </button>
               </div>
            </div>
          </div>

          {/* Main Content: Analysis Dashboard */}
          <div className="lg:col-span-9 space-y-6">
             {selectedIds.length > 0 ? (
                <div className="space-y-6">
                    {/* Charts Import */}
                    <ComparisonCharts data={chartData} />
                    
                    {/* Metrics Grid (Restored Neutral Colors) */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {['Density', 'Band Gap', 'Stability'].map((metric, i) => (
                             <div key={i} className="bg-white p-4 rounded-xl border border-zinc-200 shadow-sm flex items-center justify-between hover:shadow-md transition-shadow">
                                 <div>
                                     <div className="text-[10px] uppercase font-bold text-zinc-400 mb-1">{metric} Leader</div>
                                     <div className="font-bold text-zinc-800 text-lg">
                                         {chartData.length > 0 ? chartData[0].name : '-'}
                                     </div>
                                 </div>
                                 <div className="w-8 h-8 rounded-full flex items-center justify-center bg-zinc-50 border border-zinc-100 text-zinc-400">
                                     <Atom className="w-4 h-4" />
                                 </div>
                             </div>
                        ))}
                    </div>
                </div>
             ) : (
               <div className="h-96 flex flex-col items-center justify-center border-2 border-dashed border-zinc-200 rounded-2xl bg-zinc-50/50 text-zinc-400">
                 <BarChart3 className="w-12 h-12 mb-4 opacity-20" />
                 <p className="font-medium">No materials selected</p>
                 <p className="text-sm mt-1">Select at least one candidate from the sidebar</p>
               </div>
             )}

             {/* Scientific Data Table (Neutral Theme) */}
             {selectedIds.length > 0 && (
                <div className="bg-white border border-zinc-200 rounded-xl overflow-hidden shadow-sm">
                   <div className="px-6 py-4 border-b border-zinc-100 flex justify-between items-center bg-zinc-50/30">
                        <h3 className="font-semibold text-zinc-700 text-sm flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-zinc-400"></span>
                            Property Matrix
                        </h3>
                        <button className="text-xs text-zinc-500 font-medium hover:text-zinc-900 hover:underline">Download CSV</button>
                   </div>
                   <table className="w-full text-sm text-left">
                      <thead className="bg-white text-xs text-zinc-400 uppercase font-bold tracking-wider border-b border-zinc-100">
                        <tr>
                          <th className="px-6 py-4 font-semibold">Formula</th>
                          <th className="px-6 py-4 font-semibold">Type</th>
                          <th className="px-6 py-4 text-right font-semibold">Density <span className="text-[9px] lowercase opacity-50">(g/cm³)</span></th>
                          <th className="px-6 py-4 text-right font-semibold">Stability <span className="text-[9px] lowercase opacity-50">(0-1)</span></th>
                          <th className="px-6 py-4 text-right font-semibold">Band Gap <span className="text-[9px] lowercase opacity-50">(eV)</span></th>
                          <th className="px-6 py-4 text-right font-semibold">Form. Energy <span className="text-[9px] lowercase opacity-50">(eV)</span></th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-zinc-50">
                        {chartData.map(mat => (
                          <tr key={mat.id} className="group hover:bg-zinc-50 transition-colors">
                            <td className="px-6 py-4 font-bold text-zinc-800 flex items-center gap-2">
                                <div className="w-1.5 h-1.5 rounded-full bg-zinc-300 group-hover:bg-zinc-800 transition-colors"></div>
                                {mat.name}
                            </td>
                            <td className="px-6 py-4 text-zinc-500 text-xs font-medium">{mat.type}</td>
                            <td className="px-6 py-4 text-zinc-600 font-mono text-right text-xs">{mat.density.toFixed(2)}</td>
                            <td className="px-6 py-4 font-mono text-right text-xs">
                                <span className="text-zinc-700 font-semibold">{mat.stability.toFixed(2)}</span>
                            </td>
                            <td className="px-6 py-4 text-zinc-600 font-mono text-right text-xs">{mat.band_gap.toFixed(2)}</td>
                            <td className="px-6 py-4 text-zinc-600 font-mono text-right text-xs">{mat.formation_energy.toFixed(2)}</td>
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
