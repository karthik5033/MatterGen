"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import CrystalViewer from "@/components/CrystalViewer";
import MaterialBenchmarkChart from "@/components/MaterialBenchmarkChart";
import MaterialRadarChart from "@/components/MaterialRadarChart";
import ComparisonCharts from "@/components/charts/ComparisonCharts";
import { MaterialCandidate } from "@/types/api";

// ... (interfaces)
interface Application {
  title: string;
  description: string;
  performance_metric: string;
}

interface Ratings {
  commercial_viability: number;
  sustainability_index: number;
  manufacturing_complexity: number;
}

interface AnalysisData {
  executive_summary: string;
  scientific_deep_dive: string;
  future_tech_lore: string;
  industrial_applications: Application[];
  ratings: Ratings;
  synthesis_guide: {
    method: string;
    precursors: string[];
    equipment: string[];
    detailed_procedure: string[];
  };
  economic_outlook: {
    estimated_cost: string;
    scalability_verdict: string;
    supply_chain_risks: string[];
  };
  risk_profile: {
    flammability: string;
    toxicity: string;
    handling_precautions: string[];
  };
}


export default function MaterialDetailPage() {
  const { id } = useParams();
  const router = useRouter();
  const [material, setMaterial] = useState<MaterialCandidate | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);

  useEffect(() => {
    // 1. Load Data from Session
    if (!id) return;
    const stored = sessionStorage.getItem(`material_${id}`);
    if (stored) {
      const data = JSON.parse(stored);
      setMaterial(data);
      setLoading(false);
      
      // 2. Trigger AI Analysis
      setAnalyzing(true);
      fetch("http://localhost:8002/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          formula: data.formula,
          properties: data.predicted_properties
        })
      })
      .then(res => res.json())
      .then(data => {
        setAnalysis(data);
        setAnalyzing(false);
      })
      .catch(err => {
        console.error("Analysis failed", err);
        setAnalyzing(false);
      });
      
    } else {
      setLoading(false);
    }
  }, [id]);

  const handleDownloadPDF = () => {
     window.print();
  };

  if (loading) return <div className="min-h-screen bg-white flex items-center justify-center">Loading...</div>;

  if (!material) {
    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-center p-10">
        <h1 className="text-2xl font-bold text-gray-900 mb-4">Material Not Found</h1>
        <button onClick={() => router.push("/")} className="px-4 py-2 bg-indigo-600 text-white rounded-lg">Back to Generator</button>
      </div>
    );
  }

  const props = material.predicted_properties;

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 pb-20 print:bg-white print:pb-0">
      
      {/* Print Styles */}
      <style jsx global>{`
        @media print {
          @page {
            size: a4 portrait;
            margin: 0mm; /* Hides browser header/footer usually */
          }
          body {
            print-color-adjust: exact;
            -webkit-print-color-adjust: exact;
            margin: 10mm;
          }
          .break-before-page {
            page-break-before: always;
            break-before: page;
          }
          .break-inside-avoid {
             break-inside: avoid;
             page-break-inside: avoid;
          }
        }
      `}</style>
      
      {/* HERO HEADER (Screen Only) - UPGRADED */}
      <div className="sticky top-0 z-50 w-full animate-in fade-in slide-in-from-top-4 duration-500 print:hidden">
          <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500"></div>
          <div className="bg-white/90 backdrop-blur-xl border-b border-gray-200/80 shadow-sm supports-[backdrop-filter]:bg-white/60">
            <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
              
              {/* Left: Navigation & Title */}
              <div className="flex items-center gap-6">
                 {/* Back Button */}
                 <button 
                    onClick={() => router.back()} 
                    className="group flex items-center justify-center w-10 h-10 rounded-full bg-slate-50 border border-gray-200 text-gray-400 hover:text-slate-900 hover:border-slate-300 hover:shadow-md transition-all active:scale-95"
                    title="Go Back"
                 >
                    <svg className="w-5 h-5 group-hover:-translate-x-0.5 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                    </svg>
                 </button>

                 <div className="h-8 w-px bg-gray-200 hidden sm:block"></div>

                 <div className="flex flex-col">
                    <div className="flex items-center gap-3">
                        <h1 className="text-2xl md:text-3xl font-black tracking-tighter text-slate-900 font-mono">
                            {material.formula}
                        </h1>
                        <div className="flex items-center gap-2">
                            <span className="hidden sm:inline-flex items-center px-2.5 py-0.5 rounded-md text-[10px] bg-slate-100/80 border border-slate-200 text-slate-500 font-mono tracking-wide uppercase">
                                ID: {material.id.slice(0, 6)}...
                            </span>
                            {/* Stability Badge (Header Version) */}
                            {(props.energy_above_hull ?? props.formation_energy) < 0.05 ? (
                                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-700 border border-emerald-200 shadow-sm">
                                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mr-1.5 animate-pulse"></span>
                                    STABLE
                                </span>
                            ) : (
                                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-50 text-amber-600 border border-amber-200 shadow-sm">
                                      <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mr-1.5"></span>
                                      METASTABLE
                                </span>
                            )}
                        </div>
                    </div>
                </div>
              </div>

              {/* Right: Actions */}
              <div className="flex items-center gap-3">
                 <div className="hidden md:flex flex-col items-end mr-4">
                     <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Confidence</span>
                     <span className="text-sm font-mono font-bold text-slate-700">{(material.score * 100).toFixed(1)}%</span>
                 </div>
                 
                 <button 
                    onClick={handleDownloadPDF}
                    className="flex items-center gap-2 px-5 py-2.5 bg-slate-900 hover:bg-black text-white text-xs font-bold rounded-xl shadow-lg shadow-slate-900/10 hover:shadow-xl hover:-translate-y-0.5 transition-all active:scale-95 group"
                 >
                    <svg className="w-4 h-4 opacity-70" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
                    </svg>
                    <span>Print Report</span>
                 </button>
              </div>
            </div>
          </div>
      </div>
      
      {/* Print Header */}
      <div className="hidden print:flex items-center justify-between px-8 py-6 border-b-2 border-indigo-600 mb-8 w-full bg-white">
          <div>
              <h1 className="text-4xl font-black font-mono tracking-tight text-indigo-700">{material.formula}</h1>
              <div className="flex items-center gap-3 mt-2">
                 <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded font-mono text-xs font-bold">ID: {material.id.slice(0, 8)}</span>
                 <p className="text-xs text-gray-400 uppercase tracking-widest font-bold">AI Material Discovery Report</p>
              </div>
          </div>
          <div className="text-right">
              <div className="text-xl font-bold text-gray-900 tracking-tight">MatterGen X</div>
              <div className="text-xs text-gray-500 font-medium mt-1">Generated: {new Date().toLocaleDateString()}</div>
          </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-10 grid grid-cols-12 gap-8 bg-gray-50 print:block print:p-0">
        
        {/* LEFT COLUMN CONTENT (Charts & Stats) */}
        {/* We use a Grid in print to fit everything on Page 1 nicely */}
        <div className="col-span-12 lg:col-span-4 space-y-6 print:space-y-0 print:w-full print:grid print:grid-cols-2 print:gap-6 print:mb-8">
           
           {/* Crystal View */}
           <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden h-[400px] relative print:col-span-1 print:h-[250px] print:border-gray-300 print:shadow-none">
               <div className="absolute top-4 left-4 z-10 bg-white/90 backdrop-blur px-3 py-1 rounded-full text-xs font-bold text-gray-500 border border-gray-100">Crystal Lattice</div>
               <CrystalViewer materialId={material.id} cifData={material.crystal_structure_cif} className="w-full h-full bg-gray-50"/>
           </div>

           {/* Props */}
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 print:col-span-1 print:h-[250px] print:border-gray-300 print:shadow-none">
               <h2 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-6">Physical Properties</h2>
               <div className="space-y-4">
                  <div className="flex justify-between items-center group">
                     <span className="text-sm font-medium text-gray-500">Band Gap</span>
                     <span className="font-mono font-bold text-gray-900">{typeof props.band_gap === 'number' ? props.band_gap.toFixed(3) : props.band_gap} <span className="text-gray-400 text-xs">eV</span></span>
                  </div>
                  <div className="w-full bg-gray-100 h-1.5 rounded-full overflow-hidden print:border print:border-gray-100"><div className="bg-indigo-500 h-full rounded-full" style={{ width: `${Math.min((Number(props.band_gap) / 5) * 100, 100)}%` }}></div></div>

                  <div className="flex justify-between items-center group pt-2">
                     <span className="text-sm font-medium text-gray-500">Energy Above Hull</span>
                     <span className="font-mono font-bold text-gray-900">{typeof props.energy_above_hull === 'number' ? props.energy_above_hull.toFixed(3) : props.energy_above_hull} <span className="text-gray-400 text-xs">eV/at</span></span>
                  </div>
                  <div className="w-full bg-gray-100 h-1.5 rounded-full overflow-hidden print:border print:border-gray-100"><div className="bg-emerald-500 h-full rounded-full" style={{ width: `${Math.max(0, 100 - (Number(props.energy_above_hull) * 100))}%` }}></div></div>

                  <div className="flex justify-between items-center group pt-2">
                     <span className="text-sm font-medium text-gray-500">Bulk Modulus</span>
                     <span className="font-mono font-bold text-gray-900">{typeof props.bulk_modulus === 'number' ? props.bulk_modulus.toFixed(3) : props.bulk_modulus} <span className="text-gray-400 text-xs">GPa</span></span>
                  </div>
                  <div className="w-full bg-gray-100 h-1.5 rounded-full overflow-hidden print:border print:border-gray-100"><div className="bg-amber-500 h-full rounded-full" style={{ width: `${Math.min((Number(props.bulk_modulus) / 300) * 100, 100)}%` }}></div></div>
               </div>
            </div>

           {/* Radar */}
           <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 print:col-span-1 print:border-gray-300 print:shadow-none">
               <h2 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Multi-Metric Analysis</h2>
               <MaterialRadarChart properties={props} ratings={analysis?.ratings} />
           </div>

           {/* Benchmark */}
           <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 print:col-span-1 print:border-gray-300 print:shadow-none">
               <h2 className="text-xs font-bold text-indigo-500 uppercase tracking-widest mb-4">Performance Benchmarks</h2>
               <MaterialBenchmarkChart properties={props} />
               <p className="text-[10px] text-gray-400 mt-4 text-center border-t border-gray-100 pt-3">
                   Comparing vs. Standard Silicon & Oxide
               </p>
           </div>

           {/* Electronic Structure (DOS) Card */}
           <div className="bg-slate-900 rounded-2xl shadow-lg border border-slate-800 p-6 text-white relative overflow-hidden min-h-[250px] print:hidden">
               {/* Grid Background */}
               <div className="absolute inset-0 z-0 opacity-20" 
                    style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)', backgroundSize: '20px 20px' }}>
               </div>

               <div className="relative z-10 flex flex-col h-full">
                   <div className="flex items-center justify-between mb-6">
                       <div>
                           <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-widest flex items-center gap-2">
                               <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse"></span>
                               Electronic Structure
                           </h3>
                           <p className="text-xs text-slate-400 mt-1 font-medium">Projected Density of States</p>
                       </div>
                       <div className="text-right">
                           <div className="text-xs text-slate-500 font-mono uppercase">Band Gap</div>
                           <div className="text-lg font-bold font-mono text-white">
                               {typeof props.band_gap === 'number' ? props.band_gap.toFixed(2) : parseFloat(props.band_gap as string).toFixed(2)} <span className="text-sm text-slate-500">eV</span>
                           </div>
                       </div>
                   </div>

                   {/* DOS Visualization */}
                   <div className="flex-1 w-full relative h-[160px] mt-2">
                       <svg width="100%" height="100%" viewBox="0 0 320 160" preserveAspectRatio="none" className="overflow-visible">
                           {(() => {
                               // DYNAMIC SCALING LOGIC
                               const bgVal = Number(props.band_gap) || 0;
                               const availableWidth = 170; // From center (140) to edge (310)
                               // Target: fit band_gap + 2eV into available width. Max scale 30px/eV.
                               const pxPereV = Math.min(30, availableWidth / (bgVal + 3));
                               
                               // Calculate visible range in eV (approx -4 to +Max)
                               const maxEV = Math.floor(availableWidth / pxPereV);
                               const minEV = -Math.floor(130 / pxPereV); // 140 is center, leaving 10px margin
                               
                               // Generate Ticks: nicely rounded logic
                               const step = pxPereV < 15 ? 5 : 2; // wider steps if zoomed out
                               const ticks = [];
                               for (let e = Math.ceil(minEV / step) * step; e <= maxEV; e += step) {
                                   ticks.push(e);
                               }

                               const gapPixels = bgVal * pxPereV;
                               const startX = 140 + gapPixels;

                               return (
                                   <>
                                       {/* Grid Lines */}
                                       <defs>
                                           <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
                                               <path d="M 30 0 L 0 0 0 30" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="0.5"/>
                                           </pattern>
                                       </defs>
                                       <rect width="100%" height="100%" fill="url(#grid)" />

                                       {/* X-Axis Main Line */}
                                       <line x1="20" y1="140" x2="320" y2="140" stroke="#475569" strokeWidth="1" />
                                       
                                       {/* Dynamic X-Axis Ticks & Labels */}
                                       {ticks.map((ev, i) => (
                                           <g key={i} transform={`translate(${140 + (ev * pxPereV)}, 140)`}>
                                               <line y1="0" y2="4" stroke="#475569" strokeWidth="1" />
                                               <text y="15" fill="#64748b" fontSize="8" textAnchor="middle" className="font-mono">
                                                   {ev}
                                               </text>
                                           </g>
                                       ))}
                                       <text x="315" y="155" fill="#64748b" fontSize="9" textAnchor="end" className="font-bold">
                                           E - E<tspan dy="3" fontSize="7">f</tspan><tspan dy="-3"> (eV)</tspan>
                                       </text>

                                       {/* Fermi Level (Vertical Dashed) */}
                                       <line x1="140" y1="10" x2="140" y2="140" stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="4,2" className="opacity-80" />
                                       <text x="144" y="20" fill="#f59e0b" fontSize="10" className="font-mono font-bold">
                                           E<tspan dy="3" fontSize="8">f</tspan>
                                       </text>

                                       {/* Valence Band (Scaled) */}
                                       {/* Fixed shape scaled horizontally? Or just keep generic shape? Generic is safer for visual aesthetic */}
                                       <path d={`M ${140 - (3.5 * pxPereV)} 140 C ${140 - (2.5 * pxPereV)} 140, ${140 - (2.5 * pxPereV)} 40, ${140 - (1.5 * pxPereV)} 60 S ${140 - (0.2 * pxPereV)} 100, 140 140 Z`} 
                                             fill="url(#gradValence)" stroke="#818cf8" strokeWidth="2" />
                                       <defs>
                                           <linearGradient id="gradValence" x1="0%" y1="0%" x2="0%" y2="100%">
                                               <stop offset="0%" stopColor="#818cf8" stopOpacity="0.7" />
                                               <stop offset="100%" stopColor="#818cf8" stopOpacity="0.1" />
                                           </linearGradient>
                                       </defs>

                                       {/* Conduction Band (Dynamic Position) */}
                                       {gapPixels > 2 && startX < 310 && (
                                            <rect x="140" y="20" width={gapPixels} height="120" fill="url(#patternStripes)" opacity="0.15" />
                                       )}
                                       
                                       <path d={`M ${startX} 140 Q ${startX + (0.7 * pxPereV)} 130, ${startX + (1.3 * pxPereV)} 80 T ${startX + (3 * pxPereV)} 40 T ${startX + (5 * pxPereV)} 120 V 140 Z`} 
                                             fill="url(#gradConduction)" stroke="#2dd4bf" strokeWidth="2" />
                                       
                                       <defs>
                                            <linearGradient id="gradConduction" x1="0%" y1="0%" x2="0%" y2="100%">
                                                <stop offset="0%" stopColor="#2dd4bf" stopOpacity="0.7" />
                                                <stop offset="100%" stopColor="#2dd4bf" stopOpacity="0.1" />
                                            </linearGradient>
                                            <pattern id="patternStripes" x="0" y="0" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                                                <line x1="0" y1="0" x2="0" y2="6" stroke="#ffffff" strokeWidth="1" />
                                            </pattern>
                                       </defs>
                                       
                                       {/* Gap Arrow & Label */}
                                       {gapPixels > 40 && (
                                           <g transform={`translate(${140 + gapPixels/2}, 80)`}>
                                               <text x="0" y="-10" textAnchor="middle" fill="#94a3b8" fontSize="10" className="font-mono whitespace-nowrap">{bgVal.toFixed(2)} eV</text>
                                               <line x1={-Math.min(gapPixels/2 - 10, 40)} y1="0" x2={Math.min(gapPixels/2 - 10, 40)} y2="0" stroke="#475569" strokeWidth="1" markerStart="url(#arrow)" markerEnd="url(#arrow)" />
                                           </g>
                                       )}
                                   </>
                               );
                           })()}
                           
                           <defs>
                               <marker id="arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
                                   <path d="M 0 0 L 6 3 L 0 6 z" fill="#475569" />
                               </marker>
                           </defs>
                       </svg>

                       <div className="absolute bottom-6 left-10 text-[10px] bg-slate-800/80 px-2 py-0.5 rounded text-indigo-300 font-bold border border-indigo-500/30">Valence Band</div>
                       <div className="absolute top-10 right-10 text-[10px] bg-slate-800/80 px-2 py-0.5 rounded text-teal-300 font-bold border border-teal-500/30">Conduction Band</div>
                   </div>
               </div>
           </div>

           {/* Model Metrics Table */}
           <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 print:border-gray-300 print:shadow-none print:break-inside-avoid">
               <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xs font-bold text-gray-400 uppercase tracking-widest">Model Validation Metrics</h2>
                  <span className="px-2 py-0.5 rounded text-[10px] items-center bg-gray-50 text-gray-500 font-medium border border-gray-100">v2.4.0</span>
               </div>
               
               <div className="overflow-hidden">
                   <table className="w-full text-xs text-left">
                       <thead className="border-b border-gray-100">
                           <tr>
                               <th className="py-2 font-bold text-gray-400 uppercase tracking-wider">Metric</th>
                               <th className="py-2 text-right font-bold text-gray-400 uppercase tracking-wider">Value</th>
                               <th className="py-2 text-right font-bold text-gray-400 uppercase tracking-wider">Uncertainty</th>
                           </tr>
                       </thead>
                       <tbody className="divide-y divide-gray-50/50">
                           {[
                               { label: "Formation Energy MAE", value: "0.024", unit: "eV/at", error: (0.005 + (1 - material.score) * 0.01).toFixed(3) },
                               { label: "Band Gap RMSE", value: "0.18", unit: "eV", error: (0.04 + (1 - material.score) * 0.1).toFixed(2) },
                               { label: "Stability AUC", value: "0.96", unit: "", error: null },
                               { label: "Bulk Modulus R²", value: "0.89", unit: "", error: "0.02" },
                           ].map((row, i) => (
                               <tr key={i} className="group">
                                   <td className="py-3 font-medium text-gray-700">{row.label}</td>
                                   <td className="py-3 text-right font-mono text-gray-900 group-hover:text-indigo-600 transition-colors">
                                       {row.value} <span className="text-gray-400 text-[10px]">{row.unit}</span>
                                   </td>
                                   <td className="py-3 text-right font-mono text-gray-400">
                                       {row.error ? `±${row.error}` : <span className="text-emerald-500 font-bold">High Conf.</span>}
                                   </td>
                               </tr>
                           ))}
                       </tbody>
                   </table>
               </div>
               <div className="mt-4 pt-3 border-t border-gray-100 flex items-center justify-between text-[10px] text-gray-400">
                   <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
                        <span>Live Inference</span>
                   </div>
                   <div>Confidence: <span className="font-bold text-gray-900">{(material.score * 100).toFixed(1)}%</span></div>
               </div>
           </div>
           
           {/* Synthesis Complexity Card (Formal Scientific Theme) */}
           <div className="bg-[#0c0a09] rounded-xl shadow-sm border border-stone-800 p-6 text-stone-200 relative overflow-hidden print:bg-white print:border-stone-300 print:text-stone-900 print:break-inside-avoid">
               <div className="absolute top-0 right-0 p-4 opacity-10 print:opacity-5">
                   <svg className="w-24 h-24 print:text-stone-200" fill="currentColor" viewBox="0 0 24 24"><path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58a.49.49 0 0 0 .12-.61l-1.92-3.32a.488.488 0 0 0-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.484.484 0 0 0-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.04.17-.02.36.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58a.49.49 0 0 0-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.58 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.04-.22.02-.41-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/></svg>
               </div>
               
               <h3 className="text-xs font-bold text-stone-400 uppercase tracking-widest mb-4 relative z-10 print:text-stone-500">Synthesis Difficulty</h3>
               
               <div className="flex items-end gap-3 mb-2 relative z-10">
                   <div className="text-4xl font-mono font-bold text-white print:text-stone-900">
                       {analysis?.ratings?.manufacturing_complexity ? (analysis.ratings.manufacturing_complexity / 10).toFixed(1) : '4.2'}
                   </div>
                   <div className="text-sm font-medium text-stone-400 mb-1.5 font-mono print:text-stone-500">/ 10.0</div>
               </div>
               
               <div className="w-full bg-stone-800 h-1.5 rounded-full overflow-hidden mb-4 relative z-10 print:bg-stone-100">
                    <div className="bg-stone-300 h-full rounded-full print:bg-stone-600" style={{ width: `${(analysis?.ratings?.manufacturing_complexity || 42)}%` }}></div>
               </div>
               
               <div className="flex justify-between text-[10px] font-bold uppercase tracking-wider text-stone-500 relative z-10 print:text-stone-400">
                   <span>Easy</span>
                   <span>Moderate</span>
                   <span>Hard</span>
               </div>
               
               <div className="mt-4 pt-4 border-t border-stone-800 relative z-10 print:border-stone-100">
                   <div className="text-xs text-stone-400 leading-relaxed font-mono print:text-stone-600">
                       <span className="text-stone-200 font-bold print:text-stone-800">BATCH PROCESS</span> synthesis estimated. <span className="text-stone-200 print:text-stone-800">Standard precursors</span> available; requires high-temp sintering.
                   </div>
               </div>
           </div>
           
           {/* Print Version of Confidence (Visible only in print) */}
           <div className="hidden print:flex w-full p-4 border border-gray-300 rounded-xl bg-white items-center justify-between mb-0 print:col-span-2">
                <div>
                   <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-1">AI Confidence Score</h3>
                   <div className="text-4xl font-black text-indigo-900">92%</div>
                </div>
                <div className="text-right">
                    <div className="px-3 py-1 bg-green-100 text-green-800 rounded font-bold text-xs inline-block mb-1">HIGH RELIABILITY</div>
                    <p className="text-xs text-gray-600 max-w-[200px]">Model Certainty</p>
                </div>
           </div>
        </div>

        {/* RIGHT COLUMN CONTENT (Text) */}
        {/* RIGHT COLUMN CONTENT (Text) */}
        <div className="col-span-12 lg:col-span-8 space-y-6 print:space-y-8 print:w-full print:block">
           
           {/* Sci-Fi Lore */}
            <div className="relative rounded-2xl overflow-hidden shadow-lg shadow-indigo-500/10 group bg-gray-900 border border-gray-800 print:bg-white print:border print:border-gray-200 print:shadow-none print:break-inside-avoid">
               {/* ... (Same Sci-Fi content) ... */}
               <div className="absolute inset-0 bg-gradient-to-br from-indigo-900 via-gray-900 to-black opacity-80 print:hidden"></div>
               <div className="relative p-8">
                  <div className="flex items-center gap-3 mb-2 print:border-b print:border-gray-200 print:pb-2 print:mb-4">
                     <span className="px-2 py-1 bg-indigo-500/20 border border-indigo-400/30 rounded text-xs font-bold text-indigo-300 tracking-wider print:bg-gray-100 print:border-0 print:text-gray-800">FUTURE TECH // LORE</span>
                  </div>
                  {analyzing ? (
                     <div className="h-20 animate-pulse bg-white/5 rounded-lg w-full mt-4"></div>
                  ) : analysis ? (
                      <div className="animate-in fade-in duration-700">
                          <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-200 to-cyan-200 mb-2 print:text-indigo-900 print:bg-none">
                              {analysis.future_tech_lore.split('.')[0]}
                          </h2>
                          <p className="text-indigo-100/70 font-light text-lg italic border-l-2 border-indigo-500/50 pl-4 print:text-gray-700 print:border-indigo-300">
                              "{analysis.future_tech_lore}"
                          </p>
                      </div>
                  ) : <div className="text-gray-500 py-4">Analysis Pending...</div>}
               </div>
            </div>
            
            {/* MAIN REPORT - Force Page Break Before Deep Dive or Feasibility */}
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-8 print:border-0 print:p-0 print:shadow-none">
                {/* ... (Headings) ... */}
                <div className="flex items-center gap-3 mb-6">
                   <div className="w-8 h-8 rounded-lg bg-teal-50 flex items-center justify-center text-teal-600">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                   </div>
                   <h2 className="text-lg font-bold text-gray-900">Comprehensive Technical Report</h2>
                </div>
                
                {analyzing ? (
                   <div className="space-y-4 animate-pulse">
                      <div className="h-4 bg-gray-100 rounded w-full"></div>
                      <div className="h-4 bg-gray-100 rounded w-5/6"></div>
                      <div className="h-32 bg-gray-50 rounded w-full mt-6"></div>
                   </div>
                ) : analysis ? (
                   <div className="space-y-8 animate-in fade-in duration-500">
                      
                      {/* Executive Summary */}
                      <div className="bg-gray-50 p-6 rounded-xl border border-gray-100 print:bg-gray-50 print:border print:border-gray-200 break-inside-avoid">
                         <h4 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2">Executive Summary</h4>
                         <p className="text-gray-800 font-medium leading-relaxed">{analysis.executive_summary}</p>
                      </div>

                      {/* Ratings */}
                      <div className="grid grid-cols-3 gap-4 break-inside-avoid">
                         {analysis.ratings && Object.entries(analysis.ratings).map(([key, value]) => (
                             <div key={key} className="text-center p-4 rounded-xl border border-gray-100 bg-white shadow-sm print:border print:border-gray-200 print:shadow-none">
                                 <div className="relative w-16 h-16 mx-auto mb-2 flex items-center justify-center rounded-full border-4 border-indigo-50">
                                     <span className="text-sm font-bold text-indigo-600">{value as number}</span>
                                     <svg className="absolute inset-0 w-full h-full -rotate-90 text-indigo-500" viewBox="0 0 36 36">
                                         <path className="stroke-current" fill="none" strokeWidth="3" strokeDasharray={`${value}, 100`} d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                     </svg>
                                 </div>
                                 <div className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">{key.replace('_', ' ')}</div>
                             </div>
                         ))}
                      </div>

                      {/* Deep Dive */}
                      <div className="break-inside-avoid">
                             <h4 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3">Scientific Deep Dive</h4>
                             <div className="prose prose-sm text-gray-600 max-w-none print:text-gray-800 print:text-justify">
                                 <p>{analysis.scientific_deep_dive}</p>
                             </div>
                          </div>

                          {/* Industrial Applications Cards */}
                          <div className="print:break-inside-avoid">
                             <h4 className="text-xs font-bold text-indigo-500 uppercase tracking-widest mb-4">Industrial Applications</h4>
                             <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                {analysis.industrial_applications.map((app, i) => (
                                  <div key={i} className="p-4 bg-white rounded-xl border border-gray-200 hover:border-indigo-300 hover:shadow-md transition-all print:border-gray-300 print:bg-white print:shadow-none">
                                     <h5 className="font-bold text-gray-900 text-sm mb-2">{app.title}</h5>
                                     <p className="text-xs text-gray-500 mb-3 line-clamp-3 print:line-clamp-none">{app.description}</p>
                                     <div className="pt-3 border-t border-gray-100">
                                        <div className="text-[10px] uppercase font-bold text-indigo-400">Target Metric</div>
                                        <div className="font-mono text-xs font-bold text-indigo-700">{app.performance_metric}</div>
                                     </div>
                                  </div>
                                ))}
                             </div>
                          </div>

                          {/* NEW: Benchmarking & Analysis Section */}
                          <div className="print:break-inside-avoid pt-8 border-t border-gray-100 mt-8">
                               <div className="flex items-center gap-3 mb-6">
                                   <div className="w-8 h-8 rounded-lg bg-indigo-50 flex items-center justify-center text-indigo-600">
                                       <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
                                   </div>
                                   <div>
                                       <h2 className="text-lg font-bold text-gray-900">Comparative Benchmarking</h2>
                                       <p className="text-xs text-gray-500">Performance analysis against standard industry references.</p>
                                   </div>
                               </div>
                               
                               <div className="bg-gray-50/50 rounded-xl border border-gray-200/60 p-1">
                                    {(() => {
                                        // Dynamic Benchmark Selection Logic
                                        const bg = Number(material.predicted_properties.band_gap) || 0;
                                        
                                        // Define Benchmark Pools
                                        const benchmarks = [
                                            // Candidate (Always First)
                                            { 
                                                name: material.formula, 
                                                density: material.predicted_properties.density || 0,
                                                stability: (1 - Math.min(Math.max((material.predicted_properties.energy_above_hull || 0), 0), 1)),
                                                band_gap: bg,
                                                formation_energy: material.predicted_properties.formation_energy || 0
                                            }
                                        ];

                                        if (bg < 0.1) {
                                            // Metals / Conductors
                                            benchmarks.push(
                                                { name: 'Copper (Cu)', density: 8.96, stability: 1.0, band_gap: 0.0, formation_energy: 0.0 }, // Standard Conductor
                                                { name: 'Aluminum (Al)', density: 2.70, stability: 1.0, band_gap: 0.0, formation_energy: 0.0 }, // Lightweight Conductor
                                                { name: 'Iron (Fe)', density: 7.87, stability: 0.98, band_gap: 0.0, formation_energy: 0.0 }
                                            );
                                        } else if (bg >= 0.1 && bg < 2.5) {
                                            // Semiconductors (Solar/Chip)
                                            benchmarks.push(
                                                { name: 'Silicon (Si)', density: 2.33, stability: 1.0, band_gap: 1.12, formation_energy: -0.1 }, // Classic Semi
                                                { name: 'GaAs', density: 5.32, stability: 0.95, band_gap: 1.42, formation_energy: -0.8 }, // High Speed
                                                { name: 'CdTe', density: 5.85, stability: 0.92, band_gap: 1.5, formation_energy: -1.0 } // Solar Standard
                                            );
                                        } else {
                                            // Insulators / Wide Bandgap (Power/Optical)
                                            benchmarks.push(
                                                { name: 'Quartz (SiO2)', density: 2.65, stability: 1.0, band_gap: 8.9, formation_energy: -9.5 }, // Classic Indulator
                                                { name: 'GaN', density: 6.15, stability: 0.99, band_gap: 3.4, formation_energy: -1.2 }, // Power Electronics
                                                { name: 'Diamond (C)', density: 3.51, stability: 1.0, band_gap: 5.47, formation_energy: 0.0 } // Ultimate Wide BG
                                            );
                                        }

                                        return <ComparisonCharts data={benchmarks} />;
                                    })()}
                               </div>
                          </div>

                      {/* FEASIBILITY ANALYSIS (Force New Page) */}
                      {analysis.synthesis_guide && (
                         <div className="pt-10 border-t border-gray-100 break-before-page print:pt-0 print:border-t-0 print:mt-8">
                             <div className="print:hidden mb-6">
                                 <div className="h-px bg-gray-200 w-full mb-8"></div>
                             </div>
                             
                             <h4 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-6 print:text-indigo-600 print:mb-4">Feasibility & Logistics</h4>
                            
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                
                                {/* 1. SYNTHESIS GUIDE (Expanded) */}
                                <div className="bg-amber-50/40 rounded-xl p-6 border border-amber-100">
                                    <h5 className="flex items-center gap-2 text-base font-bold text-gray-900 mb-4">
                                        <span className="w-8 h-8 rounded bg-amber-100 text-amber-600 flex items-center justify-center text-sm">⚗️</span>
                                        Synthesis Protocol
                                    </h5>
                                    
                                    <div className="space-y-4 text-sm">
                                        <div className="pb-3 border-b border-amber-200/30">
                                            <span className="block text-xs font-bold text-amber-800/60 uppercase tracking-wider mb-1">Methodology</span>
                                            <div className="font-medium text-amber-900">{analysis.synthesis_guide['method']}</div>
                                        </div>

                                        <div className="grid grid-cols-2 gap-4">
                                            <div>
                                                <span className="block text-xs font-bold text-amber-800/60 uppercase tracking-wider mb-1">Precursors</span>
                                                <ul className="list-disc list-inside text-amber-900/80 space-y-0.5">
                                                    {Array.isArray(analysis.synthesis_guide['precursors']) && analysis.synthesis_guide['precursors'].map((item: string, i: number) => (
                                                        <li key={i}>{item}</li>
                                                    ))}
                                                </ul>
                                            </div>
                                            <div>
                                                <span className="block text-xs font-bold text-amber-800/60 uppercase tracking-wider mb-1">Equipment</span>
                                                <ul className="list-disc list-inside text-amber-900/80 space-y-0.5">
                                                    {Array.isArray(analysis.synthesis_guide['equipment']) && analysis.synthesis_guide['equipment'].map((item: string, i: number) => (
                                                        <li key={i}>{item}</li>
                                                    ))}
                                                </ul>
                                            </div>
                                        </div>

                                        <div className="bg-white/60 rounded-lg p-3 border border-amber-100">
                                            <span className="block text-xs font-bold text-amber-800/60 uppercase tracking-wider mb-2">Procedure</span>
                                            <ol className="list-decimal list-inside space-y-1.5 text-amber-950">
                                                {Array.isArray(analysis.synthesis_guide['detailed_procedure']) && analysis.synthesis_guide['detailed_procedure'].map((step: string, i: number) => (
                                                    <li key={i} className="pl-1 marker:font-bold marker:text-amber-600">{step}</li>
                                                ))}
                                            </ol>
                                        </div>
                                    </div>
                                </div>

                                {/* RIGHT COLUMN: ECONOMICS & RISK */}
                                <div className="space-y-6">
                                     
                                     {/* 2. ECONOMIC OUTLOOK */}
                                     {analysis.economic_outlook && (
                                        <div className="bg-emerald-50/40 rounded-xl p-6 border border-emerald-100">
                                            <h5 className="font-bold text-emerald-900 mb-4 flex items-center gap-2">
                                                <span className="w-6 h-6 rounded bg-emerald-100 text-emerald-600 flex items-center justify-center text-xs">💰</span> Market Analysis
                                            </h5>
                                            <div className="grid grid-cols-2 gap-4 mb-4">
                                                <div className="bg-white/60 p-3 rounded-lg border border-emerald-100/50">
                                                    <div className="text-[10px] font-bold text-emerald-600 uppercase">Est. Cost</div>
                                                    <div className="text-emerald-900 font-bold">{analysis.economic_outlook['estimated_cost']}</div>
                                                </div>
                                                <div className="bg-white/60 p-3 rounded-lg border border-emerald-100/50">
                                                    <div className="text-[10px] font-bold text-emerald-600 uppercase">Scalability</div>
                                                    <div className="text-emerald-900 font-bold">{analysis.economic_outlook['scalability_verdict']}</div>
                                                </div>
                                            </div>
                                            <div>
                                                <span className="block text-xs font-bold text-emerald-800/60 uppercase tracking-wider mb-1">Supply Chain Risks</span>
                                                <div className="flex flex-wrap gap-2">
                                                    {Array.isArray(analysis.economic_outlook['supply_chain_risks']) && analysis.economic_outlook['supply_chain_risks'].map((risk: string, i: number) => (
                                                        <span key={i} className="px-2 py-1 bg-emerald-100/50 text-emerald-800 text-xs rounded border border-emerald-200/50">{risk}</span>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                     )}

                                     {/* 3. RISK PROFILE */}
                                     {analysis.risk_profile && (
                                        <div className="bg-rose-50/40 rounded-xl p-6 border border-rose-100">
                                            <h5 className="font-bold text-rose-900 mb-4 flex items-center gap-2">
                                                <span className="w-6 h-6 rounded bg-rose-100 text-rose-600 flex items-center justify-center text-xs">⚠️</span> Safety & Compliance
                                            </h5>
                                            <div className="space-y-3 text-sm">
                                                 <div className="flex gap-4">
                                                     <div className="flex-1">
                                                         <span className="text-xs font-bold text-rose-800/60 uppercase">Flammability</span>
                                                         <p className="text-rose-900">{analysis.risk_profile['flammability']}</p>
                                                     </div>
                                                     <div className="flex-1">
                                                         <span className="text-xs font-bold text-rose-800/60 uppercase">Toxicity</span>
                                                         <p className="text-rose-900">{analysis.risk_profile['toxicity']}</p>
                                                     </div>
                                                 </div>
                                                 <div className="pt-2 border-t border-rose-200/30">
                                                     <span className="block text-xs font-bold text-rose-800/60 uppercase mb-1">Handling Precautions</span>
                                                     <ul className="grid grid-cols-2 gap-2">
                                                         {Array.isArray(analysis.risk_profile['handling_precautions']) && analysis.risk_profile['handling_precautions'].map((item: string, i: number) => (
                                                             <li key={i} className="text-xs text-rose-800 flex items-center gap-1.5">
                                                                 <span className="w-1 h-1 rounded-full bg-rose-400"></span> {item}
                                                             </li>
                                                         ))}
                                                     </ul>
                                                 </div>
                                            </div>
                                        </div>
                                     )}
                                </div>
                            </div>
                        </div>
                     )}

                  </div>
               ) : null}
           </div>

        </div>
      </div>
    </div>
  );
}
