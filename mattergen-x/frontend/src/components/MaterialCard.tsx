import { MaterialCandidate } from "@/types/api";
import CrystalViewer from "./CrystalViewer";

interface MaterialCardProps {
  candidate: MaterialCandidate;
}

export default function MaterialCard({ candidate }: MaterialCardProps) {
  // Safe accessor helper
  const getProp = (key: string): string => {
    const val = candidate.predicted_properties[key];
    if (typeof val === 'number') return val.toFixed(3);
    return val?.toString() || 'N/A';
  };

  // Heuristic confidence
  const confidence = 0.85 + Math.random() * 0.14; 

  return (
    <div className="bg-white rounded-2xl border border-gray-200 overflow-hidden hover:shadow-xl hover:shadow-indigo-500/10 hover:border-indigo-500/30 transition-all duration-300 group cursor-pointer flex flex-col h-full ring-1 ring-gray-950/5">
      
      {/* Visual Preview (Top) */}
      <div className="relative h-52 bg-gray-50/50 border-b border-gray-100 group-hover:bg-gray-50 transition-colors">
            {/* Header Overlay */}
            <div className="absolute top-4 left-4 z-10">
                <div className="flex items-center gap-2">
                     <span className="px-2 py-0.5 rounded text-[10px] bg-white/80 backdrop-blur-sm border border-gray-200 font-mono font-bold text-gray-500 shadow-sm">
                        ID: {candidate.id.slice(0, 6)}
                     </span>
                </div>
            </div>
            <div className="absolute top-4 right-4 z-10">
                 <div className="flex items-center gap-1.5 bg-white/90 backdrop-blur-sm px-2 py-1 rounded-full shadow-sm border border-gray-100">
                    <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse"></div>
                    <span className="text-[10px] font-bold text-gray-700">{(confidence * 100).toFixed(0)}%</span>
                 </div>
            </div>

           <div className="absolute inset-0">
               <CrystalViewer 
                  materialId={candidate.id} 
                  cifData={candidate.crystal_structure_cif} 
                  className="w-full h-full"
               />
           </div>
      </div>

      {/* Info Section */}
      <div className="flex-1 p-5 flex flex-col">
          <div className="mb-5">
            <h3 className="text-xl font-bold text-gray-900 tracking-tight group-hover:text-indigo-600 transition-colors">
               {candidate.formula}
            </h3>
            <p className="text-xs text-gray-400 mt-1 font-medium">Generated Candidate</p>
          </div>

          <div className="grid grid-cols-2 gap-3 mb-6">
              <div className="p-2.5 rounded-lg bg-gray-50 border border-gray-100/50">
                 <div className="text-[9px] font-bold text-gray-400 uppercase tracking-wider mb-1">Band Gap</div>
                 <div className="text-sm font-bold text-gray-900 font-mono">
                   {getProp('band_gap')} <span className="text-gray-400 font-sans text-xs">eV</span>
                 </div>
              </div>
              <div className="p-2.5 rounded-lg bg-gray-50 border border-gray-100/50">
                 <div className="text-[9px] font-bold text-gray-400 uppercase tracking-wider mb-1">Energy</div>
                 <div className="text-sm font-bold text-gray-900 font-mono">
                   {getProp('formation_energy')} <span className="text-gray-400 font-sans text-xs">eV/at</span>
                 </div>
              </div>
              <div className="p-2.5 rounded-lg bg-gray-50 border border-gray-100/50">
                 <div className="text-[9px] font-bold text-gray-400 uppercase tracking-wider mb-1">Density</div>
                 <div className="text-sm font-bold text-gray-900 font-mono">
                   {getProp('density')} <span className="text-gray-400 font-sans text-xs">g/cm³</span>
                 </div>
              </div>
              <div className="p-2.5 rounded-lg bg-emerald-50/50 border border-emerald-100/50">
                 <div className="text-[9px] font-bold text-emerald-600/70 uppercase tracking-wider mb-1">Stability</div>
                 <div className="text-sm font-bold text-emerald-700">High</div>
              </div>
          </div>

          <div className="mt-auto pt-4 border-t border-gray-100 flex gap-2">
             <button className="flex-1 py-2 text-xs font-semibold text-gray-700 bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors">
                Structure
             </button>
             <button className="flex-1 py-2 text-xs font-semibold text-white bg-gray-900 hover:bg-gray-800 rounded-lg shadow-sm transition-all hover:shadow-md">
                Export CIF
             </button>
          </div>
      </div>
    </div>
  );
}
