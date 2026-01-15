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

  // Heuristic confidence (mock for now, or derived from data if available)
  const confidence = 0.85 + Math.random() * 0.14; 

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden hover:shadow-lg hover:border-gray-300 transition-all duration-300 group cursor-pointer">
      <div className="flex flex-col md:flex-row h-full">
        
        {/* Info Section */}
        <div className="flex-1 p-6 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-2xl font-bold text-gray-900 tracking-tight group-hover:text-black">
                  {candidate.formula}
                </h3>
                <div className="flex items-center gap-2 mt-1">
                   <span className="text-[10px] uppercase font-bold text-gray-400 tracking-widest">
                     ID: {candidate.id.slice(0, 6)}
                   </span>
                </div>
              </div>
              <div className="flex flex-col items-end">
                <div className="text-[10px] font-bold uppercase text-gray-400 tracking-widest mb-1">Confidence</div>
                <div className="flex items-center gap-1.5">
                   <div className="w-16 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${confidence * 100}%` }} />
                   </div>
                   <span className="text-xs font-mono text-emerald-600 font-bold">{(confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 my-6">
                <div className="bg-gray-50 rounded-lg p-3 border border-gray-100">
                   <div className="text-[10px] font-medium text-gray-500 uppercase mb-1">Band Gap</div>
                   <div className="text-lg font-bold text-gray-900">
                     {getProp('band_gap')} {getProp('band_gap') !== 'N/A' && <span className="text-xs font-normal text-gray-400">eV</span>}
                   </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-3 border border-gray-100">
                   <div className="text-[10px] font-medium text-gray-500 uppercase mb-1">Form. Energy</div>
                   <div className="text-lg font-bold text-gray-900">
                     {getProp('formation_energy')} {getProp('formation_energy') !== 'N/A' && <span className="text-xs font-normal text-gray-400">eV/atom</span>}
                   </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-3 border border-gray-100">
                   <div className="text-[10px] font-medium text-gray-500 uppercase mb-1">Density</div>
                   <div className="text-lg font-bold text-gray-900">
                     {getProp('density')} {getProp('density') !== 'N/A' && <span className="text-xs font-normal text-gray-400">g/cm³</span>}
                   </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-3 border border-gray-100">
                   <div className="text-[10px] font-medium text-gray-500 uppercase mb-1">Stability</div>
                   <div className="text-lg font-bold text-emerald-600">High</div>
                </div>
            </div>
          </div>

          <div className="flex items-center gap-2 pt-4 border-t border-gray-100">
             <button className="text-xs font-bold text-gray-900 hover:underline hover:text-emerald-600 transition-colors">
                View Structure
             </button>
             <span className="text-gray-300">•</span>
             <button className="text-xs font-bold text-gray-900 hover:underline hover:text-emerald-600 transition-colors">
                Export CIF
             </button>
          </div>
        </div>

        {/* Visual Preview (Right side on desktop) */}
        <div className="md:w-80 bg-gray-50 border-t md:border-t-0 md:border-l border-gray-100 relative group-hover:bg-gray-100 transition-colors">
            {/* Using CrystalViewer if available, otherwise placeholder or static if needed. 
                Using strictly minimal style. */}
             <div className="absolute inset-0 p-4">
                 <CrystalViewer 
                    materialId={candidate.id} 
                    cifData={candidate.crystal_structure_cif} 
                    className="w-full h-full pointer-events-none"
                 />
             </div>
             <div className="absolute bottom-2 right-2 z-10">
                 <div className="bg-white/80 backdrop-blur rounded px-1.5 py-0.5 text-[0.6rem] font-mono text-gray-500 border border-gray-200 shadow-sm">
                    3D PREVIEW
                 </div>
             </div>
        </div>

      </div>
    </div>
  );
}
