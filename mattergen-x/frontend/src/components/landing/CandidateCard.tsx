"use client";

interface CandidateCardProps {
  material: {
    id: string;
    formula: string;
    properties: Record<string, number>;
    crystal_structure: string;
  };
}

export default function CandidateCard({ material }: CandidateCardProps) {
  return (
    <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-5 hover:border-emerald-500/50 transition-all group">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-xl font-bold font-mono text-emerald-400">{material.formula}</h3>
          <p className="text-[10px] text-zinc-500 font-mono tracking-widest mt-1 uppercase">ID: {material.id}</p>
        </div>
        <div className="bg-zinc-800 px-2 py-1 rounded text-[10px] font-mono text-zinc-400">
          CRYSTAL_READY
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-6">
        {Object.entries(material.properties).map(([key, val]) => (
          <div key={key} className="bg-zinc-950/50 p-2 rounded border border-zinc-800/50">
            <p className="text-[9px] text-zinc-500 uppercase font-mono">{key.replace('_', ' ')}</p>
            <p className="text-sm font-semibold text-zinc-200">{val.toFixed(2)}</p>
          </div>
        ))}
      </div>

      <div className="flex space-x-2">
        <button className="flex-1 bg-zinc-800 hover:bg-zinc-700 text-xs py-2 rounded transition-all font-medium">
          3D VIEW
        </button>
        <button className="flex-1 bg-zinc-800 hover:bg-zinc-700 text-xs py-2 rounded transition-all font-medium">
          DOWNLOAD CIF
        </button>
      </div>
    </div>
  );
}
