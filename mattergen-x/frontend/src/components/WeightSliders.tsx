import React from 'react';

interface WeightSlidersProps {
  weights: Record<string, number>;
  onChange: (newWeights: Record<string, number>) => void;
}

export const WeightSliders: React.FC<WeightSlidersProps> = ({ weights, onChange }) => {
  const handleChange = (key: string, value: number) => {
    onChange({
      ...weights,
      [key]: value,
    });
  };

  return (
    <div className="space-y-6 pt-2">
      <div className="flex justify-between items-center border-b border-gray-100 pb-2">
         <label className="text-xs font-bold tracking-widest text-gray-400 uppercase">Optimization Priorities</label>
      </div>
      
      {Object.keys(weights).map((key) => {
        const val = weights[key];
        return (
          <div key={key} className="group relative">
            <div className="flex justify-between mb-2">
              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                {key.replace('_', ' ')}
              </span>
              <span className="text-xs font-mono font-bold text-slate-900 tabular-nums">
                {val.toFixed(1)}
              </span>
            </div>
            
            <div className="relative h-2 w-full rounded-full bg-slate-100 overflow-visible group-hover:bg-slate-200 transition-colors">
               {/* Custom Track Fill */}
               <div 
                  className="absolute left-0 top-0 h-full bg-slate-900 rounded-full transition-all duration-75 ease-out"
                  style={{ width: `${val * 100}%` }}
               />
               
               <input 
                type="range" 
                min="0" max="1" step="0.1"
                value={val}
                onChange={(e) => handleChange(key as keyof typeof weights, parseFloat(e.target.value))}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              />
              
              {/* Thumb Indicator (Visual Only, moves with value) */}
              <div 
                  className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-white border-2 border-slate-900 rounded-full shadow-md pointer-events-none transition-all duration-75 ease-out group-hover:scale-110 group-active:scale-95 z-20"
                  style={{ left: `calc(${val * 100}% - 8px)` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
};
