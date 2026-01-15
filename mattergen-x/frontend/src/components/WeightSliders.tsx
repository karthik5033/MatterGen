import React from 'react';

interface WeightSlidersProps {
  weights: {
    density: number;
    stability: number;
    band_gap: number;
  };
  onChange: (newWeights: {
    density: number;
    stability: number;
    band_gap: number;
  }) => void;
}

export const WeightSliders: React.FC<WeightSlidersProps> = ({ weights, onChange }) => {
  const handleChange = (key: keyof typeof weights, value: number) => {
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
      
      {['density', 'stability', 'band_gap'].map((key) => {
        const val = weights[key as keyof typeof weights];
        return (
          <div key={key} className="group">
            <div className="flex justify-between mb-3">
              <span className="text-sm font-medium text-gray-700 capitalize">
                {key.replace('_', ' ')}
              </span>
              <span className="text-xs font-mono text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                {(val * 100).toFixed(0)}%
              </span>
            </div>
            
            <div className="relative h-6 flex items-center">
               <input 
                type="range" 
                min="0" max="1" step="0.05"
                value={val}
                onChange={(e) => handleChange(key as keyof typeof weights, parseFloat(e.target.value))}
                className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/30"
              />
            </div>
          </div>
        );
      })}
    </div>
  );
};
