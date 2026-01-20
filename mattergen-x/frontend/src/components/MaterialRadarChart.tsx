"use client";

import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";

interface MaterialRadarChartProps {
  properties: Record<string, any>;
  ratings?: {
    commercial_viability: number;
    sustainability_index: number;
    manufacturing_complexity: number;
  };
}

export default function MaterialRadarChart({ properties, ratings }: MaterialRadarChartProps) {
  // Normalize Data for the Chart (Scale 0-100)
  
  // 1. Band Gap (Target: ~0-5eV commonly). Scale: x * 20 (cap at 100)
  const bandGap = Number(properties.band_gap || 0);
  const bandGapScore = Math.min(bandGap * 20, 100);

  // 2. Stability (Energy Above Hull). 
  // Lower is better (0 is stable).
  // 0 eV/at -> 100% Stability
  // 0.2 eV/at -> 0% Stability (Very metastable)
  const hullE = Number(properties.energy_above_hull || 0);
  let stabilityScore = 100 - (hullE * 500); 
  stabilityScore = Math.max(0, Math.min(100, stabilityScore));

  // 3. Bulk Modulus (Mechanical Strength). 
  // Range ~0 - 300 GPa? 
  // Map 150 GPa -> 100?
  const bulkModulus = Number(properties.bulk_modulus || properties.ml_bulk_modulus || 50);
  const strengthScore = Math.min((bulkModulus / 200) * 100, 100);

  // 4. Ratings (Already 0-100)
  const commercial = ratings?.commercial_viability || 50;
  const sustainability = ratings?.sustainability_index || 50;
  
  // 5. Mfg Ease (100 - complexity)
  const complexity = ratings?.manufacturing_complexity || 50;
  const manufacturability = 100 - complexity;

  const data = [
    { subject: "Stability", A: stabilityScore, fullMark: 100 },
    { subject: "Band Gap", A: bandGapScore, fullMark: 100 },
    { subject: "Strength", A: strengthScore, fullMark: 100 },
    { subject: "Commercial", A: commercial, fullMark: 100 },
    { subject: "Sustainable", A: sustainability, fullMark: 100 },
    { subject: "Mfg Ease", A: manufacturability, fullMark: 100 },
  ];

  return (
    <div className="w-full h-[350px] relative">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid stroke="#e5e7eb" />
          <PolarAngleAxis 
            dataKey="subject" 
            tick={{ fill: '#6b7280', fontSize: 10, fontWeight: 600 }} 
          />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
          <Radar
            name="Material Score"
            dataKey="A"
            stroke="#4f46e5"
            strokeWidth={2}
            fill="#6366f1"
            fillOpacity={0.4}
          />
          <Tooltip 
            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
            itemStyle={{ color: '#4338ca', fontWeight: 'bold' }}
          />
        </RadarChart>
      </ResponsiveContainer>
      
      {/* Overlay Stats/Legend maybe? */}
      <div className="absolute top-0 right-0 text-[10px] text-gray-400 font-mono">
        Multi-Metric Analysis
      </div>
    </div>
  );
}
