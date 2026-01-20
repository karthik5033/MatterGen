"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine
} from "recharts";

interface BenchmarkChartProps {
  properties: Record<string, number | boolean>;
}

export default function MaterialBenchmarkChart({ properties }: BenchmarkChartProps) {
  // Safely extract values
  const bandGap = typeof properties['band_gap'] === 'number' ? properties['band_gap'] : 0;
  const hullEnergy = typeof properties['energy_above_hull'] === 'number' ? properties['energy_above_hull'] : 0;
  const bulkModulus = typeof properties['bulk_modulus'] === 'number' ? properties['bulk_modulus'] : 100;

  // Mock standard values for comparison (e.g., standard silicon or common oxide)
  const data = [
    {
      name: "Gap (eV)",
      Material: Number(bandGap.toFixed(3)),
      Standard: 1.1, // Silicon
    },
    {
      name: "Hull (eV/at)",
      Material: Number(hullEnergy.toFixed(3)),
      Standard: 0.05, // Typical meta-stable limit
    },
    {
      name: "Bulk (GPa)",
      Material: Number((bulkModulus / 1).toFixed(1)),
      Standard: 100.0, // Typical ceramic
    }
  ];

  return (
    <div className="w-full h-[250px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{
            top: 20,
            right: 30,
            left: 0,
            bottom: 5,
          }}
          barGap={8}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
          <XAxis 
            dataKey="name" 
            tick={{ fontSize: 11, fill: '#4b5563', fontWeight: 500 }} 
            axisLine={false}
            tickLine={false}
            dy={10}
          />
          <YAxis 
            tick={{ fontSize: 10, fill: '#9ca3af' }} 
            axisLine={false}
            tickLine={false}
          />
          <Tooltip 
            cursor={{ fill: '#f9fafb' }}
            contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
            itemStyle={{ fontSize: '12px', fontWeight: 600 }}
            formatter={(value: number) => [value, '']}
          />
          <Legend 
            wrapperStyle={{ fontSize: '11px', paddingTop: '20px' }} 
            iconType="circle"
          />
          <ReferenceLine y={0} stroke="#e5e7eb" />
          <Bar name="Candidate Material" dataKey="Material" fill="#4f46e5" radius={[4, 4, 4, 4]} barSize={40} animationDuration={1500} />
          <Bar name="Industry Standard" dataKey="Standard" fill="#cbd5e1" radius={[4, 4, 4, 4]} barSize={40} animationDuration={1500} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
