"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";

interface ComparisonItem {
  name: string;
  density: number;
  stability: number;
  band_gap: number;
  formation_energy: number;
  fill?: string;
}

interface ComparisonChartsProps {
  data: ComparisonItem[];
}

export default function ComparisonCharts({ data }: ComparisonChartsProps) {
  // Transform data for Radar Chart (normalize if needed, but for now raw values)
  // Radar chart usually needs specific format: { subject: 'Density', A: 120, B: 110 }
  // We can construct this dynamically.

  const radarData = [
    { subject: 'Density', fullMark: 10 },
    { subject: 'Stability', fullMark: 1 },
    { subject: 'Band Gap', fullMark: 5 },
    { subject: 'Form. Energy', fullMark: 0 }, // Lower is usually better, might need inversion
  ].map(metric => {
    const entry: any = { subject: metric.subject };
    data.forEach(item => {
        // Simple mapping, might need normalization in real app
        if (metric.subject === 'Density') entry[item.name] = item.density;
        if (metric.subject === 'Stability') entry[item.name] = item.stability; // Assuming 0-1 score
        if (metric.subject === 'Band Gap') entry[item.name] = item.band_gap;
        if (metric.subject === 'Form. Energy') entry[item.name] = Math.abs(item.formation_energy); // Visualizing magnitude
    });
    return entry;
  });

  const colors = ["#111827", "#10b981", "#3b82f6", "#ef4444", "#f59e0b"];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* Bar Chart - Side by Side Property Comparison */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <h3 className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-6">Property Analysis</h3>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
              <XAxis dataKey="name" tick={{fontSize: 10}} axisLine={false} tickLine={false} />
              <YAxis tick={{fontSize: 10}} axisLine={false} tickLine={false} />
              <Tooltip 
                contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                cursor={{fill: '#f9fafb'}}
              />
              <Legend wrapperStyle={{fontSize: '12px', paddingTop: '20px'}} />
              <Bar dataKey="band_gap" name="Band Gap (eV)" fill="#111827" radius={[4, 4, 0, 0]} barSize={20} />
              <Bar dataKey="formation_energy" name="Form. Energy (eV)" fill="#9ca3af" radius={[4, 4, 0, 0]} barSize={20} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Radar Chart - Trade-off Analysis */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <h3 className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-6">Trade-off Profiler</h3>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
              <PolarGrid stroke="#e5e7eb" />
              <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: '#6b7280' }} />
              <PolarRadiusAxis angle={30} domain={[0, 'auto']} tick={false} axisLine={false} />
              {data.map((item, index) => (
                <Radar
                  key={item.name}
                  name={item.name}
                  dataKey={item.name}
                  stroke={colors[index % colors.length]}
                  fill={colors[index % colors.length]}
                  fillOpacity={0.1}
                />
              ))}
              <Legend wrapperStyle={{fontSize: '12px', paddingTop: '20px'}} />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
