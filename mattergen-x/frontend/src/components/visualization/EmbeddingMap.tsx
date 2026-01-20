"use client";

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ApiService } from '@/lib/api';

interface MapPoint {
  id: string;
  formula: string;
  x: number;
  y: number;
  targets?: {
    band_gap?: number;
    formation_energy?: number;
  };
  neighbors: string[];
}

interface EmbeddingMapProps {
  onSelect?: (materialId: string) => void;
  height?: number;
}

export function EmbeddingMap({ onSelect, height = 500 }: EmbeddingMapProps) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: height });
  const [points, setPoints] = useState<MapPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [hovered, setHovered] = useState<MapPoint | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    async function fetchMap() {
      try {
        const data = await ApiService.getMaterialMap();
        setPoints(data.points || []);
      } catch (e) {
        console.error("Failed to load map", e);
      } finally {
        setLoading(false);
      }
    }
    fetchMap();
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
        const { width, height: h } = entries[0].contentRect;
        setDimensions({ width, height: h });
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const { width, height: containerHeight } = dimensions;

  // Compute bounds
  const getBounds = () => {
    if (points.length === 0) return { minX: 0, maxX: 100, minY: 0, maxY: 100 };
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys)
    };
  };
  
  const bounds = getBounds();

  // Margins form the axis box
  const xPadding = 60;
  const topPadding = 90; 
  const bottomPadding = 60;
  const dataPadding = 15; // Buffer so points don't touch axes

  const toSvgX = (val: number) => {
    const norm = (val - bounds.minX) / (bounds.maxX - bounds.minX || 1);
    const availableWidth = width - 2 * xPadding - 2 * dataPadding;
    return xPadding + dataPadding + norm * availableWidth;
  };

  const toSvgY = (val: number) => {
    const norm = (val - bounds.minY) / (bounds.maxY - bounds.minY || 1);
    const availableHeight = containerHeight - topPadding - bottomPadding - 2 * dataPadding;
    // FLIP Y: 0 at bottom
    // Bottom of data area = containerHeight - bottomPadding - dataPadding
    // Top of data area = topPadding + dataPadding
    const yBottom = containerHeight - bottomPadding - dataPadding;
    return yBottom - norm * availableHeight;
  };

  return (
    <div ref={containerRef} className="relative w-full bg-white rounded-xl overflow-hidden border border-gray-200 shadow-sm" style={{ height }}>
      <div className="absolute top-6 left-6 z-10 pointer-events-none">
        <h3 className="text-gray-900 font-bold text-sm tracking-tight flex items-center gap-2">
           <span className="w-2 h-2 rounded-full bg-indigo-600"></span>
           Latent Space Projection (UMAP)
        </h3>
        <p className="text-[11px] text-gray-500 max-w-sm mt-1.5 leading-relaxed">
           2D visualization of 64-dimensional material embeddings. Proximal points share structural and chemical similarities.
        </p>
        {loading && <div className="flex items-center gap-2 mt-3"><span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse"/> <span className="text-emerald-700 text-[10px] font-medium">Computing Manifold...</span></div>}
      </div>

      <svg 
        className="w-full h-full cursor-crosshair bg-gray-50/30"
        viewBox={`0 0 ${width || 800} ${containerHeight}`}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Scientific Grid */}
        <defs>
          <pattern id="sci-grid" width="100" height="100" patternUnits="userSpaceOnUse">
            <path d="M 100 0 L 0 0 0 100" fill="none" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="4 4"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#sci-grid)" />

        {/* Axes */}
        <line x1={xPadding} y1={containerHeight - bottomPadding} x2={width - xPadding} y2={containerHeight - bottomPadding} stroke="#94a3b8" strokeWidth="1" />
        <line x1={xPadding} y1={topPadding} x2={xPadding} y2={containerHeight - bottomPadding} stroke="#94a3b8" strokeWidth="1" />
        
        {/* Axis Labels */}
        <text x={width/2} y={containerHeight - 20} textAnchor="middle" fontSize="10" fill="#64748b" fontFamily="sans-serif" fontWeight="500">Principal Component 1</text>
        <text x={25} y={(containerHeight + topPadding)/2} textAnchor="middle" fontSize="10" fill="#64748b" fontFamily="sans-serif" fontWeight="500" transform={`rotate(-90, 25, ${(containerHeight + topPadding)/2})`}>Principal Component 2</text>


        {points.map((p) => {
            const cx = toSvgX(p.x);
            const cy = toSvgY(p.y);
            const isHovered = hovered?.id === p.id;
            const isSelected = selected === p.id;
            
            // Color mapping: Band Gap (Scientific Palette)
            const bg = p.targets?.band_gap || 0;
            let color = '#3b82f6'; // Default Blue
            if (bg <= 0.5) color = '#6366f1'; // Indigo (Metal)
            else if (bg <= 2.5) color = '#10b981'; // Emerald (Semi)
            else color = '#f59e0b'; // Amber (Insulator)

            return (
              <circle
                key={p.id}
                cx={cx}
                cy={cy}
                r={isHovered ? 6 : (isSelected ? 4 : 1.8)}
                fill={isSelected ? '#1e293b' : color} 
                stroke={isSelected ? '#fff' : 'transparent'}
                strokeWidth={isSelected ? 2 : 0}
                opacity={isHovered || isSelected ? 1 : 0.6}
                onMouseEnter={() => setHovered(p)}
                onMouseLeave={() => setHovered(null)}
                onClick={() => {
                  setSelected(p.id);
                  onSelect?.(p.id);
                }}
                className="transition-all duration-200"
                style={{ cursor: 'pointer' }}
              />
            );
          })}
        
      </svg>

      {/* Stats & Tooltip */}
      <div className="absolute bottom-4 left-6 pointer-events-none z-10">
          <div className="text-[10px] font-mono text-gray-400 bg-white/50 backdrop-blur px-2 py-1 rounded border border-gray-100">
             POINTS: <span className="text-gray-900 font-bold">{points.length}</span>
             {points.length > 100 && <span className="ml-2 text-emerald-600 font-bold">● LIVE DATA</span>}
          </div>
      </div>

      <AnimatePresence>
        {hovered && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="absolute z-20 pointer-events-none bg-white/95 backdrop-blur border border-gray-200 px-3 py-2 rounded shadow-xl flex flex-col gap-0.5 min-w-[140px]"
            style={{
              left: Math.min(toSvgX(hovered.x) + 15, width - 160), // Prevent overflow right
              top: Math.min(toSvgY(hovered.y) - 40, containerHeight - 80) // Prevent overflow bottom
            }}
          >
            <div className="text-gray-900 font-bold text-xs">{hovered.formula}</div>
            <div className="text-[10px] text-gray-500 font-mono border-t border-gray-100 pt-1 mt-1">
               Band Gap: {hovered.targets?.band_gap?.toFixed(3)} eV
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Legend */}
      <div className="absolute top-4 right-4 flex flex-col gap-1.5 bg-white/90 p-3 rounded border border-gray-100 shadow-sm text-[10px] text-gray-600">
        <span className="font-bold text-gray-900 uppercase tracking-wider mb-1">Band Gap (eV)</span>
        <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-indigo-500"></div> &lt; 0.5 (Metal)</div>
        <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-emerald-500"></div> 0.5 - 2.5 (Semi)</div>
        <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-amber-500"></div> &gt; 2.5 (Insulator)</div>
      </div>
    </div>
  );
}
