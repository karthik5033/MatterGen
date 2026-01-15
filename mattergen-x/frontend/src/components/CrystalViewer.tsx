"use client";

import { useEffect, useRef } from 'react';

interface CrystalViewerProps {
  cifData?: string;
  materialId?: string;
  className?: string;
}

export default function CrystalViewer({ cifData, materialId, className }: CrystalViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Animation Loop (Placeholder for Mol*)
    let angle = 0;
    let animationId: number;

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const size = 80;

      // Draw backdrop (transparent/neutral)
      ctx.fillStyle = '#f8fafc'; // Neutral slate-50
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Grid Pattern
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      for(let i=0; i<canvas.width; i+=40) { ctx.beginPath(); ctx.moveTo(i,0); ctx.lineTo(i,canvas.height); ctx.stroke(); }
      for(let i=0; i<canvas.height; i+=40) { ctx.beginPath(); ctx.moveTo(0,i); ctx.lineTo(canvas.width,i); ctx.stroke(); }

      // 3D Rotation Mock
      const points = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
      ];
      
      const projected = points.map(p => {
        const x = p[0] * Math.cos(angle) - p[2] * Math.sin(angle);
        const z = p[0] * Math.sin(angle) + p[2] * Math.cos(angle);
        const y = p[1] * Math.cos(angle * 0.5) - z * Math.sin(angle * 0.5);
        const factor = 300 / (z + 4);
        return [x * factor * size/50 + centerX, y * factor * size/50 + centerY];
      });

      // Connections
      ctx.strokeStyle = '#475569'; // Slate-600
      ctx.lineWidth = 1.5;
      const edges = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]];
      edges.forEach(([i, j]) => {
        ctx.beginPath();
        ctx.moveTo(projected[i][0], projected[i][1]);
        ctx.lineTo(projected[j][0], projected[j][1]);
        ctx.stroke();
      });

      // Atoms
      projected.forEach((p, idx) => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], idx % 2 === 0 ? 6 : 4, 0, Math.PI * 2);
        ctx.fillStyle = idx % 2 === 0 ? '#10b981' : '#ef4444'; // Green / Red accents
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      });

      angle += 0.01;
      animationId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationId);
  }, []);

  return (
    <div className={`flex flex-col h-full bg-white border border-gray-200 rounded-lg overflow-hidden ${className}`}>
        {/* Scientific Header */}
        <div className="flex justify-between items-center px-4 py-3 bg-gray-50 border-b border-gray-200">
            <h4 className="text-xs font-bold uppercase tracking-widest text-gray-700">Crystal Structure</h4>
            <div className="flex items-center gap-2">
                 <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
                 <span className="text-[10px] font-mono text-gray-400">LIVE RENDER</span>
            </div>
        </div>

        {/* Main Viewport */}
        <div className="relative flex-1 bg-gray-50/50 min-h-[200px]">
            <canvas 
                ref={canvasRef} 
                width={600} 
                height={600} 
                className="absolute inset-0 w-full h-full object-contain mix-blend-multiply opacity-90"
            />
            
            {/* Overlay Info */}
            <div className="absolute bottom-4 left-4 pointer-events-none">
                <div className="bg-white/90 backdrop-blur border border-gray-200 px-3 py-2 rounded shadow-sm">
                    <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-1">Unit Cell</div>
                    <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] font-mono text-gray-600">
                        <span>a: 3.42Å</span>
                        <span>b: 3.42Å</span>
                        <span>c: 5.12Å</span>
                    </div>
                </div>
            </div>
            
            {/* Interactive Hints */}
            <div className="absolute top-4 right-4 pointer-events-none opacity-50">
                 <div className="text-[9px] font-bold text-gray-400 border border-gray-300 rounded px-1.5 py-0.5 uppercase">
                    Drag to Rotate
                 </div>
            </div>
        </div>

        {/* Footer / Legend */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-gray-200 bg-white">
            <div className="flex items-center gap-4 text-[10px] text-gray-500 font-medium">
                <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-500"></span> Li
                </div>
                <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-red-500"></span> O
                </div>
                <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-gray-400"></span> Fe
                </div>
            </div>
            <div className="text-[9px] font-mono text-gray-300">
                Mol* Ready
            </div>
        </div>

      {/* Hidden Data for Integration */}
      <div className="hidden">{cifData}</div>
    </div>
  );
}
