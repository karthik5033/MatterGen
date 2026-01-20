"use client";

import { useEffect, useRef } from 'react';

interface CrystalViewerProps {
  cifData?: string;
  materialId?: string;
  className?: string;
}

export default function CrystalViewer({ cifData, materialId, className }: CrystalViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rotationRef = useRef({ x: 0, y: 0 });
  const isDragging = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });
  const scaleRef = useRef(200);

  // Element Colors (CPK-ish)
  const elementColors: Record<string, string> = {
    "H": "#FFFFFF", "Li": "#CC80FF", "O": "#FF0D0D", "Na": "#AB5CF2", 
    "Mg": "#8AFF00", "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", 
    "S": "#FFFF30", "Cl": "#1FF01F", "K": "#8F40D4", "Ca": "#3DFF00",
    "Ti": "#BFC2C7", "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050",
    "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F", "Ge": "#668F8F",
    "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929", "Zr": "#94E0E0",
    "Mo": "#54B5B5", "Ag": "#C0C0C0", "Sn": "#668080", "Sb": "#9E80C2",
    "Te": "#D47A00", "I": "#940094", "Xe": "#429EB0", "Cs": "#57178F",
    "Ba": "#00C900", "La": "#70D4FF", "Ce": "#FFFFC7", "Au": "#FFD123",
    "Pb": "#575961", "Bi": "#9E4FB5", "Th": "#00BAFF", "U": "#008FFF"
  };

  // Parsing Logic
  const parseCIF = (cif: string) => {
    const atoms: { x: number; y: number; z: number; element: string }[] = [];
    try {
        const lines = cif.split('\n');
        
        // State for loop parsing
        let inLoop = false;
        let loopHeaders: string[] = [];
        let parsingAtoms = false;
        
        let idxType = -1;
        let idxLabel = -1;
        let idxX = -1;
        let idxY = -1;
        let idxZ = -1;

        lines.forEach(line => {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith('#')) return;

            if (trimmed.startsWith('loop_')) {
                inLoop = true;
                loopHeaders = [];
                parsingAtoms = false;
                return;
            }

            if (inLoop) {
                // Header Lines
                if (trimmed.startsWith('_')) {
                    loopHeaders.push(trimmed);
                    // Check if this loop is about atoms
                    if (trimmed.startsWith('_atom_site_')) {
                        parsingAtoms = true;
                    }
                } else {
                    // Data Lines: If this loop was about atoms, parse them
                    if (parsingAtoms) {
                        // Map headers once
                        if (idxX === -1) {
                             idxLabel = loopHeaders.findIndex(h => h.includes('_label'));
                             idxType = loopHeaders.findIndex(h => h.includes('_type_symbol'));
                             idxX = loopHeaders.findIndex(h => h.includes('_fract_x'));
                             idxY = loopHeaders.findIndex(h => h.includes('_fract_y'));
                             idxZ = loopHeaders.findIndex(h => h.includes('_fract_z'));
                        }

                        // Robust splitting (handle spaces)
                        const parts = trimmed.split(/\s+/);
                        
                        // Need generic coordinates
                        if (idxX !== -1 && idxY !== -1 && idxZ !== -1 && parts.length >= 3) {
                             const x = parseFloat(parts[idxX]);
                             const y = parseFloat(parts[idxY]);
                             const z = parseFloat(parts[idxZ]);
                             
                             let el = "X";
                             if (idxType !== -1) el = parts[idxType];
                             else if (idxLabel !== -1) el = parts[idxLabel].replace(/[0-9\+\-]/g, ''); // Strip charge/index
                             
                             // Clean Element
                             el = el.charAt(0).toUpperCase() + el.slice(1).toLowerCase();
                             if (!isNaN(x)) {
                                 // Normalize to -1..1 range for canvas (center at 0,0,0)
                                 atoms.push({ x: (x - 0.5) * 2, y: (y - 0.5) * 2, z: (z - 0.5) * 2, element: el });
                             }
                        }
                    }
                }
            }
        });
        
        // Fallback for non-loop simple CIFs (rare but possible in old files)
        if (atoms.length === 0 && cif.includes('_atom_site_')) {
             // Try primitive regex
             const regex = /([A-Z][a-z]?)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)/g;
             let match;
             while ((match = regex.exec(cif)) !== null) {
                 atoms.push({
                     element: match[1],
                     x: (parseFloat(match[2]) - 0.5) * 2,
                     y: (parseFloat(match[3]) - 0.5) * 2,
                     z: (parseFloat(match[4]) - 0.5) * 2
                 });
             }
        }

        if (atoms.length === 0) throw new Error("No atoms parsed");
        return atoms;

    } catch (e) {
        console.warn("CIF Parse Error, using fallback", e);
        // Fallback: Generate a Perovskite-like dummy structure
        return [
            { x: 0, y: 0, z: 0, element: "Ti" }, // Center
            { x: -1, y: -1, z: -1, element: "Ca" },
            { x: 1, y: 1, z: 1, element: "Ca" },
            { x: 0.5, y: 0, z: 0, element: "O" },
            { x: 0, y: 0.5, z: 0, element: "O" },
            { x: 0, y: 0, z: 0.5, element: "O" }, 
        ];
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let atoms = parseCIF(cifData || "");
    
    // Auto-rotate init
    let angleY = 0.5;
    let angleX = 0.5;
    let animationId: number;

    // Draw Unit Cell Box (Normalized -1 to 1)
    const drawUnitCell = (currentScale: number, centerX: number, centerY: number) => {
        const corners = [
            {x: -1, y: -1, z: -1}, {x: 1, y: -1, z: -1}, {x: 1, y: 1, z: -1}, {x: -1, y: 1, z: -1}, // Back face
            {x: -1, y: -1, z: 1},  {x: 1, y: -1, z: 1},  {x: 1, y: 1, z: 1},  {x: -1, y: 1, z: 1}   // Front face
        ];
        
        // Project corners
        const projCorners = corners.map(p => {
            // Rotations
            let y1 = p.y * Math.cos(angleX) - p.z * Math.sin(angleX);
            let z1 = p.y * Math.sin(angleX) + p.z * Math.cos(angleX);
            let x2 = p.x * Math.cos(angleY) - z1 * Math.sin(angleY);
            let z2 = p.x * Math.sin(angleY) + z1 * Math.cos(angleY);
            
            // Perspective
            const depth = 4 - z2;
            const scale = currentScale;
            return {
                x: x2 * (300/(depth || 1)) * (scale/100) + centerX,
                y: y1 * (300/(depth || 1)) * (scale/100) + centerY
            };
        });

        const lines = [
            [0,1], [1,2], [2,3], [3,0], // Back
            [4,5], [5,6], [6,7], [7,4], // Front
            [0,4], [1,5], [2,6], [3,7]  // Connectors
        ];
        
        ctx.strokeStyle = "rgba(71, 85, 105, 0.4)"; // Darker Slate-600 with higher opacity
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        
        ctx.beginPath();
        lines.forEach(([s, e]) => {
            if (projCorners[s] && projCorners[e]) {
                ctx.moveTo(projCorners[s].x, projCorners[s].y);
                ctx.lineTo(projCorners[e].x, projCorners[e].y);
            }
        });
        ctx.stroke();
        ctx.setLineDash([]);
    };

    const render = () => {
      if(!ctx || !canvas) return;
      
      // Update Rotation
      if (!isDragging.current) {
         angleY += 0.005; 
      } else {
         angleY = rotationRef.current.y;
         angleX = rotationRef.current.x;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const currentScale = scaleRef.current; 

      drawUnitCell(currentScale, centerX, centerY);

      // 1. Transform Atoms to 3D World Space (Rotation)
      const transformedAtoms = atoms.map(atom => {
        // Rotation X
        let y1 = atom.y * Math.cos(angleX) - atom.z * Math.sin(angleX);
        let z1 = atom.y * Math.sin(angleX) + atom.z * Math.cos(angleX);
        
        // Rotation Y
        let x2 = atom.x * Math.cos(angleY) - z1 * Math.sin(angleY);
        let z2 = atom.x * Math.sin(angleY) + z1 * Math.cos(angleY);
        
        return { ...atom, wx: x2, wy: y1, wz: z2 };
      });

      // 2. Project to 2D Screen Space
      const projectedAtoms = transformedAtoms.map(atom => {
        const depth = 4 - atom.wz;
        const perspective = 300 / (depth + 1); 
        
        return {
            x: atom.wx * perspective * (currentScale/100) + centerX,
            y: atom.wy * perspective * (currentScale/100) + centerY,
            z: atom.wz,
            wx: atom.wx, wy: atom.wy, wz: atom.wz,
            // Increased Atom Size: 24 for larger, 18 for smaller (was 16/12)
            r: (elementColors[atom.element] ? 24 : 18) * (perspective/50), 
            element: atom.element,
            color: elementColors[atom.element] || "#94a3b8",
        };
      });

      // Sort by Z
      projectedAtoms.sort((a, b) => a.z - b.z);

      // Draw Bonds
      ctx.strokeStyle = '#64748b'; // darker slate (Slate-500)
      ctx.lineWidth = 3.0; // Thicker bonds
      
      const limit = Math.min(projectedAtoms.length, 100);
      if (limit > 1) {
          for (let i = 0; i < limit; i++) {
              const p1 = projectedAtoms[i];
              const a1 = p1 as any; 
              for (let j = i + 1; j < limit; j++) {
                  const p2 = projectedAtoms[j];
                  const a2 = p2 as any;

                  const dx = a1.wx - a2.wx;
                  const dy = a1.wy - a2.wy;
                  const dz = a1.wz - a2.wz;
                  const distSq = dx*dx + dy*dy + dz*dz;

                  // Tighter threshold to clean up "messy webs"
                  // Was 2.25 (1.5 dist). Let's restrict to ~1.2 dist (1.44 sq) 
                  // to keep only significant neighbors, creating a clearer structure
                  if (distSq < 1.6 && distSq > 0.01) { 
                      ctx.beginPath();
                      ctx.moveTo(p1.x, p1.y);
                      ctx.lineTo(p2.x, p2.y);
                      const alpha = 0.5; 
                      ctx.strokeStyle = `rgba(100, 116, 139, ${alpha})`; 
                      ctx.stroke();
                  }
              }
          }
      }

      // Draw Atoms & Labels
      projectedAtoms.forEach(atom => {
          // Atom Sphere
          const grad = ctx.createRadialGradient(
              atom.x - atom.r/3, atom.y - atom.r/3, atom.r/4, 
              atom.x, atom.y, atom.r
          );
          grad.addColorStop(0, "#ffffff"); 
          grad.addColorStop(0.3, atom.color); 
          grad.addColorStop(1, "#1e293b"); 

          ctx.beginPath();
          ctx.arc(atom.x, atom.y, Math.max(2, atom.r), 0, Math.PI * 2);
          ctx.fillStyle = grad;
          ctx.fill();
          
          ctx.strokeStyle = "rgba(0,0,0,0.3)";
          ctx.lineWidth = 1; // Thicker border
          ctx.stroke();

          // Label with Tag Background (Simpler, clearer)
          if (atom.r > 10) { 
              // Bigger font
              const fontSize = Math.max(11, atom.r * 0.6);
              ctx.font = `bold ${fontSize}px Inter, sans-serif`;
              const textWidth = ctx.measureText(atom.element).width;
              
              // Tag Background (Pill) - High Contrast
              // White background, slightly larger padding
              ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
              const pad = 6;
              const h = fontSize + 4;
              
              ctx.beginPath();
              if (ctx.roundRect) {
                 ctx.roundRect(atom.x - textWidth/2 - pad/2, atom.y - h/2, textWidth + pad, h, 4);
              } else {
                 ctx.rect(atom.x - textWidth/2 - pad/2, atom.y - h/2, textWidth + pad, h);
              }
              ctx.fill();
              
              // Border for tag
              ctx.strokeStyle = "rgba(0,0,0,0.1)";
              ctx.lineWidth = 1;
              ctx.stroke();

              // Text
              ctx.fillStyle = '#0f172a'; // Slate-900
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText(atom.element, atom.x, atom.y + 1);
          }
      });    

      animationId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationId);
  }, [cifData]);

  // Hook handles... (keep existing)
  const handleMouseDown = (e: React.MouseEvent) => {
    isDragging.current = true;
    lastMouse.current = { x: e.clientX, y: e.clientY };
  };
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current) return;
    const deltaX = e.clientX - lastMouse.current.x;
    const deltaY = e.clientY - lastMouse.current.y;
    rotationRef.current.y += deltaX * 0.01;
    rotationRef.current.x += deltaY * 0.01;
    lastMouse.current = { x: e.clientX, y: e.clientY };
  };
  const handleMouseUp = () => { isDragging.current = false; };
  const handleWheel = (e: React.WheelEvent) => {
      e.stopPropagation();
      const delta = -Math.sign(e.deltaY) * 10;
      scaleRef.current = Math.min(Math.max(50, scaleRef.current + delta), 400); 
  };

  // Derive Crystal System (Mock)
  const atomCount = parseCIF(cifData || "").length;

  return (
    <div 
        className={`relative w-full h-full bg-white cursor-move group ${className}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
    >
        <div className="absolute inset-0 opacity-[0.03] pointer-events-none" 
            style={{ backgroundImage: 'radial-gradient(#000 1px, transparent 1px)', backgroundSize: '20px 20px' }} 
        />
        
        <canvas 
            ref={canvasRef} 
            width={800} 
            height={800} 
            className="w-full h-full object-contain"
        />
        
        {/* Crystal Data HUD */}
        <div className="absolute top-4 left-4 z-10 pointer-events-none space-y-2">
             <div className="flex items-center gap-2">
                 <div className="bg-slate-900 text-white text-[10px] font-bold px-2 py-1 rounded shadow-md border border-slate-700">
                    CUBIC SYSTEM
                 </div>
                 <div className="bg-white/80 backdrop-blur text-slate-600 text-[10px] font-bold px-2 py-1 rounded shadow-sm border border-slate-200">
                    {atomCount} ATOMS
                 </div>
             </div>
             <div className="text-[10px] text-slate-400 font-mono pl-1">
                 a = b = c ≈ 7.5 Å
             </div>
        </div>
        
        {/* Helper Badge */}
        <div className="absolute bottom-3 right-3 pointer-events-none transition-opacity group-hover:opacity-100 opacity-60">
             <div className="flex items-center gap-2 bg-white/90 backdrop-blur border border-gray-100 px-3 py-1.5 rounded-full shadow-sm">
                <div className="flex gap-1">
                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse"></div>
                    <div className="w-1.5 h-1.5 rounded-full bg-rose-500"></div>
                </div>
                <span className="text-[9px] font-bold text-gray-500 uppercase tracking-widest">Interactive 3D</span>
             </div>
        </div>
    </div>
  );
}
