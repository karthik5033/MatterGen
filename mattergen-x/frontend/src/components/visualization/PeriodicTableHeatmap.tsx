
"use client";

import React, { useEffect, useState } from 'react';
import { ApiService } from '@/lib/api';
import { motion, AnimatePresence } from 'framer-motion';

// Simplistic Periodic Table Layout
// Grouped by row to make grid easy
const LAYOUT = [
    // Row 1
    ["H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"],
    // Row 2
    ["Li", "Be", "", "", "", "", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne"],
    // Row 3
    ["Na", "Mg", "", "", "", "", "", "", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar"],
    // Row 4
    ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
    // Row 5
    ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
    // Row 6
    ["Cs", "Ba", "La*", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
    // Row 7 (Incomplete for brevity/layout)
    ["Fr", "Ra", "Ac*", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
];

// Lanthanides / Actinides
const ROW_L = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"];
const ROW_A = ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"];

interface ElementData {
    count: number;
    avg_energy_above_hull: number;
    avg_bulk_modulus: number;
}

export const PeriodicTableHeatmap = () => {
    const [stats, setStats] = useState<Record<string, ElementData>>({});
    const [loading, setLoading] = useState(true);
    const [hoveredEl, setHoveredEl] = useState<string | null>(null);

    useEffect(() => {
        const load = async () => {
            try {
                const data = await ApiService.getElementStats();
                setStats(data);
            } catch (e) {
                console.error("Failed to load element stats", e);
            } finally {
                setLoading(false);
            }
        };
        load();
    }, []);

    const getColor = (el: string) => {
        if (!stats[el]) return 'bg-white border-gray-100 text-gray-200 pointer-events-none';
        
        // Stability based on Energy Above Hull (eV/atom)
        // 0.0 = Stable (On Hull)
        // < 0.05 = Highly Stable
        // < 0.1 = Metastable
        // > 0.1 = Unstable
        const val = stats[el].avg_energy_above_hull ?? 0;
        
        // Ideally 0 or very close to 0
        if (val <= 0.02) return 'bg-slate-900 border-slate-900 text-white shadow-md';
        if (val <= 0.05) return 'bg-slate-700 border-slate-700 text-white';
        if (val <= 0.10) return 'bg-slate-400 border-slate-400 text-white';
        if (val <= 0.20) return 'bg-slate-100 border-slate-200 text-slate-500';
        
        // Highly Unstable
        return 'bg-white border-red-100 text-red-300';
    };

    // Determine HUD Position (Quadtree logic simplified)
    const getHudPosition = (el: string) => {
         // Find column index in LAYOUT
         let colIndex = -1;
         LAYOUT.forEach(row => {
             const idx = row.indexOf(el);
             if (idx !== -1) colIndex = idx;
         });
         
         if (colIndex === -1) {
             // Check F-block
             if (ROW_L.includes(el) || ROW_A.includes(el)) colIndex = 4; // Approx left-ish
             if (ROW_L.slice(8).includes(el)) colIndex = 12; // Approx right-ish
         }

         // If on left side (0-8), show on RIGHT
         // If on right side (9-17), show on LEFT
         if (colIndex > 8) {
             return { left: 0, right: 'auto' }; // Show on Left
         }
         return { right: 0, left: 'auto' }; // Show on Right
    };

    const ElementCell = ({ el }: { el: string }) => {
        if (!el) return <div className="aspect-square"></div>;
        
        if (el === "La*" || el === "Ac*") {
             return <div className="aspect-square flex items-center justify-center text-[10px] text-gray-300 font-mono select-none">{el}</div>;
        }

        const data = stats[el];
        const baseClass = "aspect-square rounded-sm flex flex-col items-center justify-center text-xs font-bold transition-all duration-200 border cursor-pointer relative group";
        const colorClass = getColor(el);
        
        return (
            <motion.div 
               className={`${baseClass} ${colorClass}`}
               onMouseEnter={() => setHoveredEl(el)}
               onMouseLeave={() => setHoveredEl(null)}
               whileHover={{ scale: 1.15, zIndex: 30, boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1)" }}
            >
                <div className="font-mono tracking-tighter">{el}</div>
                {data && <div className={`text-[8px] font-medium leading-none mt-0.5 ${data.avg_energy_above_hull < 0.05 ? 'opacity-60' : 'opacity-100'}`}>{data.count}</div>}
            </motion.div>
        );
    };

    return (
        <div className="w-full relative">
            <h3 className="text-xl font-black text-gray-900 mb-6 flex items-center gap-3">
                <span className="w-8 h-8 rounded-lg bg-gray-900 text-white flex items-center justify-center text-sm font-bold shadow-lg shadow-gray-200">Pr</span>
                <span className="bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600">Elemental Stability Matrix</span>
            </h3>
            
            {loading ? (
                <div className="h-[400px] flex flex-col gap-3 items-center justify-center text-gray-400">
                    <div className="relative">
                        <div className="w-12 h-12 rounded-xl border-2 border-indigo-500/20 border-t-indigo-500 animate-spin"></div>
                        <div className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-indigo-500">AI</div>
                    </div>
                    <div className="text-xs font-mono uppercase tracking-widest animate-pulse">Processing 45,229 Materials...</div>
                    <div className="text-[10px] text-gray-300">Synchronizing chemical space intelligence</div>
                </div>
            ) : (
                <div className="flex flex-col gap-6 select-none">
                    {/* Main Table */}
                    <div className="grid grid-cols-[repeat(18,minmax(0,1fr))] gap-1.5 pr-4">
                        {LAYOUT.map((row, rI) => (
                           <React.Fragment key={rI}>
                               {row.map((el, cI) => <ElementCell key={`${rI}-${cI}`} el={el} />)}
                           </React.Fragment> 
                        ))}
                    </div>
                    
                    {/* F-Block */}
                    <div className="pl-20 mt-2">
                         <div className="grid grid-cols-[repeat(15,minmax(0,1fr))] gap-1.5 max-w-[83%]">
                             {ROW_L.map(el => <ElementCell key={el} el={el} />)}
                         </div>
                         <div className="grid grid-cols-[repeat(15,minmax(0,1fr))] gap-1.5 max-w-[83%] mt-1.5">
                             {ROW_A.map(el => <ElementCell key={el} el={el} />)}
                         </div>
                    </div>
                </div>
            )}

            {/* Scientific Overlay / HUD */}
            <AnimatePresence>
                {hoveredEl && stats[hoveredEl] && (
                    <motion.div 
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        transition={{ duration: 0.1 }}
                        className="absolute top-10 w-72 bg-white/80 backdrop-blur-xl p-6 rounded-2xl border border-white/50 shadow-2xl shadow-indigo-500/10 z-50 ring-1 ring-gray-900/5"
                        style={getHudPosition(hoveredEl)}
                    >
                         <div className="flex justify-between items-start mb-6">
                            <div>
                                <h4 className="text-5xl font-black tracking-tighter text-gray-900 mb-1">{hoveredEl}</h4>
                                <div className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Properties</div>
                            </div>
                            <div className={`w-12 h-12 rounded-xl flex items-center justify-center text-xl font-bold border-2 ${getColor(hoveredEl).replace('shadow-', '').replace('text-white', 'text-gray-900 bg-transparent')}`}>
                                {hoveredEl}
                            </div>
                         </div>
                         
                         <div className="space-y-5">
                             <div>
                                 <div className="flex justify-between text-xs font-bold mb-1.5">
                                     <span className="text-gray-400 uppercase tracking-wider">Frequency</span>
                                     <span className="font-mono text-gray-900">{stats[hoveredEl].count} <span className="text-gray-400 font-normal">samples</span></span>
                                 </div>
                                 <div className="w-full bg-gray-100 h-1.5 rounded-full overflow-hidden"><div className="bg-gray-900 h-full rounded-full" style={{ width: `${Math.min(stats[hoveredEl].count / 2, 100)}%` }}></div></div>
                             </div>

                             <div>
                                 <div className="flex justify-between text-xs font-bold mb-1.5">
                                     <span className="text-gray-400 uppercase tracking-wider">Avg. Stability (E_h)</span>
                                     <span className={`font-mono ${(stats[hoveredEl].avg_energy_above_hull ?? 0) < 0.05 ? 'text-emerald-600' : 'text-amber-600'}`}>{(stats[hoveredEl].avg_energy_above_hull ?? 0).toFixed(3)} eV/atom</span>
                                 </div>
                                 {/* Bar: For Eh, closer to 0 is FULL bar (better). So 1 - (val * scale) */}
                                 <div className="w-full bg-gray-100 h-1.5 rounded-full overflow-hidden"><div className={`${(stats[hoveredEl].avg_energy_above_hull ?? 0) < 0.05 ? 'bg-emerald-500' : 'bg-amber-500'} h-full rounded-full`} style={{ width: `${Math.max(0, 100 - ((stats[hoveredEl].avg_energy_above_hull ?? 0) * 500))}%` }}></div></div>
                             </div>

                             <div>
                                 <div className="flex justify-between text-xs font-bold mb-1.5">
                                     <span className="text-gray-400 uppercase tracking-wider">Bulk Modulus</span>
                                     <span className="font-mono text-indigo-600">{(stats[hoveredEl].avg_bulk_modulus ?? 0).toFixed(1)} GPa</span>
                                 </div>
                                 <div className="w-full bg-gray-100 h-1.5 rounded-full overflow-hidden"><div className="bg-indigo-500 h-full rounded-full" style={{ width: `${Math.min((stats[hoveredEl].avg_bulk_modulus ?? 0) / 3, 100)}%` }}></div></div>
                             </div>
                         </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
