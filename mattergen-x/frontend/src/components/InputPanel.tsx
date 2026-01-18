import React, { useState } from 'react';
import { WeightSliders } from './WeightSliders';

interface InputPanelProps {
  onGenerate: (prompt: string, weights: Record<string, number>) => Promise<void>;
  isLoading: boolean;
  error: string | null;
}

/**
 * Component for the project specification input panel.
 * Encapsulates the prompt textarea and the submission button.
 */
export default function InputPanel({ onGenerate, isLoading, error }: InputPanelProps) {
  const [prompt, setPrompt] = useState("");
  const [weights, setWeights] = useState({
    density: 0.5,
    stability: 0.5,
    band_gap: 0.5,
    shear_modulus: 0.5,
    thermal_conductivity: 0.5,
    refractive_index: 0.5
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onGenerate(prompt, weights);
  };

  /* 
    Categorized Prompt Library (50 Items)
  */
  const promptCategories = {
    "Energy & Battery": [
      { label: "Li-Ion Cathode", prompt: "High-capacity lithium cobalt oxide cathode" },
      { label: "Solid Electrolyte", prompt: "High ionic conductivity garnet-type electrolyte" },
      { label: "Perovskite Solar", prompt: "Stable lead-free perovskite for photovoltaics" },
      { label: "Hydrogen Storage", prompt: "Metal-organic framework for high-density H2 storage" },
      { label: "Fuel Cell Catalyst", prompt: "Pt-free catalyst for oxygen reduction reaction" },
      { label: "Supercapacitor", prompt: "High surface area porous carbon electrode" },
      { label: "Thermometric", prompt: "High ZT thermoelectric material for waste heat recovery" },
      { label: "Nuclear Fuel", prompt: "High thermal conductivity uranium nitride fuel" },
      { label: "Radiation Shield", prompt: "Boron-rich neutron absorbing ceramic" },
      { label: "Sodium-Ion Anode", prompt: "Hard carbon anode for sodium-ion batteries" }
    ],
    "Electronics & Computing": [
      { label: "High-k Dielectric", prompt: "High dielectric constant oxide for transistors" },
      { label: "Topological Insulator", prompt: "Bi2Se3-based topological insulator candidate" },
      { label: "2D Semiconductor", prompt: "Monolayer transition metal dichalcogenide" },
      { label: "Transparent Conductor", prompt: "High mobility indium tin oxide alternative" },
      { label: "Superconductor", prompt: "High-Tc cuprate superconductor structure" },
      { label: "Piezoelectric", prompt: "Lead-free piezoelectric ceramic for sensors" },
      { label: "Ferroelectric", prompt: "Room temperature ferroelectric memory material" },
      { label: "Spintronic", prompt: "Half-metallic ferromagnet for spintronics" },
      { label: "Quantum Dot", prompt: "Cadmium-free quantum dot emitter" },
      { label: "Low-k Interconnect", prompt: "Ultra-low dielectric constant porous silica" }
    ],
    "Structural & Alloys": [
      { label: "Ultra-hard Carbon", prompt: "Superhard carbon polymorph harder than diamond" },
      { label: "High-Entropy Alloy", prompt: "Single-phase FCC high-entropy alloy" },
      { label: "Shape Memory", prompt: "NiTi-based shape memory alloy" },
      { label: "Lightweight Armor", prompt: "High strength-to-weight boron carbide ceramic" },
      { label: "Refractory Metal", prompt: "Ultra-high melting point tungsten alloy" },
      { label: "Self-Healing", prompt: "Polymer-mimetic self-healing ceramic" },
      { label: "Bulk Metallic Glass", prompt: "Zr-based bulk metallic glass composition" },
      { label: "Aerospace Superalloy", prompt: "Creep-resistant nickel-based superalloy" },
      { label: "Biocompatible Implant", prompt: "Low modulus titanium alloy for implants" },
      { label: "Thermal Barrier", prompt: "Low thermal conductivity zirconia coating" }
    ],
    "Magnetic & Optical": [
      { label: "Permanent Magnet", prompt: "High coercivity rare-earth free magnet" },
      { label: "Magnetocaloric", prompt: "Giant magnetocaloric effect material" },
      { label: "Skyrmion Host", prompt: "Chiral magnet hosting magnetic skyrmions" },
      { label: "Multiferroic", prompt: "Room temperature magnetoelectric multiferroic" },
      { label: "Photocatalyst", prompt: "Visible light active photocatalyst for water splitting" },
      { label: "Laser Host", prompt: "Nd-doped yttrium aluminum garnet host" },
      { label: "Non-linear Optical", prompt: "High SHG efficiency crystal" },
      { label: "Scintillator", prompt: "Fast decay lutetium oxyorthosilicate scintillator" },
      { label: "Plasmonic", prompt: "Low-loss plasmonic metal nitride" },
      { label: "Electrochromic", prompt: "Fast switching tungsten oxide window" }
    ],
    "Novel & Exotic": [
      { label: "Weyl Semimetal", prompt: "TaAs-family Weyl semimetal candidate" },
      { label: "Axion Insulator", prompt: "Magnetic topological insulator axion state" },
      { label: "Spin Liquid", prompt: "Quantum spin liquid candidate in Kagome lattice" },
      { label: "Hyperuniform", prompt: "Disordered hyperuniform photonic structure" },
      { label: "Metamaterial", prompt: "Negative refractive index metamaterial" },
      { label: "Time Crystal", prompt: "Discrete time crystal phase candidate" },
      { label: "Superionic", prompt: "Superionic ice phase at high pressure" },
      { label: "Quasicrystal", prompt: "Icosahedral quasicrystal alloy" },
      { label: "Heavy Fermion", prompt: "Heavy fermion superconductor" },
      { label: "Excitonic", prompt: "Excitonic insulator condensate candidate" }
    ]
  };

  const [activeCategory, setActiveCategory] = useState<keyof typeof promptCategories>("Energy & Battery");
  const [showPresets, setShowPresets] = useState(false);
  const dropdownRef = React.useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  React.useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowPresets(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="bg-white border border-gray-200 rounded-2xl p-8 shadow-xl relative overflow-hidden">
      {/* Subtle Texture Background */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none" 
            style={{ backgroundImage: 'linear-gradient(#000 1px, transparent 1px), linear-gradient(90deg, #000 1px, transparent 1px)', backgroundSize: '24px 24px' }}>
      </div>
      
      <form onSubmit={handleSubmit} className="relative space-y-8 z-10">
        
        {/* Prompt Section */}
        <div className="space-y-4">
          <div className="flex justify-between items-center relative z-20">
            <label htmlFor="prompt" className="text-xs font-bold tracking-widest text-slate-500 uppercase">
              Target Specification
            </label>
            
            <div className="relative" ref={dropdownRef}>
                <button
                    type="button"
                    onClick={() => setShowPresets(!showPresets)}
                    className={`flex items-center gap-2 text-xs font-semibold px-4 py-2 rounded-lg border transition-all ${showPresets ? 'bg-slate-800 text-white border-slate-800' : 'bg-white text-slate-600 border-gray-300 hover:border-slate-400 hover:text-slate-900'}`}
                >
                    <span>Browse Presets</span>
                    <svg className={`w-3 h-3 transition-transform ${showPresets ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                </button>

                {/* MEGA MENU DROPDOWN */}
                {showPresets && (
                    <div className="absolute right-0 top-12 w-[600px] bg-white rounded-xl shadow-2xl border border-gray-200 ring-1 ring-black/5 overflow-hidden flex flex-col md:flex-row z-50 animate-in fade-in zoom-in-95 duration-200">
                        {/* Categories Sidebar */}
                        <div className="w-48 bg-slate-50 border-r border-gray-200 p-2 space-y-1">
                            {Object.keys(promptCategories).map((cat) => (
                                <button
                                    key={cat}
                                    type="button"
                                    onClick={() => setActiveCategory(cat as keyof typeof promptCategories)}
                                    className={`w-full text-left px-3 py-2.5 rounded-lg text-xs font-medium transition-all flex justify-between items-center ${
                                        activeCategory === cat 
                                        ? 'bg-white text-slate-900 shadow-sm ring-1 ring-gray-200' 
                                        : 'text-slate-500 hover:bg-gray-100 hover:text-slate-900'
                                    }`}
                                >
                                    {cat}
                                    {activeCategory === cat && <div className="w-1.5 h-1.5 rounded-full bg-slate-800"></div>}
                                </button>
                            ))}
                        </div>

                        {/* Prompts Grid */}
                        <div className="flex-1 p-4 bg-white max-h-[320px] overflow-y-auto scrollbar-thin scrollbar-thumb-gray-200">
                           <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 sticky top-0 bg-white pb-2 border-b border-gray-50">
                               {activeCategory} Candidates
                           </h4>
                           <div className="grid grid-cols-2 gap-2">
                                {promptCategories[activeCategory].map((p, i) => (
                                    <button
                                        key={i}
                                        type="button"
                                        onClick={() => {
                                            setPrompt(p.prompt);
                                            setShowPresets(false);
                                        }}
                                        className="text-left p-2.5 rounded-lg border border-gray-100 hover:border-slate-300 hover:bg-slate-50 transition-all group"
                                    >
                                        <div className="text-xs font-semibold text-slate-700 group-hover:text-slate-900 mb-0.5">{p.label}</div>
                                        <div className="text-[10px] text-gray-400 leading-tight line-clamp-2 group-hover:text-slate-600">{p.prompt}</div>
                                    </button>
                                ))}
                           </div>
                        </div>
                    </div>
                )}
            </div>
          </div>
          
          <div className="relative group">
              <textarea
                id="prompt"
                rows={3}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe your ideal material properties (e.g. 'A stable semiconductor with band gap > 2.0 eV')..."
                className="w-full bg-slate-50 border-0 ring-1 ring-gray-200 rounded-lg p-4 text-base text-gray-900 placeholder:text-gray-400 focus:ring-2 focus:ring-slate-500/20 shadow-inner resize-none transition-all group-hover:ring-gray-300 focus:bg-white"
              />
              <div className="absolute bottom-3 right-3 text-[10px] text-gray-300 pointer-events-none font-mono">
                  AI-POWERED
              </div>
          </div>
        </div>

        <div className="h-px bg-gray-100"></div>

        {/* Sliders Section */}
        <div className="space-y-8">
             <WeightSliders weights={weights} onChange={setWeights} />
             
             <button
               type="submit"
               disabled={isLoading || !prompt.trim()}
               className="w-full py-4 bg-slate-900 hover:bg-black text-white text-sm font-bold rounded-lg shadow-lg hover:shadow-xl hover:-translate-y-0.5 active:translate-y-0 transition-all disabled:opacity-50 disabled:pointer-events-none relative overflow-hidden group"
             >
               <div className="relative flex items-center justify-center gap-2">
                   {isLoading ? (
                     <>
                         <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                         <span>Processing...</span>
                     </>
                   ) : (
                     <>
                         <span>GENERATE CANDIDATES</span>
                         <svg className="w-4 h-4 opacity-70 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                         </svg>
                     </>
                   )}
               </div>
             </button>
        </div>

        {error && (
          <div className="bg-red-50/50 backdrop-blur border border-red-100 text-red-600 text-xs p-3 rounded-lg flex items-center gap-2 animate-in fade-in slide-in-from-top-1">
            <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {error}
          </div>
        )}
      </form>
    </div>
  );
}
