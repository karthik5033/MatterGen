import React, { useState, useRef, useEffect } from 'react';
import { WeightSliders } from './WeightSliders';

interface InputPanelProps {
  onGenerate: (prompt: string, weights: Record<string, number>) => Promise<void>;
  isLoading: boolean;
  error: string | null;
  prompt: string;
  setPrompt: (value: string) => void;
  weights: Record<string, number>;
  setWeights: React.Dispatch<React.SetStateAction<Record<string, number>>>;
}

type PromptItem = { label: string; prompt: string; weights?: Record<string, number> };
type SubCategory = { title: string; items: PromptItem[] };
type Category = { [subCat: string]: PromptItem[] };

/**
 * Component for the project specification input panel.
 * Encapsulates the prompt textarea, preset browser, and submission button.
 */
export default function InputPanel({ 
  onGenerate, 
  isLoading, 
  error,
  prompt,
  setPrompt,
  weights,
  setWeights
}: InputPanelProps) {

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onGenerate(prompt, weights);
  };

  /* 
    Expanded Categorized Prompt Library 
  */
  const promptCategories: Record<string, Category> = {
    "Energy & Sustainability": {
      "Batteries (Li/Na/Solid)": [
        { label: "Li-Ion Cathode", prompt: "High-capacity lithium cobalt oxide cathode material for EV batteries", weights: { stability: 0.8, density: 0.7 } },
        { label: "Solid Electrolyte", prompt: "High ionic conductivity garnet-type solid electrolyte (LLZO)", weights: { band_gap: 0.9, stability: 0.6 } },
        { label: "Na-Ion Anode", prompt: "Hard carbon anode optimized for sodium-ion storage", weights: { density: 0.4, stability: 0.7 } },
        { label: "Sulfur Cathode", prompt: "Microporous carbon host for lithium-sulfur battery cathode", weights: { density: 0.3 } },
        { label: "Supercapacitor", prompt: "High surface area graphene-based supercapacitor electrode", weights: { density: 0.2, stability: 0.9 } }
      ],
      "Solar & Photovoltaics": [
        { label: "Perovskite Solar", prompt: "Stable lead-free halide perovskite for high-efficiency solar cells", weights: { band_gap: 0.6, stability: 0.5 } },
        { label: "Tandem Cell Absorber", prompt: "Wide bandgap semiconductor (1.7 eV) for tandem solar cells", weights: { band_gap: 0.8 } },
        { label: "Organic PV", prompt: "Non-fullerene acceptor molecule for organic photovoltaics", weights: { flexibility: 0.9 } },
        { label: "Transparent PV", prompt: "Ultrawide bandgap oxide for transparent solar windows", weights: { band_gap: 0.95 } }
      ],
      "Green Tech": [
        { label: "Hydrogen Storage", prompt: "Metal-organic framework (MOF) with high volumetric hydrogen uptake", weights: { density: 0.1, porosity: 0.9 } },
        { label: "Fuel Cell Catalyst", prompt: "Platinum-free cathode catalyst for oxygen reduction reaction (ORR)", weights: { stability: 0.8 } },
        { label: "CO2 Capture", prompt: "Porous zeolite optimized for selective CO2 adsorption", weights: { density: 0.4 } },
        { label: "Thermoelectric", prompt: "High ZT thermoelectric material for waste heat recovery", weights: { thermal_conductivity: 0.1 } }
      ]
    },
    "Electronics & Computing": {
      "Semiconductors": [
        { label: "High-k Dielectric", prompt: "High dielectric constant oxide for next-gen MOSFET gates", weights: { band_gap: 0.9 } },
        { label: "2D Semiconductor", prompt: "Monolayer transition metal dichalcogenide (TMDC) with direct band gap", weights: { stability: 0.6 } },
        { label: "Power Electronics", prompt: "Ultra-wide bandgap Gallium Oxide polymorph for high voltage switching", weights: { band_gap: 0.85, thermal_conductivity: 0.7 } },
        { label: "Neuromorphic", prompt: "Memristive oxide for neuromorphic computing synapses", weights: { stability: 0.7 } }
      ],
      "Quantum Materials": [
        { label: "Topological Insulator", prompt: "Bi2Se3-based topological insulator with protected surface states", weights: { band_gap: 0.3 } },
        { label: "Superconductor", prompt: "High-Tc cuprate superconductor structure with layered perovskite motif", weights: { stability: 0.5 } },
        { label: "Spin Liquid", prompt: "Quantum spin liquid candidate on a Kagome lattice", weights: { magnetic_moment: 0.8 } },
        { label: "Qubit Host", prompt: "Isotopically pure silicon carbide for quantum vacancy centers", weights: { stability: 0.9 } },
        { label: "Majorana Fermion", prompt: "Nanowire hybrid structure hosting Majorana fermions", weights: { band_gap: 0.2 } }
      ],
      "Memory & Logic": [
        { label: "Ferroelectric RAM", prompt: "Hafnium oxide based ferroelectric for non-volatile memory", weights: { stability: 0.8 } },
        { label: "Spintronic", prompt: "Half-metallic Heusler alloy for spin-transfer torque RAM", weights: { magnetic_moment: 0.9, conductivity: 0.8 } },
        { label: "Phase Change", prompt: "Chalcogenide glass for fast phase-change memory switching", weights: { stability: 0.4 } }
      ]
    },
    "Advanced Materials": {
      "Carbon & Nanomaterials": [
        { label: "Superhard Carbon", prompt: "Superhard carbon polymorph harder than diamond", weights: { shear_modulus: 1.0, stability: 0.9 } },
        { label: "MXene", prompt: "2D titanium carbide MXene for EMI shielding", weights: { conductivity: 0.9 } },
        { label: "Nanotube", prompt: "Single-walled carbon nanotube with specific chirality", weights: { stability: 0.8 } },
        { label: "Aerogel", prompt: "Ultralight graphene aerogel for thermal insulation", weights: { density: 0.1, thermal_conductivity: 0.1 } }
      ],
       "Magnetic & Spintronic": [
        { label: "Rare-Earth Free Magnet", prompt: "High coercivity iron-nitride permanent magnet", weights: { magnetic_moment: 0.9 } },
        { label: "Magnetocaloric", prompt: "Giant magnetocaloric effect material for solid-state cooling", weights: { stability: 0.6 } },
        { label: "Skyrmion Host", prompt: "Chiral magnet capable of hosting magnetic skyrmions at room temp", weights: { magnetic_moment: 0.7 } }
      ],
      "Catalysis & Chemical": [
        { label: "Ammonia Catalyst", prompt: "Low-temperature Haber-Bosch catalyst for ammonia synthesis", weights: { stability: 0.7 } },
        { label: "Water Splitting", prompt: "Visible light photocatalyst for overall water splitting", weights: { band_gap: 0.5 } },
        { label: "CO2 Reduction", prompt: "Electro-catalyst for efficient CO2 reduction to ethylene", weights: { stability: 0.6 } }
      ]
    },
    "Structural & Aerospace": {
      "High-Performance Alloys": [
        { label: "Superalloy", prompt: "Single-crystal nickel-based superalloy for turbine blades", weights: { shear_modulus: 0.9, thermal_conductivity: 0.3 } },
        { label: "High-Entropy Alloy", prompt: "Single-phase FCC high-entropy alloy with high ductility", weights: { shear_modulus: 0.8 } },
        { label: "Lightweight Mg", prompt: "High-strength magnesium alloy with corrosion resistance", weights: { density: 0.3, shear_modulus: 0.6 } },
        { label: "Bulk Metallic Glass", prompt: "Zr-based bulk metallic glass with high elastic limit", weights: { shear_modulus: 0.7 } }
      ],
      "Extreme Environments": [
        { label: "Hypersonic Ceramic", prompt: "Ultra-high melting point carbide (UHTC) for hypersonic leading edges", weights: { stability: 1.0, thermal_conductivity: 0.8 } },
        { label: "Radiation Shield", prompt: "Boron-rich ceramic for neutron absorption in nuclear reactors", weights: { density: 0.6 } },
        { label: "Fusion Wall", prompt: "Tungsten alloy resistant to plasma erosion and blistering", weights: { stability: 0.9, thermal_conductivity: 0.9 } },
        { label: "Space Shielding", prompt: "Generative polymer composite for cosmic ray shielding", weights: { density: 0.2 } }
      ]
    },
    "Optical & Photonic": {
      "Light Manipulation": [
        { label: "Transparent Conductor", prompt: "High mobility indium-free transparent conducting oxide", weights: { band_gap: 0.8, conductivity: 0.9 } },
        { label: "Metamaterial", prompt: "Negative refractive index structure for optical cloaking", weights: { refractive_index: 0.1 } },
        { label: "Photonic Crystal", prompt: "3D photonic crystal with complete bandgap in visible range", weights: { refractive_index: 0.8 } },
        { label: "Variable Index", prompt: "Phase-change material for reconfigurable photonics", weights: { refractive_index: 0.6 } }
      ],
      "Emitters & Sensors": [
        { label: "Laser Host", prompt: "Nd-doped garnet host crystal with high thermal conductivity", weights: { thermal_conductivity: 0.8, band_gap: 0.9 } },
        { label: "Scintillator", prompt: "Fast decay lutetium silicate scintillator for medical imaging", weights: { density: 0.8 } },
        { label: "Quantum Emitter", prompt: "Single-photon emitter defect utilizing hexagonal boron nitride", weights: { band_gap: 0.9 } }
      ]
    },
    "Theoretical & Exotic": {
       "Fundamental Physics": [
          { label: "Time Crystal", prompt: "Discrete time crystal phase in a driven disorder system", weights: { stability: 0.2 } },
          { label: "Superionic Ice", prompt: "Superionic ice phase stabilized at high pressure", weights: { density: 0.8 } },
          { label: "Metallic Hydrogen", prompt: "Metastable metallic hydrogen phase at ambient pressure", weights: { density: 0.1, conductivity: 1.0 } },
          { label: "Axion Insulator", prompt: "Magnetic topological insulator axion state", weights: { magnetic_moment: 0.5 } }
       ],
       "Hypothetical": [
          { label: "Warp Field", prompt: "Negative energy density exotic matter for warp drive stabilization", weights: { density: 0.0 } },
          { label: "Room Temp Superconductor", prompt: "Hydride-based room temperature superconductor at low pressure", weights: { conductivity: 1.0 } }
       ]
    }
  };

  const [activeCategory, setActiveCategory] = useState<string>("Energy & Sustainability");
  const [showPresets, setShowPresets] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowPresets(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="relative z-10 text-slate-900">
      <div className="bg-white border border-gray-200 rounded-2xl shadow-xl relative">
        {/* Background Layer (Clipped) */}
        <div className="absolute inset-0 rounded-2xl overflow-hidden pointer-events-none z-0">
          <div 
            className="absolute inset-0 opacity-[0.03]" 
            style={{ 
              backgroundImage: 'linear-gradient(#000 1px, transparent 1px), linear-gradient(90deg, #000 1px, transparent 1px)', 
              backgroundSize: '24px 24px' 
            }}
          />
        </div>

        <form onSubmit={handleSubmit} className="relative z-10 p-8 space-y-8">
          {/* Prompt Section */}
          <div className="space-y-4">
            <div className="flex justify-between items-center relative z-20">
              <label htmlFor="prompt" className="text-xs font-bold tracking-widest text-slate-500 uppercase">
                Target Specification
              </label>

              <div className="flex items-center gap-3 relative z-30">


                <div className="relative" ref={dropdownRef}>
                  <button
                    type="button"
                    onClick={() => setShowPresets(!showPresets)}
                    className={`flex items-center gap-2 text-xs font-semibold px-4 py-2 rounded-lg border transition-all ${
                      showPresets 
                      ? 'bg-slate-800 text-white border-slate-800' 
                      : 'bg-white text-slate-600 border-gray-300 hover:border-slate-400 hover:text-slate-900'
                    }`}
                  >
                    <span>Browse Presets</span>
                    <svg className={`w-3 h-3 transition-transform ${showPresets ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {/* MEGA MENU DROPDOWN */}
                  {showPresets && (
                    <div className="absolute left-0 top-12 w-[650px] bg-white rounded-xl shadow-2xl border border-gray-200 ring-1 ring-black/5 overflow-hidden flex flex-col md:flex-row z-50 animate-in fade-in zoom-in-95 duration-200 origin-top-left">
                      {/* Categories Sidebar */}
                      <div className="w-56 bg-slate-50 border-r border-gray-200 p-2 space-y-1 shrink-0">
                        {Object.keys(promptCategories).map((cat) => (
                          <button
                            key={cat}
                            type="button"
                            onClick={() => setActiveCategory(cat)}
                            className={`w-full text-left px-3 py-2.5 rounded-lg text-xs font-medium transition-all flex justify-between items-center ${
                              activeCategory === cat 
                              ? 'bg-white text-slate-900 shadow-sm ring-1 ring-gray-200' 
                              : 'text-slate-500 hover:bg-gray-100 hover:text-slate-900'
                            }`}
                          >
                            {cat}
                            {activeCategory === cat && <div className="w-1.5 h-1.5 rounded-full bg-slate-800" />}
                          </button>
                        ))}
                      </div>

                      {/* Prompts Grid */}
                      <div className="flex-1 p-5 bg-white max-h-[400px] overflow-y-auto custom-scrollbar">
                        <style jsx>{`
                          .custom-scrollbar::-webkit-scrollbar {
                            width: 6px;
                          }
                          .custom-scrollbar::-webkit-scrollbar-track {
                            background: transparent;
                          }
                          .custom-scrollbar::-webkit-scrollbar-thumb {
                            background-color: #cbd5e1;
                            border-radius: 20px;
                            border: 2px solid transparent;
                            background-clip: content-box;
                          }
                          .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                            background-color: #94a3b8;
                          }
                        `}</style>
                        <div className="space-y-6">
                            {Object.entries(promptCategories[activeCategory]).map(([subCat, items]) => (
                                <div key={subCat}>
                                    <h4 className="text-[10px] font-bold text-indigo-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                                        <span className="w-1 h-1 rounded-full bg-indigo-500"></span>
                                        {subCat}
                                    </h4>
                                    <div className="grid grid-cols-2 gap-2">
                                        {items.map((p, i) => (
                                            <button
                                                key={i}
                                                type="button"
                                                onClick={() => {
                                                    setPrompt(p.prompt);
                                                    if (p.weights) setWeights(prev => ({...prev, ...p.weights}));
                                                    setShowPresets(false);
                                                }}
                                                className="text-left p-2.5 rounded-lg border border-gray-100 hover:border-indigo-200 hover:bg-indigo-50/50 transition-all group h-full"
                                            >
                                                <div className="text-xs font-semibold text-slate-700 group-hover:text-indigo-900 mb-0.5">{p.label}</div>
                                                <div className="text-[10px] text-gray-400 leading-tight line-clamp-2 group-hover:text-indigo-600/70">{p.prompt}</div>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
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

          <div className="h-px bg-gray-100" />

          {/* Sliders Section */}
          <div className="space-y-8">
            <WeightSliders weights={weights} onChange={(newWeights) => setWeights(newWeights as typeof weights)} />
            
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
    </div>
  );
}
