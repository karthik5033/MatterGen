"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect } from "react";

interface ReportData {
  title: string;
  formula: string;
  abstract: string;
  properties: {
    label: string;
    value: string | number;
    unit?: string;
  }[];
  applications: string[];
}

interface ReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  data: ReportData | null;
}

export default function ReportModal({ isOpen, onClose, data }: ReportModalProps) {
  // Prevent background scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => { document.body.style.overflow = "unset"; };
  }, [isOpen]);

  if (!data) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 text-gray-900">
          
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/40 backdrop-blur-sm transition-colors"
          />

          {/* Modal Content */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 10 }}
            transition={{ duration: 0.2, ease: [0.0, 0.0, 0.2, 1] as const }}
            className="relative w-full max-w-2xl bg-white rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
          >
            
            {/* Header */}
            <div className="flex items-center justify-between px-8 py-6 border-b border-gray-100 bg-white sticky top-0 z-10">
              <div>
                 <h2 className="text-xl font-bold font-serif tracking-tight text-gray-900">
                   Automated Characterization Report
                 </h2>
                 <p className="text-xs text-gray-500 font-mono mt-1 uppercase tracking-wider">
                   Target: <span className="text-black font-bold">{data.formula}</span> // GenAI-v2.0
                 </p>
              </div>
              <button 
                onClick={onClose}
                className="p-2 -mr-2 text-gray-400 hover:text-gray-900 transition-colors rounded-full hover:bg-gray-50"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Scrollable Body */}
            <div className="overflow-y-auto px-8 py-8 space-y-8 scrollbar-thin scrollbar-thumb-gray-200">
              
              {/* Abstract */}
              <section>
                <h3 className="text-sm font-bold uppercase tracking-widest text-gray-400 mb-3 border-b border-gray-100 pb-2">
                   01. Abstract
                </h3>
                <p className="text-gray-700 leading-relaxed text-justify font-serif">
                  {data.abstract}
                </p>
              </section>

              {/* Key Properties Grid */}
              <section>
                <h3 className="text-sm font-bold uppercase tracking-widest text-gray-400 mb-4 border-b border-gray-100 pb-2">
                   02. Computed Properties
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  {data.properties.map((prop, idx) => (
                    <div key={idx} className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                       <div className="text-[10px] text-gray-500 uppercase font-bold">{prop.label}</div>
                       <div className="text-lg font-mono font-bold text-black mt-1">
                         {prop.value} <span className="text-xs text-gray-400 font-normal">{prop.unit}</span>
                       </div>
                    </div>
                  ))}
                </div>
              </section>

              {/* Applications */}
              <section>
                <h3 className="text-sm font-bold uppercase tracking-widest text-gray-400 mb-3 border-b border-gray-100 pb-2">
                   03. Potential Applications
                </h3>
                <ul className="list-disc list-inside space-y-1 text-gray-700">
                  {data.applications.map((app, idx) => (
                    <li key={idx} className="marker:text-emerald-500">{app}</li>
                  ))}
                </ul>
              </section>

            </div>

            {/* Footer */}
            <div className="bg-gray-50 px-8 py-4 border-t border-gray-100 flex items-center justify-between">
               <div className="text-[10px] text-gray-400 font-mono">
                 Generated via MatterGen X Neural Architecture
               </div>
               <button className="flex items-center gap-2 bg-black text-white text-xs font-bold px-4 py-2 rounded-lg hover:bg-emerald-600 transition-colors shadow-sm">
                 <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                 </svg>
                 Download PDF
               </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
