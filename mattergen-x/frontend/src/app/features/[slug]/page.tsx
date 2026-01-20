"use client";

import React from 'react';
import Link from 'next/link';
import { notFound, useParams } from 'next/navigation';
import { motion } from 'framer-motion';
import { ArrowLeft, Zap, Cpu, Fingerprint, Pencil, Settings2, Sparkles, BookOpen, Share2 } from 'lucide-react';

// Data for each feature page
const featureData: Record<string, {
    title: string;
    subtitle: string;
    icon: React.ElementType;
    content: React.ReactNode;
    color: string;
}> = {
    "inverse-design": {
        title: "Inverse Design",
        subtitle: "From Desired Properties to realized Structures",
        icon: Zap,
        color: "text-amber-500",
        content: (
            <div className="space-y-6 text-slate-600 leading-relaxed">
                <p>
                    Traditional materials discovery relies on screening known databases or trial-and-error experimentation. 
                    <strong>Inverse Design</strong> flips this paradigm: you define the properties you want—such as a specific 
                    band gap, stability threshold, or bulk modulus—and our AI generates the crystal structures that meet them.
                </p>
                <h3 className="text-xl font-semibold text-slate-900 mt-8 mb-4">How It Works</h3>
                <ul className="list-disc pl-6 space-y-2">
                    <li><strong>Conditioning:</strong> Our DiffCSP model takes scalar property targets as input vectors.</li>
                    <li><strong>Diffusion Process:</strong> It iteratively denoises a random distribution of atoms into a stable crystal lattice.</li>
                    <li><strong>Validation:</strong> Generated structures are instantly validated against physical laws using ML potentials.</li>
                </ul>
                <div className="bg-amber-50 border border-amber-100 p-6 rounded-xl mt-8">
                    <h4 className="font-semibold text-amber-900 mb-2">Key Benefit</h4>
                    <p className="text-amber-800 text-sm">
                        Reduces discovery time from months to minutes by bypassing the need to search through millions of candidates.
                    </p>
                </div>
            </div>
        )
    },
    "high-throughput": {
        title: "High Throughput Screening",
        subtitle: "Massive Parallel Inference for Rapid Discovery",
        icon: Cpu,
        color: "text-blue-500",
        content: (
            <div className="space-y-6 text-slate-600 leading-relaxed">
                <p>
                    MatterGen X isn't just fast; it's scalable. Our inference engine is optimized to generate and evaluate 
                    thousands of candidate materials in parallel, utilizing advanced GPU acceleration and batch processing.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-8">
                    <div className="bg-white p-5 rounded-lg border shadow-sm">
                        <div className="text-3xl font-bold text-slate-900 mb-1">10k+</div>
                        <div className="text-sm text-slate-500">Candidates per Hour</div>
                    </div>
                    <div className="bg-white p-5 rounded-lg border shadow-sm">
                        <div className="text-3xl font-bold text-slate-900 mb-1">&lt;100ms</div>
                        <div className="text-sm text-slate-500">Per-Structure Inference</div>
                    </div>
                </div>
                <p>
                    Whether you are searching for a new battery cathode or a superhard alloy, High Throughput Screening ensures 
                    you explore the entire chemical space effectively without computational bottlenecks.
                </p>
            </div>
        )
    },
    "structure-identity": {
        title: "Structure Identity & Fingerprinting",
        subtitle: "Ensuring Novelty and Uniqueness",
        icon: Fingerprint,
        color: "text-rose-500",
        content: (
            <div className="space-y-6 text-slate-600 leading-relaxed">
                <p>
                    In generative AI, avoiding mode collapse (generating the same thing over and over) is critical. 
                    Our <strong>Structure Identity</strong> module uses crystallographic fingerprinting to ensure diversity.
                </p>
                <p>
                   We compute unique hashes based on:
                </p>
                <ul className="list-disc pl-6 space-y-2">
                    <li><strong>Space Group Symmetry:</strong> The mathematical backbone of the crystal.</li>
                    <li><strong>Coordination Environment:</strong> How atoms connect to their neighbors.</li>
                    <li><strong>Stoichiometry:</strong> The precise elemental ratios.</li>
                </ul>
                <p className="mt-4">
                    This guarantees that every candidate presented to you is structurally distinct, maximizing your chances of finding a breakthrough material.
                </p>
            </div>
        )
    },
    "property-tuning": {
        title: "Precision Property Tuning",
        subtitle: "Fine-grained Control over Material Behavior",
        icon: Pencil,
        color: "text-emerald-500",
        content: (
            <div className="space-y-6 text-slate-600 leading-relaxed">
                <p>
                    Generic generative models often fail at specific constraints. MatterGen X allows for 
                    <strong>Precision Tuning</strong>. Does your application require a material closer to an insulator 
                    but with high thermal conductivity? You can specify these conflicting constraints.
                </p>
                <h3 className="text-xl font-semibold text-slate-900 mt-8 mb-4">Supported Properties</h3>
                 <div className="flex flex-wrap gap-2">
                    {["Band Gap", "Formation Energy", "Bulk Modulus", "Shear Modulus", "density", "Fermi Energy"].map((tag) => (
                        <span key={tag} className="px-3 py-1 bg-slate-100 text-slate-700 rounded-full text-sm font-medium">
                            {tag}
                        </span>
                    ))}
                </div>
            </div>
        )
    },
    "full-control": {
        title: "Full Control & Weights",
        subtitle: "Steer the Generation Process",
        icon: Settings2,
        color: "text-indigo-500",
        content: (
            <div className="space-y-6 text-slate-600 leading-relaxed">
                <p>
                    The "Black Box" era of AI is over. MatterGen X gives you knobs and dials to control the generation process. 
                    Adjust the <strong>Weights</strong> of different objectives to prioritize what matters most to your research.
                </p>
                <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 mt-4">
                    <h4 className="font-semibold text-slate-900 mb-4">Example Workflow</h4>
                    <div className="flex items-center gap-4 text-sm">
                        <div className="flex-1 bg-white p-3 rounded shadow-sm text-center">
                            Stability (50%)
                        </div>
                        <div className="text-slate-400">+</div>
                        <div className="flex-1 bg-white p-3 rounded shadow-sm text-center">
                            Hardness (30%)
                        </div>
                        <div className="text-slate-400">+</div>
                        <div className="flex-1 bg-white p-3 rounded shadow-sm text-center">
                            Novelty (20%)
                        </div>
                    </div>
                </div>
            </div>
        )
    },
    "ai-native": {
        title: "AI Native Architecture",
        subtitle: "Built on CDVAE and DiffCSP",
        icon: Sparkles,
        color: "text-violet-500",
        content: (
            <div className="space-y-6 text-slate-600 leading-relaxed">
                <p>
                    MatterGen X utilizes state-of-the-art <strong>Crystal Diffusion Variational Autoencoders (CDVAE)</strong> and 
                    <strong>DiffCSP</strong> architectures. These models treat the crystal structure generation as a denoising process in continuous space.
                </p>
                <p>
                    Unlike voxel-based or graph-based approaches which are often limited by resolution or fixed topology, 
                    our diffusion approach allows for infinite lattice resolution and arbitrary atom placement, capturing true physical validities.
                </p>
                <div className="mt-8 p-4 bg-violet-50 text-violet-800 rounded-lg text-sm border border-violet-100">
                    "The convergence of Physics and AI is happening here."
                </div>
            </div>
        )
    }
};

export default function FeaturePage() {
    const params = useParams();
    const slug = params?.slug as string;
    const data = featureData[slug];

    if (!data) {
        return (
            <div className="min-h-screen flex flex-col items-center justify-center p-6 bg-slate-50">
               <h1 className="text-2xl font-bold text-slate-900">Feature Not Found</h1>
               <Link href="/" className="mt-4 text-indigo-600 hover:underline">Return Home</Link>
            </div>
        );
    }

    const Icon = data.icon;

    return (
        <div className="min-h-screen bg-white">
            {/* Header / Nav */}
            <div className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100">
                <div className="max-w-4xl mx-auto px-6 h-16 flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-indigo-600 transition-colors">
                        <ArrowLeft className="size-4" />
                        Back to Home
                    </Link>
                    <div className="text-sm font-semibold text-slate-900">
                        MatterGen X Documentation
                    </div>
                </div>
            </div>

            {/* Hero Section */}
            <header className="relative bg-slate-50 border-b border-slate-200 overflow-hidden">
                <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-100 contrast-150 grayscale"></div>
                <div className="max-w-4xl mx-auto px-6 py-20 relative z-10">
                    <div className="inline-flex items-center justify-center p-3 bg-white rounded-xl shadow-sm border border-slate-200 mb-6">
                        <Icon className={`size-8 ${data.color}`} />
                    </div>
                    <motion.h1 
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight mb-4"
                    >
                        {data.title}
                    </motion.h1>
                    <motion.p 
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="text-xl text-slate-500 font-light"
                    >
                        {data.subtitle}
                    </motion.p>
                </div>
            </header>

            {/* Content Body */}
            <main className="max-w-4xl mx-auto px-6 py-12">
                <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-12">
                    {/* Main Text */}
                    <article className="prose prose-slate prose-lg max-w-none">
                        {data.content}
                    </article>

                    {/* Sidebar */}
                    <aside className="space-y-6">
                        <div className="bg-slate-50 rounded-xl p-6 border border-slate-100 sticky top-24">
                            <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
                                <BookOpen className="size-4" />
                                Related Topics
                            </h3>
                            <ul className="space-y-3 text-sm">
                                {Object.entries(featureData).filter(([k]) => k !== slug).map(([key, item]) => (
                                    <li key={key}>
                                        <Link href={`/features/${key}`} className="block text-slate-600 hover:text-indigo-600 transition-colors p-2 hover:bg-white rounded-lg">
                                            {item.title}
                                        </Link>
                                    </li>
                                )).slice(0, 4)}
                            </ul>
                            
                            <div className="mt-8 pt-6 border-t border-slate-200">
                                <button className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg bg-indigo-600 text-white font-medium hover:bg-indigo-700 transition-colors shadow-sm">
                                    <Share2 className="size-4" />
                                    Share Concept
                                </button>
                            </div>
                        </div>
                    </aside>
                </div>
            </main>
        </div>
    );
}
