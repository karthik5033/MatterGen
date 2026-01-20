"use client";
import React from 'react'

import { Cpu, Fingerprint, Pencil, Settings2, Sparkles, Zap } from 'lucide-react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

import Link from 'next/link'

const features = [
    {
        icon: <Zap className="size-5" />,
        title: "Inverse Design",
        description: "Generate structures directly from validation metrics like stability and band gap.",
        href: "/features/inverse-design"
    },
    {
        icon: <Cpu className="size-5" />,
        title: "High Throughput",
        description: "Screen thousands of candidates in seconds with our optimized inference engine.",
        href: "/features/high-throughput"
    },
    {
        icon: <Fingerprint className="size-5" />,
        title: "Structure Identity",
        description: "Unique fingerprinting ensures novel candidates are distinct from training data.",
        href: "/features/structure-identity"
    },
    {
        icon: <Pencil className="size-5" />,
        title: "Property Tuning",
        description: "Fine-tune density, formation energy, and electronic properties with precision.",
        href: "/features/property-tuning"
    },
    {
        icon: <Settings2 className="size-5" />,
        title: "Full Control",
        description: "Adjust weights and constraints to steer the generative model towards your goals.",
        href: "/features/full-control"
    },
    {
        icon: <Sparkles className="size-5" />,
        title: "AI Native",
        description: "Built on state-of-the-art architectures (CDVAE/DiffCSP) for reliable generation.",
        href: "/features/ai-native"
    }
]

const SpotlightCard = ({ children, className = "" }: { children: React.ReactNode, className?: string }) => {
    const divRef = React.useRef<HTMLDivElement>(null);
    const [isFocused, setIsFocused] = React.useState(false);
    const [position, setPosition] = React.useState({ x: 0, y: 0 });
    const [opacity, setOpacity] = React.useState(0);

    const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
        if (!divRef.current || isFocused) return;

        const div = divRef.current;
        const rect = div.getBoundingClientRect();

        setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    };

    const handleFocus = () => {
        setIsFocused(true);
        setOpacity(1);
    };

    const handleBlur = () => {
        setIsFocused(false);
        setOpacity(0);
    };

    const handleMouseEnter = () => {
        setOpacity(1);
    };

    const handleMouseLeave = () => {
        setOpacity(0);
    };

    return (
        <motion.div
            ref={divRef}
            onMouseMove={handleMouseMove}
            onFocus={handleFocus}
            onBlur={handleBlur}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            className={cn(
                "group relative rounded-2xl border border-gray-100 bg-white/60 backdrop-blur-sm overflow-hidden",
                "hover:border-indigo-100 hover:shadow-xl hover:shadow-indigo-500/5 transition-all duration-300",
                className
            )}
            whileHover={{ y: -5 }}
        >
            <div
                className="pointer-events-none absolute -inset-px opacity-0 transition duration-300 group-hover:opacity-100"
                style={{
                    background: `radial-gradient(600px circle at ${position.x}px ${position.y}px, rgba(99, 102, 241, 0.06), transparent 40%)`,
                }}
            />
            <div className="relative h-full">{children}</div>
        </motion.div>
    );
};
export default function Features() {
    return (
        <section className="py-12 md:py-24 relative overflow-hidden">
             {/* Background Gradients */}
            <div className="absolute top-0 right-0 -mr-20 -mt-20 size-[500px] rounded-full bg-indigo-50/50 blur-[100px] -z-10" />
            <div className="absolute bottom-0 left-0 -ml-20 -mb-20 size-[500px] rounded-full bg-emerald-50/50 blur-[100px] -z-10" />

            <div className="mx-auto max-w-6xl px-6">
                <div className="relative z-10 mx-auto max-w-2xl space-y-6 text-center mb-12">
                    <motion.h2 
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.5 }}
                        className="text-balance text-4xl font-medium lg:text-5xl text-gray-900"
                    >
                        The foundation for <span className="text-indigo-600">materials innovation</span>
                    </motion.h2>
                    <motion.p 
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.5, delay: 0.1 }}
                        className="text-lg text-gray-500 text-pretty"
                    >
                        MatterGen X evolves beyond traditional screening. It enables an entire ecosystem—from inverse design to property validation—helping researchers and labs innovate.
                    </motion.p>
                </div>

                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {features.map((feature, i) => (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.5, delay: i * 0.1 }}
                        >
                            <Link href={feature.href} className="block h-full">
                                <SpotlightCard className="p-6 h-full flex flex-col items-start justify-between border-gray-200/60 bg-white/80 hover:bg-white/95"> 
                                    <div>
                                        <div className="mb-4 inline-flex items-center justify-center size-10 rounded-lg bg-stone-100 text-stone-700 group-hover:bg-stone-700 group-hover:text-white transition-all duration-300 group-hover:scale-110 group-hover:rotate-3 shadow-sm border border-stone-200">
                                            {feature.icon}
                                        </div>
                                        <h3 className="mb-2 text-lg font-semibold text-gray-900 group-hover:text-stone-800 transition-colors">
                                            {feature.title}
                                        </h3>
                                        <p className="text-sm text-gray-500 leading-relaxed group-hover:text-gray-600">
                                            {feature.description}
                                        </p>
                                    </div>
                                    
                                    <div className="mt-6 flex items-center text-sm font-medium text-stone-600 opacity-60 group-hover:opacity-100 transition-opacity">
                                        Learn more
                                        <svg className="ml-1 w-4 h-4 transition-transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                        </svg>
                                    </div>
                                </SpotlightCard>
                            </Link>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    )
}
