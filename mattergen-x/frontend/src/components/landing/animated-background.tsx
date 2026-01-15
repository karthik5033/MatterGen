'use client'

import { motion, useScroll, useTransform } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'

const BackgroundParticles = () => {
    const [particles, setParticles] = useState<any[]>([]);

    useEffect(() => {
        setParticles(Array.from({ length: 20 }).map((_, i) => ({
            id: i,
            left: Math.random() * 100 + '%',
            top: Math.random() * 100 + '%',
            size: Math.random() * 4 + 1 + 'px',
            duration: Math.random() * 10 + 10,
            delay: Math.random() * 20,
            endY: Math.random() * -100 - 50
        })));
    }, []);

    return (
        <>
            {particles.map((p) => (
                <motion.div
                    key={p.id}
                    className="absolute rounded-full bg-primary/20"
                    style={{
                        left: p.left,
                        top: p.top,
                        width: p.size,
                        height: p.size,
                    }}
                    animate={{
                        y: [0, p.endY],
                        opacity: [0, 0.5, 0],
                        scale: [0, 1.5, 0],
                    }}
                    transition={{
                        duration: p.duration,
                        repeat: Infinity,
                        ease: "linear",
                        delay: p.delay,
                    }}
                />
            ))}
        </>
    );
};

export const AnimatedBackground = () => {
    // Mouse interaction for subtle parallax
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
    
    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            setMousePosition({
                x: e.clientX / window.innerWidth,
                y: e.clientY / window.innerHeight,
            })
        }
        window.addEventListener('mousemove', handleMouseMove)
        return () => window.removeEventListener('mousemove', handleMouseMove)
    }, [])

    return (
        <div className="absolute inset-0 -z-50 overflow-hidden bg-white">
            {/* 1. Base Grid Pattern */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]" />
            
            {/* 2. Secondary Larger Grid for Depth */}
             <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808008_1px,transparent_1px),linear-gradient(to_bottom,#80808008_1px,transparent_1px)] bg-[size:96px_96px]" />

            {/* 3. Moving Gradient Orbs */}
            <motion.div 
                className="absolute -left-[10%] -top-[10%] h-[500px] w-[500px] rounded-full bg-purple-500/10 blur-[100px] filter"
                animate={{
                    x: mousePosition.x * 50,
                    y: mousePosition.y * 50,
                    scale: [1, 1.1, 1],
                }}
                transition={{ duration: 5, repeat: Infinity, repeatType: "reverse" }}
            />
            
            <motion.div 
                className="absolute right-[0%] top-[20%] h-[400px] w-[400px] rounded-full bg-blue-500/10 blur-[120px] filter"
                animate={{
                    x: mousePosition.x * -30,
                    y: mousePosition.y * -30,
                    opacity: [0.5, 0.8, 0.5]
                }}
                transition={{ duration: 7, repeat: Infinity, repeatType: "reverse" }}
            />

             <motion.div 
                className="absolute bottom-[0%] left-[20%] h-[600px] w-[600px] rounded-full bg-emerald-500/05 blur-[100px] filter"
                animate={{
                    x: mousePosition.x * 20,
                    y: mousePosition.y * 20,
                }}
                transition={{ type: "spring", damping: 50, stiffness: 50 }}
            />

            {/* 4. Subtle Noise Texture Overlay for "Pro" Texture */}
            <div className="absolute inset-0 opacity-[0.03] mix-blend-overlay" style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")` }} />
            
            {/* 5. Floating Background Particles - "More Animation" */}
             <div className="absolute inset-0 overflow-hidden">
                <BackgroundParticles />
            </div>

            {/* 6. Vignette for Focus - Light Only */}
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,transparent_0%,rgba(0,0,0,0.02)_100%)]" />
        </div>
    )
}
