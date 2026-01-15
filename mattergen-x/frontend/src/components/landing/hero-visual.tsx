'use client'

import { motion } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'

export const HeroVisual = () => {
    return (
        <div className="relative flex h-[500px] w-full items-center justify-center lg:h-[700px] perspective-[1000px]">
            <div className="relative size-[400px] lg:size-[600px] flex items-center justify-center">
                {/* Ambient Glow */}
                <div className="absolute inset-0 rounded-full bg-blue-500/10 blur-[120px] filter" />
                
                {/* Unified 3D Atom Simulation */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
                    <AtomSimulation />
                </div>
            </div>
        </div>
    )
}

const AtomSimulation = () => {
    const [rotation, setRotation] = useState({ x: 0, y: 0 })
    const requestRef = useRef<number>(0)
    const startTimeRef = useRef<number>(Date.now())

    // --- CONFIGURATION ---
    const nucleonData = [
        { x: 0, y: 0, z: 0, color: 'white', size: 50 }, 
        { x: 22, y: 22, z: 22, color: 'blue', size: 38 },
        { x: -22, y: -22, z: -22, color: 'blue', size: 38 },
        { x: 22, y: -22, z: 22, color: 'indigo', size: 38 },
        { x: -22, y: 22, z: -22, color: 'indigo', size: 38 },
        { x: 22, y: 22, z: -22, color: 'indigo', size: 38 },
        { x: -22, y: -22, z: 22, color: 'indigo', size: 38 },
        { x: 32, y: 0, z: 0, color: 'blue', size: 34 },
        { x: -32, y: 0, z: 0, color: 'indigo', size: 34 },
        { x: 0, y: 32, z: 0, color: 'blue', size: 34 },
        { x: 0, y: -32, z: 0, color: 'indigo', size: 34 },
        { x: 0, y: 0, z: 32, color: 'blue', size: 34 },
        { x: 0, y: 0, z: -32, color: 'indigo', size: 34 },
    ]

    const orbitData = [
        { radius: 180, tiltX: 75, tiltY: 15, speed: 1.2, color: 'cyan', offset: 0 },
        { radius: 180, tiltX: 120, tiltY: -15, speed: 0.9, color: 'sky', offset: 2 },
        { radius: 180, tiltX: 45, tiltY: 85, speed: 1.1, color: 'blue', offset: 4 },
        { radius: 160, tiltX: 90, tiltY: 45, speed: 0.8, color: 'indigo', offset: 1 },
    ];

    // --- CALCULATIONS ---
    const getElectronPos = (orbit: typeof orbitData[0], time: number) => {
        const theta = (time * orbit.speed) + orbit.offset;
        return calculateOrbitPoint(orbit, theta);
    }

    const calculateOrbitPoint = (orbit: typeof orbitData[0], theta: number) => {
        // Flat circle
        const x = orbit.radius * Math.cos(theta);
        const y = orbit.radius * Math.sin(theta);
        const z = 0;

        // Apply Orbital Tilt (Local Rotation)
        const radX = (orbit.tiltX * Math.PI) / 180;
        const radY = (orbit.tiltY * Math.PI) / 180;

        // Rotate X
        const y1 = y * Math.cos(radX) - z * Math.sin(radX);
        const z1 = y * Math.sin(radX) + z * Math.cos(radX);
        
        // Rotate Y
        const x2 = x * Math.cos(radY) - z1 * Math.sin(radY);
        const z2 = x * Math.sin(radY) + z1 * Math.cos(radY);

        return { x: x2, y: y1, z: z2 };
    }

    // --- ANIMATION LOOP ---
    const animate = () => {
        const now = Date.now();
        const elapsed = (now - startTimeRef.current) * 0.001; // Seconds

        setRotation({
            x: (elapsed * 0.2) % (Math.PI * 2),
            y: (elapsed * 0.4) % (Math.PI * 2),
        });

        requestRef.current = requestAnimationFrame(animate);
    }

    const [mounted, setMounted] = useState(false)

    useEffect(() => {
        setMounted(true)
        requestRef.current = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(requestRef.current);
    }, [])

    // --- RENDERING HELPERS ---
    const rotatePoint = (point: {x: number, y: number, z: number}, rotX: number, rotY: number) => {
        // Apply Global Rotation
        // Rotate Y
        let x = point.x * Math.cos(rotY) - point.z * Math.sin(rotY);
        let z = point.x * Math.sin(rotY) + point.z * Math.cos(rotY);
        let y = point.y;

        // Rotate X
        let y_new = y * Math.cos(rotX) - z * Math.sin(rotX);
        let z_new = y * Math.sin(rotX) + z * Math.cos(rotX);
        
        return { x, y: y_new, z: z_new };
    }

    if (!mounted) return null;

    const elapsed = (Date.now() - startTimeRef.current) * 0.001;

    // 1. PROJECT NUCLEUS
    const projectedNucleons = nucleonData.map((n, i) => {
        const rotated = rotatePoint(n, rotation.x, rotation.y);
        return { ...n, ...rotated, originalIndex: i, type: 'nucleon', opacity: 1 };
    });

    // 2. PROJECT ELECTRONS & TRAILS
    const electrons: any[] = [];
    const trails: any[] = [];
    
    for(let i=0; i<orbitData.length; i++) {
        const orbit = orbitData[i];
        
        // Current Position
        const rawPos = getElectronPos(orbit, elapsed);
        const projectedPos = rotatePoint(rawPos, rotation.x, rotation.y); 
        
        // Add Electron
        electrons.push({
            ...projectedPos,
            color: orbit.color,
            size: 10,
            type: 'electron',
            opacity: 1,
            id: `e-${i}`
        });

        // Add Trails (History)
        for(let j=1; j<=8; j++) {
            const timeLag = elapsed - (j * 0.04);
            const rawTrailPos = getElectronPos(orbit, timeLag);
            const projectedTrail = rotatePoint(rawTrailPos, rotation.x, rotation.y);
            trails.push({
                ...projectedTrail,
                color: orbit.color,
                size: 10 - j, 
                opacity: 0.6 - (j * 0.07),
                type: 'trail',
                id: `t-${i}-${j}`
            });
        }
    }

    // 3. PROJECT ORBIT PATHS (RINGS)
    const projectedOrbits = orbitData.map(orbit => {
        const points = [];
        // Generate points along the orbit to create a smooth path
        const steps = 120; // Higher resolution for smoothness
        for (let i = 0; i <= steps; i++) {
            const theta = (i / steps) * Math.PI * 2;
            const rawPos = calculateOrbitPoint(orbit, theta);
            const rotated = rotatePoint(rawPos, rotation.x, rotation.y);
            points.push(rotated);
        }
        
        // Convert points to SVG path 'd' string
        const pathData = points.reduce((acc, p, i) => {
            return acc + (i === 0 ? `M ${p.x} ${p.y}` : ` L ${p.x} ${p.y}`);
        }, "");

        return { d: pathData, color: orbit.color };
    });

    // 4. SORT ALL PARTICLES BY Z
    const allParticles = [...projectedNucleons, ...electrons, ...trails].sort((a, b) => a.z - b.z);

    return (
        <div className="relative size-[400px] lg:size-[600px] flex items-center justify-center">
            {/* Core Glow */}
             <div className="absolute inset-0 bg-blue-600/20 blur-[60px] rounded-full animate-pulse z-0" />
            
             {/* 3D Projected Orbit Lines - Restored to be visible and solid */}
             <svg className="absolute inset-0 size-full overflow-visible" viewBox="-300 -300 600 600" style={{ pointerEvents: 'none', zIndex: 0 }}>
                {projectedOrbits.map((orbit, i) => (
                    <g key={i}>
                        {/* Outer Glow */}
                        <path 
                            d={orbit.d}
                            fill="none"
                            stroke={orbit.color === 'cyan' ? '#22d3ee' : orbit.color === 'sky' ? '#38bdf8' : orbit.color === 'blue' ? '#60a5fa' : '#818cf8'}
                            strokeWidth="3"
                            strokeOpacity="0.1"
                        />
                        {/* Main Line */}
                        <path 
                            d={orbit.d}
                            fill="none"
                            stroke={orbit.color === 'cyan' ? '#22d3ee' : orbit.color === 'sky' ? '#38bdf8' : orbit.color === 'blue' ? '#60a5fa' : '#818cf8'}
                            strokeWidth="1"
                            strokeOpacity="0.5"
                        />
                    </g>
                ))}
             </svg>

             {/* Render Particles */}
            {allParticles.map((p, i) => {
                let bgStyle = "";
                let shadow = "";
                
                if (p.type === 'nucleon') {
                    if (p.color === 'white') bgStyle = "bg-[radial-gradient(circle_at_30%_30%,white,theme(colors.blue.300),theme(colors.blue.900))]"
                    else if (p.color === 'blue') bgStyle = "bg-[radial-gradient(circle_at_30%_30%,#dbeafe,theme(colors.blue.400),theme(colors.blue.900))]"
                    else bgStyle = "bg-[radial-gradient(circle_at_30%_30%,#e0e7ff,theme(colors.indigo.400),theme(colors.indigo.900))]"
                    shadow = "shadow-lg"
                } else if (p.type === 'electron') {
                    // Bright Electron Gradient
                    if(p.color === 'cyan') bgStyle = "bg-[radial-gradient(circle_at_30%_30%,#cffafe,theme(colors.cyan.400),theme(colors.cyan.600))]"
                    if(p.color === 'sky') bgStyle = "bg-[radial-gradient(circle_at_30%_30%,#e0f2fe,theme(colors.sky.400),theme(colors.sky.600))]"
                    if(p.color === 'blue') bgStyle = "bg-[radial-gradient(circle_at_30%_30%,#dbeafe,theme(colors.blue.400),theme(colors.blue.600))]"
                    if(p.color === 'indigo') bgStyle = "bg-[radial-gradient(circle_at_30%_30%,#e0e7ff,theme(colors.indigo.400),theme(colors.indigo.600))]"
                    shadow = "shadow-[0_0_15px_currentColor]"
                } else {
                    // Trail - Simpler color
                    if(p.color === 'cyan') bgStyle = "bg-cyan-400"
                    if(p.color === 'sky') bgStyle = "bg-sky-400"
                    if(p.color === 'blue') bgStyle = "bg-blue-400"
                    if(p.color === 'indigo') bgStyle = "bg-indigo-400"
                }

                return (
                    <div
                        key={i}
                        className={`absolute rounded-full ${bgStyle} ${shadow}`}
                        style={{
                            width: p.size,
                            height: p.size,
                            transform: `translate3d(${p.x}px, ${p.y}px, 0px) scale(${0.8 + (p.z / 300)})`, // Perspective scale
                            zIndex: Math.floor(p.z + 200),
                            opacity: p.opacity,
                        }} 
                    />
                )
            })}
        </div>
    )
}

const Ring = ({ size, rotateX, rotateY, duration, delay, particleColor }: { size: number, rotateX: number, rotateY: number, duration: number, delay: number, particleColor: string }) => {
    return (
        <motion.div 
            className="absolute left-1/2 top-1/2 rounded-full border-[1px] border-primary/15 preserve-3d"
            style={{ 
                width: size,
                height: size,
                marginTop: -size/2,
                marginLeft: -size/2,
            }}
            initial={{ rotateX, rotateY, rotateZ: 0 }}
            animate={{ 
                rotateZ: 360 
            }}
            transition={{ 
                duration: duration, 
                repeat: Infinity, 
                ease: "linear",
                delay: delay
            }}
        >
             {/* Main Electron */}
            <div className={`absolute -top-1.5 left-1/2 size-3 -ml-1.5 rounded-full ${particleColor} shadow-[0_0_20px_currentColor] z-10`} />
            
            {/* Trail Particles for Speed Effect */}
            <div className={`absolute -top-1 left-[48%] size-2 rounded-full ${particleColor} opacity-60 blur-[1px]`} />
            <div className={`absolute -top-0.5 left-[46%] size-1.5 rounded-full ${particleColor} opacity-40 blur-[2px]`} />
             <div className={`absolute top-0 left-[44%] size-1 rounded-full ${particleColor} opacity-20 blur-[3px]`} />
        </motion.div>
    )
}


