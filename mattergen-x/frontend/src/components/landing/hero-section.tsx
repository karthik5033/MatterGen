import React from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import Image from 'next/image'
import { HeroHeader } from './header'
import { HeroVisual } from './hero-visual'
import { AnimatedBackground } from './animated-background'

export default function HeroSection() {
    return (
        <div className="relative">
            <AnimatedBackground />
            <HeroHeader />
            <main className="overflow-x-hidden">
                <section>
                    <div className="pb-24 pt-8 md:pb-32 lg:pb-40 lg:pt-20">
                        <div className="relative mx-auto flex max-w-6xl flex-col px-6 lg:block">
                            <div className="mx-auto max-w-lg text-center lg:ml-0 lg:w-1/2 lg:text-left z-10 relative">
                                <h1 className="mt-8 max-w-2xl text-balance text-5xl font-medium tracking-tight md:text-6xl lg:mt-16 xl:text-7xl">
                                    Accelerate Discovery with <span className="text-primary">Generative AI</span>
                                </h1>
                                <p className="mt-8 max-w-2xl text-pretty text-lg text-muted-foreground/90">
                                    Uncover novel stable crystals with target properties 10x faster. MatterGen X combines inverse design with high-fidelity property prediction.
                                </p>

                                <div className="mt-12 flex flex-col items-center justify-center gap-4 sm:flex-row lg:justify-start">
                                    <Button
                                        asChild
                                        size="lg"
                                        className="h-12 px-8 text-base shadow-md transition-all hover:scale-[1.02] bg-zinc-900 text-white hover:bg-zinc-800 border border-zinc-900">
                                        <Link href="/generate">
                                            <span className="text-nowrap font-medium">Start Generating</span>
                                            <svg className="ml-2 size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                                            </svg>
                                        </Link>
                                    </Button>
                                    <Button
                                        key={2}
                                        asChild
                                        size="lg"
                                        variant="outline"
                                        className="h-12 px-8 text-base bg-white text-zinc-700 border-zinc-200 hover:bg-zinc-50 hover:text-zinc-900 shadow-sm transition-all hover:scale-[1.02]">
                                        <Link href="/compare">
                                            <span className="text-nowrap font-medium">View Benchmarks</span>
                                        </Link>
                                    </Button>
                                </div>
                            </div>
                            <div className="-z-10 order-first ml-auto w-full lg:absolute lg:inset-y-0 lg:right-0 lg:h-full lg:w-[60%] lg:order-last flex items-center justify-center pointer-events-none">
                                <HeroVisual />
                            </div>
                        </div>
                    </div>
                </section>

            </main>
        </div>
    )
}
