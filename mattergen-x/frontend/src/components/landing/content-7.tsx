import { Cpu, Zap } from 'lucide-react'
import Image from 'next/image'

export default function ContentSection() {
    return (
        <section className="py-16 md:py-32">
            <div className="mx-auto max-w-5xl space-y-8 px-6 md:space-y-16">
                <h2 className="relative z-10 max-w-xl text-4xl font-medium lg:text-5xl">The MatterGen ecosystem powers your research.</h2>
                <div className="grid gap-6 sm:grid-cols-2 md:gap-12 lg:gap-24 items-center">
                    <div className="relative space-y-4">
                        <p className="text-muted-foreground">
                            MatterGen X is more than just a model. <span className="text-accent-foreground font-bold">It's a complete discovery platform</span> — integrating generation, simulation, and analysis.
                        </p>
                        <p className="text-muted-foreground">From initial candidate generation to final DFT validation, our unified pipeline accelerates every stage of the materials design process.</p>

                        <div className="grid grid-cols-2 gap-3 pt-6 sm:gap-4">
                            <div className="space-y-3">
                                <div className="flex items-center gap-2">
                                    <Zap className="size-4" />
                                    <h3 className="text-sm font-medium">Fast</h3>
                                </div>
                                <p className="text-muted-foreground text-sm">Generate thousands of stable candidates in minutes, not months.</p>
                            </div>
                            <div className="space-y-2">
                                <div className="flex items-center gap-2">
                                    <Cpu className="size-4" />
                                    <h3 className="text-sm font-medium">Accurate</h3>
                                </div>
                                <p className="text-muted-foreground text-sm">Pre-trained on massive datasets for high-fidelity property prediction.</p>
                            </div>
                        </div>
                    </div>
                    <div className="relative mt-6 sm:mt-0">
                        <div className="bg-linear-to-b aspect-[4/3] relative rounded-2xl from-zinc-300 to-transparent p-px dark:from-zinc-700">
                            <Image src="/ecosystem-visual.png" className="size-full rounded-[15px] object-cover shadow-2xl border border-white/10" alt="MatterGen Ecosystem Visualization" width={1024} height={1024} />
                        </div>
                    </div>
                </div>
            </div>
        </section>
    )
}
