'use client'
import Link from 'next/link'
import { Logo } from './logo'
import { Menu, X, ArrowRight, ChevronDown, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import React, { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'

const menuItems = [
    { name: 'Research', href: '/research', hasDropdown: true },
    { name: 'Impact', href: '/impact' },
    { name: 'Discoveries', href: '/discoveries' },
    { name: 'About', href: '/about' },
]

export const HeroHeader = () => {
    const [menuState, setMenuState] = React.useState(false)
    const [scrolled, setScrolled] = useState(false)

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 20)
        }
        window.addEventListener('scroll', handleScroll)
        return () => window.removeEventListener('scroll', handleScroll)
    }, [])

    return (
        <header className="fixed top-6 z-50 w-full flex justify-center px-4">
            <nav
                className={cn(
                    "w-full max-w-5xl transition-all duration-500 ease-out border rounded-full px-2 pl-6",
                    scrolled 
                        ? "bg-white/80 dark:bg-zinc-950/80 backdrop-blur-xl border-zinc-200/50 dark:border-zinc-800/50 shadow-lg py-2" 
                        : "bg-white/40 dark:bg-zinc-950/40 backdrop-blur-md border-white/20 shadow-sm py-2"
                )}
            >
                <div className="flex items-center justify-between">
                    {/* Logo Section */}
                    <div className="flex items-center">
                        <Link
                            href="/"
                            aria-label="home"
                            className="flex items-center gap-2 group">
                            <Logo className="text-lg transition-transform group-hover:scale-105" />
                        </Link>
                    </div>

                    {/* Desktop Navigation - Centered */}
                    <div className="hidden md:absolute md:left-1/2 md:-translate-x-1/2 md:flex items-center gap-1">
                        {menuItems.map((item, index) => (
                            <Link
                                key={index}
                                href={item.href}
                                className="px-4 py-2 text-sm font-medium text-zinc-900 hover:text-zinc-700 dark:text-zinc-100 dark:hover:text-zinc-300 rounded-full hover:bg-zinc-100/50 dark:hover:bg-zinc-800/50 transition-all flex items-center gap-1 group"
                            >
                                {item.name}
                                {item.hasDropdown && (
                                    <ChevronDown className="w-3 h-3 opacity-50 group-hover:translate-y-0.5 transition-transform" />
                                )}
                            </Link>
                        ))}
                    </div>

                    {/* Right Actions */}
                    <div className="flex items-center gap-2">
                        <Button
                            asChild
                            variant="default"
                            size="sm"
                            className="rounded-full bg-zinc-900 text-white hover:bg-zinc-800 dark:bg-white dark:text-zinc-900 px-5 h-9 transition-all hover:shadow-lg hover:shadow-indigo-500/20 active:scale-95"
                        >
                            <Link href="/generate" className="flex items-center gap-2">
                                <Sparkles className="w-3.5 h-3.5" />
                                <span className="font-medium">Try MatterGen</span>
                            </Link>
                        </Button>

                        {/* Mobile Toggle */}
                        <button
                            onClick={() => setMenuState(!menuState)}
                            className="md:hidden p-2 text-zinc-600 hover:bg-zinc-100 rounded-full transition-colors"
                        >
                            {menuState ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                        </button>
                    </div>
                </div>

                {/* Mobile Menu Dropdown */}
                <div className={cn(
                    "absolute top-full left-0 right-0 mt-2 p-4 bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-2xl shadow-xl md:hidden transition-all duration-300 origin-top overflow-hidden",
                    menuState ? "opacity-100 scale-y-100 max-h-[400px]" : "opacity-0 scale-y-95 max-h-0 pointer-events-none"
                )}>
                    <ul className="flex flex-col gap-2">
                        {menuItems.map((item, index) => (
                            <li key={index}>
                                <Link
                                    href={item.href}
                                    onClick={() => setMenuState(false)}
                                    className="block p-3 text-sm font-medium text-zinc-600 rounded-xl hover:bg-zinc-50 dark:hover:bg-zinc-900 transition-colors"
                                >
                                    {item.name}
                                </Link>
                            </li>
                        ))}
                    </ul>
                </div>
            </nav>
        </header>
    )
}
