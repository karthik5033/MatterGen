'use client'
import Link from 'next/link'
import { Logo } from './logo'
import { Menu, X, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import React, { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'

const menuItems = [
    { name: 'Generate', href: '/generate' },
    { name: 'Compare', href: '/compare' },
    { name: 'Docs', href: '/docs' },
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
        <header className="fixed top-0 z-50 w-full">
            <nav
                data-state={menuState && 'active'}
                className={cn(
                    "w-full transition-all duration-300 ease-in-out border-b border-transparent",
                    scrolled ? "bg-white/80 dark:bg-zinc-950/80 backdrop-blur-md border-zinc-200/50 dark:border-zinc-800/50 shadow-sm py-4" : "bg-transparent py-6"
                )}
            >
                <div className="mx-auto max-w-7xl px-6 md:px-12">
                    <div className="relative flex flex-wrap items-center justify-between gap-6">
                        {/* Logo and Mobile Toggle */}
                        <div className="flex w-full items-center justify-between lg:w-auto">
                            <Link
                                href="/"
                                aria-label="home"
                                className="flex items-center space-x-2 transition-opacity hover:opacity-80">
                                <Logo />
                            </Link>

                            <button
                                onClick={() => setMenuState(!menuState)}
                                aria-label={menuState == true ? 'Close Menu' : 'Open Menu'}
                                className="relative z-20 -m-2.5 -mr-4 block cursor-pointer p-2.5 lg:hidden"
                            >
                                <Menu className="size-6 transition-transform duration-200" />
                            </button>
                        </div>

                        {/* Desktop Navigation */}
                        <div className="hidden lg:flex lg:items-center lg:gap-8">
                            <ul className="flex items-center gap-8 text-sm font-medium">
                                {menuItems.map((item, index) => (
                                    <li key={index}>
                                        <Link
                                            href={item.href}
                                            className="text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 transition-colors"
                                        >
                                            {item.name}
                                        </Link>
                                    </li>
                                ))}
                            </ul>
                        </div>

                        {/* Desktop CTA */}
                        <div className="hidden lg:flex lg:items-center lg:gap-4">
                            <Button
                                asChild
                                variant="default"
                                size="sm"
                                className="bg-zinc-900 text-white hover:bg-zinc-800 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200 rounded-full px-6 transition-all hover:scale-105"
                            >
                                <Link href="/generate">
                                    <span>Start App</span>
                                    <ArrowRight className="ml-2 size-4" />
                                </Link>
                            </Button>
                        </div>

                        {/* Mobile Menu Dropdown */}
                        <div className={cn(
                            "absolute top-full left-0 w-full bg-white dark:bg-zinc-950 border-b border-zinc-200 dark:border-zinc-800 p-6 shadow-xl lg:hidden transition-all duration-300 ease-in-out origin-top",
                            menuState ? "opacity-100 scale-y-100 translate-y-0 visible" : "opacity-0 scale-y-95 -translate-y-4 invisible"
                        )}>
                            <ul className="flex flex-col gap-4 text-base font-medium">
                                {menuItems.map((item, index) => (
                                    <li key={index}>
                                        <Link
                                            href={item.href}
                                            onClick={() => setMenuState(false)}
                                            className="block py-2 text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
                                        >
                                            {item.name}
                                        </Link>
                                    </li>
                                ))}
                                <li className="pt-4 border-t border-zinc-100 dark:border-zinc-800">
                                    <Button asChild className="w-full justify-center">
                                        <Link href="/generate">Start App</Link>
                                    </Button>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
            </nav>
        </header>
    )
}
