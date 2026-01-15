import { Atom } from 'lucide-react'
import { cn } from '@/lib/utils'

export const Logo = ({ className, uniColor }: { className?: string; uniColor?: boolean }) => {
    return (
        <div className={cn('flex items-center gap-2 font-bold text-xl tracking-tight', className)}>
            <div className="bg-primary/10 rounded-lg p-1">
                <Atom className="size-5 text-primary" />
            </div>
            <span>MatterGen X</span>
        </div>
    )
}

export const LogoIcon = ({ className, uniColor }: { className?: string; uniColor?: boolean }) => {
    return (
        <div className={cn('bg-primary/10 rounded-lg p-1', className)}>
            <Atom className="size-5 text-primary" />
        </div>
    )
}

export const LogoStroke = ({ className }: { className?: string }) => {
    return (
        <Atom className={cn('size-7 w-7 text-muted-foreground', className)} />
    )
}
