import { cva } from 'class-variance-authority'
import { cn } from '../../lib/utils'

const badgeVariants = cva(
  'inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors',
  {
    variants: {
      variant: {
        default: 'bg-primary-soft text-primary border border-primary/20',
        secondary: 'bg-secondary text-secondary-foreground',
        outline: 'border border-border text-foreground',
        accent: 'bg-accent-soft text-accent-foreground border border-accent/30',
        success: 'bg-success/10 text-success border border-success/30',
        destructive: 'bg-destructive/10 text-destructive border border-destructive/30',
        muted: 'bg-muted text-muted-foreground',
      },
    },
    defaultVariants: { variant: 'default' },
  }
)

export function Badge({ className, variant, ...props }) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />
}
