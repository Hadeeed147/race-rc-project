import { forwardRef } from 'react'
import * as ToastPrimitive from '@radix-ui/react-toast'
import { X } from 'lucide-react'
import { cva } from 'class-variance-authority'
import { cn } from '../../lib/utils'

const ToastProvider = ToastPrimitive.Provider

const ToastViewport = forwardRef(({ className, ...props }, ref) => (
  <ToastPrimitive.Viewport
    ref={ref}
    className={cn(
      'fixed bottom-0 right-0 z-[100] flex max-h-screen w-full flex-col-reverse gap-2 p-4 sm:bottom-4 sm:right-4 sm:top-auto sm:flex-col md:max-w-[420px]',
      className
    )}
    {...props}
  />
))
ToastViewport.displayName = 'ToastViewport'

const toastVariants = cva(
  'group pointer-events-auto relative flex w-full items-center justify-between gap-3 ' +
  'overflow-hidden rounded-xl border p-4 pr-7 shadow-soft data-[state=open]:animate-in ' +
  'data-[state=closed]:animate-out data-[swipe=end]:animate-out data-[state=closed]:fade-out-80 ' +
  'data-[state=closed]:slide-out-to-right-full data-[state=open]:slide-in-from-right-full',
  {
    variants: {
      variant: {
        default: 'border-border bg-card text-card-foreground',
        success: 'border-success/30 bg-success/10 text-foreground',
        destructive: 'border-destructive/40 bg-destructive/10 text-foreground',
        accent: 'border-accent/40 bg-accent/10 text-foreground',
      },
    },
    defaultVariants: { variant: 'default' },
  }
)

const Toast = forwardRef(({ className, variant, ...props }, ref) => (
  <ToastPrimitive.Root
    ref={ref}
    className={cn(toastVariants({ variant }), className)}
    {...props}
  />
))
Toast.displayName = 'Toast'

const ToastTitle = forwardRef(({ className, ...props }, ref) => (
  <ToastPrimitive.Title ref={ref} className={cn('text-sm font-semibold', className)} {...props} />
))
ToastTitle.displayName = 'ToastTitle'

const ToastDescription = forwardRef(({ className, ...props }, ref) => (
  <ToastPrimitive.Description ref={ref} className={cn('text-sm text-muted-foreground', className)} {...props} />
))
ToastDescription.displayName = 'ToastDescription'

const ToastClose = forwardRef(({ className, ...props }, ref) => (
  <ToastPrimitive.Close
    ref={ref}
    toast-close=""
    className={cn(
      'absolute right-2 top-2 rounded-md p-1 text-foreground/50 opacity-70 transition hover:text-foreground hover:opacity-100',
      className
    )}
    {...props}
  >
    <X className="h-3 w-3" />
  </ToastPrimitive.Close>
))
ToastClose.displayName = 'ToastClose'

export { ToastProvider, ToastViewport, Toast, ToastTitle, ToastDescription, ToastClose }
