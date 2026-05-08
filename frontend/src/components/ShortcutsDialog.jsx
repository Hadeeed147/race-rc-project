import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription,
} from './ui/Dialog'

const SHORTCUTS = [
  { keys: ['?'],         label: 'Open this dialog' },
  { keys: ['G'],         label: 'Go home / Article input' },
  { keys: ['Q'],         label: 'Go to Quiz' },
  { keys: ['H'],         label: 'Go to Hints' },
  { keys: ['A'],         label: 'Go to Analytics' },
  { keys: ['D'],         label: 'Toggle dark mode' },
  { keys: ['Esc'],       label: 'Close any dialog' },
]

function Kbd({ children }) {
  return (
    <kbd className="inline-flex min-w-[1.6rem] items-center justify-center rounded-md border border-border bg-secondary px-1.5 py-0.5 text-[10px] font-mono font-semibold text-foreground shadow-sm">
      {children}
    </kbd>
  )
}

export default function ShortcutsDialog({ open, onOpenChange }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Keyboard shortcuts</DialogTitle>
          <DialogDescription>Single-key shortcuts. Most keys ignore typing in textareas.</DialogDescription>
        </DialogHeader>
        <div className="mt-2 grid gap-2">
          {SHORTCUTS.map((s) => (
            <div key={s.label} className="flex items-center justify-between rounded-md border border-border bg-card/50 px-3 py-2">
              <span className="text-sm text-foreground">{s.label}</span>
              <span className="flex gap-1">
                {s.keys.map((k) => <Kbd key={k}>{k}</Kbd>)}
              </span>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}
