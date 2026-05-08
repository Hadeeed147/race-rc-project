import { motion, AnimatePresence } from 'framer-motion'
import { Info, X } from 'lucide-react'
import { useUI } from '../lib/store'

export default function Banner() {
  const dismissed = useUI((s) => s.bannerDismissed)
  const dismiss = useUI((s) => s.dismissBanner)
  return (
    <AnimatePresence>
      {!dismissed && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.25 }}
          className="overflow-hidden bg-accent-soft/80 dark:bg-accent-soft/50 border-b border-accent/30"
        >
          <div className="mx-auto flex w-full max-w-6xl items-center gap-3 px-4 md:px-6 py-2 text-sm">
            <Info className="h-4 w-4 text-accent-foreground shrink-0" />
            <span className="text-accent-foreground">
              Answers are AI-generated and may be wrong. This is a classical-ML
              demo using TF-IDF + a soft-vote ensemble — not GPT.
            </span>
            <button
              onClick={dismiss}
              aria-label="Dismiss banner"
              className="ml-auto rounded p-1 text-accent-foreground/60 hover:bg-accent/20 hover:text-accent-foreground"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
