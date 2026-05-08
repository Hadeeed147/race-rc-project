import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Compass, Home } from 'lucide-react'
import { Button } from '../components/ui/Button'

export default function NotFound() {
  return (
    <div className="mx-auto flex min-h-[60vh] w-full max-w-2xl flex-col items-center justify-center px-4 text-center">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4 }}
        className="grid h-20 w-20 place-items-center rounded-2xl bg-primary-soft text-primary shadow-glow"
      >
        <Compass className="h-9 w-9" />
      </motion.div>
      <h1 className="mt-6 text-4xl font-bold tracking-tight">Off the page</h1>
      <p className="mt-3 max-w-md text-muted-foreground">
        That route doesn't exist. Head back to the article input and start a fresh quiz.
      </p>
      <div className="mt-6">
        <Button asChild>
          <Link to="/"><Home className="h-4 w-4" /> Back home</Link>
        </Button>
      </div>
    </div>
  )
}
