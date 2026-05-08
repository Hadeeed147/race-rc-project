import { useEffect, useState } from 'react'
import { Routes, Route, useLocation, useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'

import NavBar from './components/NavBar.jsx'
import Banner from './components/Banner.jsx'
import Footer from './components/Footer.jsx'
import ShortcutsDialog from './components/ShortcutsDialog.jsx'
import { Toaster } from './components/ui/Toaster.jsx'

import ArticleInput from './pages/ArticleInput.jsx'
import Quiz from './pages/Quiz.jsx'
import Hints from './pages/Hints.jsx'
import Analytics from './pages/Analytics.jsx'
import NotFound from './pages/NotFound.jsx'

import { useThemeSync } from './lib/useTheme.js'
import { useUI } from './lib/store.js'

function isTypingTarget(target) {
  if (!target) return false
  const tag = (target.tagName || '').toLowerCase()
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return true
  if (target.isContentEditable) return true
  return false
}

export default function App() {
  useThemeSync()
  const navigate = useNavigate()
  const location = useLocation()
  const toggleTheme = useUI((s) => s.toggleTheme)
  const [shortcutsOpen, setShortcutsOpen] = useState(false)

  // Global keyboard shortcuts
  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (isTypingTarget(e.target)) return
      const k = e.key
      if (k === '?' || (k === '/' && e.shiftKey)) { e.preventDefault(); setShortcutsOpen(true); return }
      if (k === 'Escape') { setShortcutsOpen(false); return }
      switch (k.toLowerCase()) {
        case 'g': navigate('/'); break
        case 'q': navigate('/quiz'); break
        case 'h': navigate('/hints'); break
        case 'a': navigate('/analytics'); break
        case 'd': toggleTheme(); break
        default: return
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [navigate, toggleTheme])

  return (
    <div className="flex min-h-screen flex-col">
      <Banner />
      <NavBar onShortcutsOpen={() => setShortcutsOpen(true)} />

      <main className="flex-1">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
          >
            <Routes location={location}>
              <Route path="/"          element={<ArticleInput />} />
              <Route path="/quiz"      element={<Quiz />} />
              <Route path="/hints"     element={<Hints />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="*"          element={<NotFound />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>

      <Footer />
      <ShortcutsDialog open={shortcutsOpen} onOpenChange={setShortcutsOpen} />
      <Toaster />
    </div>
  )
}
