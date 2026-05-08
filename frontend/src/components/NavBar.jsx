import { NavLink, Link } from 'react-router-dom'
import { BookOpenText, BarChart3, MessageCircleQuestion, Wand2, Moon, Sun, Keyboard } from 'lucide-react'
import { Button } from './ui/Button'
import { useUI } from '../lib/store'
import { cn } from '../lib/utils'

const links = [
  { to: '/',          label: 'Article',    icon: BookOpenText },
  { to: '/quiz',      label: 'Quiz',       icon: Wand2 },
  { to: '/hints',     label: 'Hints',      icon: MessageCircleQuestion },
  { to: '/analytics', label: 'Analytics',  icon: BarChart3 },
]

export default function NavBar({ onShortcutsOpen }) {
  const theme = useUI((s) => s.theme)
  const toggle = useUI((s) => s.toggleTheme)
  return (
    <header className="sticky top-0 z-30 w-full border-b border-border glass">
      <div className="mx-auto flex h-14 w-full max-w-6xl items-center gap-3 px-4 md:px-6">
        <Link to="/" className="flex items-center gap-2.5 font-semibold tracking-tight">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-primary text-primary-foreground shadow-glow">
            <BookOpenText className="h-4 w-4" />
          </span>
          <span className="hidden sm:inline">RACE RC</span>
          <span className="text-xs text-muted-foreground hidden md:inline ml-1">/ Reading AI</span>
        </Link>

        <nav className="ml-2 flex items-center gap-1">
          {links.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                cn(
                  'inline-flex items-center gap-2 rounded-md px-2.5 py-1.5 text-sm font-medium transition',
                  isActive
                    ? 'bg-secondary text-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-secondary/60'
                )
              }
            >
              <Icon className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            aria-label="Keyboard shortcuts"
            onClick={onShortcutsOpen}
            title="Keyboard shortcuts (?)"
          >
            <Keyboard className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Toggle dark mode"
            onClick={toggle}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </div>
      </div>
    </header>
  )
}
