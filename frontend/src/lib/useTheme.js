import { useEffect } from 'react'
import { useUI } from './store'

/** Sync the theme preference into <html class="dark"> for Tailwind. */
export function useThemeSync() {
  const theme = useUI((s) => s.theme)
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'dark') root.classList.add('dark')
    else root.classList.remove('dark')
  }, [theme])
  return theme
}
