import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

/** Standard shadcn cn() helper — merges Tailwind classes safely. */
export function cn(...inputs) {
  return twMerge(clsx(inputs))
}

export function formatPercent(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  return `${(x * 100).toFixed(digits)}%`
}

export function formatNumber(x, digits = 4) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  return Number(x).toFixed(digits)
}

export function readingTimeMinutes(text, wpm = 220) {
  const words = (text || '').trim().split(/\s+/).filter(Boolean).length
  return Math.max(1, Math.round(words / wpm))
}

export function wordCount(text) {
  return (text || '').trim().split(/\s+/).filter(Boolean).length
}
