import { create } from 'zustand'
import { persist } from 'zustand/middleware'

/** Quiz session state — populated by /generate or /sample. */
export const useQuiz = create((set) => ({
  source: null,                 // 'generated' | 'sample'
  article: '',
  question: '',
  answer: null,                 // gold A/B/C/D when known (sample), null for generated
  answerText: null,             // gold text for /distractors fallback
  options: { A: '', B: '', C: '', D: '' },
  template: null,
  generationLatencyMs: null,
  hintsUsed: 0,
  setQuiz: (q) => set({
    source: q.source,
    article: q.article,
    question: q.question,
    answer: q.answer ?? null,
    answerText: q.answerText ?? q.options?.[q.answer] ?? null,
    options: q.options,
    template: q.template ?? null,
    generationLatencyMs: q.generationLatencyMs ?? null,
    hintsUsed: 0,
  }),
  bumpHints: () => set((s) => ({ hintsUsed: Math.min(3, s.hintsUsed + 1) })),
  reset: () => set({
    source: null, article: '', question: '', answer: null, answerText: null,
    options: { A: '', B: '', C: '', D: '' }, template: null,
    generationLatencyMs: null, hintsUsed: 0,
  }),
}))

/** Persisted UI preferences. */
export const useUI = create(
  persist(
    (set, get) => ({
      theme: 'dark',
      bannerDismissed: false,
      latencyHistory: [],     // last 20 latency samples in ms
      setTheme: (t) => set({ theme: t }),
      toggleTheme: () => set({ theme: get().theme === 'dark' ? 'light' : 'dark' }),
      dismissBanner: () => set({ bannerDismissed: true }),
      pushLatency: (ms) => set((s) => ({
        latencyHistory: [...s.latencyHistory.slice(-19), { ts: Date.now(), ms }],
      })),
    }),
    { name: 'race-ui-prefs' }
  )
)
