import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import confetti from 'canvas-confetti'
import {
  CheckCircle2, XCircle, Sparkles, Lightbulb, RotateCcw, Loader2,
  Cpu, Timer, BookOpen, ChevronDown, AlertTriangle, Wand2, CheckCircle,
} from 'lucide-react'

import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'
import { Progress } from '../components/ui/Progress'
import { RadioGroup, RadioGroupItem } from '../components/ui/RadioGroup'
import { Alert, AlertDescription } from '../components/ui/Alert'
import { toast } from '../components/ui/use-toast'

import { predictAnswer } from '../lib/api'
import { useQuiz, useUI } from '../lib/store'
import { cn, formatPercent } from '../lib/utils'

const LABELS = ['A', 'B', 'C', 'D']

/**
 * Map raw scores -> visual bar widths in [12, 100] using min-max
 * normalization. Raw probabilities for the four options often cluster
 * (e.g. 0.41 / 0.41 / 0.41 / 0.42) because the article+question dominates
 * the TF-IDF signal. Min-max stretching makes the relative ranking
 * obvious without misrepresenting the absolute % (we keep that separate).
 */
function normalizedWidths(scores) {
  if (!scores) return null
  const vals = LABELS.map((L) => Number(scores?.[L] ?? 0))
  const min = Math.min(...vals)
  const max = Math.max(...vals)
  const span = max - min
  // If everything is identical, give them all the same medium width.
  if (span < 1e-6) return LABELS.reduce((acc, L) => ({ ...acc, [L]: 65 }), {})
  return LABELS.reduce((acc, L, i) => {
    const norm = (vals[i] - min) / span        // 0..1
    acc[L] = 12 + norm * 88                    // 12..100
    return acc
  }, {})
}

/** Three-step opacity ramp by rank. Highest = full primary; lowest = pale. */
function rankOpacityClass(score, scores) {
  if (score === undefined || !scores) return 'bg-primary/40'
  const sorted = LABELS.map((L) => scores[L]).sort((a, b) => b - a)
  if (score === sorted[0]) return 'bg-primary'           // top
  if (score === sorted[1]) return 'bg-primary/70'        // 2nd
  if (score === sorted[2]) return 'bg-primary/45'        // 3rd
  return 'bg-primary/25'                                  // bottom
}

export default function Quiz() {
  const navigate = useNavigate()
  const quiz = useQuiz()
  const reset = useQuiz((s) => s.reset)
  const pushLatency = useUI((s) => s.pushLatency)

  const [choice, setChoice] = useState(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [articleOpen, setArticleOpen] = useState(true)

  // No quiz in store? push back home.
  useEffect(() => {
    if (!quiz.question) {
      const t = setTimeout(() => navigate('/', { replace: true }), 0)
      return () => clearTimeout(t)
    }
  }, [quiz.question, navigate])

  const correctLabel = quiz.answer
  const isCorrect = result && choice && choice === correctLabel
  const isWrong = result && choice && choice !== correctLabel

  const widths = useMemo(() => normalizedWidths(result?.scores), [result?.scores])

  const onCheck = async () => {
    if (!choice) return
    setBusy(true)
    try {
      const r = await predictAnswer({
        article: quiz.article,
        question: quiz.question,
        options: quiz.options,
      })
      pushLatency(r.latency_ms)
      setResult(r)
      if (correctLabel && choice === correctLabel) {
        confetti({
          particleCount: 80, spread: 70, origin: { y: 0.45 },
          colors: ['#6366f1', '#f59e0b', '#10b981'],
          disableForReducedMotion: true,
        })
      }
    } catch (e) {
      toast({ title: 'Predict failed', description: e.message, variant: 'destructive' })
    } finally {
      setBusy(false)
    }
  }

  const onTryAnother = () => {
    reset()
    navigate('/')
  }

  if (!quiz.question) return null

  const isReal = quiz.source === 'real'
  const hasGold = !!correctLabel

  return (
    <div className="mx-auto w-full max-w-6xl px-4 md:px-6 py-8 md:py-12">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
        className="grid gap-6 lg:grid-cols-[1.1fr_1fr]"
      >
        {/* LEFT — Article */}
        <Card>
          <CardHeader>
            <button
              onClick={() => setArticleOpen((o) => !o)}
              className="flex w-full items-center justify-between text-left"
            >
              <div className="flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-primary" />
                <CardTitle>Passage</CardTitle>
              </div>
              <ChevronDown className={cn('h-4 w-4 text-muted-foreground transition-transform', articleOpen ? '' : '-rotate-90')} />
            </button>
          </CardHeader>
          <AnimatePresence initial={false}>
            {articleOpen && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.25 }}
                className="overflow-hidden"
              >
                <CardContent>
                  <div className="max-h-[460px] overflow-y-auto pr-2 scrollbar-thin">
                    <p className="text-[15px] leading-7 text-foreground/90 whitespace-pre-wrap">
                      {quiz.article}
                    </p>
                  </div>
                </CardContent>
              </motion.div>
            )}
          </AnimatePresence>
        </Card>

        {/* RIGHT — Question + options + result */}
        <div className="flex flex-col gap-4">
          {/* Source / quality badges + warning banner */}
          <div className="flex flex-wrap items-center gap-2">
            {isReal ? (
              <Badge variant="success" className="gap-1.5">
                <CheckCircle className="h-3 w-3" /> Real RACE question
              </Badge>
            ) : (
              <Badge variant="accent" className="gap-1.5">
                <Wand2 className="h-3 w-3" /> AI-generated
              </Badge>
            )}
            {quiz.template && (
              <Badge variant="outline" className="capitalize">
                template: {quiz.template}
              </Badge>
            )}
          </div>

          {quiz.qualityWarning && (
            <Alert variant="accent">
              <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
              <AlertDescription>
                This is a low-confidence AI-generated question. For a stronger demo,
                switch to <strong>Real RACE Question</strong> mode.
              </AlertDescription>
            </Alert>
          )}

          <Card>
            <CardHeader>
              <CardTitle className="leading-snug">{quiz.question}</CardTitle>
            </CardHeader>
            <CardContent>
              <RadioGroup value={choice ?? ''} onValueChange={(v) => { setChoice(v); setResult(null) }}>
                <ul className="grid gap-2">
                  {LABELS.map((L) => {
                    const text = quiz.options?.[L]
                    const score = result?.scores?.[L]
                    const isSel = choice === L
                    const isAns = result && hasGold && correctLabel === L
                    const isPicked = result && choice === L
                    const stateClass = result
                      ? isAns ? 'border-success bg-success/10'
                      : isPicked && hasGold && choice !== correctLabel ? 'border-destructive bg-destructive/10'
                      : 'border-border'
                      : isSel ? 'border-primary bg-primary-soft' : 'border-border hover:border-primary/40'
                    const visualWidth = widths?.[L] ?? 0
                    const barClass = isAns ? 'bg-success' : rankOpacityClass(score, result?.scores)
                    return (
                      <li key={L}>
                        <label className={cn(
                          'group relative flex cursor-pointer items-start gap-3 rounded-xl border bg-card px-4 py-3 transition-all',
                          stateClass,
                        )}>
                          <RadioGroupItem id={`opt-${L}`} value={L} className="mt-1" disabled={busy || result !== null} />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-xs font-bold text-primary">{L}</span>
                              {result && isAns && (
                                <Badge variant="success" className="ml-auto"><CheckCircle2 className="h-3 w-3" /> correct</Badge>
                              )}
                              {result && isPicked && hasGold && !isAns && (
                                <Badge variant="destructive" className="ml-auto"><XCircle className="h-3 w-3" /> your pick</Badge>
                              )}
                              {result && !hasGold && result.predicted === L && (
                                <Badge variant="default" className="ml-auto">model's top pick</Badge>
                              )}
                            </div>
                            <p className="mt-1 text-sm leading-relaxed text-foreground">{text}</p>
                            {score !== undefined && (
                              <div className="mt-2.5 flex items-center gap-3">
                                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                                  <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${visualWidth}%` }}
                                    transition={{ duration: 0.6, ease: 'easeOut' }}
                                    className={cn('h-full rounded-full transition-colors', barClass)}
                                  />
                                </div>
                                <span className="font-mono text-[11px] text-muted-foreground tabular-nums w-12 text-right">
                                  {(score * 100).toFixed(1)}%
                                </span>
                              </div>
                            )}
                          </div>
                        </label>
                      </li>
                    )
                  })}
                </ul>
              </RadioGroup>

              {busy && <Progress indeterminate className="mt-4" />}

              <div className="mt-5 flex flex-wrap items-center gap-2">
                {!result ? (
                  <Button onClick={onCheck} disabled={!choice || busy}>
                    {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                    Check Answer
                  </Button>
                ) : (
                  <>
                    <Button onClick={onTryAnother} variant="outline">
                      <RotateCcw className="h-4 w-4" />
                      Try another article
                    </Button>
                    <Button asChild variant="ghost">
                      <Link to="/quiz">Reset</Link>
                    </Button>
                  </>
                )}
                <Button asChild variant="ghost" className="ml-auto">
                  <Link to="/hints">
                    <Lightbulb className="h-4 w-4" />
                    Need a hint?
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Verdict + meta */}
          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
              >
                <Card>
                  <CardContent className="p-4 flex flex-wrap items-center gap-3">
                    {hasGold && isCorrect && (
                      <Badge variant="success" className="text-sm py-1 px-2.5">
                        <CheckCircle2 className="h-3.5 w-3.5" /> You got it right
                      </Badge>
                    )}
                    {hasGold && isWrong && (
                      <Badge variant="destructive" className="text-sm py-1 px-2.5">
                        <XCircle className="h-3.5 w-3.5" /> Gold answer:&nbsp;
                        <span className="font-mono">{correctLabel}</span>
                      </Badge>
                    )}
                    {!hasGold && (
                      <Badge variant="default" className="text-sm py-1 px-2.5">
                        Model picked&nbsp;<span className="font-mono">{result.predicted}</span>
                      </Badge>
                    )}
                    <Badge variant="secondary" className="gap-1">
                      <Cpu className="h-3 w-3" /> {result.model_used.split('(')[0].trim()}
                    </Badge>
                    <Badge variant="outline" className="gap-1">
                      <Timer className="h-3 w-3" /> {result.latency_ms} ms
                    </Badge>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  )
}
