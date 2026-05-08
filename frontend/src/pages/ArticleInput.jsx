import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Sparkles, Shuffle, Loader2, ArrowRight, BookOpen, Clock, FileText } from 'lucide-react'

import { Card, CardContent } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { Progress } from '../components/ui/Progress'
import { Badge } from '../components/ui/Badge'
import { Skeleton } from '../components/ui/Skeleton'
import { toast } from '../components/ui/use-toast'

import { generateQuiz, getSample } from '../lib/api'
import { useQuiz, useUI } from '../lib/store'
import { wordCount, readingTimeMinutes } from '../lib/utils'

const PLACEHOLDER = `Paste an English reading passage here…

For example: a short article about science, history, or daily life.
The model will generate a multiple-choice question from it.`

export default function ArticleInput() {
  const navigate = useNavigate()
  const setQuiz = useQuiz((s) => s.setQuiz)
  const pushLatency = useUI((s) => s.pushLatency)
  const [article, setArticle] = useState('')
  const [busy, setBusy] = useState(null) // 'sample' | 'generate' | null

  const onLoadSample = async () => {
    setBusy('sample')
    try {
      const r = await getSample()
      setArticle(r.article)
      toast({ title: 'Loaded sample passage', description: `id: ${r.id}`, variant: 'default' })
    } catch (e) {
      toast({ title: 'Could not load sample', description: e.message, variant: 'destructive' })
    } finally {
      setBusy(null)
    }
  }

  const onGenerate = async () => {
    if (article.trim().length < 50) {
      toast({ title: 'Article too short', description: 'Need at least 50 characters.', variant: 'destructive' })
      return
    }
    setBusy('generate')
    try {
      const r = await generateQuiz(article.trim())
      pushLatency(r.latency_ms)
      setQuiz({
        source: 'generated',
        article: article.trim(),
        question: r.question,
        answer: r.answer,
        answerText: r.answer_text,
        options: r.options,
        template: r.template,
        generationLatencyMs: r.latency_ms,
      })
      toast({ title: 'Quiz generated', description: `Template: ${r.template} · ${r.latency_ms} ms` })
      navigate('/quiz')
    } catch (e) {
      toast({ title: 'Generation failed', description: e.message, variant: 'destructive' })
    } finally {
      setBusy(null)
    }
  }

  const wc = wordCount(article)
  const cc = article.length
  const rt = readingTimeMinutes(article)

  return (
    <div className="mx-auto w-full max-w-4xl px-4 md:px-6 py-10 md:py-14">
      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="mb-8 md:mb-10"
      >
        <div className="inline-flex items-center gap-2 mb-4">
          <Badge variant="accent" className="gap-1.5">
            <Sparkles className="h-3 w-3" />
            Classical ML · TF-IDF Ensemble
          </Badge>
        </div>
        <h1 className="text-4xl md:text-5xl font-bold leading-tight tracking-tight">
          Reading Comprehension <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">AI</span>
        </h1>
        <p className="mt-3 max-w-2xl text-base md:text-lg text-muted-foreground leading-relaxed">
          Paste a passage. Our soft-vote ensemble (Logistic Regression + LinearSVC + ComplementNB)
          generates a multiple-choice question, scores each option, and offers graduated hints —
          all in under a hundred milliseconds.
        </p>
      </motion.div>

      {/* Editor */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.06 }}
      >
        <Card className="overflow-hidden">
          <CardContent className="p-0">
            <div className="flex items-center gap-2 px-4 py-2 border-b border-border bg-muted/40">
              <FileText className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs font-medium text-muted-foreground">Article</span>
              <div className="ml-auto flex items-center gap-3 text-[11px] text-muted-foreground">
                <span>{cc.toLocaleString()} chars</span>
                <span className="opacity-40">·</span>
                <span>{wc} words</span>
                <span className="opacity-40">·</span>
                <span className="inline-flex items-center gap-1"><Clock className="h-3 w-3" /> {rt} min read</span>
              </div>
            </div>
            <textarea
              className="w-full min-h-[300px] resize-y bg-transparent px-5 py-4 text-[15px] leading-relaxed text-foreground placeholder:text-muted-foreground/60 focus:outline-none scrollbar-thin"
              placeholder={PLACEHOLDER}
              value={article}
              onChange={(e) => setArticle(e.target.value)}
              spellCheck={false}
            />
          </CardContent>
        </Card>

        {/* Actions */}
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <Button
            variant="outline"
            onClick={onLoadSample}
            disabled={busy !== null}
          >
            {busy === 'sample'
              ? <Loader2 className="h-4 w-4 animate-spin" />
              : <Shuffle className="h-4 w-4" />}
            Load Random Sample
          </Button>
          <Button
            onClick={onGenerate}
            disabled={busy !== null || article.trim().length < 50}
          >
            {busy === 'generate'
              ? <Loader2 className="h-4 w-4 animate-spin" />
              : <Sparkles className="h-4 w-4" />}
            Generate Quiz
            <ArrowRight className="h-4 w-4" />
          </Button>
          <span className="ml-auto text-xs text-muted-foreground">
            Min 50 chars to generate
          </span>
        </div>

        {/* Loading progress */}
        {busy === 'generate' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-6 space-y-3"
          >
            <Progress indeterminate />
            <div className="flex items-center gap-3">
              <Skeleton className="h-4 w-1/3" />
              <Skeleton className="h-4 w-1/4" />
              <Skeleton className="h-4 w-1/5" />
            </div>
          </motion.div>
        )}
      </motion.div>

      {/* Stats footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.15 }}
        className="mt-10 grid gap-3 sm:grid-cols-3"
      >
        {[
          { label: 'Models in ensemble', value: '3', sub: 'LR · SVC · NB' },
          { label: 'Vocab size',          value: '20 000', sub: 'TF-IDF unigrams + bigrams' },
          { label: 'Training option rows',value: '281 168', sub: '70k questions × 4 options' },
        ].map((c) => (
          <div key={c.label} className="rounded-xl border border-border bg-card/60 px-4 py-3">
            <div className="flex items-baseline justify-between">
              <span className="text-xs text-muted-foreground">{c.label}</span>
            </div>
            <div className="mt-1 text-lg font-semibold tracking-tight text-foreground">{c.value}</div>
            <div className="text-[11px] text-muted-foreground">{c.sub}</div>
          </div>
        ))}
      </motion.div>
    </div>
  )
}
