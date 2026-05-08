import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  CartesianGrid, Cell,
} from 'recharts'
import {
  Activity, Target, Award, Timer, Zap, Database, Server, Layers,
} from 'lucide-react'

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { Skeleton } from '../components/ui/Skeleton'
import { toast } from '../components/ui/use-toast'

import { getAnalytics } from '../lib/api'
import { useUI } from '../lib/store'
import { formatNumber, formatPercent } from '../lib/utils'

const stagger = { animate: { transition: { staggerChildren: 0.06 } } }
const fadeUp = { initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.35 } }

export default function Analytics() {
  const [data, setData] = useState(null)
  const [busy, setBusy] = useState(true)
  const latencyHistory = useUI((s) => s.latencyHistory)

  useEffect(() => {
    let mounted = true
    ;(async () => {
      try {
        const r = await getAnalytics()
        if (mounted) setData(r)
      } catch (e) {
        toast({ title: 'Analytics failed', description: e.message, variant: 'destructive' })
      } finally {
        if (mounted) setBusy(false)
      }
    })()
    return () => { mounted = false }
  }, [])

  const ensembleRow = useMemo(
    () => data?.model_a?.find((m) => m.name?.toLowerCase().includes('ensemble')) || null,
    [data]
  )
  const lrRow = useMemo(
    () => data?.model_a?.find((m) => m.name?.toLowerCase().includes('logistic')) || null,
    [data]
  )

  const avgLatency = latencyHistory.length
    ? Math.round(latencyHistory.reduce((s, h) => s + h.ms, 0) / latencyHistory.length)
    : null

  const cards = [
    {
      label: 'Accuracy', icon: Target,
      value: ensembleRow ? formatPercent(ensembleRow.accuracy, 1) : '—',
      sub: ensembleRow ? `Ensemble vs LR: +${formatNumber((ensembleRow.accuracy - (lrRow?.accuracy || 0)), 4)}` : 'val set',
    },
    {
      label: 'Macro F1', icon: Activity,
      value: ensembleRow ? formatNumber(ensembleRow.macro_f1, 4) : '—',
      sub: ensembleRow ? 'per-question argmax' : 'val set',
    },
    {
      label: 'Exact Match', icon: Award,
      value: ensembleRow ? formatNumber(ensembleRow.exact_match, 4) : '—',
      sub: ensembleRow ? `vs random 0.250: +${formatNumber(ensembleRow.exact_match - 0.25, 3)}` : 'baseline 0.25',
    },
    {
      label: 'Avg latency', icon: Timer,
      value: avgLatency != null ? `${avgLatency} ms` : '— ms',
      sub: latencyHistory.length ? `n=${latencyHistory.length} requests this session` : 'no requests yet',
    },
  ]

  // BarChart data
  const chartData = (data?.model_a || []).map((m) => ({
    name: m.name?.split('(')[0]?.trim() || m.name,
    'Macro F1': Number(m.macro_f1?.toFixed(4) ?? 0),
    'Exact Match': Number(m.exact_match?.toFixed(4) ?? 0),
  }))

  // Confusion matrix (use ensemble row, fallback to LR)
  const cm = ensembleRow?.confusion_matrix || lrRow?.confusion_matrix || [[0, 0], [0, 0]]
  const flat = cm.flat()
  const cmMax = Math.max(...flat) || 1

  // Model B
  const b = data?.model_b
  const bD = b?.distractors || {}
  const bH = b?.hints || {}

  return (
    <motion.div variants={stagger} initial="initial" animate="animate"
      className="mx-auto w-full max-w-6xl px-4 md:px-6 py-8 md:py-12 space-y-6">
      {/* Header */}
      <motion.div {...fadeUp}>
        <div className="flex items-center gap-2">
          <Badge variant="default" className="gap-1.5"><Zap className="h-3 w-3" /> Live val-set metrics</Badge>
        </div>
        <h1 className="mt-3 text-3xl md:text-4xl font-bold tracking-tight">Model dashboard</h1>
        <p className="mt-2 text-muted-foreground max-w-2xl">
          Performance of each Model A variant on the held-out validation split (8 787 questions /
          35 148 option rows), plus Model B distractor + hint scores on a 200-sample subset.
        </p>
      </motion.div>

      {/* Metric cards */}
      <motion.div {...fadeUp} className="grid gap-4 grid-cols-2 lg:grid-cols-4">
        {cards.map((c) => (
          <Card key={c.label}>
            <CardContent className="p-5">
              <div className="flex items-center gap-2 text-muted-foreground">
                <c.icon className="h-4 w-4" />
                <span className="text-xs uppercase tracking-wide">{c.label}</span>
              </div>
              <div className="mt-2 text-2xl md:text-3xl font-bold tabular-nums tracking-tight">
                {busy ? <Skeleton className="h-8 w-24" /> : c.value}
              </div>
              <div className="mt-1 text-[11px] text-muted-foreground">{c.sub}</div>
            </CardContent>
          </Card>
        ))}
      </motion.div>

      {/* Bar chart */}
      <motion.div {...fadeUp}>
        <Card>
          <CardHeader>
            <CardTitle>Model A — Macro F1 vs Exact Match</CardTitle>
            <CardDescription>All three classifiers + the soft-vote ensemble. EM is the metric that matters.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              {busy ? (
                <Skeleton className="h-full w-full" />
              ) : (
                <ResponsiveContainer>
                  <BarChart data={chartData} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                    <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} domain={[0, 0.7]} />
                    <Tooltip
                      contentStyle={{
                        background: 'hsl(var(--card))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="Macro F1" fill="hsl(var(--primary))" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="Exact Match" fill="hsl(var(--accent))" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* CM + Model B */}
      <motion.div {...fadeUp} className="grid gap-4 lg:grid-cols-[1fr_1.4fr]">
        <Card>
          <CardHeader>
            <CardTitle>Confusion matrix — Ensemble</CardTitle>
            <CardDescription>Per-option (after per-question argmax).</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-[auto_1fr_1fr] gap-2 items-center">
              <div />
              <div className="text-center text-[11px] text-muted-foreground">pred 0</div>
              <div className="text-center text-[11px] text-muted-foreground">pred 1</div>
              {[0, 1].map((i) => (
                <FragmentRow key={i}>
                  <div className="text-right pr-2 text-[11px] text-muted-foreground">true {i}</div>
                  {[0, 1].map((j) => {
                    const v = cm[i]?.[j] ?? 0
                    const ratio = v / cmMax
                    return (
                      <div
                        key={j}
                        className="rounded-lg border border-border p-3 text-center font-mono text-sm tabular-nums"
                        style={{
                          background: `hsl(var(--primary) / ${0.05 + ratio * 0.5})`,
                          color: ratio > 0.6 ? 'hsl(var(--primary-foreground))' : 'hsl(var(--foreground))',
                        }}
                      >
                        {v.toLocaleString()}
                      </div>
                    )
                  })}
                </FragmentRow>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model B — distractors + hints</CardTitle>
            <CardDescription>200 val samples; n-gram overlap metrics.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {busy ? <Skeleton className="h-32 w-full" /> : (
              <>
                <Row label="BLEU"      value={formatNumber(bD.bleu, 4)} />
                <Row label="ROUGE-1 F" value={formatNumber(bD.rouge1_f, 4)} />
                <Row label="ROUGE-2 F" value={formatNumber(bD.rouge2_f, 4)} />
                <Row label="ROUGE-L F" value={formatNumber(bD.rougeL_f, 4)} />
                <Row label="METEOR"    value={formatNumber(bD.meteor, 4)} />
                <Row label="Distractor F1" value={formatNumber(bD.f1, 4)} />
                <div className="border-t border-border pt-3 mt-2 space-y-2">
                  <Row label="Hints — Precision @ 1" value={formatNumber(bH.precision_at_1, 4)} />
                  <Row label="Hints — Precision @ 3" value={formatNumber(bH.precision_at_3, 4)} />
                  <Row label="Hints — R² (scorer)"   value={formatNumber(bH.r2_scorer, 3)} muted />
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* System info */}
      <motion.div {...fadeUp}>
        <Card>
          <CardHeader>
            <CardTitle>System</CardTitle>
            <CardDescription>Reproducibility surface.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Stat icon={Database} label="Vocab size"        value="20 000" sub="TF-IDF unigrams + bigrams" />
              <Stat icon={Layers}   label="Train option rows" value="281 168" sub="70 292 questions × 4" />
              <Stat icon={Server}   label="Models"            value="3 + 1"   sub="LR · SVC · NB · Ensemble" />
              <Stat icon={Activity} label="Frameworks"        value="sklearn 1.6.1" sub="pandas 2.2.3 · numpy 2.2.2" />
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  )
}

function FragmentRow({ children }) {
  return <>{children}</>
}

function Row({ label, value, muted }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className={`font-mono text-sm tabular-nums ${muted ? 'text-muted-foreground' : 'text-foreground'}`}>{value}</span>
    </div>
  )
}

function Stat({ icon: Icon, label, value, sub }) {
  return (
    <div className="rounded-xl border border-border bg-card/60 p-4">
      <div className="flex items-center gap-2 text-muted-foreground">
        <Icon className="h-3.5 w-3.5" />
        <span className="text-[11px] uppercase tracking-wide">{label}</span>
      </div>
      <div className="mt-1 text-base font-semibold">{value}</div>
      <div className="text-[11px] text-muted-foreground">{sub}</div>
    </div>
  )
}
