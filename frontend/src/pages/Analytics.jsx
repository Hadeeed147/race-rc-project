import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  CartesianGrid,
} from 'recharts'
import {
  Activity, Target, Award, Timer, Zap, Database, Server, Layers,
  Wand2, BrainCircuit, Sparkles, CheckCircle2,
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
      ; (async () => {
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
      label: 'Acc (Verification)', icon: Target,
      value: ensembleRow ? formatPercent(ensembleRow.accuracy, 1) : '—',
      sub: ensembleRow ? `Ensemble vs LR: +${formatNumber((ensembleRow.accuracy - (lrRow?.accuracy || 0)), 4)}` : 'val set',
    },
    {
      label: 'Macro F1', icon: Activity,
      value: ensembleRow ? formatNumber(ensembleRow.macro_f1, 4) : '—',
      sub: ensembleRow ? 'per-question argmax' : 'val set',
    },
    {
      label: 'Question BLEU', icon: Sparkles,
      value: data?.model_a_gen?.bleu ? formatNumber(data.model_a_gen.bleu, 4) : '—',
      sub: `ROUGE-L: ${data?.model_a_gen?.rouge_l ? formatNumber(data.model_a_gen.rouge_l, 3) : '—'}`,
    },
    {
      label: 'Label Prop F1', icon: BrainCircuit,
      value: data?.unsupervised?.label_prop?.f1_lp ? formatNumber(data.unsupervised.label_prop.f1_lp, 4) : '—',
      sub: `Gain: +${data?.unsupervised?.label_prop?.improvement ? formatNumber(data.unsupervised.label_prop.improvement, 1) : '0'}% vs small LR`,
    },
  ]

  // BarChart data
  const chartData = (data?.model_a || []).map((m) => ({
    name: m.name?.split('(')[0]?.trim() || m.name,
    'Macro F1': Number(m.macro_f1?.toFixed(4) ?? 0),
    'Exact Match': Number(m.exact_match?.toFixed(4) ?? 0),
  }))

  // Confusion matrix
  const cm = ensembleRow?.confusion_matrix || lrRow?.confusion_matrix || [[0, 0], [0, 0]]
  const flat = cm.flat()
  const cmMax = Math.max(...flat) || 1

  const b = data?.model_b
  const bD = b?.distractors || {}
  const bH = b?.hints || {}
  const genOurs = data?.model_a_gen || {}
  const genT5 = data?.baselines?.t5_small || {}

  return (
    <motion.div variants={stagger} initial="initial" animate="animate"
      className="mx-auto w-full max-w-6xl px-4 md:px-6 py-8 md:py-12 space-y-8">

      {/* Header */}
      <motion.div {...fadeUp} className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Badge variant="default" className="gap-1.5"><Zap className="h-3 w-3" /> Live performance dashboard</Badge>
          </div>
          <h1 className="mt-3 text-3xl md:text-4xl font-bold tracking-tight">Analytics Dashboard</h1>
          <p className="mt-2 text-muted-foreground max-w-2xl">
            Formal evaluation results across the RACE test/val splits. Comparison between
            classical ensemble models and neural baselines.
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs text-muted-foreground bg-muted/30 px-3 py-2 rounded-lg border border-border">
          <div className="flex items-center gap-1.5"><CheckCircle2 className="h-3.5 w-3.5 text-success" /> Verification</div>
          <div className="flex items-center gap-1.5"><Sparkles className="h-3.5 w-3.5 text-primary" /> Generation</div>
        </div>
      </motion.div>

      {/* Metric Cards */}
      <motion.div {...fadeUp} className="grid gap-4 grid-cols-2 lg:grid-cols-4">
        {cards.map((c) => (
          <Card key={c.label} className="overflow-hidden">
            <CardContent className="p-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <c.icon className="h-4 w-4" />
                  <span className="text-xs uppercase tracking-widest font-medium">{c.label}</span>
                </div>
              </div>
              <div className="mt-2 text-2xl md:text-3xl font-bold tabular-nums tracking-tight">
                {busy ? <Skeleton className="h-8 w-24" /> : c.value}
              </div>
              <div className="mt-1 text-[11px] text-muted-foreground flex items-center gap-1">
                {c.sub}
              </div>
            </CardContent>
          </Card>
        ))}
      </motion.div>

      {/* Section 1: Unsupervised Learning (Rubric 4.2.2) */}
      <motion.div {...fadeUp} className="space-y-4">
        <div className="flex items-center gap-2">
          <Database className="h-5 w-5 text-success" />
          <h2 className="text-xl font-bold">Model A: Unsupervised & Semi-Supervised</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-success">K-Means Clustering</CardTitle>
              <CardDescription>Latent pattern discovery in Q-A pairs</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-end">
                <div>
                  <div className="text-2xl font-bold">{formatNumber(data?.unsupervised?.kmeans?.silhouette, 4)}</div>
                  <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Silhouette Score</div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold">{formatPercent(data?.unsupervised?.kmeans?.purity, 1)}</div>
                  <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Cluster Purity</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-success">Label Propagation</CardTitle>
              <CardDescription>Semi-supervised learning improvement</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-end">
                <div>
                  <div className="text-2xl font-bold">{formatNumber(data?.unsupervised?.label_prop?.f1_lp, 4)}</div>
                  <div className="text-[10px] text-muted-foreground uppercase tracking-wider">LP Macro F1</div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-success">+{formatNumber(data?.unsupervised?.label_prop?.improvement, 1)}%</div>
                  <div className="text-[10px] text-muted-foreground uppercase tracking-wider text-success">Gain vs small-supervised</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </motion.div>

      {/* Section 2: Verification Section */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <BrainCircuit className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-bold">Model A: Option Verification</h2>
          <Badge variant="outline" className="ml-auto">Supervised Ensemble</Badge>
        </div>

        <motion.div {...fadeUp} className="grid gap-4 lg:grid-cols-[1.5fr_1fr]">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Classifier Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[280px] w-full mt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                    <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} />
                    <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} domain={[0, 0.7]} />
                    <Tooltip cursor={{ fill: 'hsl(var(--muted) / 0.4)' }} contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 8, fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
                    <Bar dataKey="Macro F1" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} barSize={40} />
                    <Bar dataKey="Exact Match" fill="hsl(var(--accent))" radius={[4, 4, 0, 0]} barSize={40} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Confusion Matrix</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-[auto_1fr_1fr] gap-2 items-center mt-6">
                <div />
                <div className="text-center text-[10px] uppercase tracking-wider text-muted-foreground">pred 0</div>
                <div className="text-center text-[10px] uppercase tracking-wider text-muted-foreground">pred 1</div>
                {[0, 1].map((i) => (
                  <div key={i} className="contents">
                    <div className="text-right pr-2 text-[10px] uppercase tracking-wider text-muted-foreground">true {i}</div>
                    {[0, 1].map((j) => {
                      const v = cm[i]?.[j] ?? 0
                      const ratio = v / cmMax
                      return (
                        <div key={j} className="rounded-lg border border-border p-4 text-center font-mono text-sm tabular-nums" style={{ background: `hsl(var(--primary) / ${0.03 + ratio * 0.4})`, borderLeft: ratio > 0.5 ? '2px solid hsl(var(--primary))' : '1px solid hsl(var(--border))' }}>
                          {v.toLocaleString()}
                        </div>
                      )
                    })}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Section 3: Text Generation (Family 2) */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-bold">Generation Tasks (Model A + B)</h2>
          <Badge variant="outline" className="ml-auto">ROUGE / BLEU / METEOR</Badge>
        </div>

        <motion.div {...fadeUp} className="grid gap-4 lg:grid-cols-2">
          {/* Model A Question Generation */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base text-primary">Model A: Wh-Question Generation</CardTitle>
              <CardDescription>Template-based Wh-Inversion vs Neural baseline.</CardDescription>
            </CardHeader>
            <CardContent>
              <MetricTable
                headers={['Metric', 'Ours (Improved)', 'Baseline (T5)']}
                rows={[
                  ['BLEU', genOurs.bleu, genT5.bleu],
                  ['ROUGE-L', genOurs.rouge_l, genT5.rouge_l],
                  ['METEOR', genOurs.meteor, genT5.meteor],
                ]}
              />
            </CardContent>
          </Card>

          {/* Model B Distractors + Hints */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base text-primary">Model B: Distractors & Hints</CardTitle>
              <CardDescription>Quality of distractor spans and hint relevance.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <MetricTable
                headers={['Metric (Distractors)', 'Score']}
                rows={[
                  ['BLEU', bD.bleu],
                  ['ROUGE-L', bD.rougeL_f],
                  ['METEOR', bD.meteor],
                  ['F1 Token', bD.f1],
                ]}
              />
              <MetricTable
                headers={['Metric (Hints)', 'Score']}
                rows={[
                  ['Prec @ 1', bH.precision_at_1],
                  ['Prec @ 3', bH.precision_at_3],
                  ['R² Scorer', bH.r2_scorer, true],
                ]}
              />
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Footer / System Info */}
      <motion.div {...fadeUp}>
        <Card className="bg-muted/10 border-dashed">
          <CardContent className="p-4">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Stat icon={Database} label="Vocab size" value="20 000" sub="TF-IDF unigrams + bigrams" />
              <Stat icon={Layers} label="Total Rows" value="316 316" sub="Option-level training set" />
              <Stat icon={Timer} label="Avg Latency" value={avgLatency != null ? `${avgLatency} ms` : '—'} sub="API response time" />
              <Stat icon={Activity} label="Environment" value="Python 3.12" sub="scikit-learn · FastAPI" />
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  )
}

function MetricTable({ headers, rows }) {
  return (
    <div className="rounded-lg border border-border overflow-hidden mb-2">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-muted/50 text-left border-b border-border">
            {headers.map((h, i) => (
              <th key={i} className={`p-2 font-medium text-[10px] uppercase tracking-widest ${i > 0 ? 'text-right' : ''}`}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((row, i) => (
            <tr key={i} className="hover:bg-muted/20 transition-colors">
              {row.map((cell, j) => {
                if (typeof cell === 'boolean') return null;
                const isLabel = j === 0;
                return (
                  <td key={j} className={`p-2 font-mono text-[11px] tabular-nums ${isLabel ? 'text-muted-foreground font-sans' : 'text-foreground font-semibold text-right'}`}>
                    {isLabel ? cell : formatNumber(cell, cell > 1 ? 1 : 4)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Stat({ icon: Icon, label, value, sub }) {
  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-2 text-muted-foreground mb-1">
        <Icon className="h-3.5 w-3.5" />
        <span className="text-[10px] uppercase tracking-wider font-semibold">{label}</span>
      </div>
      <div className="text-lg font-bold tabular-nums">{value}</div>
      <div className="text-[10px] text-muted-foreground leading-tight">{sub}</div>
    </div>
  )
}
