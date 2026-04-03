'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { Activity, ArrowLeft, BarChart3, Bug, Database, FileText, Gauge, Layers3, Loader2, Plus, RefreshCw, Search } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'

interface Document {
  id: string
  name: string
  pages: number
  chunk_count: number
}

interface DocumentDiagnostics {
  chunk_count: number
  avg_chunk_tokens: number
  too_short_chunk_count: number
  too_long_chunk_count: number
  duplicate_pair_count: number
  code_block_cut_count: number
}

interface TraceSummary {
  trace_id: string
  created_at: string
  status: string
  question: string
  retrieval_method?: string
  final_chunks?: number
  total_candidates?: number
  total_latency_ms?: number
}

interface TraceDetail extends TraceSummary {
  answer?: string
  spans?: Array<{
    span_id: string
    name: string
    type: string
    latency_ms: number
  }>
  candidates?: Array<{
    chunk_id: string
    stage: string
    rank: number
    chunk_index?: number
    rerank_score?: number
    selected_for_context?: boolean
    content_preview: string
  }>
  final_context?: Array<{
    rank: number
    chunk_id: string
    chunk_index?: number
    score?: number
    content_preview: string
  }>
}

interface EvaluationMetric {
  hit_at_3: number
  hit_at_5: number
  hit_at_10: number
  mrr: number
  avg_response_time: number
  errors: number
}

interface EvaluationRunSummary {
  run_id: string
  timestamp: string
  dataset_id?: string
  dataset_name?: string
  document_id?: string
  methods: string[]
  metrics: Record<string, EvaluationMetric>
  case_count: number
  status: string
}

interface EvaluationRunDetail extends EvaluationRunSummary {
  document_name?: string
  duration_seconds?: number
  results: Record<string, Array<{
    test_case_id: string
    question: string
    retrieved_chunks: string[]
    hit: boolean
    rank?: number
    response_time: number
  }>>
}

interface EvaluationCase {
  case_id: string
  question: string
  expected_keywords?: string[]
  enabled: boolean
}

interface EvaluationDataset {
  dataset_id: string
  name: string
  description?: string
  document_id?: string
  enabled: boolean
  case_count?: number
  enabled_case_count?: number
  cases?: EvaluationCase[]
}

interface Badcase {
  badcase_id: string
  trace_id?: string
  question: string
  failure_type: string
  severity: string
  expected_behavior?: string
  status: string
  created_at: string
}

interface DashboardSummary {
  online_quality: {
    trace_count: number
    success_count: number
    error_count: number
    success_rate: number
    avg_latency_ms: number
    p95_latency_ms: number
    avg_final_chunks: number
    avg_total_candidates: number
    avg_retrieval_seconds: number
    avg_llm_seconds: number
    total_tokens: number
    estimated_cost: number
    priced_trace_count: number
  }
  knowledge_base: {
    document_count: number
    chunk_count: number
    avg_chunks_per_doc: number
    too_short_chunk_count: number
    too_long_chunk_count: number
    duplicate_pair_count: number
    code_block_cut_count: number
    empty_chunk_count: number
  }
  offline_evaluation: {
    run_count: number
    latest_run_id?: string
    latest_dataset_name?: string
    latest_case_count: number
    best_method?: string
    best_hit_at_5: number
    best_mrr: number
  }
  badcases: {
    total_count: number
    open_count: number
    resolved_count: number
    by_failure_type: Record<string, number>
  }
}

export default function RAGOpsPage() {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

  const [dashboard, setDashboard] = useState<DashboardSummary | null>(null)
  const [documents, setDocuments] = useState<Document[]>([])
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null)
  const [scopeMode, setScopeMode] = useState<'document' | 'all'>('document')
  const [diagnostics, setDiagnostics] = useState<DocumentDiagnostics | null>(null)
  const [traces, setTraces] = useState<TraceSummary[]>([])
  const [selectedTrace, setSelectedTrace] = useState<TraceDetail | null>(null)
  const [datasets, setDatasets] = useState<EvaluationDataset[]>([])
  const [selectedDataset, setSelectedDataset] = useState<EvaluationDataset | null>(null)
  const [evaluationRuns, setEvaluationRuns] = useState<EvaluationRunSummary[]>([])
  const [selectedEvaluationRun, setSelectedEvaluationRun] = useState<EvaluationRunDetail | null>(null)
  const [badcases, setBadcases] = useState<Badcase[]>([])
  const [datasetName, setDatasetName] = useState('')
  const [caseQuestion, setCaseQuestion] = useState('')
  const [caseKeywords, setCaseKeywords] = useState('')
  const [badcaseExpected, setBadcaseExpected] = useState('')
  const [isRunningEvaluation, setIsRunningEvaluation] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)

  const selectedDocument = documents.find(document => document.id === selectedDocumentId)
  const activeDocumentId = scopeMode === 'document' ? selectedDocumentId : null
  const latestRun = selectedEvaluationRun || evaluationRuns[0]
  const latestMetrics = latestRun?.metrics || {}

  const overview = useMemo(() => {
    const traceCount = traces.length
    const avgLatency = traceCount
      ? Math.round(traces.reduce((sum, trace) => sum + (trace.total_latency_ms || 0), 0) / traceCount)
      : 0
    const badcaseOpenCount = badcases.filter(item => item.status === 'open').length
    const bestHitAt5 = Math.max(0, ...Object.values(latestMetrics).map(metric => metric.hit_at_5 || 0))
    return {
      traceCount,
      avgLatency,
      badcaseOpenCount,
      bestHitAt5,
    }
  }, [badcases, latestMetrics, traces])

  const formatPercent = (value?: number) => `${Math.round((value || 0) * 100)}%`
  const formatCurrency = (value?: number) => `$${(value || 0).toFixed(4)}`

  const loadDashboard = async () => {
    const response = await fetch(`${backendUrl}/ops/dashboard`)
    const data = await response.json()
    if (data.success) setDashboard(data.data)
  }

  const loadDocuments = async () => {
    const response = await fetch(`${backendUrl}/documents`)
    const data = await response.json()
    if (!data.success) return
    const items = data.data || []
    setDocuments(items)
    setSelectedDocumentId(previous => previous || items[0]?.id || null)
  }

  const loadDiagnostics = async (documentId: string | null) => {
    if (!documentId) {
      setDiagnostics(null)
      return
    }
    const response = await fetch(`${backendUrl}/documents/${documentId}/diagnostics`)
    const data = await response.json()
    if (data.success) setDiagnostics(data.data.diagnostics)
  }

  const loadTraces = async () => {
    const response = await fetch(`${backendUrl}/ops/traces?limit=40`)
    const data = await response.json()
    if (data.success) setTraces(data.data || [])
  }

  const loadTraceDetail = async (traceId: string) => {
    const response = await fetch(`${backendUrl}/ops/traces/${traceId}`)
    const data = await response.json()
    if (data.success) setSelectedTrace(data.data)
  }

  const loadDatasets = async () => {
    const response = await fetch(`${backendUrl}/evaluation/datasets`)
    const data = await response.json()
    if (!data.success) return
    const items = data.data || []
    setDatasets(items)
    if (!selectedDataset && items[0]) loadDatasetDetail(items[0].dataset_id)
  }

  const loadDatasetDetail = async (datasetId: string) => {
    const response = await fetch(`${backendUrl}/evaluation/datasets/${datasetId}`)
    const data = await response.json()
    if (data.success) setSelectedDataset(data.data)
  }

  const loadEvaluationRuns = async () => {
    const response = await fetch(`${backendUrl}/evaluation/runs?limit=20`)
    const data = await response.json()
    if (data.success) setEvaluationRuns(data.data || [])
  }

  const loadEvaluationRunDetail = async (runId: string) => {
    const response = await fetch(`${backendUrl}/evaluation/runs/${runId}`)
    const data = await response.json()
    if (data.success) setSelectedEvaluationRun(data.data)
  }

  const loadBadcases = async () => {
    const response = await fetch(`${backendUrl}/ops/badcases?limit=50`)
    const data = await response.json()
    if (data.success) setBadcases(data.data || [])
  }

  const refreshAll = async () => {
    setIsRefreshing(true)
    try {
      await Promise.all([
        loadDashboard(),
        loadDocuments(),
        loadTraces(),
        loadDatasets(),
        loadEvaluationRuns(),
        loadBadcases(),
      ])
    } finally {
      setIsRefreshing(false)
    }
  }

  const createDataset = async () => {
    const name = datasetName.trim()
    if (!name) return
    const response = await fetch(`${backendUrl}/evaluation/datasets`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, document_id: activeDocumentId }),
    })
    const data = await response.json()
    if (!data.success) return
    setDatasetName('')
    await loadDatasets()
    await loadDatasetDetail(data.data.dataset_id)
  }

  const addCase = async () => {
    if (!selectedDataset || !caseQuestion.trim()) return
    const response = await fetch(`${backendUrl}/evaluation/datasets/${selectedDataset.dataset_id}/cases`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: caseQuestion.trim(),
        document_id: activeDocumentId,
        expected_keywords: caseKeywords.split(',').map(item => item.trim()).filter(Boolean),
      }),
    })
    const data = await response.json()
    if (!data.success) return
    setCaseQuestion('')
    setCaseKeywords('')
    await loadDatasets()
    await loadDatasetDetail(selectedDataset.dataset_id)
  }

  const toggleCase = async (item: EvaluationCase) => {
    if (!selectedDataset) return
    await fetch(`${backendUrl}/evaluation/datasets/${selectedDataset.dataset_id}/cases/${item.case_id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: !item.enabled }),
    })
    await loadDatasets()
    await loadDatasetDetail(selectedDataset.dataset_id)
  }

  const runEvaluation = async () => {
    if ((scopeMode === 'document' && !selectedDocumentId) || !selectedDataset || isRunningEvaluation) return
    setIsRunningEvaluation(true)
    try {
      const response = await fetch(`${backendUrl}/evaluation/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: activeDocumentId,
          test_set_id: selectedDataset.dataset_id,
          methods: ['baseline', 'hybrid', 'hybrid_rerank'],
        }),
      })
      const data = await response.json()
      if (data.success) {
        await loadDashboard()
        await loadEvaluationRuns()
        await loadEvaluationRunDetail(data.data.run_id)
      }
    } finally {
      setIsRunningEvaluation(false)
    }
  }

  const markBadcase = async () => {
    if (!selectedTrace) return
    const response = await fetch(`${backendUrl}/ops/badcases`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        trace_id: selectedTrace.trace_id,
        failure_type: 'other',
        severity: 'medium',
        expected_behavior: badcaseExpected.trim(),
      }),
    })
    const data = await response.json()
    if (data.success) {
      setBadcaseExpected('')
      await loadBadcases()
      await loadDashboard()
    }
  }

  useEffect(() => {
    refreshAll()
  }, [])

  useEffect(() => {
    loadDiagnostics(activeDocumentId)
  }, [activeDocumentId])

  return (
    <div className="min-h-screen bg-[#eef2f1] text-slate-950">
      <header className="border-b border-slate-200 bg-[#fbfcfb]/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-5 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-600 hover:text-slate-950">
              <ArrowLeft className="h-4 w-4" />
            </Link>
            <div>
              <div className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-emerald-600" />
                <h1 className="text-xl font-semibold tracking-tight">RAGOps Console</h1>
              </div>
              <p className="mt-1 text-sm text-slate-500">在线质量观测 · 知识库质量治理 · 离线基准与回归评测</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <ToggleGroup
              type="single"
              value={scopeMode}
              onValueChange={(value) => {
                if (value === 'document' || value === 'all') setScopeMode(value)
              }}
              className="grid w-[260px] grid-cols-2"
              variant="outline"
              size="sm"
            >
              <ToggleGroupItem value="document" className="bg-white text-xs">Current document</ToggleGroupItem>
              <ToggleGroupItem value="all" className="bg-white text-xs">All documents</ToggleGroupItem>
            </ToggleGroup>
            <select
              value={selectedDocumentId || ''}
              onChange={(event) => setSelectedDocumentId(event.target.value || null)}
              disabled={scopeMode === 'all'}
              className="h-9 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-700"
            >
              {documents.map(document => (
                <option key={document.id} value={document.id}>{document.name}</option>
              ))}
            </select>
            <Button variant="outline" className="bg-white" onClick={refreshAll} disabled={isRefreshing}>
              {isRefreshing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
              Refresh
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        <section className="grid gap-3 md:grid-cols-4">
          <MetricCard
            icon={<Search className="h-4 w-4" />}
            label="Recent traces"
            value={(dashboard?.online_quality.trace_count ?? overview.traceCount).toString()}
            hint={`${dashboard?.online_quality.avg_latency_ms ?? overview.avgLatency}ms avg · p95 ${dashboard?.online_quality.p95_latency_ms ?? 0}ms`}
          />
          <MetricCard
            icon={<Gauge className="h-4 w-4" />}
            label="Knowledge chunks"
            value={(scopeMode === 'all' ? dashboard?.knowledge_base.chunk_count ?? documents.reduce((sum, document) => sum + document.chunk_count, 0) : diagnostics?.chunk_count || selectedDocument?.chunk_count || 0).toString()}
            hint={scopeMode === 'all' ? `${dashboard?.knowledge_base.document_count ?? documents.length} documents` : `${diagnostics?.avg_chunk_tokens || 0} avg tokens`}
          />
          <MetricCard
            icon={<BarChart3 className="h-4 w-4" />}
            label="Best Hit@5"
            value={formatPercent(dashboard?.offline_evaluation.best_hit_at_5 ?? overview.bestHitAt5)}
            hint={dashboard?.offline_evaluation.best_method || latestRun?.run_id || 'No eval run'}
          />
          <MetricCard
            icon={<Bug className="h-4 w-4" />}
            label="Open badcases"
            value={(dashboard?.badcases.open_count ?? overview.badcaseOpenCount).toString()}
            hint={`${dashboard?.badcases.total_count ?? badcases.length} total`}
          />
        </section>

        <Tabs defaultValue="online" className="mt-6">
          <TabsList className="grid h-auto w-full grid-cols-3 bg-slate-200/70 p-1">
            <TabsTrigger value="online" className="text-xs sm:text-sm">Online Quality</TabsTrigger>
            <TabsTrigger value="kb" className="text-xs sm:text-sm">Knowledge Base</TabsTrigger>
            <TabsTrigger value="offline" className="text-xs sm:text-sm">Offline Evaluation</TabsTrigger>
          </TabsList>

          <TabsContent value="online" className="mt-4">
            <div className="grid gap-4 lg:grid-cols-[360px_1fr]">
              <Panel title="Trace Stream" description="真实请求链路和候选上下文">
                <ScrollArea className="h-[520px] pr-3">
                  <div className="space-y-2">
                    {traces.map(trace => (
                      <button
                        key={trace.trace_id}
                        type="button"
                        className={`w-full rounded-lg border p-3 text-left text-sm transition-colors ${
                          selectedTrace?.trace_id === trace.trace_id ? 'border-slate-950 bg-white' : 'border-slate-200 bg-slate-50 hover:bg-white'
                        }`}
                        onClick={() => loadTraceDetail(trace.trace_id)}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="line-clamp-2 font-medium text-slate-800">{trace.question}</span>
                          <Badge variant="outline">{trace.status}</Badge>
                        </div>
                        <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-500">
                          {trace.retrieval_method && <span>{trace.retrieval_method}</span>}
                          {typeof trace.total_latency_ms === 'number' && <span>{trace.total_latency_ms}ms</span>}
                          {typeof trace.final_chunks === 'number' && <span>{trace.final_chunks}/{trace.total_candidates} chunks</span>}
                        </div>
                      </button>
                    ))}
                    {traces.length === 0 && <EmptyState text="暂无 trace，先在首页发送一次问题" />}
                  </div>
                </ScrollArea>
              </Panel>

              <div className="space-y-4">
                <Panel title="Trace Detail" description={selectedTrace ? `${selectedTrace.trace_id} · ${selectedTrace.question}` : '选择一条 trace 查看详情'}>
                  {selectedTrace ? (
                    <div className="space-y-4">
                      <div className="flex flex-col gap-2 sm:flex-row">
                        <Input value={badcaseExpected} onChange={(event) => setBadcaseExpected(event.target.value)} placeholder="标注期望行为" />
                        <Button variant="outline" className="bg-white" onClick={markBadcase}>
                          <Bug className="mr-2 h-4 w-4" />
                          Mark badcase
                        </Button>
                      </div>
                      <div className="grid gap-4 lg:grid-cols-3">
                        <TraceColumn title="Timeline">
                          {(selectedTrace.spans || []).map(span => (
                            <div key={span.span_id} className="flex items-center justify-between rounded-md bg-slate-50 px-3 py-2 text-xs">
                              <span className="truncate text-slate-700">{span.name}</span>
                              <span className="text-slate-400">{span.latency_ms}ms</span>
                            </div>
                          ))}
                        </TraceColumn>
                        <TraceColumn title="Final Context">
                          {(selectedTrace.final_context || []).map(item => (
                            <div key={item.chunk_id} className="rounded-md bg-slate-50 p-3 text-xs">
                              <div className="mb-1 flex items-center gap-2 text-slate-500">
                                <Badge variant="outline">#{item.rank}</Badge>
                                {typeof item.chunk_index === 'number' && <span>chunk {item.chunk_index + 1}</span>}
                              </div>
                              <p className="line-clamp-4 text-slate-700">{item.content_preview}</p>
                            </div>
                          ))}
                        </TraceColumn>
                        <TraceColumn title="Candidates">
                          {(selectedTrace.candidates || []).slice(0, 30).map((item, index) => (
                            <div key={`${item.stage}-${item.chunk_id}-${index}`} className="rounded-md bg-slate-50 p-3 text-xs">
                              <div className="mb-1 flex flex-wrap items-center gap-2 text-slate-500">
                                <Badge variant={item.selected_for_context ? 'default' : 'outline'}>{item.stage}</Badge>
                                <span>rank {item.rank}</span>
                              </div>
                              <p className="line-clamp-3 text-slate-700">{item.content_preview}</p>
                            </div>
                          ))}
                        </TraceColumn>
                      </div>
                    </div>
                  ) : (
                    <EmptyState text="从左侧选择一条 trace" />
                  )}
                </Panel>

                <Panel title="Online Summary" description="请求质量、延迟和成本概览">
                  <div className="grid gap-3 md:grid-cols-4">
                    <SummaryCell label="Success rate" value={formatPercent(dashboard?.online_quality.success_rate)} />
                    <SummaryCell label="Errors" value={(dashboard?.online_quality.error_count || 0).toString()} />
                    <SummaryCell label="Avg context" value={`${dashboard?.online_quality.avg_final_chunks || 0}/${dashboard?.online_quality.avg_total_candidates || 0}`} />
                    <SummaryCell label="Est. cost" value={formatCurrency(dashboard?.online_quality.estimated_cost)} />
                    <SummaryCell label="Retrieval avg" value={`${dashboard?.online_quality.avg_retrieval_seconds || 0}s`} />
                    <SummaryCell label="LLM avg" value={`${dashboard?.online_quality.avg_llm_seconds || 0}s`} />
                    <SummaryCell label="Tokens" value={(dashboard?.online_quality.total_tokens || 0).toString()} />
                    <SummaryCell label="Priced traces" value={(dashboard?.online_quality.priced_trace_count || 0).toString()} />
                  </div>
                </Panel>

                <Panel title="Badcases" description="真实请求失败样例池">
                  <div className="grid gap-3 md:grid-cols-2">
                    {badcases.map(item => (
                      <div key={item.badcase_id} className="rounded-lg border border-slate-200 bg-white p-3 text-sm">
                        <div className="mb-2 flex items-center justify-between gap-2">
                          <Badge variant="outline">{item.failure_type}</Badge>
                          <span className="text-xs text-slate-400">{item.status}</span>
                        </div>
                        <p className="line-clamp-2 font-medium text-slate-800">{item.question}</p>
                        {item.expected_behavior && <p className="mt-2 line-clamp-2 text-xs text-slate-500">{item.expected_behavior}</p>}
                      </div>
                    ))}
                    {badcases.length === 0 && <EmptyState text="暂无 badcase" />}
                  </div>
                </Panel>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="kb" className="mt-4">
            <div className="grid gap-4 lg:grid-cols-[360px_1fr]">
              <Panel title="Documents" description="选择文档查看 chunk 质量">
                <div className="space-y-2">
                  {documents.map(document => (
                    <button
                      key={document.id}
                      type="button"
                      className={`w-full rounded-lg border p-3 text-left text-sm ${
                        selectedDocumentId === document.id ? 'border-slate-950 bg-white' : 'border-slate-200 bg-slate-50 hover:bg-white'
                      }`}
                      onClick={() => setSelectedDocumentId(document.id)}
                    >
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-blue-500" />
                        <span className="truncate font-medium text-slate-800">{document.name}</span>
                      </div>
                      <div className="mt-2 flex gap-3 text-xs text-slate-500">
                        <span>{document.pages} pages</span>
                        <span>{document.chunk_count} chunks</span>
                      </div>
                    </button>
                  ))}
                </div>
              </Panel>

              <Panel title="Knowledge Base Quality" description={scopeMode === 'all' ? '全部文档概览' : selectedDocument?.name || '选择文档'}>
                {scopeMode === 'all' && dashboard ? (
                  <div className="grid gap-3 md:grid-cols-3">
                    <MetricCard icon={<FileText className="h-4 w-4" />} label="Documents" value={dashboard.knowledge_base.document_count.toString()} hint={`${dashboard.knowledge_base.avg_chunks_per_doc} chunks/doc`} />
                    <MetricCard icon={<Layers3 className="h-4 w-4" />} label="Chunks" value={dashboard.knowledge_base.chunk_count.toString()} hint="indexed units" />
                    <MetricCard icon={<Database className="h-4 w-4" />} label="Duplicates" value={dashboard.knowledge_base.duplicate_pair_count.toString()} hint="similar chunk pairs" />
                    <MetricCard icon={<FileText className="h-4 w-4" />} label="Code cuts" value={dashboard.knowledge_base.code_block_cut_count.toString()} hint="broken code blocks" />
                    <MetricCard icon={<Search className="h-4 w-4" />} label="Too short" value={dashboard.knowledge_base.too_short_chunk_count.toString()} hint="weak semantic units" />
                    <MetricCard icon={<Search className="h-4 w-4" />} label="Too long" value={dashboard.knowledge_base.too_long_chunk_count.toString()} hint="high context cost" />
                  </div>
                ) : diagnostics ? (
                  <div className="grid gap-3 md:grid-cols-3">
                    <MetricCard icon={<Layers3 className="h-4 w-4" />} label="Chunks" value={diagnostics.chunk_count.toString()} hint="indexed units" />
                    <MetricCard icon={<Gauge className="h-4 w-4" />} label="Avg tokens" value={diagnostics.avg_chunk_tokens.toString()} hint="chunk size" />
                    <MetricCard icon={<Database className="h-4 w-4" />} label="Duplicates" value={diagnostics.duplicate_pair_count.toString()} hint="similar chunk pairs" />
                    <MetricCard icon={<FileText className="h-4 w-4" />} label="Code cuts" value={diagnostics.code_block_cut_count.toString()} hint="broken code blocks" />
                    <MetricCard icon={<Search className="h-4 w-4" />} label="Too short" value={diagnostics.too_short_chunk_count.toString()} hint="weak semantic units" />
                    <MetricCard icon={<Search className="h-4 w-4" />} label="Too long" value={diagnostics.too_long_chunk_count.toString()} hint="high context cost" />
                  </div>
                ) : (
                  <EmptyState text="暂无文档诊断数据" />
                )}
              </Panel>
            </div>
          </TabsContent>

          <TabsContent value="offline" className="mt-4">
            <div className="grid gap-4 lg:grid-cols-[360px_1fr]">
              <Panel title="Datasets" description="维护可周期运行的评测集">
                <div className="mb-3 flex gap-2">
                  <Input value={datasetName} onChange={(event) => setDatasetName(event.target.value)} placeholder="新评测集名称" />
                  <Button variant="outline" className="bg-white" onClick={createDataset}>
                    <Plus className="h-4 w-4" />
                  </Button>
                </div>
                <div className="space-y-2">
                  {datasets.map(dataset => (
                    <button
                      key={dataset.dataset_id}
                      type="button"
                      className={`w-full rounded-lg border p-3 text-left text-sm ${
                        selectedDataset?.dataset_id === dataset.dataset_id ? 'border-slate-950 bg-white' : 'border-slate-200 bg-slate-50 hover:bg-white'
                      }`}
                      onClick={() => loadDatasetDetail(dataset.dataset_id)}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate font-medium text-slate-800">{dataset.name}</span>
                        <Badge variant="outline">{dataset.enabled_case_count ?? dataset.case_count ?? 0}</Badge>
                      </div>
                      <div className="mt-1 truncate text-xs text-slate-400">{dataset.dataset_id}</div>
                    </button>
                  ))}
                </div>
              </Panel>

              <div className="space-y-4">
                <Panel title="Dataset Cases" description={selectedDataset?.name || '选择评测集'}>
                  {selectedDataset ? (
                    <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
                      <ScrollArea className="h-[360px] pr-3">
                        <div className="space-y-2">
                          {(selectedDataset.cases || []).map(item => (
                            <button key={item.case_id} type="button" className="w-full rounded-lg bg-slate-50 p-3 text-left text-sm hover:bg-slate-100" onClick={() => toggleCase(item)}>
                              <div className="flex items-start justify-between gap-2">
                                <span className="line-clamp-2 font-medium text-slate-800">{item.question}</span>
                                <Badge variant={item.enabled ? 'default' : 'outline'}>{item.enabled ? 'on' : 'off'}</Badge>
                              </div>
                              <p className="mt-2 truncate text-xs text-slate-400">{(item.expected_keywords || []).join(', ') || 'no keywords'}</p>
                            </button>
                          ))}
                        </div>
                      </ScrollArea>
                      <div className="space-y-2">
                        <Textarea value={caseQuestion} onChange={(event) => setCaseQuestion(event.target.value)} placeholder="新增评测问题" />
                        <Input value={caseKeywords} onChange={(event) => setCaseKeywords(event.target.value)} placeholder="期望关键词，用英文逗号分隔" />
                        <Button className="w-full bg-slate-950 text-white hover:bg-slate-800" onClick={addCase}>Add case</Button>
                        <Button className="w-full" variant="outline" onClick={runEvaluation} disabled={(scopeMode === 'document' && !selectedDocumentId) || isRunningEvaluation}>
                          {isRunningEvaluation ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <BarChart3 className="mr-2 h-4 w-4" />}
                          Run evaluation
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <EmptyState text="选择或创建一个评测集" />
                  )}
                </Panel>

                <Panel title="Evaluation Runs" description="策略基准和回归结果">
                  {dashboard && (
                    <div className="mb-4 grid gap-3 md:grid-cols-4">
                      <SummaryCell label="Runs" value={dashboard.offline_evaluation.run_count.toString()} />
                      <SummaryCell label="Latest dataset" value={dashboard.offline_evaluation.latest_dataset_name || '-'} />
                      <SummaryCell label="Cases" value={dashboard.offline_evaluation.latest_case_count.toString()} />
                      <SummaryCell label="Best MRR" value={dashboard.offline_evaluation.best_mrr.toFixed(3)} />
                    </div>
                  )}
                  <div className="grid gap-4 lg:grid-cols-[300px_1fr]">
                    <ScrollArea className="h-[360px] pr-3">
                      <div className="space-y-2">
                        {evaluationRuns.map(run => (
                          <button
                            key={run.run_id}
                            type="button"
                            className={`w-full rounded-lg border p-3 text-left text-sm ${
                              selectedEvaluationRun?.run_id === run.run_id ? 'border-slate-950 bg-white' : 'border-slate-200 bg-slate-50 hover:bg-white'
                            }`}
                            onClick={() => loadEvaluationRunDetail(run.run_id)}
                          >
                            <div className="truncate font-medium text-slate-800">{run.run_id}</div>
                            <div className="mt-1 text-xs text-slate-400">{run.case_count} cases · {run.timestamp}</div>
                          </button>
                        ))}
                      </div>
                    </ScrollArea>

                    {latestRun ? (
                      <div className="space-y-4">
                        <div className="grid gap-3 md:grid-cols-3">
                          {Object.entries(latestRun.metrics || {}).map(([method, metric]) => (
                            <div key={method} className="rounded-lg border border-slate-200 bg-white p-3 text-sm">
                              <div className="font-medium text-slate-800">{method}</div>
                              <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                                <div><span className="text-slate-400">Hit@5</span><div className="font-medium">{formatPercent(metric.hit_at_5)}</div></div>
                                <div><span className="text-slate-400">MRR</span><div className="font-medium">{metric.mrr.toFixed(3)}</div></div>
                              </div>
                            </div>
                          ))}
                        </div>
                        {selectedEvaluationRun && (
                          <div className="grid gap-3 md:grid-cols-3">
                            {Object.entries(selectedEvaluationRun.results || {}).map(([method, results]) => (
                              <div key={method} className="rounded-lg border border-slate-200 bg-white p-3">
                                <div className="mb-2 flex items-center justify-between text-xs">
                                  <span className="font-medium text-slate-600">{method}</span>
                                  <Badge variant="outline">{results.filter(item => item.hit).length}/{results.length}</Badge>
                                </div>
                                <ScrollArea className="h-[220px] pr-3">
                                  <div className="space-y-2">
                                    {results.map(item => (
                                      <div key={`${method}-${item.test_case_id}`} className="rounded-md bg-slate-50 p-2 text-xs">
                                        <div className="mb-1 flex items-center justify-between">
                                          <Badge variant={item.hit ? 'default' : 'outline'}>{item.hit ? `hit #${item.rank}` : 'miss'}</Badge>
                                          <span className="text-slate-400">{item.response_time.toFixed(3)}s</span>
                                        </div>
                                        <p className="line-clamp-2 text-slate-700">{item.question}</p>
                                      </div>
                                    ))}
                                  </div>
                                </ScrollArea>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ) : (
                      <EmptyState text="暂无评测运行" />
                    )}
                  </div>
                </Panel>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}

function Panel({ title, description, children }: { title: string; description?: string; children: React.ReactNode }) {
  return (
    <Card className="border border-slate-200 bg-[#fbfcfb] shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{title}</CardTitle>
        {description && <CardDescription className="line-clamp-2">{description}</CardDescription>}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  )
}

function MetricCard({ icon, label, value, hint }: { icon: React.ReactNode; label: string; value: string; hint: string }) {
  return (
    <Card className="border border-slate-200 bg-[#fbfcfb] shadow-sm">
      <CardContent className="p-4">
        <div className="flex items-center justify-between gap-3">
          <div className="text-slate-500">{icon}</div>
          <Badge variant="outline" className="bg-white">{label}</Badge>
        </div>
        <div className="mt-4 text-2xl font-semibold tracking-tight text-slate-950">{value}</div>
        <div className="mt-1 truncate text-xs text-slate-400">{hint}</div>
      </CardContent>
    </Card>
  )
}

function SummaryCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="truncate text-xs text-slate-400">{label}</div>
      <div className="mt-1 truncate text-sm font-semibold text-slate-900">{value}</div>
    </div>
  )
}

function TraceColumn({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">{title}</div>
      <ScrollArea className="h-[260px] pr-3">
        <div className="space-y-2">{children}</div>
      </ScrollArea>
    </div>
  )
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-200 bg-white p-6 text-center text-sm text-slate-400">
      {text}
    </div>
  )
}
