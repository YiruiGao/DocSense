'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import {
  Upload,
  FileText,
  Trash2,
  Send,
  Bot,
  User,
  Loader2,
  AlertCircle,
  CheckCircle,
  Search,
  Clock,
  Database,
  Layers3,
} from 'lucide-react'

interface Document {
  id: string
  name: string
  pages: number
  file_size: number
  chunk_count: number
  created_at: string
}

interface Source {
  chunk_id: string
  document_id?: string
  document_name?: string
  page_number: number
  chunk_index?: number
  content: string
  score: number
}

interface QueryMetadata {
  retrieval_method: string
  total_candidates: number
  final_chunks: number
  response_time_seconds: number
  trace_id?: string
  query_rewrite_used?: boolean
  timings?: Record<string, number>
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  metadata?: QueryMetadata
  isLoading?: boolean
}

export default function RAGPage() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null)
  const [scopeMode, setScopeMode] = useState<'document' | 'all'>('document')
  const [isUploading, setIsUploading] = useState(false)
  const [isSending, setIsSending] = useState(false)
  const [isLoadingDocs, setIsLoadingDocs] = useState(true)
  const [uploadStatus, setUploadStatus] = useState<{ type: 'success' | 'error' | null; message: string }>({ type: null, message: '' })
  const fileInputRef = useRef<HTMLInputElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const selectedDocument = documents.find(doc => doc.id === selectedDocumentId) || null
  const activeDocumentId = scopeMode === 'document' ? selectedDocumentId : null
  const exampleQuestions = [
    '这份文档的核心观点是什么？',
    '文档里有哪些关键结论？',
    '请根据文档总结主要信息'
  ]
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

  // 加载文档列表
  const loadDocuments = async () => {
    try {
      setIsLoadingDocs(true)
      const response = await fetch(`${backendUrl}/documents`)
      const data = await response.json()
      if (data.success) {
        const docs = data.data || []
        setDocuments(docs)
        setSelectedDocumentId(prev => {
          if (prev && docs.some((doc: Document) => doc.id === prev)) return prev
          return docs[0]?.id || null
        })
      }
    } catch (error) {
      console.error('Load documents error:', error)
    } finally {
      setIsLoadingDocs(false)
    }
  }

  // 初始加载文档
  useEffect(() => {
    loadDocuments()
  }, [])

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // 上传文档
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // 检查文件类型
    const ext = file.name.toLowerCase().split('.').pop()
    const supportedTypes = ['pdf', 'txt', 'md']
    if (!ext || !supportedTypes.includes(ext)) {
      setUploadStatus({ type: 'error', message: '只支持 PDF、TXT、MD 文件' })
      return
    }

    setIsUploading(true)
    setUploadStatus({ type: null, message: '' })

    try {
      const formData = new FormData()
      formData.append('file', file)

      // 调用Python后端API
      const response = await fetch(`${backendUrl}/documents/upload`, {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (data.success) {
        setUploadStatus({ type: 'success', message: `文档 "${data.data.name}" 上传成功，共 ${data.data.pages} 页， ${data.data.chunk_count} 个分块` })
        setSelectedDocumentId(data.data.id)
        loadDocuments()
      } else {
        setUploadStatus({ type: 'error', message: data.error || '上传失败' })
      }
    } catch (error) {
      setUploadStatus({ type: 'error', message: '上传失败，请重试' })
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  // 删除文档
  const handleDelete = async (id: string) => {
    try {
      const response = await fetch(`${backendUrl}/documents/${id}`, {
        method: 'DELETE'
      })

      const data = await response.json()

      if (data.success) {
        setSelectedDocumentId(prev => prev === id ? null : prev)
        loadDocuments()
      }
    } catch (error) {
      console.error('Delete error:', error)
    }
  }

  // 发送消息
  const handleSend = async (questionOverride?: string) => {
    const question = (questionOverride || inputValue).trim()
    if (!question || isSending) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: question
    }

    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: '',
      isLoading: true
    }

    setMessages(prev => [...prev, userMessage, loadingMessage])
    setInputValue('')
    setIsSending(true)

    try {
      // 调用Python后端API
      const response = await fetch(`${backendUrl}/chat/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          question,
          document_id: activeDocumentId,
          options: {
            use_hybrid_search: true,
            use_rerank: true,
            top_k: 3
          }
        })
      })

      const data = await response.json()

      if (data.success) {
        const assistantMessage: Message = {
          id: (Date.now() + 2).toString(),
          role: 'assistant',
          content: data.data.answer,
          metadata: data.data.metadata,
          sources: data.data.sources.map((s: Source) => ({
            chunk_id: s.chunk_id,
            document_id: s.document_id,
            document_name: s.document_name,
            page_number: s.page_number,
            chunk_index: s.chunk_index,
            content: s.content,
            score: s.score
          }))
        }
        setMessages(prev => prev.filter(m => !m.isLoading).concat(assistantMessage))
      } else {
        const errorMessage: Message = {
          id: (Date.now() + 2).toString(),
          role: 'assistant',
          content: data.error || '回答生成失败，请重试'
        }
        setMessages(prev => prev.filter(m => !m.isLoading).concat(errorMessage))
      }
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        role: 'assistant',
        content: '网络错误，请重试'
      }
      setMessages(prev => prev.filter(m => !m.isLoading).concat(errorMessage))
    } finally {
      setIsSending(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#f3f6f5] text-slate-900">
      {/* 顶部标题栏 */}
      <header className="border-b border-slate-200 bg-[#fbfcfb]/90 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-slate-950 flex items-center justify-center">
                <Search className="w-5 h-5 text-amber-300" />
              </div>
              <div>
                <h1 className="text-xl font-semibold tracking-tight text-slate-950">DocSense</h1>
                <p className="text-sm text-slate-500">Document-grounded RAG with hybrid retrieval and reranking</p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
              <a
                href="/ragops"
                className="inline-flex h-7 items-center rounded-md border border-slate-300 bg-white/80 px-3 font-medium text-slate-700 hover:border-slate-900 hover:text-slate-950"
              >
                RAGOps Console
              </a>
              <Badge variant="outline" className="border-slate-300 bg-white/70">
                ChromaDB
              </Badge>
              <Badge variant="outline" className="border-slate-300 bg-white/70">
                BM25 + RRF
              </Badge>
              <Badge variant="outline" className="border-slate-300 bg-white/70">
                Reranker
              </Badge>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 左侧：文档管理 */}
          <div className="lg:col-span-1">
            <Card className="border border-slate-200 shadow-sm bg-[#fbfcfb]">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-blue-500" />
                  Document Library
                </CardTitle>
                <CardDescription>
                上传 PDF/TXT/MD，系统会提取文本、分块并建立向量与 BM25 索引
              </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* 上传区域 */}
                <div className="relative">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.txt,.md"
                    onChange={handleUpload}
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className={`flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-xl cursor-pointer transition-all
                      ${isUploading
                        ? 'border-blue-300 bg-blue-50'
                        : 'border-slate-200 hover:border-blue-400 hover:bg-blue-50/50'
                      }`}
                  >
                    {isUploading ? (
                      <div className="flex flex-col items-center gap-2">
                        <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
                        <span className="text-sm text-blue-600">处理中...</span>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-2">
                        <Upload className="w-8 h-8 text-slate-400" />
                        <span className="text-sm text-slate-500">点击上传文档</span>
                        <span className="text-xs text-slate-400">支持 PDF、TXT、MD 格式</span>
                      </div>
                    )}
                  </label>
                </div>

                {/* 上传状态 */}
                {uploadStatus.type && (
                  <div className={`flex items-center gap-2 p-3 rounded-lg text-sm
                    ${uploadStatus.type === 'success'
                      ? 'bg-green-50 text-green-700'
                      : 'bg-red-50 text-red-700'
                    }`}
                  >
                    {uploadStatus.type === 'success' ? (
                      <CheckCircle className="w-4 h-4 flex-shrink-0" />
                    ) : (
                      <AlertCircle className="w-4 h-4 flex-shrink-0" />
                    )}
                    <span>{uploadStatus.message}</span>
                  </div>
                )}

                <Separator />

                <div className="rounded-lg border border-slate-200 bg-white p-3">
                  <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                    <Database className="h-3.5 w-3.5" />
                    Scope
                  </div>
                  <ToggleGroup
                    type="single"
                    value={scopeMode}
                    onValueChange={(value) => {
                      if (value === 'document' || value === 'all') setScopeMode(value)
                    }}
                    className="grid w-full grid-cols-2"
                    variant="outline"
                    size="sm"
                  >
                    <ToggleGroupItem value="document" className="text-xs">Current document</ToggleGroupItem>
                    <ToggleGroupItem value="all" className="text-xs">All documents</ToggleGroupItem>
                  </ToggleGroup>
                </div>

                {scopeMode === 'all' ? (
                  <div className="rounded-lg border border-emerald-200 bg-emerald-50/80 p-3">
                    <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-emerald-700">
                      <Database className="h-3.5 w-3.5" />
                      Current context
                    </div>
                    <p className="mt-1 text-sm font-medium text-emerald-950">全部文档</p>
                    <div className="mt-2 flex items-center gap-3 text-xs text-emerald-700">
                      <span>{documents.length} documents</span>
                      <span>{documents.reduce((sum, doc) => sum + doc.chunk_count, 0)} chunks</span>
                    </div>
                  </div>
                ) : selectedDocument && (
                  <div className="rounded-lg border border-emerald-200 bg-emerald-50/80 p-3">
                    <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-emerald-700">
                      <Database className="h-3.5 w-3.5" />
                      Current context
                    </div>
                    <p className="mt-1 truncate text-sm font-medium text-emerald-950">{selectedDocument.name}</p>
                    <div className="mt-2 flex items-center gap-3 text-xs text-emerald-700">
                      <span>{selectedDocument.pages} 页</span>
                      <span>{selectedDocument.chunk_count} chunks</span>
                    </div>
                  </div>
                )}

                {/* 文档列表 */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-500">已上传文档</span>
                    <Badge variant="secondary">{documents.length}</Badge>
                  </div>

                  <ScrollArea className="h-[300px] pr-3">
                    {isLoadingDocs ? (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
                      </div>
                    ) : documents.length === 0 ? (
                      <div className="text-center py-8 text-slate-400">
                        <FileText className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p className="text-sm">暂无文档</p>
                        <p className="text-xs mt-1">上传文档开始使用</p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {documents.map(doc => (
                          <div
                            key={doc.id}
                            className={`group flex items-start justify-between p-3 rounded-lg border transition-colors ${
                              selectedDocumentId === doc.id
                                ? 'border-slate-900 bg-white'
                                : 'border-transparent bg-slate-50 hover:bg-slate-100'
                            }`}
                            onClick={() => setSelectedDocumentId(doc.id)}
                          >
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                <FileText className="w-4 h-4 text-blue-500 flex-shrink-0" />
                                <span className="text-sm font-medium text-slate-700 truncate">
                                  {doc.name}
                                </span>
                              </div>
                              <div className="flex items-center gap-2 mt-1 text-xs text-slate-400">
                                <span>{doc.pages} 页</span>
                                <span>•</span>
                                <span>{doc.chunk_count} 个分块</span>
                              </div>
                            </div>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="opacity-0 group-hover:opacity-100 transition-opacity h-8 w-8 text-slate-400 hover:text-red-500"
                              onClick={(event) => {
                                event.stopPropagation()
                                handleDelete(doc.id)
                              }}
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          </div>
                        ))}
                      </div>
                    )}
                  </ScrollArea>
                </div>
              </CardContent>
            </Card>

          </div>

          {/* 右侧：对话区域 */}
          <div className="lg:col-span-2">
            <Card className="border border-slate-200 shadow-sm bg-[#fbfcfb] h-[calc(100vh-180px)] flex flex-col">
              <CardHeader className="pb-3 flex-shrink-0">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Bot className="w-5 h-5 text-slate-900" />
                      Grounded Q&A
                    </CardTitle>
                    <CardDescription>
                      {scopeMode === 'all'
                        ? `当前基于全部 ${documents.length} 个文档回答`
                        : selectedDocument
                          ? `当前基于 "${selectedDocument.name}" 回答`
                          : '请选择或上传文档后开始问答'}
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <Layers3 className="h-4 w-4" />
                    hybrid search · rerank · citations
                  </div>
                </div>
              </CardHeader>

              <CardContent className="flex-1 flex flex-col min-h-0 overflow-hidden">
                {/* 消息列表 */}
                <ScrollArea className="flex-1 pr-3 mb-4 h-0">
                  {messages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-center py-12">
                      <div className="w-16 h-16 rounded-2xl bg-slate-950 flex items-center justify-center mb-4">
                        <Bot className="w-8 h-8 text-amber-300" />
                      </div>
                      <h3 className="text-lg font-medium text-slate-800 mb-2">Ask against the selected document</h3>
                      <p className="text-sm text-slate-400 max-w-xs">
                        上传并选择文档后，系统会检索证据片段、重排序，并基于来源回答
                      </p>
                      {documents.length > 0 && (
                        <div className="mt-5 flex flex-wrap justify-center gap-2">
                          {exampleQuestions.map(question => (
                            <Button
                              key={question}
                              variant="outline"
                              size="sm"
                              className="bg-white"
                              onClick={() => handleSend(question)}
                              disabled={isSending || (scopeMode === 'document' && !selectedDocumentId) || documents.length === 0}
                            >
                              {question}
                            </Button>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {messages.map(message => (
                        <div
                          key={message.id}
                          className={`flex gap-3 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
                        >
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0
                            ${message.role === 'user'
                              ? 'bg-blue-500 text-white'
                              : 'bg-gradient-to-br from-indigo-500 to-purple-500 text-white'
                            }`}
                          >
                            {message.role === 'user' ? (
                              <User className="w-4 h-4" />
                            ) : (
                              <Bot className="w-4 h-4" />
                            )}
                          </div>

                          <div className={`flex-1 ${message.role === 'user' ? 'text-right' : ''}`}>
                            <div className={`inline-block max-w-[85%] text-left rounded-2xl px-4 py-3
                              ${message.role === 'user'
                                ? 'bg-blue-500 text-white rounded-tr-sm'
                                : 'bg-slate-100 text-slate-700 rounded-tl-sm'
                              }`}
                            >
                              {message.isLoading ? (
                                <div className="flex items-center gap-2">
                                  <Loader2 className="w-4 h-4 animate-spin" />
                                  <span className="text-sm">Retrieving · Reranking · Generating</span>
                                </div>
                              ) : (
                                <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                              )}
                            </div>

                            {message.metadata && (
                              <div className={`mt-2 flex flex-wrap gap-2 text-xs text-slate-500 ${message.role === 'user' ? 'justify-end' : ''}`}>
                                <span className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1">
                                  <Database className="h-3 w-3" />
                                  {message.metadata.retrieval_method}
                                </span>
                                <span className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1">
                                  <Layers3 className="h-3 w-3" />
                                  {message.metadata.final_chunks}/{message.metadata.total_candidates} chunks
                                </span>
                                <span className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1">
                                  <Clock className="h-3 w-3" />
                                  {message.metadata.response_time_seconds}s
                                </span>
                                {message.metadata.timings?.llm_seconds !== undefined && (
                                  <span className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1">
                                    LLM {message.metadata.timings.llm_seconds}s
                                  </span>
                                )}
                                {message.metadata.trace_id && (
                                  <a
                                    className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1 hover:border-slate-400"
                                    href={`/ragops?trace_id=${encodeURIComponent(message.metadata.trace_id)}`}
                                  >
                                    Trace
                                  </a>
                                )}
                              </div>
                            )}

                            {/* 引用来源 */}
                            {message.sources && message.sources.length > 0 && (
                              <div className="mt-2 max-w-[85%]">
                                <p className="text-xs text-slate-400 mb-1">Evidence sources</p>
                                <div className="space-y-1">
                                  {message.sources.map((source, idx) => (
                                    <div
                                      key={idx}
                                      className="text-xs bg-white border border-slate-200 rounded-lg p-3"
                                    >
                                      <div className="flex flex-wrap items-center gap-2 mb-1">
                                        {source.document_name && (
                                          <span className="max-w-[180px] truncate text-slate-500">
                                            {source.document_name}
                                          </span>
                                        )}
                                        <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                                          第{source.page_number}页
                                        </Badge>
                                        {typeof source.chunk_index === 'number' && (
                                          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                                            chunk {source.chunk_index + 1}
                                          </Badge>
                                        )}
                                        <span className="text-slate-400">score {(source.score * 100).toFixed(0)}%</span>
                                      </div>
                                      <p className="text-slate-600 leading-relaxed">{source.content}</p>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                      <div ref={messagesEndRef} />
                    </div>
                  )}
                </ScrollArea>

                {/* 输入区域 */}
                <div className="flex gap-2 flex-shrink-0">
                  <Input
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        handleSend()
                      }
                    }}
                    placeholder={scopeMode === 'all' ? '输入一个基于全部文档的问题...' : selectedDocument ? '输入一个基于当前文档的问题...' : '请先上传或选择文档'}
                    className="flex-1"
                    disabled={isSending || documents.length === 0 || (scopeMode === 'document' && !selectedDocumentId)}
                  />
                  <Button
                    onClick={() => handleSend()}
                    disabled={!inputValue.trim() || isSending || documents.length === 0 || (scopeMode === 'document' && !selectedDocumentId)}
                    className="bg-slate-950 text-white hover:bg-slate-800"
                  >
                    {isSending ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Send className="w-4 h-4" />
                    )}
                  </Button>
                </div>
                {scopeMode === 'document' && !selectedDocumentId && (
                  <p className="text-xs text-slate-400 mt-2 text-center">
                    请先上传或选择一个文档，问答会限定在当前文档范围内
                  </p>
                )}
              </CardContent>
            </Card>

          </div>
        </div>
      </main>
    </div>
  )
}
