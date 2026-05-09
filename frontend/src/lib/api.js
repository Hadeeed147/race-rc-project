/**
 * Centralized API client for the RACE RC backend.
 * Base URL: http://localhost:8000. Configure via VITE_API_BASE if needed.
 */

const BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:8000').replace(/\/+$/, '')
const DEFAULT_TIMEOUT_MS = 15_000

export class ApiError extends Error {
  constructor(message, { status, body, cause } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
    if (cause) this.cause = cause
  }
}

async function fetchJSON(path, { method = 'GET', body, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(`${BASE}${path}`, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })
    let data = null
    const text = await res.text()
    try { data = text ? JSON.parse(text) : null } catch { data = text }
    if (!res.ok) {
      const detail = (data && data.detail) || res.statusText
      throw new ApiError(`${method} ${path} failed: ${detail}`, { status: res.status, body: data })
    }
    return data
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new ApiError(`${method} ${path} timed out after ${timeoutMs}ms.`, { cause: err })
    }
    if (err instanceof ApiError) throw err
    throw new ApiError(`${method} ${path} network error: ${err.message || err}`, { cause: err })
  } finally {
    clearTimeout(timer)
  }
}

export const getHealth             = ()                          => fetchJSON('/healthz')
export const getSample             = ()                          => fetchJSON('/sample')
export const getSampleWithQuestion = ()                          => fetchJSON('/sample_with_question')
export const generateQuiz          = (article)                   => fetchJSON('/generate',    { method: 'POST', body: { article } })
export const predictAnswer         = ({ article, question, options }) =>
  fetchJSON('/predict', { method: 'POST', body: { article, question, options } })
export const getDistractors        = (article, correct_answer) =>
  fetchJSON('/distractors', { method: 'POST', body: { article, correct_answer } })
export const getHints              = (article, question)         =>
  fetchJSON('/hints', { method: 'POST', body: { article, question } })
export const getAnalytics          = ()                          => fetchJSON('/analytics')
