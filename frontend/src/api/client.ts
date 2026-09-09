import type {
  AssignmentCases,
  ClaimGraph,
  DocumentListItem,
  ExtractedFacts,
  PageWordBBoxesResponse,
  PipelineJobStatus,
  SystemCapabilities,
} from './types';

const API_BASE = '/api';

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, options);
  if (!res.ok) {
    let errorDetail = `Request failed (${res.status})`;
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || errJson.message || errorDetail;
    } catch {
      // fallback to status text
    }
    throw new ApiError(res.status, errorDetail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  // Capabilities
  getCapabilities: () => request<SystemCapabilities>('/system/capabilities'),

  // Seed demo
  seedDemo: () => request<{ status: string; total_facts: number; clusters_count: number }>('/system/seed-demo', {
    method: 'POST',
  }),

  // Documents
  listDocuments: () => request<DocumentListItem[]>('/documents'),

  uploadPdf: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return request<DocumentListItem>('/upload', {
      method: 'POST',
      body: formData,
    });
  },

  getDocumentFacts: (docId: string) => request<ExtractedFacts>(`/documents/${docId}/facts`),

  deleteDocument: (docId: string) =>
    request<{ deleted: boolean; doc_id: string }>(`/documents/${docId}`, {
      method: 'DELETE',
    }),

  clearAllDocuments: () =>
    request<{ deleted_count: number; message: string }>('/documents', {
      method: 'DELETE',
    }),

  deleteDocumentFacts: (docId: string) =>
    request<{ deleted: boolean; doc_id: string; message: string }>(`/documents/${docId}/facts`, {
      method: 'DELETE',
    }),

  clearAllFacts: () =>
    request<{ deleted_count: number; message: string }>('/facts', {
      method: 'DELETE',
    }),

  // Pipeline
  startPipeline: (
    docIds?: string[],
    targetPages?: Record<string, number[]>,
    forceReextract?: boolean
  ) =>
    request<{ job_id: string; status: string; total_docs: number }>('/pipeline/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        doc_ids: docIds,
        target_pages: targetPages,
        force_reextract: forceReextract ?? false,
      }),
    }),

  getPipelineStatus: (jobId: string) => request<PipelineJobStatus>(`/pipeline/status/${jobId}`),

  // Reconciliation & Cases
  getReconciliationCases: () => request<AssignmentCases>('/reconcile/cases'),
  getLatestReconciliation: () => request<ClaimGraph>('/reconcile/latest'),

  reconcile: (docIds?: string[]) =>
    request<ClaimGraph>('/reconcile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc_ids: docIds }),
    }),

  // Evidence PDF & BBoxes
  getPdfUrl: (docId: string) => `${API_BASE}/documents/${docId}/file`,

  getPageImageUrl: (docId: string, pageNum: number) =>
    `${API_BASE}/documents/${docId}/page/${pageNum}/image`,

  getPageWordBBoxes: (docId: string, pageNum: number) =>
    request<PageWordBBoxesResponse>(`/documents/${docId}/page/${pageNum}/word-bboxes`),
};
