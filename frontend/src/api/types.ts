/**
 * TypeScript definitions mapping directly to backend Pydantic models in app/models.py
 */

export interface Context {
  temporal?: string | null;
  scope?: string | null;
  conditions?: string | null;
}

export type MatchType = 'exact' | 'normalized' | 'fuzzy' | 'unverified';

export interface Provenance {
  doc_id: string;
  page?: number | null;
  evidence_quote: string;
  verified: boolean;
  match_type: MatchType;
}

export interface Fact {
  subject: string;
  predicate: string;
  value: string;
  numeric_value?: number | null;
  unit?: string | null;
  base_unit?: string | null;
  scale: number;
  dimension?: string | null;
  canonical_value?: number | null;
  canonical_unit?: string | null;
  context: Context;
  confidence: number;
  provenance: Provenance;
}

export interface ExtractedFacts {
  doc_id: string;
  page_number?: number | null;
  facts: Fact[];
  model_used: string;
  fallback_attempts: number;
  skipped_pages: number[];
  warnings: string[];
}

export interface EvidenceEntry {
  doc_id: string;
  doc_filename: string;
  page?: number | null;
  value: string;
  canonical_value?: number | null;
  canonical_unit?: string | null;
  temporal?: string | null;
  scope?: string | null;
  conditions?: string | null;
  evidence_quote: string;
  confidence: number;
  match_type: MatchType | string;
}

export interface ClaimEdge {
  source_fact_idx: number;
  target_fact_idx: number;
  edge_type: 'contradicts' | 'corroborates' | 'supersedes';
  detection_method: string;
  confidence: number;
  explanation: string;
}

export interface FactCluster {
  cluster_id: string;
  subject: string;
  predicate: string;
  fact_indices: number[];
  edges: ClaimEdge[];
  case_type:
    | 'corroborated'
    | 'contradicted'
    | 'reconciled_temporal'
    | 'reconciled_scope'
    | 'reconciled_unit'
    | 'reconciled_conditions'
    | 'extraction_failure';
  dispute_code: string;
  dispute_detail?: string | null;
  evidence: EvidenceEntry[];
  explanation: string;
  credibility_scores: Record<string | number, number>;
  consensus_value?: number | null;
  consensus_unit?: string | null;
  doc_count: number;
}

export interface ExtractionFailure {
  failure_type: string;
  fact_index?: number | null;
  doc_id: string;
  page?: number | null;
  description: string;
  mitigation: string;
}

export interface AssignmentCases {
  case_1_corroborated?: {
    cluster_id: string;
    subject: string;
    predicate: string;
    case_type: string;
    dispute_code?: string;
    dispute_detail?: string | null;
    documents_involved: number;
    evidence: EvidenceEntry[];
    corroborating_edges: ClaimEdge[];
    explanation: string;
  } | null;
  case_2_contradicted?: {
    cluster_id: string;
    subject: string;
    predicate: string;
    case_type: string;
    dispute_code?: string;
    dispute_detail?: string | null;
    documents_involved: number;
    evidence: EvidenceEntry[];
    edges: ClaimEdge[];
    explanation: string;
    consensus_value?: number | null;
    credibility_scores?: Record<string, number>;
  } | null;
  case_3_reconciled?: {
    cluster_id: string;
    subject: string;
    predicate: string;
    case_type: string;
    dispute_code?: string;
    dispute_detail?: string | null;
    reconciliation_dimension?: string;
    documents_involved: number;
    evidence: EvidenceEntry[];
    edges: ClaimEdge[];
    explanation: string;
  } | null;
  case_4_extraction_failure?: ExtractionFailure | null;
}

export interface ClaimGraph {
  documents: Record<string, string>;
  total_facts: number;
  clusters: FactCluster[];
  extraction_failures: ExtractionFailure[];
  unmatched_facts: number[];
  case_summary: Record<string, number>;
  pipeline_metadata?: Record<string, any>;
}

export interface DocumentListItem {
  doc_id: string;
  filename: string;
  page_count: number;
  created_at?: string;
  text_blocks?: number;
  tables?: number;
  scanned_pages?: number[];
  warnings?: string[];
  has_extracted_facts?: boolean;
}

export interface SystemCapabilities {
  gemini_api_key_configured: boolean;
  voyage_api_key_configured: boolean;
  docling_available: boolean;
  nli_model_available: boolean;
  loaded_documents: number;
  extracted_documents: number;
  has_reconciliation: boolean;
}

export interface PipelineJobStatus {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  current_doc_name: string;
  current_doc_index: number;
  total_docs: number;
  current_step: string;
  facts_extracted_so_far: number;
  elapsed_seconds: number;
  stages_completed: string[];
  progress_percent?: number;
  current_page?: number;
  total_pages?: number;
  error?: string | null;
}

export interface ParseJobStatus {
  job_id: string;
  status: 'running' | 'completed' | 'failed';
  filename: string;
  current_page: number;
  total_pages: number;
  progress_percent: number;
  current_step: string;
  elapsed_seconds: number;
  error?: string | null;
  document?: DocumentListItem | null;
}

export interface BoundingBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface WordBBox {
  word: string;
  bbox: BoundingBox;
}

export interface PageWordBBoxesResponse {
  doc_id: string;
  page_num: number;
  page_width?: number;
  page_height?: number;
  words: WordBBox[];
}
