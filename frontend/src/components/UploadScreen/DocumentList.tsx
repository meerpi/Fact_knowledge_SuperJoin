import React, { useState } from 'react';
import { FileText, AlertTriangle, X, Loader2, Trash2, RefreshCw } from 'lucide-react';
import type { DocumentListItem } from '../../api/types';

interface DocumentListProps {
  documents: DocumentListItem[];
  onAnalyze: (forceReextract?: boolean) => void;
  isAnalyzing: boolean;
  onRemoveDocument?: (docId: string) => void;
  onClearAll?: () => void;
  onResetDocFacts?: (docId: string) => void;
  deletingDocId?: string | null;
  isClearingAll?: boolean;
}

export const DocumentList: React.FC<DocumentListProps> = ({
  documents,
  onAnalyze,
  isAnalyzing,
  onRemoveDocument,
  onClearAll,
  onResetDocFacts,
  deletingDocId,
  isClearingAll,
}) => {
  const [forceReextract, setForceReextract] = useState(false);

  if (documents.length === 0) {
    return null;
  }

  const totalPages = documents.reduce((sum, d) => sum + d.page_count, 0);
  const cachedCount = documents.filter((d) => d.has_extracted_facts).length;

  return (
    <div className="mt-6 border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900">
      {/* Header */}
      <div className="p-4 border-b border-neutral-200 dark:border-neutral-800 flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="font-semibold text-neutral-900 dark:text-neutral-100">
            {documents.length} {documents.length === 1 ? 'Document' : 'Documents'} Loaded
          </span>
          <span className="text-neutral-400">•</span>
          <span className="text-neutral-500">{totalPages} pages total</span>
          {cachedCount > 0 && (
            <>
              <span className="text-neutral-400">•</span>
              <span className="text-emerald-600 dark:text-emerald-400">
                {cachedCount} cached in SQLite
              </span>
            </>
          )}
        </div>

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5 cursor-pointer select-none text-[11px] font-mono text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-200">
            <input
              type="checkbox"
              checked={forceReextract}
              onChange={(e) => setForceReextract(e.target.checked)}
              disabled={isAnalyzing}
              className="rounded border-neutral-300 dark:border-neutral-600 text-neutral-900 focus:ring-0"
            />
            <span>Force Re-extract (Bypass Cache)</span>
          </label>

          {onClearAll && (
            <button
              onClick={onClearAll}
              disabled={isAnalyzing || isClearingAll}
              className="px-3 py-2 text-xs font-mono font-medium text-neutral-500 hover:text-red-600 dark:text-neutral-400 dark:hover:text-red-400 border border-neutral-300 dark:border-neutral-700 hover:border-red-300 dark:hover:border-red-800 transition-colors disabled:opacity-40 flex items-center gap-1.5"
              title="Unload all documents"
            >
              {isClearingAll ? (
                <Loader2 className="w-3 h-3 animate-spin text-neutral-400" />
              ) : (
                <Trash2 className="w-3 h-3 text-neutral-400 hover:text-red-500" />
              )}
              <span>{isClearingAll ? 'Clearing...' : 'Clear All'}</span>
            </button>
          )}

          <button
            onClick={() => onAnalyze(forceReextract)}
            disabled={isAnalyzing || isClearingAll || documents.length === 0}
            className={`px-4 py-2 text-xs font-mono font-medium uppercase tracking-wider transition-colors flex items-center gap-1.5 ${
              isAnalyzing
                ? 'bg-neutral-200 text-neutral-500 cursor-not-allowed dark:bg-neutral-800 dark:text-neutral-500'
                : forceReextract
                ? 'bg-amber-600 text-white hover:bg-amber-700 dark:bg-amber-600 dark:hover:bg-amber-500'
                : 'bg-neutral-900 text-white hover:bg-neutral-800 dark:bg-neutral-100 dark:text-neutral-900 dark:hover:bg-white'
            }`}
          >
            {forceReextract && !isAnalyzing && <RefreshCw className="w-3 h-3" />}
            <span>
              {isAnalyzing
                ? 'Extracting & Reconciling...'
                : forceReextract
                ? 'Re-extract All Facts'
                : 'Analyze Documents'}
            </span>
          </button>
        </div>
      </div>

      {/* Document Table */}
      <div className="divide-y divide-neutral-200 dark:divide-neutral-800 text-xs">
        {documents.map((doc, idx) => {
          const hasScannedPages = doc.scanned_pages && doc.scanned_pages.length > 0;
          const isDeletingThis = deletingDocId === doc.doc_id;

          return (
            <div key={doc.doc_id || idx} className="p-4 hover:bg-neutral-50/60 dark:hover:bg-neutral-850/40 transition-colors">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <FileText className="w-4 h-4 text-neutral-500 shrink-0" />
                  <span className="font-medium text-neutral-900 dark:text-neutral-100 truncate" title={doc.filename}>
                    {doc.filename}
                  </span>
                  <span className="text-neutral-400 font-mono text-[11px]">
                    ({doc.page_count} {doc.page_count === 1 ? 'page' : 'pages'})
                  </span>
                </div>

                <div className="flex items-center gap-3 text-neutral-500 font-mono text-[11px]">
                  {doc.has_extracted_facts && (
                    <span className="px-1.5 py-0.5 rounded bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800 text-[10px] font-mono flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" />
                      Cached in DB
                    </span>
                  )}
                  {doc.has_extracted_facts && onResetDocFacts && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onResetDocFacts(doc.doc_id);
                      }}
                      disabled={isAnalyzing}
                      className="px-2 py-0.5 text-[10px] font-mono text-neutral-500 hover:text-amber-600 dark:hover:text-amber-400 border border-neutral-200 dark:border-neutral-700 hover:border-amber-300 rounded transition-colors"
                      title="Delete extracted facts for this document from SQLite (preserves uploaded PDF)"
                    >
                      Clear Facts
                    </button>
                  )}
                  {doc.text_blocks !== undefined && (
                    <span>{doc.text_blocks} text blocks</span>
                  )}
                  {doc.tables !== undefined && (
                    <span>{doc.tables} tables</span>
                  )}
                  {onRemoveDocument && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onRemoveDocument(doc.doc_id);
                      }}
                      disabled={isAnalyzing || isDeletingThis || isClearingAll}
                      className="p-1.5 text-neutral-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/30 rounded border border-transparent hover:border-red-200 dark:hover:border-red-900/50 transition-colors disabled:opacity-40 flex items-center justify-center"
                      title={`Unload and remove ${doc.filename}`}
                      aria-label={`Unload ${doc.filename}`}
                    >
                      {isDeletingThis ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin text-neutral-400" />
                      ) : (
                        <X className="w-3.5 h-3.5 stroke-[2.2]" />
                      )}
                    </button>
                  )}
                </div>
              </div>

              {/* Degraded State: Scanned or low-text warning */}
              {hasScannedPages && (
                <div className="mt-2.5 p-2 bg-amber-50 border border-amber-200 text-amber-900 dark:bg-amber-950/20 dark:border-amber-900/40 dark:text-amber-300 flex items-start gap-1.5 text-[11px]">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-medium">Scanned image pages detected:</span>{' '}
                    Pages {doc.scanned_pages?.join(', ')} contain minimal selectable text (&lt;20 characters).
                    Extraction will rely on OCR or layout tables for these pages.
                  </div>
                </div>
              )}

              {doc.warnings && doc.warnings.length > 0 && !hasScannedPages && (
                <div className="mt-2 text-[11px] text-neutral-500 font-mono">
                  Notice: {doc.warnings.join('; ')}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

