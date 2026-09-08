import React from 'react';
import { FileText, AlertTriangle } from 'lucide-react';
import type { DocumentListItem } from '../../api/types';

interface DocumentListProps {
  documents: DocumentListItem[];
  onAnalyze: () => void;
  isAnalyzing: boolean;
}

export const DocumentList: React.FC<DocumentListProps> = ({
  documents,
  onAnalyze,
  isAnalyzing,
}) => {
  if (documents.length === 0) {
    return null;
  }

  const totalPages = documents.reduce((sum, d) => sum + d.page_count, 0);

  return (
    <div className="mt-6 border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900">
      {/* Header */}
      <div className="p-4 border-b border-neutral-200 dark:border-neutral-800 flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="font-semibold text-neutral-900 dark:text-neutral-100">
            {documents.length} {documents.length === 1 ? 'Document' : 'Documents'} Loaded
          </span>
          <span className="text-neutral-400">•</span>
          <span className="text-neutral-500">{totalPages} pages total</span>
        </div>

        <button
          onClick={onAnalyze}
          disabled={isAnalyzing || documents.length === 0}
          className={`px-4 py-2 text-xs font-mono font-medium uppercase tracking-wider transition-colors ${
            isAnalyzing
              ? 'bg-neutral-200 text-neutral-500 cursor-not-allowed dark:bg-neutral-800 dark:text-neutral-500'
              : 'bg-neutral-900 text-white hover:bg-neutral-800 dark:bg-neutral-100 dark:text-neutral-900 dark:hover:bg-white'
          }`}
        >
          {isAnalyzing ? 'Extracting & Reconciling...' : 'Analyze Documents'}
        </button>
      </div>

      {/* Document Table */}
      <div className="divide-y divide-neutral-200 dark:divide-neutral-800 text-xs">
        {documents.map((doc, idx) => {
          const hasScannedPages = doc.scanned_pages && doc.scanned_pages.length > 0;
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

                <div className="flex items-center gap-4 text-neutral-500 font-mono text-[11px]">
                  {doc.text_blocks !== undefined && (
                    <span>{doc.text_blocks} text blocks</span>
                  )}
                  {doc.tables !== undefined && (
                    <span>{doc.tables} tables</span>
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
