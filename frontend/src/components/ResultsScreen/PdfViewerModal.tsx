import React, { useEffect, useState } from 'react';
import { X, ExternalLink, Loader2 } from 'lucide-react';
import { api } from '../../api/client';
import type { WordBBox } from '../../api/types';

interface PdfViewerModalProps {
  docId: string | null;
  pageNum: number | null;
  onClose: () => void;
  highlightQuote?: string;
}

export const PdfViewerModal: React.FC<PdfViewerModalProps> = ({
  docId,
  pageNum,
  onClose,
  highlightQuote,
}) => {
  const [words, setWords] = useState<WordBBox[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!docId || pageNum === null) return;
    setLoading(true);
    setError(null);

    api
      .getPageWordBBoxes(docId, pageNum)
      .then((res) => {
        setWords(res.words || []);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load word coordinates');
      })
      .finally(() => {
        setLoading(false);
      });
  }, [docId, pageNum]);

  if (!docId || pageNum === null) return null;

  const pdfUrl = `${api.getPdfUrl(docId)}#page=${pageNum + 1}`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-[2px] p-4">
      <div className="w-full max-w-5xl h-[90vh] bg-white dark:bg-neutral-900 border border-neutral-300 dark:border-neutral-700 shadow-2xl flex flex-col">
        {/* Header */}
        <div className="p-3 border-b border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-850 flex items-center justify-between text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className="font-bold uppercase text-neutral-500">Source PDF Exhibit</span>
            <span>•</span>
            <span className="font-semibold text-neutral-900 dark:text-neutral-100">
              Page {pageNum} (Display Page {pageNum + 1})
            </span>
            {words.length > 0 && (
              <span className="text-neutral-400 text-[11px]">
                ({words.length} grounded word bboxes)
              </span>
            )}
          </div>

          <div className="flex items-center gap-3">
            <a
              href={api.getPdfUrl(docId)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-[11px] text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100"
            >
              Open raw PDF <ExternalLink className="w-3 h-3" />
            </a>
            <button
              onClick={onClose}
              className="p-1 text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Highlight Banner if quote provided */}
        {highlightQuote && (
          <div className="px-4 py-2 bg-amber-50 dark:bg-amber-950/30 border-b border-amber-200 dark:border-amber-900/40 text-xs font-mono text-amber-900 dark:text-amber-200 truncate">
            <span className="font-semibold">Searching quote:</span> &ldquo;{highlightQuote}&rdquo;
          </div>
        )}

        {/* Viewer Body */}
        <div className="flex-1 relative bg-neutral-200 dark:bg-neutral-950 flex flex-col items-center justify-center overflow-hidden">
          {loading && (
            <div className="absolute inset-0 bg-white/70 dark:bg-neutral-900/70 flex items-center justify-center z-10">
              <div className="flex items-center gap-2 font-mono text-xs text-neutral-600 dark:text-neutral-400">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Loading word bounding boxes...</span>
              </div>
            </div>
          )}

          {error && (
            <div className="p-3 bg-red-50 text-red-900 text-xs font-mono border border-red-200 mb-2">
              Bounding box notice: {error} (Displaying standard PDF renderer)
            </div>
          )}

          {/* Embedded native PDF viewer */}
          <iframe
            src={pdfUrl}
            title="PDF Document"
            className="w-full h-full border-0"
          />
        </div>

        {/* Footer */}
        <div className="p-2 border-t border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-850 flex items-center justify-between text-[11px] font-mono text-neutral-500">
          <div>Document ID: {docId}</div>
          <button
            onClick={onClose}
            className="px-3 py-1 border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 hover:bg-neutral-100 dark:hover:bg-neutral-800"
          >
            Close Viewer
          </button>
        </div>
      </div>
    </div>
  );
};
