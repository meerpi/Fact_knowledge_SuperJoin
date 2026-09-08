import React, { useEffect, useState, useMemo } from 'react';
import {
  X,
  ExternalLink,
  Loader2,
  ChevronLeft,
  ChevronRight,
  Image as ImageIcon,
  FileText,
  ZoomIn,
  ZoomOut,
  RotateCcw,
} from 'lucide-react';
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
  const [currentPage, setCurrentPage] = useState<number>(pageNum ?? 0);
  const [words, setWords] = useState<WordBBox[]>([]);
  const [pageDim, setPageDim] = useState<{ width: number; height: number }>({
    width: 612,
    height: 792,
  });
  const [viewMode, setViewMode] = useState<'image' | 'pdf'>('image');
  const [loading, setLoading] = useState(false);
  const [imageLoading, setImageLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [zoom, setZoom] = useState<number>(100);

  // Sync initial pageNum
  useEffect(() => {
    if (pageNum !== null) {
      setCurrentPage(pageNum);
      setZoom(100);
    }
  }, [pageNum]);

  // Load word bboxes for the active page
  useEffect(() => {
    if (!docId || currentPage === null) return;
    setLoading(true);
    setImageLoading(true);
    setError(null);

    api
      .getPageWordBBoxes(docId, currentPage)
      .then((res: any) => {
        setWords(res.words || []);
        if (res.page_width && res.page_height) {
          setPageDim({ width: res.page_width, height: res.page_height });
        }
      })
      .catch((err) => {
        setError(err.message || 'Failed to load word coordinates');
      })
      .finally(() => {
        setLoading(false);
      });
  }, [docId, currentPage]);

  // Match quote words to bounding boxes
  const highlightedBBoxes = useMemo(() => {
    if (!highlightQuote || words.length === 0) return [];
    const quoteWords = highlightQuote
      .toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter((w) => w.length > 2);

    if (quoteWords.length === 0) return [];

    return words.filter((wb) => {
      const cleanWord = wb.word.toLowerCase().replace(/[^\w]/g, '');
      return cleanWord.length > 2 && quoteWords.includes(cleanWord);
    });
  }, [highlightQuote, words]);

  if (!docId || pageNum === null) return null;

  const pdfUrl = `${api.getPdfUrl(docId)}#page=${currentPage + 1}`;
  const imageUrl = api.getPageImageUrl(docId, currentPage);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-xs p-2 sm:p-4">
      <div className="w-full max-w-5xl h-[92vh] bg-white dark:bg-neutral-900 border border-neutral-300 dark:border-neutral-700 shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="p-3 border-b border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-850 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
          <div className="flex items-center gap-3">
            <span className="font-bold uppercase text-neutral-500">Source PDF Exhibit</span>
            <span>•</span>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
                disabled={currentPage <= 0}
                className="p-1 border border-neutral-300 dark:border-neutral-700 disabled:opacity-30 hover:bg-neutral-100 dark:hover:bg-neutral-800"
                title="Previous page"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
              <span className="font-semibold text-neutral-900 dark:text-neutral-100 px-1">
                Page {currentPage + 1}
              </span>
              <button
                onClick={() => setCurrentPage((p) => p + 1)}
                className="p-1 border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800"
                title="Next page"
              >
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>

            {words.length > 0 && (
              <span className="text-neutral-400 text-[11px] hidden sm:inline">
                ({words.length} grounded word bboxes)
              </span>
            )}
          </div>

          <div className="flex items-center gap-2 sm:gap-3">
            {/* View Mode Toggle */}
            <div className="flex items-center border border-neutral-300 dark:border-neutral-700 rounded-sm overflow-hidden text-[11px]">
              <button
                onClick={() => setViewMode('image')}
                className={`px-2 py-1 flex items-center gap-1 transition-colors ${
                  viewMode === 'image'
                    ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900 font-semibold'
                    : 'bg-white dark:bg-neutral-900 text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-800'
                }`}
              >
                <ImageIcon className="w-3 h-3" />
                <span>Page View</span>
              </button>
              <button
                onClick={() => setViewMode('pdf')}
                className={`px-2 py-1 flex items-center gap-1 transition-colors ${
                  viewMode === 'pdf'
                    ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900 font-semibold'
                    : 'bg-white dark:bg-neutral-900 text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-800'
                }`}
              >
                <FileText className="w-3 h-3" />
                <span>Native PDF</span>
              </button>
            </div>

            {/* Zoom Controls for Image View */}
            {viewMode === 'image' && (
              <div className="hidden md:flex items-center gap-1 text-[11px]">
                <button
                  onClick={() => setZoom((z) => Math.max(50, z - 15))}
                  className="p-1 border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800"
                  title="Zoom Out"
                >
                  <ZoomOut className="w-3 h-3" />
                </button>
                <span className="w-10 text-center font-mono">{zoom}%</span>
                <button
                  onClick={() => setZoom((z) => Math.min(200, z + 15))}
                  className="p-1 border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800"
                  title="Zoom In"
                >
                  <ZoomIn className="w-3 h-3" />
                </button>
                <button
                  onClick={() => setZoom(100)}
                  className="p-1 border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800"
                  title="Reset Zoom"
                >
                  <RotateCcw className="w-3 h-3" />
                </button>
              </div>
            )}

            <a
              href={api.getPdfUrl(docId)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-[11px] text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100 border border-neutral-300 dark:border-neutral-700 px-2 py-1 bg-white dark:bg-neutral-900 hover:bg-neutral-50 dark:hover:bg-neutral-800"
            >
              <span>Open PDF</span>
              <ExternalLink className="w-3 h-3" />
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
          <div className="px-4 py-2 bg-amber-50 dark:bg-amber-950/30 border-b border-amber-200 dark:border-amber-900/40 text-xs font-mono text-amber-900 dark:text-amber-200 flex items-center justify-between gap-2">
            <div className="truncate">
              <span className="font-semibold">Grounding Quote:</span> &ldquo;{highlightQuote}&rdquo;
            </div>
            {highlightedBBoxes.length > 0 && (
              <span className="text-[10px] bg-amber-200 dark:bg-amber-800 px-1.5 py-0.5 rounded-sm font-semibold shrink-0">
                {highlightedBBoxes.length} words highlighted
              </span>
            )}
          </div>
        )}

        {/* Viewer Body */}
        <div className="flex-1 relative bg-neutral-200 dark:bg-neutral-950 overflow-auto flex items-center justify-center p-4">
          {loading && (
            <div className="absolute inset-0 bg-white/70 dark:bg-neutral-900/70 flex items-center justify-center z-20">
              <div className="flex items-center gap-2 font-mono text-xs text-neutral-600 dark:text-neutral-400">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Loading page exhibit...</span>
              </div>
            </div>
          )}

          {error && (
            <div className="absolute top-3 left-3 right-3 p-3 bg-red-50 text-red-900 text-xs font-mono border border-red-200 z-10">
              Notice: {error}
            </div>
          )}

          {viewMode === 'image' ? (
            /* High-Resolution Rendered Page Exhibit */
            <div
              className="relative shadow-xl border border-neutral-300 dark:border-neutral-700 bg-white transition-transform origin-top"
              style={{
                width: `${(pageDim.width * zoom) / 100}px`,
                height: `${(pageDim.height * zoom) / 100}px`,
                maxWidth: '100%',
              }}
            >
              {imageLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-neutral-100">
                  <Loader2 className="w-6 h-6 animate-spin text-neutral-400" />
                </div>
              )}
              <img
                src={imageUrl}
                alt={`PDF Page ${currentPage + 1}`}
                onLoad={() => setImageLoading(false)}
                className="w-full h-full object-contain select-none"
              />

              {/* Bounding Box Overlays */}
              {highlightedBBoxes.map((wb, idx) => {
                const scaleX = (pageDim.width * zoom) / 100 / pageDim.width;
                const scaleY = (pageDim.height * zoom) / 100 / pageDim.height;
                const left = wb.bbox.x0 * scaleX;
                const top = wb.bbox.y0 * scaleY;
                const width = (wb.bbox.x1 - wb.bbox.x0) * scaleX;
                const height = (wb.bbox.y1 - wb.bbox.y0) * scaleY;

                return (
                  <div
                    key={idx}
                    className="absolute bg-amber-400/40 border border-amber-500 rounded-[1px] pointer-events-none transition-all hover:bg-amber-400/60"
                    style={{
                      left: `${left}px`,
                      top: `${top}px`,
                      width: `${width}px`,
                      height: `${height}px`,
                    }}
                    title={wb.word}
                  />
                );
              })}
            </div>
          ) : (
            /* Native PDF Viewer */
            <object
              data={pdfUrl}
              type="application/pdf"
              className="w-full h-full border-0 bg-white"
            >
              <iframe
                src={pdfUrl}
                title="PDF Document"
                className="w-full h-full border-0"
              />
            </object>
          )}
        </div>

        {/* Footer */}
        <div className="p-2.5 border-t border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-850 flex items-center justify-between text-[11px] font-mono text-neutral-500">
          <div>Document ID: {docId} • Page {currentPage + 1}</div>
          <button
            onClick={onClose}
            className="px-3 py-1 border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-800 dark:text-neutral-200"
          >
            Close Viewer
          </button>
        </div>
      </div>
    </div>
  );
};
