import React, { useEffect, useState, useMemo, useRef } from 'react';
import {
  Loader2,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  ExternalLink,
  FileText,
  X,
} from 'lucide-react';
import { api } from '../../api/client';
import type { WordBBox } from '../../api/types';

interface InlinePdfViewerProps {
  docId: string;
  pageNum: number;
  docFilename?: string;
  highlightQuote?: string;
  highlightValue?: string | number;
  initialZoom?: number;
  maxHeight?: string;
  onClose?: () => void;
  showCloseButton?: boolean;
}

export const InlinePdfViewer: React.FC<InlinePdfViewerProps> = ({
  docId,
  pageNum,
  docFilename,
  highlightQuote,
  highlightValue,
  initialZoom = 95,
  maxHeight = '420px',
  onClose,
  showCloseButton = false,
}) => {
  const [currentPage, setCurrentPage] = useState<number>(pageNum);
  const [words, setWords] = useState<WordBBox[]>([]);
  const [pageDim, setPageDim] = useState<{ width: number; height: number }>({
    width: 612,
    height: 792,
  });
  const [loading, setLoading] = useState(false);
  const [imageLoading, setImageLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [zoom, setZoom] = useState<number>(initialZoom);

  const containerRef = useRef<HTMLDivElement>(null);
  const firstHighlightRef = useRef<HTMLDivElement>(null);

  // Sync initial pageNum
  useEffect(() => {
    setCurrentPage(pageNum);
  }, [pageNum, docId]);

  // Load word bboxes for the active page
  useEffect(() => {
    if (!docId) return;
    setLoading(true);
    setImageLoading(true);
    setError(null);

    api
      .getPageWordBBoxes(docId, currentPage)
      .then((res) => {
        setWords(res.words || []);
        if (res.page_width && res.page_height) {
          setPageDim({ width: res.page_width, height: res.page_height });
        }
      })
      .catch((err) => {
        setError(err.message || 'Coordinates unavailable');
      })
      .finally(() => {
        setLoading(false);
      });
  }, [docId, currentPage]);

  // Match quote and value words to bounding boxes
  const highlightedBBoxes = useMemo(() => {
    if (words.length === 0) return [];

    const targetTokens = new Set<string>();

    if (highlightValue !== undefined && highlightValue !== null) {
      const valStr = String(highlightValue)
        .toLowerCase()
        .replace(/[^\w]/g, '');
      if (valStr.length > 0) targetTokens.add(valStr);
    }

    if (highlightQuote) {
      const quoteWords = highlightQuote
        .toLowerCase()
        .replace(/[^\w\s]/g, '')
        .split(/\s+/)
        .filter((w) => w.length > 2);
      quoteWords.forEach((w) => targetTokens.add(w));
    }

    if (targetTokens.size === 0) return [];

    return words.filter((wb) => {
      const clean = wb.word.toLowerCase().replace(/[^\w]/g, '');
      return clean.length > 0 && targetTokens.has(clean);
    });
  }, [highlightQuote, highlightValue, words]);

  // Auto-scroll to center on first highlight when image loads or when highlights change
  const scrollToHighlight = () => {
    if (firstHighlightRef.current && containerRef.current) {
      const container = containerRef.current;
      const target = firstHighlightRef.current;
      const containerRect = container.getBoundingClientRect();
      const targetRect = target.getBoundingClientRect();
      const offsetTop = targetRect.top - containerRect.top + container.scrollTop;
      container.scrollTo({
        top: Math.max(0, offsetTop - container.clientHeight / 2 + 40),
        behavior: 'smooth',
      });
    }
  };

  const handleImageLoaded = () => {
    setImageLoading(false);
    setTimeout(scrollToHighlight, 100);
  };

  useEffect(() => {
    if (!imageLoading && highlightedBBoxes.length > 0) {
      const timer = setTimeout(scrollToHighlight, 80);
      return () => clearTimeout(timer);
    }
  }, [highlightedBBoxes, imageLoading]);

  const imageUrl = api.getPageImageUrl(docId, currentPage);
  const pdfUrl = `${api.getPdfUrl(docId)}#page=${currentPage + 1}`;

  return (
    <div className="border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 flex flex-col overflow-hidden text-xs font-mono shadow-xs">
      {/* Mini Control Header */}
      <div className="p-2 border-b border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-850 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 truncate">
          <FileText className="w-3.5 h-3.5 text-neutral-500 shrink-0" />
          <span className="font-semibold text-neutral-900 dark:text-neutral-100 truncate max-w-[200px]" title={docFilename}>
            {docFilename || docId}
          </span>
          <span className="text-neutral-400">•</span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
              disabled={currentPage <= 0}
              className="p-0.5 border border-neutral-300 dark:border-neutral-700 disabled:opacity-30 hover:bg-neutral-100 dark:hover:bg-neutral-800"
              title="Previous page"
            >
              <ChevronLeft className="w-3 h-3" />
            </button>
            <span className="font-bold text-neutral-800 dark:text-neutral-200 px-1">
              p. {currentPage + 1}
            </span>
            <button
              onClick={() => setCurrentPage((p) => p + 1)}
              className="p-0.5 border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800"
              title="Next page"
            >
              <ChevronRight className="w-3 h-3" />
            </button>
          </div>
          {highlightedBBoxes.length > 0 && (
            <span className="px-1.5 py-0.2 bg-amber-100 dark:bg-amber-900/60 text-amber-800 dark:text-amber-200 text-[10px] rounded-xs font-semibold shrink-0">
              {highlightedBBoxes.length} highlights
            </span>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-1.5 shrink-0">
          <button
            onClick={() => setZoom((z) => Math.max(50, z - 15))}
            className="p-1 border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800"
            title="Zoom Out"
          >
            <ZoomOut className="w-3 h-3" />
          </button>
          <span className="w-8 text-center text-[10px]">{zoom}%</span>
          <button
            onClick={() => setZoom((z) => Math.min(200, z + 15))}
            className="p-1 border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800"
            title="Zoom In"
          >
            <ZoomIn className="w-3 h-3" />
          </button>
          <button
            onClick={() => setZoom(initialZoom)}
            className="p-1 border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800"
            title="Reset Zoom"
          >
            <RotateCcw className="w-3 h-3" />
          </button>
          <a
            href={pdfUrl}
            target="_blank"
            rel="noreferrer"
            className="p-1 border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-600 dark:text-neutral-400"
            title="Open raw PDF page in new tab"
          >
            <ExternalLink className="w-3 h-3" />
          </a>
          {showCloseButton && onClose && (
            <button
              onClick={onClose}
              className="p-1 text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100"
              title="Close PDF view"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Quote callout strip */}
      {highlightQuote && (
        <div className="px-3 py-1.5 bg-amber-50/80 dark:bg-amber-950/30 border-b border-amber-200 dark:border-amber-900/40 text-[11px] text-amber-900 dark:text-amber-200 truncate flex items-center justify-between gap-2">
          <div className="truncate">
            <span className="font-semibold">Grounding Quote:</span> &ldquo;{highlightQuote}&rdquo;
          </div>
          {highlightValue && (
            <span className="font-bold text-amber-950 dark:text-amber-100 shrink-0">
              Value: {highlightValue}
            </span>
          )}
        </div>
      )}

      {/* Canvas Viewport */}
      <div
        ref={containerRef}
        style={{ maxHeight }}
        className="relative bg-neutral-200 dark:bg-neutral-950 overflow-auto flex items-center justify-center p-3"
      >
        {(loading || imageLoading) && (
          <div className="absolute inset-0 bg-white/70 dark:bg-neutral-900/70 flex items-center justify-center z-20">
            <div className="flex items-center gap-2 text-neutral-600 dark:text-neutral-400 text-xs">
              <Loader2 className="w-4 h-4 animate-spin text-neutral-500" />
              <span>Rendering segment...</span>
            </div>
          </div>
        )}

        {error && (
          <div className="absolute top-2 left-2 right-2 p-2 bg-red-50 border border-red-200 text-red-800 text-[11px] z-10">
            {error}
          </div>
        )}

        {/* High-res rendered page exhibit */}
        <div
          className="relative shadow-md border border-neutral-300 dark:border-neutral-700 bg-white transition-all origin-top shrink-0 my-auto"
          style={{
            width: `${(pageDim.width * zoom) / 100}px`,
            height: `${(pageDim.height * zoom) / 100}px`,
            maxWidth: 'none',
          }}
        >
          <img
            src={imageUrl}
            alt={`PDF Page ${currentPage + 1}`}
            onLoad={handleImageLoaded}
            className="w-full h-full object-contain select-none"
          />

          {/* Bounding box highlights */}
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
                ref={idx === 0 ? firstHighlightRef : undefined}
                className="absolute bg-amber-400/40 border border-amber-500 rounded-[1px] pointer-events-none transition-all"
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
      </div>
    </div>
  );
};
