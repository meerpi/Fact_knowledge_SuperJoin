import React, { useEffect, useState } from 'react';
import {
  Layers,
  AlertTriangle,
  RotateCcw,
  CheckCircle2,
  Upload,
  ChevronRight,
} from 'lucide-react';
import { api } from './api/client';
import type {
  ClaimGraph,
  DocumentListItem,
  Fact,
  PipelineJobStatus,
  SystemCapabilities,
} from './api/types';
import { SystemCapabilitiesBanner } from './components/SystemCapabilitiesBanner';
import { UploadDropzone } from './components/UploadScreen/UploadDropzone';
import { DocumentList } from './components/UploadScreen/DocumentList';
import { ExtractionProgress } from './components/UploadScreen/ExtractionProgress';
import { DemoDatasetBanner } from './components/UploadScreen/DemoDatasetBanner';
import { FactsLedger } from './components/ResultsScreen/FactsLedger';
import { RelationshipsGraph } from './components/ResultsScreen/RelationshipsGraph';
import { PdfViewerModal } from './components/ResultsScreen/PdfViewerModal';

type ActiveTab = 'disputed' | 'facts';

export const App: React.FC = () => {
  // Navigation & Screen state
  const [currentScreen, setCurrentScreen] = useState<'upload' | 'results'>(() => {
    return (localStorage.getItem('fkl_screen') as 'upload' | 'results') || 'upload';
  });
  const [activeTab, setActiveTab] = useState<ActiveTab>(() => {
    const saved = localStorage.getItem('fkl_tab') as any;
    return saved === 'disputed' || saved === 'facts' ? saved : 'disputed';
  });

  // System capabilities
  const [capabilities, setCapabilities] = useState<SystemCapabilities | null>(null);

  // Document states
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isSeedingDemo, setIsSeedingDemo] = useState(false);
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null);
  const [isClearingAll, setIsClearingAll] = useState(false);

  // Pipeline execution & polling
  const [activeJobId, setActiveJobId] = useState<string | null>(() => {
    return localStorage.getItem('fkl_job_id');
  });
  const [pipelineStatus, setPipelineStatus] = useState<PipelineJobStatus | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Reconciliation Results
  const [claimGraph, setClaimGraph] = useState<ClaimGraph | null>(null);
  const [allFacts, setAllFacts] = useState<Fact[]>([]);
  const [docFilenames, setDocFilenames] = useState<Record<string, string>>({});

  // PDF Viewer Modal
  const [pdfModalDocId, setPdfModalDocId] = useState<string | null>(null);
  const [pdfModalPage, setPdfModalPage] = useState<number | null>(null);
  const [pdfModalQuote, setPdfModalQuote] = useState<string | undefined>(undefined);

  // Load capabilities, existing documents, facts, and reconciliation on mount
  useEffect(() => {
    loadCapabilities();
    loadDocuments();
    checkExistingResults();
    loadAllFacts();
  }, []);

  // Save session state to localStorage
  useEffect(() => {
    localStorage.setItem('fkl_screen', currentScreen);
    localStorage.setItem('fkl_tab', activeTab);
  }, [currentScreen, activeTab]);

  const loadCapabilities = () => {
    api.getCapabilities().then(setCapabilities).catch(console.error);
  };

  const loadDocuments = () => {
    api
      .listDocuments()
      .then((docs) => {
        setDocuments(docs);
        const map: Record<string, string> = {};
        docs.forEach((d) => {
          map[d.doc_id] = d.filename;
        });
        setDocFilenames((prev) => ({ ...prev, ...map }));
      })
      .catch(console.error);
  };

  const checkExistingResults = async () => {
    try {
      const graphData = await api.getLatestReconciliation().catch(() => null);
      if (graphData) setClaimGraph(graphData);
    } catch {
      // No reconciliation results yet
    }
  };

  const loadAllFacts = async () => {
    try {
      const docs = await api.listDocuments();
      const factsAccum: Fact[] = [];
      for (const d of docs) {
        try {
          const ef = await api.getDocumentFacts(d.doc_id);
          if (ef && ef.facts) {
            factsAccum.push(...ef.facts);
          }
        } catch {
          // facts not extracted for this doc yet
        }
      }
      setAllFacts(factsAccum);
    } catch (e) {
      console.error('Failed to load facts', e);
    }
  };

  // Upload handler
  const handleUploadFile = async (file: File) => {
    setIsUploading(true);
    try {
      const newDoc = await api.uploadPdf(file);
      setDocuments((prev) => [...prev.filter((d) => d.doc_id !== newDoc.doc_id), newDoc]);
      setDocFilenames((prev) => ({ ...prev, [newDoc.doc_id]: newDoc.filename }));
      loadCapabilities();
    } finally {
      setIsUploading(false);
    }
  };

  // Remove individual document handler
  const handleRemoveDocument = async (docId: string) => {
    setDeletingDocId(docId);
    try {
      await api.deleteDocument(docId);
      const remaining = documents.filter((d) => d.doc_id !== docId);
      setDocuments(remaining);
      setDocFilenames((prev) => {
        const next = { ...prev };
        delete next[docId];
        return next;
      });
      // Remove facts associated with this document
      setAllFacts((prev) => prev.filter((f) => f.provenance?.doc_id !== docId));
      if (remaining.length === 0) {
        setClaimGraph(null);
        setAllFacts([]);
        setCurrentScreen('upload');
      }
      loadCapabilities();
    } catch (err: any) {
      console.error('Failed to remove document', err);
      alert(`Failed to remove document: ${err.message || err}`);
    } finally {
      setDeletingDocId(null);
    }
  };

  // Clear / Unload all documents handler
  const handleClearAll = async () => {
    if (documents.length === 0) return;
    if (!window.confirm('Unload all documents? This will clear all loaded files, extractions, and reconciliation results.')) {
      return;
    }
    setIsClearingAll(true);
    try {
      await api.clearAllDocuments();
      setDocuments([]);
      setDocFilenames({});
      setAllFacts([]);
      setClaimGraph(null);
      handleResetSession();
      loadCapabilities();
    } catch (err: any) {
      console.error('Failed to clear documents', err);
      alert(`Failed to clear documents: ${err.message || err}`);
    } finally {
      setIsClearingAll(false);
    }
  };

  // Seed starter dataset handler
  const handleSeedDemo = async () => {
    setIsSeedingDemo(true);
    try {
      await api.seedDemo();
      loadDocuments();
      loadCapabilities();
      try {
        const graph = await api.getLatestReconciliation();
        setClaimGraph(graph);
      } catch {
        // fallback
      }
      await loadAllFacts();
      setCurrentScreen('results');
      setActiveTab('disputed');
    } catch (err) {
      console.error('Seed demo error', err);
    } finally {
      setIsSeedingDemo(false);
    }
  };

  // Start analysis pipeline
  const handleStartAnalysis = async (forceReextract: boolean = false) => {
    if (documents.length === 0) return;
    setIsAnalyzing(true);
    try {
      const res = await api.startPipeline(
        documents.map((d) => d.doc_id),
        undefined,
        forceReextract
      );
      setActiveJobId(res.job_id);
      localStorage.setItem('fkl_job_id', res.job_id);
    } catch (err: any) {
      alert(`Analysis failed to start: ${err.message}`);
      setIsAnalyzing(false);
    }
  };

  // Delete extracted facts for a specific document from SQLite
  const handleResetDocFacts = async (docId: string) => {
    try {
      await api.deleteDocumentFacts(docId);
      await loadDocuments();
      await loadAllFacts();
    } catch (err: any) {
      alert(`Failed to delete facts: ${err.message}`);
    }
  };

  // Poll pipeline job status
  useEffect(() => {
    if (!activeJobId) return;

    let isMounted = true;
    const interval = setInterval(async () => {
      try {
        const job = await api.getPipelineStatus(activeJobId);
        if (!isMounted) return;
        setPipelineStatus(job);

        if (job.status === 'completed') {
          clearInterval(interval);
          setIsAnalyzing(false);
          localStorage.removeItem('fkl_job_id');
          setActiveJobId(null);

          // Fetch final results & transition directly to Contradictions tab
          const graphData = await api.reconcile();
          setClaimGraph(graphData);
          await loadAllFacts();
          setCurrentScreen('results');
          setActiveTab('disputed');
        } else if (job.status === 'failed') {
          clearInterval(interval);
          setIsAnalyzing(false);
          alert(`Pipeline extraction failed: ${job.error}`);
        }
      } catch (err: any) {
        console.error('Status poll error', err);
        if (err?.message?.includes('404') || err?.status === 404) {
          clearInterval(interval);
          setIsAnalyzing(false);
          localStorage.removeItem('fkl_job_id');
          setActiveJobId(null);
        }
      }
    }, 600);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [activeJobId]);

  // Reset session handler
  const handleResetSession = () => {
    localStorage.removeItem('fkl_screen');
    localStorage.removeItem('fkl_tab');
    localStorage.removeItem('fkl_job_id');
    setCurrentScreen('upload');
    setActiveTab('facts');
    setPipelineStatus(null);
    setActiveJobId(null);
    setIsAnalyzing(false);
  };

  const handleOpenPdfViewer = (docId: string, page: number, quote?: string) => {
    setPdfModalDocId(docId);
    setPdfModalPage(page);
    setPdfModalQuote(quote);
  };

  return (
    <div className="min-h-screen bg-[#fafafa] dark:bg-[#0a0a0a] text-neutral-900 dark:text-neutral-100 flex flex-col font-sans selection:bg-teal-100 selection:text-teal-900">
      {/* Top Banner: Only shown when capabilities are degraded */}
      <SystemCapabilitiesBanner capabilities={capabilities} />

      {/* Main Header / Chrome */}
      <header className="border-b border-neutral-300 dark:border-neutral-800 bg-white dark:bg-neutral-900 px-4 sm:px-6 py-3">
        <div className="max-w-6xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 border-2 border-neutral-900 dark:border-neutral-100 flex items-center justify-center font-mono font-bold text-xs">
              FK
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight text-neutral-900 dark:text-neutral-100">
                Fact Verification & Cross-Document Reconciliation
              </h1>
            </div>
          </div>

          {/* Navigation Controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setCurrentScreen('upload')}
              className={`px-3 py-1.5 text-xs font-mono font-medium border transition-colors flex items-center gap-1.5 ${
                currentScreen === 'upload'
                  ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900'
                  : 'border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800'
              }`}
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Upload Documents</span>
            </button>

            {(allFacts.length > 0 || claimGraph) && (
              <button
                onClick={() => {
                  setCurrentScreen('results');
                  setActiveTab('disputed');
                }}
                className={`px-3 py-1.5 text-xs font-mono font-medium border transition-colors flex items-center gap-1.5 ${
                  currentScreen === 'results'
                    ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900'
                    : 'border-neutral-300 dark:border-neutral-700 bg-emerald-50 text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-200 hover:bg-emerald-100 dark:hover:bg-emerald-900/60'
                }`}
              >
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                <span>View Results ({allFacts.length} Facts)</span>
              </button>
            )}

            <button
              onClick={handleResetSession}
              title="Reset session state"
              className="p-1.5 text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 border border-transparent hover:border-neutral-300 dark:hover:border-neutral-700 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </header>

      {/* Screen 2 Tab Navigation (When on Results Screen) */}
      {currentScreen === 'results' && (
        <div className="border-b border-neutral-300 dark:border-neutral-800 bg-white dark:bg-neutral-900 px-4 sm:px-6">
          <div className="max-w-6xl mx-auto flex items-center gap-6 text-xs font-mono">
            <button
              onClick={() => setActiveTab('disputed')}
              className={`py-3 font-semibold flex items-center gap-1.5 border-b-2 transition-colors ${
                activeTab === 'disputed'
                  ? 'border-neutral-900 dark:border-neutral-100 text-neutral-900 dark:text-neutral-100'
                  : 'border-transparent text-neutral-500 hover:text-neutral-800 dark:hover:text-neutral-200'
              }`}
            >
              <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />
              <span>Contradictions & Reconciliation Graph</span>
              {claimGraph?.clusters && claimGraph.clusters.filter((c) => c.case_type !== 'corroborated').length > 0 && (
                <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-amber-100 dark:bg-amber-900/60 text-amber-800 dark:text-amber-200 font-mono">
                  {claimGraph.clusters.filter((c) => c.case_type !== 'corroborated').length}
                </span>
              )}
            </button>

            <button
              onClick={() => setActiveTab('facts')}
              className={`py-3 font-semibold flex items-center gap-1.5 border-b-2 transition-colors ${
                activeTab === 'facts'
                  ? 'border-neutral-900 dark:border-neutral-100 text-neutral-900 dark:text-neutral-100'
                  : 'border-transparent text-neutral-500 hover:text-neutral-800 dark:hover:text-neutral-200'
              }`}
            >
              <Layers className="w-3.5 h-3.5 text-neutral-500" />
              <span>All Facts Ledger</span>
              {allFacts.length > 0 && (
                <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 font-mono">
                  {allFacts.length}
                </span>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-4 sm:p-6">
        {currentScreen === 'upload' ? (
          /* Screen 1: Upload & Document Ingestion */
          <div className="max-w-2xl mx-auto space-y-6 pt-4">
            {allFacts.length > 0 && (
              <div className="p-3.5 bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-300 dark:border-emerald-800 flex items-center justify-between">
                <div className="flex items-center gap-2.5 text-xs font-mono text-emerald-900 dark:text-emerald-100">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                  <div>
                    <span className="font-bold">{allFacts.length.toLocaleString()} Facts Stored in SQLite</span>
                    <span className="text-neutral-400 mx-1.5">•</span>
                    <span>Arbitrated Contradictions Ready</span>
                  </div>
                </div>
                <button
                  onClick={() => {
                    setCurrentScreen('results');
                    setActiveTab('disputed');
                  }}
                  className="px-3.5 py-1.5 text-xs font-mono font-bold uppercase tracking-wider bg-emerald-700 text-white hover:bg-emerald-800 transition-colors flex items-center gap-1.5 shrink-0"
                >
                  <span>View Contradictions & Graph</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            <div className="text-center space-y-1">
              <h2 className="text-base font-bold text-neutral-900 dark:text-neutral-100 tracking-tight">
                Document Ingestion & Fact Extraction
              </h2>
              <p className="text-xs text-neutral-500 max-w-md mx-auto">
                Upload target financial or regulatory PDFs. Atomic facts will be grounded against primary text blocks and arbitrated across sources.
              </p>
            </div>

            {/* Drag and Drop Zone */}
            <UploadDropzone onUpload={handleUploadFile} isUploading={isUploading} />

            {/* List of uploaded documents */}
            <DocumentList
              documents={documents}
              onAnalyze={handleStartAnalysis}
              isAnalyzing={isAnalyzing}
              onRemoveDocument={handleRemoveDocument}
              onClearAll={handleClearAll}
              onResetDocFacts={handleResetDocFacts}
              deletingDocId={deletingDocId}
              isClearingAll={isClearingAll}
            />

            {/* Authentic Extraction Progress Display */}
            {isAnalyzing && <ExtractionProgress status={pipelineStatus} />}

            {/* Optional 1-Click Starter Dataset */}
            <DemoDatasetBanner onLoadDemo={handleSeedDemo} isLoading={isSeedingDemo} />
          </div>
        ) : (
          /* Screen 2: Results Exhibit */
          <div className="space-y-6">
            {activeTab === 'disputed' && (
              <RelationshipsGraph
                clusters={claimGraph?.clusters || []}
                onViewPdf={handleOpenPdfViewer}
                docFilenames={docFilenames}
              />
            )}

            {activeTab === 'facts' && (
              <FactsLedger
                facts={allFacts}
                docFilenames={docFilenames}
                onViewPdf={handleOpenPdfViewer}
              />
            )}
          </div>
        )}
      </main>

      {/* In-place PDF Viewer with Word Bounding Box Highlighting */}
      <PdfViewerModal
        docId={pdfModalDocId}
        pageNum={pdfModalPage}
        highlightQuote={pdfModalQuote}
        onClose={() => {
          setPdfModalDocId(null);
          setPdfModalPage(null);
          setPdfModalQuote(undefined);
        }}
      />

      {/* Minimal footer — no self-referential copy */}
      <footer className="border-t border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 px-4 py-2 text-[11px] font-mono text-neutral-400">
        <div className="max-w-6xl mx-auto flex flex-wrap items-center justify-between gap-2">
          <div>{allFacts.length > 0 ? `${allFacts.length} facts extracted` : ''}</div>
          <div>{claimGraph?.clusters ? `${claimGraph.clusters.length} clusters` : ''}</div>
        </div>
      </footer>
    </div>
  );
};

export default App;
