import React, { useEffect, useState } from 'react';
import {
  FileText,
  Layers,
  Network,
  RotateCcw,
  CheckCircle2,
  Upload,
} from 'lucide-react';
import { api } from './api/client';
import type {
  AssignmentCases,
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
import { CasesView } from './components/ResultsScreen/CasesView';
import { FactsLedger } from './components/ResultsScreen/FactsLedger';
import { RelationshipsGraph } from './components/ResultsScreen/RelationshipsGraph';
import { PdfViewerModal } from './components/ResultsScreen/PdfViewerModal';

type ActiveTab = 'cases' | 'facts' | 'relationships';

export const App: React.FC = () => {
  // Navigation & Screen state
  const [currentScreen, setCurrentScreen] = useState<'upload' | 'results'>(() => {
    return (localStorage.getItem('fkl_screen') as 'upload' | 'results') || 'upload';
  });
  const [activeTab, setActiveTab] = useState<ActiveTab>(() => {
    return (localStorage.getItem('fkl_tab') as ActiveTab) || 'cases';
  });

  // System capabilities
  const [capabilities, setCapabilities] = useState<SystemCapabilities | null>(null);

  // Document states
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isSeedingDemo, setIsSeedingDemo] = useState(false);

  // Pipeline execution & polling
  const [activeJobId, setActiveJobId] = useState<string | null>(() => {
    return localStorage.getItem('fkl_job_id');
  });
  const [pipelineStatus, setPipelineStatus] = useState<PipelineJobStatus | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Reconciliation Results
  const [cases, setCases] = useState<AssignmentCases | null>(null);
  const [claimGraph, setClaimGraph] = useState<ClaimGraph | null>(null);
  const [allFacts, setAllFacts] = useState<Fact[]>([]);
  const [docFilenames, setDocFilenames] = useState<Record<string, string>>({});

  // PDF Viewer Modal
  const [pdfModalDocId, setPdfModalDocId] = useState<string | null>(null);
  const [pdfModalPage, setPdfModalPage] = useState<number | null>(null);

  // Load capabilities & existing documents on mount
  useEffect(() => {
    loadCapabilities();
    loadDocuments();
    checkExistingResults();
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
      const casesData = await api.getReconciliationCases();
      if (casesData) {
        setCases(casesData);
        try {
          const graph = await api.getLatestReconciliation();
          setClaimGraph(graph);
        } catch {
          // fallback
        }
        // Load all extracted facts for loaded docs
        loadAllFacts();
      }
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

  // Seed starter dataset handler
  const handleSeedDemo = async () => {
    setIsSeedingDemo(true);
    try {
      await api.seedDemo();
      loadDocuments();
      loadCapabilities();
      const casesData = await api.getReconciliationCases();
      setCases(casesData);
      try {
        const graph = await api.getLatestReconciliation();
        setClaimGraph(graph);
      } catch {
        // fallback
      }
      await loadAllFacts();
      setCurrentScreen('results');
      setActiveTab('cases');
    } catch (err) {
      console.error('Seed demo error', err);
    } finally {
      setIsSeedingDemo(false);
    }
  };

  // Start analysis pipeline
  const handleStartAnalysis = async () => {
    if (documents.length === 0) return;
    setIsAnalyzing(true);
    try {
      const res = await api.startPipeline(documents.map((d) => d.doc_id));
      setActiveJobId(res.job_id);
      localStorage.setItem('fkl_job_id', res.job_id);
    } catch (err: any) {
      alert(`Analysis failed to start: ${err.message}`);
      setIsAnalyzing(false);
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

          // Fetch final results & transition directly to Cases tab
          const [casesData, graphData] = await Promise.all([
            api.getReconciliationCases(),
            api.reconcile(),
          ]);
          setCases(casesData);
          setClaimGraph(graphData);
          await loadAllFacts();
          setCurrentScreen('results');
          setActiveTab('cases');
        } else if (job.status === 'failed') {
          clearInterval(interval);
          setIsAnalyzing(false);
          alert(`Pipeline extraction failed: ${job.error}`);
        }
      } catch (err) {
        console.error('Status poll error', err);
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
    setActiveTab('cases');
    setPipelineStatus(null);
    setActiveJobId(null);
    setIsAnalyzing(false);
  };

  const handleOpenPdfViewer = (docId: string, page: number) => {
    setPdfModalDocId(docId);
    setPdfModalPage(page);
  };

  return (
    <div className="min-h-screen bg-[#fafafa] dark:bg-[#0a0a0a] text-neutral-900 dark:text-neutral-100 flex flex-col font-sans selection:bg-teal-100 selection:text-teal-900">
      {/* Top Banner: Capabilities & Graceful Degradation Status */}
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
              <p className="text-[11px] text-neutral-500 font-mono">
                Deterministic Grounding • ArbGraph Claim Arbitration • Evidence Exhibit
              </p>
            </div>
          </div>

          {/* Navigation Controls */}
          <div className="flex items-center gap-2">
            {currentScreen === 'results' ? (
              <button
                onClick={() => setCurrentScreen('upload')}
                className="px-3 py-1.5 text-xs font-mono font-medium border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-850 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors flex items-center gap-1.5"
              >
                <Upload className="w-3 h-3" />
                <span>Upload More Documents</span>
              </button>
            ) : cases ? (
              <button
                onClick={() => setCurrentScreen('results')}
                className="px-3 py-1.5 text-xs font-mono font-medium border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-850 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors flex items-center gap-1.5"
              >
                <span>View Results</span>
                <CheckCircle2 className="w-3 h-3 text-teal-600" />
              </button>
            ) : null}

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
              onClick={() => setActiveTab('cases')}
              className={`py-3 font-semibold flex items-center gap-1.5 border-b-2 transition-colors ${
                activeTab === 'cases'
                  ? 'border-neutral-900 dark:border-neutral-100 text-neutral-900 dark:text-neutral-100'
                  : 'border-transparent text-neutral-500 hover:text-neutral-800 dark:hover:text-neutral-200'
              }`}
            >
              <FileText className="w-3.5 h-3.5" />
              <span>Cases</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 font-mono">
                Primary
              </span>
            </button>

            <button
              onClick={() => setActiveTab('facts')}
              className={`py-3 font-semibold flex items-center gap-1.5 border-b-2 transition-colors ${
                activeTab === 'facts'
                  ? 'border-neutral-900 dark:border-neutral-100 text-neutral-900 dark:text-neutral-100'
                  : 'border-transparent text-neutral-500 hover:text-neutral-800 dark:hover:text-neutral-200'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>All Facts Ledger</span>
              {allFacts.length > 0 && (
                <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 font-mono">
                  {allFacts.length}
                </span>
              )}
            </button>

            <button
              onClick={() => setActiveTab('relationships')}
              className={`py-3 font-semibold flex items-center gap-1.5 border-b-2 transition-colors ${
                activeTab === 'relationships'
                  ? 'border-neutral-900 dark:border-neutral-100 text-neutral-900 dark:text-neutral-100'
                  : 'border-transparent text-neutral-500 hover:text-neutral-800 dark:hover:text-neutral-200'
              }`}
            >
              <Network className="w-3.5 h-3.5" />
              <span>Relationships Graph</span>
            </button>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-4 sm:p-6">
        {currentScreen === 'upload' ? (
          /* Screen 1: Upload & Document Ingestion */
          <div className="max-w-2xl mx-auto space-y-6 pt-4">
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
            />

            {/* Authentic Extraction Progress Display */}
            {isAnalyzing && <ExtractionProgress status={pipelineStatus} />}

            {/* Optional 1-Click Starter Dataset */}
            <DemoDatasetBanner onLoadDemo={handleSeedDemo} isLoading={isSeedingDemo} />
          </div>
        ) : (
          /* Screen 2: Results Exhibit */
          <div className="space-y-6">
            {activeTab === 'cases' && (
              <CasesView cases={cases} onViewPdf={handleOpenPdfViewer} />
            )}

            {activeTab === 'facts' && (
              <FactsLedger
                facts={allFacts}
                docFilenames={docFilenames}
                onViewPdf={handleOpenPdfViewer}
              />
            )}

            {activeTab === 'relationships' && (
              <RelationshipsGraph
                clusters={
                  claimGraph?.clusters || [
                    ...(cases?.case_1_corroborated ? [cases.case_1_corroborated as any] : []),
                    ...(cases?.case_2_contradicted ? [cases.case_2_contradicted as any] : []),
                    ...(cases?.case_3_reconciled ? [cases.case_3_reconciled as any] : []),
                  ]
                }
              />
            )}
          </div>
        )}
      </main>

      {/* In-place PDF Viewer with Word Bounding Box Highlighting */}
      <PdfViewerModal
        docId={pdfModalDocId}
        pageNum={pdfModalPage}
        onClose={() => {
          setPdfModalDocId(null);
          setPdfModalPage(null);
        }}
      />

      {/* Exhibit Footer */}
      <footer className="border-t border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 px-4 py-3 text-[11px] font-mono text-neutral-500">
        <div className="max-w-6xl mx-auto flex flex-wrap items-center justify-between gap-2">
          <div>Fact Knowledge Layer • ArbGraph Arbitration Engine</div>
          <div className="flex items-center gap-3">
            <span>Session Persisted</span>
            <span>•</span>
            <span>Deterministic Verification</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
