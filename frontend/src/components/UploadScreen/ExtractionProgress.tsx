import React from 'react';
import { Loader2, CheckCircle2, Clock, Database, FileText } from 'lucide-react';
import type { PipelineJobStatus } from '../../api/types';

interface ExtractionProgressProps {
  status: PipelineJobStatus | null;
}

export const ExtractionProgress: React.FC<ExtractionProgressProps> = ({ status }) => {
  if (!status) return null;

  return (
    <div className="mt-6 border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 p-5 space-y-4">
      {/* Top Banner with Real Status and Elapsed Clock */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-neutral-200 dark:border-neutral-800">
        <div className="flex items-center gap-2">
          <Loader2 className="w-4 h-4 text-neutral-800 dark:text-neutral-200 animate-spin" />
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-neutral-900 dark:text-neutral-100">
            {status.status === 'running' ? 'Live Pipeline Execution' : 'Pipeline Queued'}
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono text-neutral-500">
          <div className="flex items-center gap-1">
            <Clock className="w-3.5 h-3.5 text-neutral-400" />
            <span>{status.elapsed_seconds.toFixed(1)}s elapsed</span>
          </div>
          <div className="flex items-center gap-1">
            <Database className="w-3.5 h-3.5 text-neutral-400" />
            <span>{status.facts_extracted_so_far} facts grounded</span>
          </div>
        </div>
      </div>

      {/* Current Step Description */}
      <div className="p-3 bg-neutral-50 dark:bg-neutral-850 border border-neutral-200 dark:border-neutral-800 text-xs">
        <div className="text-[10px] font-mono text-neutral-400 uppercase tracking-wider mb-1">
          Active Execution Step
        </div>
        <div className="font-mono text-neutral-900 dark:text-neutral-100 font-medium">
          {status.current_step}
        </div>
        {status.current_doc_name && (
          <div className="mt-1 text-[11px] text-neutral-500 flex items-center gap-1">
            <FileText className="w-3 h-3" />
            <span>
              Target: {status.current_doc_name} ({status.current_doc_index}/{status.total_docs})
            </span>
          </div>
        )}
      </div>

      {/* Completed Stages Chronological Log */}
      {status.stages_completed && status.stages_completed.length > 0 && (
        <div className="space-y-1.5 pt-1">
          <div className="text-[10px] font-mono uppercase tracking-wider text-neutral-400 mb-1">
            Completed Audit Stages
          </div>
          <div className="space-y-1 max-h-40 overflow-y-auto font-mono text-xs text-neutral-600 dark:text-neutral-400">
            {status.stages_completed.map((stage, i) => (
              <div key={i} className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400 shrink-0" />
                <span>{stage}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {status.error && (
        <div className="p-3 bg-red-50 border border-red-200 text-red-900 dark:bg-red-950/30 dark:border-red-900/50 dark:text-red-300 text-xs font-mono">
          Error: {status.error}
        </div>
      )}
    </div>
  );
};
