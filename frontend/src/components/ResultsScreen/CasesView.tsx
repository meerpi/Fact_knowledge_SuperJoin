import React from 'react';
import type { AssignmentCases } from '../../api/types';
import { CompareCard } from '../CompareCard';

interface CasesViewProps {
  cases: AssignmentCases | null;
  onViewPdf?: (docId: string, page: number) => void;
}

export const CasesView: React.FC<CasesViewProps> = ({ cases, onViewPdf }) => {
  if (!cases) {
    return (
      <div className="p-8 text-center border border-dashed border-neutral-300 dark:border-neutral-700 text-neutral-500 font-mono text-sm">
        No cases available. Upload documents and run analysis to populate verification cases.
      </div>
    );
  }

  const {
    case_1_corroborated,
    case_2_contradicted,
    case_3_reconciled,
    case_4_extraction_failure,
  } = cases;

  return (
    <div className="space-y-6">
      {/* Informative Header Strip */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-1 pb-1">
        <div>
          <h2 className="text-sm font-bold tracking-tight text-neutral-900 dark:text-neutral-100">
            Graded Verification Cases
          </h2>
          <p className="text-xs text-neutral-500 mt-0.5">
            Deterministic claim arbitration across independent source documents. Click any card to expand reasoning.
          </p>
        </div>
        <div className="text-xs font-mono text-neutral-400">
          4 canonical cases evaluated
        </div>
      </div>

      {/* Case 1: Independent Corroboration */}
      {case_1_corroborated ? (
        <CompareCard
          caseTitle="Case 1 • Corroboration"
          subject={case_1_corroborated.subject}
          predicate={case_1_corroborated.predicate}
          verdict={case_1_corroborated.case_type}
          disputeCode={case_1_corroborated.dispute_code || 'AGREEMENT_EXACT'}
          disputeDetail={case_1_corroborated.dispute_detail}
          evidence={case_1_corroborated.evidence}
          explanation={case_1_corroborated.explanation}
          onViewPdf={onViewPdf}
        />
      ) : (
        <div className="p-4 border border-dashed border-neutral-200 text-xs font-mono text-neutral-400">
          Case 1 (Corroborated): No cross-document corroboration found in current dataset.
        </div>
      )}

      {/* Case 2: Genuine Contradiction */}
      {case_2_contradicted ? (
        <CompareCard
          caseTitle="Case 2 • Genuine Contradiction"
          subject={case_2_contradicted.subject}
          predicate={case_2_contradicted.predicate}
          verdict={case_2_contradicted.case_type}
          disputeCode={case_2_contradicted.dispute_code || 'DISPUTE_GENUINE_CONFLICT'}
          disputeDetail={case_2_contradicted.dispute_detail}
          evidence={case_2_contradicted.evidence}
          explanation={case_2_contradicted.explanation}
          consensusValue={case_2_contradicted.consensus_value}
          credibilityScores={case_2_contradicted.credibility_scores}
          onViewPdf={onViewPdf}
        />
      ) : (
        <div className="p-4 border border-dashed border-neutral-200 text-xs font-mono text-neutral-400">
          Case 2 (Contradicted): No genuine contradictions identified in current dataset.
        </div>
      )}

      {/* Case 3: Reconciled Apparent Contradiction */}
      {case_3_reconciled ? (
        <CompareCard
          caseTitle="Case 3 • Context Reconciliation"
          subject={case_3_reconciled.subject}
          predicate={case_3_reconciled.predicate}
          verdict={case_3_reconciled.case_type}
          disputeCode={case_3_reconciled.dispute_code || 'DISPUTE_TEMPORAL_DRIFT'}
          disputeDetail={case_3_reconciled.dispute_detail}
          reconciliationDimension={case_3_reconciled.reconciliation_dimension}
          evidence={case_3_reconciled.evidence}
          explanation={case_3_reconciled.explanation}
          onViewPdf={onViewPdf}
        />
      ) : (
        <div className="p-4 border border-dashed border-neutral-200 text-xs font-mono text-neutral-400">
          Case 3 (Reconciled): No apparent contradictions requiring contextual reconciliation found.
        </div>
      )}

      {/* Case 4: Extraction / Verification Failure */}
      {case_4_extraction_failure ? (
        <CompareCard
          caseTitle="Case 4 • Handled Pipeline Failure"
          verdict="extraction_failure"
          explanation="Quote verification failure detected by the 3-tier matching engine."
          failure={case_4_extraction_failure}
          onViewPdf={onViewPdf}
        />
      ) : (
        <div className="p-4 border border-dashed border-neutral-200 text-xs font-mono text-neutral-400">
          Case 4 (Extraction Failure): Zero extraction failures detected across all facts.
        </div>
      )}
    </div>
  );
};
