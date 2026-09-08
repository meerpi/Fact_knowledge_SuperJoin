/**
 * Plain-English human labels and descriptions for dispute codes
 * strictly avoiding raw enum strings like DISPUTE_TEMPORAL_DRIFT
 */

export interface DisputeMeta {
  label: string;
  description: string;
  severity: 'urgent' | 'moderate' | 'neutral';
}

export const DISPUTE_CODE_MAP: Record<string, DisputeMeta> = {
  // Genuine conflicts (urgent)
  DISPUTE_GENUINE_CONFLICT: {
    label: 'Genuine conflict',
    description: 'Irreconcilable figures reported for the same metric and context across sources.',
    severity: 'urgent',
  },
  DISPUTE_SIGN_MISMATCH: {
    label: 'Sign mismatch (profit vs loss)',
    description: 'One document reports positive performance while another reports negative.',
    severity: 'urgent',
  },
  DISPUTE_ORDER_OF_MAGNITUDE: {
    label: 'Scale error (>10x discrepancy)',
    description: 'Values differ by more than an order of magnitude, likely due to scale conversion errors.',
    severity: 'urgent',
  },

  // Reconcilable differences (moderate)
  DISPUTE_TEMPORAL_DRIFT: {
    label: 'Different reporting period',
    description: 'Values represent different fiscal years, quarters, or balance dates.',
    severity: 'moderate',
  },
  DISPUTE_UNIT_MISMATCH: {
    label: 'Unit or currency mismatch',
    description: 'Figures are denominated in differing units or currencies (e.g. % vs bps, INR vs USD).',
    severity: 'moderate',
  },
  DISPUTE_SCOPE_DIFFERENCE: {
    label: 'Consolidated vs Standalone boundary',
    description: 'One source reports group consolidated figures while another reports standalone entity figures.',
    severity: 'moderate',
  },
  DISPUTE_ACCOUNTING_BASIS: {
    label: 'Accounting basis difference',
    description: 'Discrepancy caused by accounting standard differences (e.g., GAAP vs Non-GAAP, pre-tax vs post-tax).',
    severity: 'moderate',
  },
  DISPUTE_ROUNDING: {
    label: 'Rounding variance (<1%)',
    description: 'Minor variance within normal reporting rounding tolerance.',
    severity: 'neutral',
  },

  // Agreements (neutral / positive)
  AGREEMENT_EXACT: {
    label: 'Exact numerical match',
    description: 'Identical canonical values verified across independent documents.',
    severity: 'neutral',
  },
  AGREEMENT_APPROXIMATE: {
    label: 'Approximate match',
    description: 'Values match within standard financial reporting tolerance.',
    severity: 'neutral',
  },

  // Fallbacks
  UNRESOLVED: {
    label: 'Unresolved discrepancy',
    description: 'Reason for variance could not be definitively determined.',
    severity: 'moderate',
  },
  DISPUTE_CUSTOM: {
    label: 'Domain-specific dispute',
    description: 'Specific domain difference noted in audit context.',
    severity: 'moderate',
  },
};

export function getDisputeMeta(code?: string | null): DisputeMeta {
  if (!code) {
    return {
      label: 'Uncategorized',
      description: 'Dispute category not specified.',
      severity: 'neutral',
    };
  }
  return (
    DISPUTE_CODE_MAP[code] || {
      label: code.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase()),
      description: 'Domain dispute classification.',
      severity: 'moderate',
    }
  );
}
