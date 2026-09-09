import React from 'react';
import { ShieldAlert } from 'lucide-react';
import type { SystemCapabilities } from '../api/types';

interface SystemCapabilitiesBannerProps {
  capabilities: SystemCapabilities | null;
}

export const SystemCapabilitiesBanner: React.FC<SystemCapabilitiesBannerProps> = ({
  capabilities,
}) => {
  if (!capabilities) return null;

  const {
    gemini_api_key_configured,
    voyage_api_key_configured,
    docling_available,
    nli_model_available,
  } = capabilities;

  const degradedItems: string[] = [];
  if (!gemini_api_key_configured) {
    degradedItems.push('Gemini API key missing: live extraction disabled');
  }
  if (!voyage_api_key_configured) {
    degradedItems.push('Voyage AI offline: falling back to lexical entity/metric alignment');
  }
  if (!docling_available) {
    degradedItems.push('Docling ML parser offline: falling back to pdfplumber text parsing');
  }
  if (!nli_model_available) {
    degradedItems.push('DeBERTa-v3 NLI offline: qualitative claim NLI stage skipped');
  }

  // Only render when there are degraded capabilities — no banner when everything works
  if (degradedItems.length === 0) {
    return null;
  }

  return (
    <div className="border-b border-amber-300 dark:border-amber-800/60 bg-amber-50 dark:bg-amber-950/40 px-4 py-2 text-xs text-amber-950 dark:text-amber-200">
      <div className="flex items-start gap-2">
        <ShieldAlert className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
        <div className="space-y-0.5 flex-1">
          <span className="font-semibold font-mono">Degraded capabilities:</span>
          <ul className="list-disc list-inside font-mono text-[11px] text-amber-900/90 dark:text-amber-300/90">
            {degradedItems.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};
