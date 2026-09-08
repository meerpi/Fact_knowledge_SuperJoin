import React from 'react';
import { Database, ArrowRight } from 'lucide-react';

interface DemoDatasetBannerProps {
  onLoadDemo: () => void;
  isLoading: boolean;
}

export const DemoDatasetBanner: React.FC<DemoDatasetBannerProps> = ({
  onLoadDemo,
  isLoading,
}) => {
  return (
    <div className="mt-4 p-4 border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-850/60 flex flex-wrap items-center justify-between gap-3 text-xs">
      <div className="flex items-start gap-2.5">
        <Database className="w-4 h-4 text-neutral-600 dark:text-neutral-400 mt-0.5 shrink-0" />
        <div>
          <div className="font-semibold text-neutral-900 dark:text-neutral-100">
            Starter Dataset: India Macroeconomy (3 Reports)
          </div>
          <p className="text-neutral-500 text-[11px] mt-0.5">
            Pre-extracted baseline across Economic Survey 2024-25, RBI Annual Report, and IMF Article IV.
          </p>
        </div>
      </div>

      <button
        onClick={onLoadDemo}
        disabled={isLoading}
        className="px-3 py-1.5 font-mono text-xs font-medium border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 text-neutral-800 dark:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-800 inline-flex items-center gap-1.5 transition-colors disabled:opacity-50"
      >
        <span>{isLoading ? 'Loading Dataset...' : 'Load Starter Dataset'}</span>
        <ArrowRight className="w-3 h-3" />
      </button>
    </div>
  );
};
