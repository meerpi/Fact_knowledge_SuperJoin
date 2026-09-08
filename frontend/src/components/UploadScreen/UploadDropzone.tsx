import React, { useRef, useState } from 'react';
import { UploadCloud, AlertCircle } from 'lucide-react';

interface UploadDropzoneProps {
  onUpload: (file: File) => Promise<void>;
  isUploading: boolean;
}

export const UploadDropzone: React.FC<UploadDropzoneProps> = ({
  onUpload,
  isUploading,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndUpload = async (file: File) => {
    setErrorMessage(null);

    // Guardrail 1: file type
    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
      setErrorMessage(`'${file.name}' is not a PDF. Only PDF documents can be analyzed for grounded facts.`);
      return;
    }

    // Guardrail 2: file size (50MB limit)
    const MAX_SIZE_MB = 50;
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setErrorMessage(
        `'${file.name}' is ${(file.size / (1024 * 1024)).toFixed(1)}MB, exceeding the ${MAX_SIZE_MB}MB limit.`
      );
      return;
    }

    try {
      await onUpload(file);
    } catch (err: any) {
      setErrorMessage(err.message || 'Upload failed. Please try again.');
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      for (let i = 0; i < e.dataTransfer.files.length; i++) {
        await validateAndUpload(e.dataTransfer.files[i]);
      }
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      for (let i = 0; i < e.target.files.length; i++) {
        await validateAndUpload(e.target.files[i]);
      }
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !isUploading && fileInputRef.current?.click()}
        className={`border-2 border-dashed p-8 text-center cursor-pointer transition-colors ${
          isDragOver
            ? 'border-neutral-900 bg-neutral-100 dark:border-neutral-100 dark:bg-neutral-800'
            : 'border-neutral-300 dark:border-neutral-700 hover:border-neutral-500 dark:hover:border-neutral-500 bg-white dark:bg-neutral-900'
        } ${isUploading ? 'opacity-60 cursor-not-allowed' : ''}`}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          accept=".pdf,application/pdf"
          multiple
          className="hidden"
        />

        <div className="flex flex-col items-center justify-center gap-2">
          <UploadCloud className="w-8 h-8 text-neutral-500 stroke-[1.5]" />
          <div className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
            {isUploading ? 'Parsing document structure...' : 'Drop PDF files here, or click to browse'}
          </div>
          <p className="text-xs text-neutral-500">
            Accepts financial reports, prospectuses, filings, or transcripts (up to 50MB per file)
          </p>
        </div>
      </div>

      {errorMessage && (
        <div className="mt-3 p-3 text-xs flex items-start gap-2 border border-red-200 bg-red-50 text-red-900 dark:border-red-900/60 dark:bg-red-950/30 dark:text-red-300">
          <AlertCircle className="w-4 h-4 shrink-0 text-red-600 dark:text-red-400 mt-0.5" />
          <div className="flex-1">{errorMessage}</div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-red-600 hover:underline font-mono"
          >
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
};
