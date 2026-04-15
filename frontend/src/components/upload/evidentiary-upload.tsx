import { useState, useRef, useCallback } from "react";
import { cn } from "@/lib/utils";
import { useQueueStore } from "@/stores/queue-store";
import { HashVisualizer } from "./hash-visualizer";
import { ExifWarning } from "./exif-warning";
import { sha256 } from "@/lib/hash";
import { Upload, FileImage } from "lucide-react";

export function EvidentiaryUpload() {
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [isHashing, setIsHashing] = useState(false);
  const [hashProgress, setHashProgress] = useState(0);
  const [hash, setHash] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const addJob = useQueueStore((s) => s.addJob);

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) return;

    setSelectedFile(file);
    setHash(null);
    setIsHashing(true);
    setHashProgress(0);

    // Generate thumbnail
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);

    // Compute SHA-256
    const fileHash = await sha256(file, (_partial, pct) => {
      setHashProgress(pct);
    });
    setHash(fileHash);
    setIsHashing(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleSubmit = () => {
    if (!selectedFile || !preview) return;
    addJob(selectedFile, preview);
    // Reset
    setSelectedFile(null);
    setPreview(null);
    setHash(null);
    setIsHashing(false);
    setHashProgress(0);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    setHash(null);
    setIsHashing(false);
    setHashProgress(0);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="max-w-xl mx-auto space-y-4">
      <div>
        <h2 className="text-base font-semibold text-on-surface mb-1">
          Evidence Upload
        </h2>
        <p className="text-sm text-on-surface-muted">
          Upload media for forensic deepfake analysis
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={cn(
          "relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all",
          "hover:border-accent hover:bg-accent-subtle/20",
          isDragOver
            ? "border-primary bg-primary/5"
            : preview
              ? "border-border bg-surface-elevated"
              : "border-border-muted bg-surface-elevated"
        )}
      >
        {preview ? (
          <div className="flex flex-col items-center gap-3">
            <img
              src={preview}
              alt="Preview"
              className="max-h-48 rounded-lg object-contain"
            />
            <p className="text-sm text-on-surface-muted">
              {selectedFile?.name}
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 py-4">
            <div className="w-12 h-12 rounded-full bg-surface-sunken flex items-center justify-center">
              {isDragOver ? (
                <FileImage className="w-5 h-5 text-primary" />
              ) : (
                <Upload className="w-5 h-5 text-on-surface-faint" />
              )}
            </div>
            <div>
              <p className="text-sm font-medium text-on-surface">
                Drop image here or click to browse
              </p>
              <p className="text-xs text-on-surface-faint mt-1">
                PNG, JPG, WebP supported
              </p>
            </div>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
      </div>

      {/* Hash visualizer */}
      <HashVisualizer
        hash={hash}
        isHashing={isHashing}
        progress={hashProgress}
      />

      {/* EXIF warning */}
      {selectedFile && <ExifWarning />}

      {/* Action buttons */}
      {selectedFile && (
        <div className="flex gap-3">
          <button
            onClick={handleClear}
            className="flex-1 py-2.5 px-4 rounded-lg border border-border text-sm font-medium text-on-surface-muted hover:bg-surface-sunken transition-colors"
          >
            Clear
          </button>
          <button
            onClick={handleSubmit}
            disabled={isHashing}
            className={cn(
              "flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-colors",
              "bg-primary text-on-primary hover:bg-primary-hovered",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
          >
            {isHashing ? "Computing hash..." : "Analyze"}
          </button>
        </div>
      )}
    </div>
  );
}
