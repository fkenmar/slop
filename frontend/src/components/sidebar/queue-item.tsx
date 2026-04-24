import { cn } from "@/lib/utils";
import type { AnalysisJob } from "@/types/api";
import { ProgressSteps } from "./progress-steps";
import { EtaDisplay } from "./eta-display";
import { AlertTriangle, CheckCircle2, RefreshCw, X, FileImage } from "lucide-react";
import { useQueueStore } from "@/stores/queue-store";

interface QueueItemProps {
  job: AnalysisJob;
  isActive: boolean;
}

export function QueueItem({ job, isActive }: QueueItemProps) {
  const { setActiveJob, retryJob, removeJob } = useQueueStore();

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => setActiveJob(job.id)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          setActiveJob(job.id);
        }
      }}
      className={cn(
        "w-full text-left p-3 border-b border-border-muted transition-colors cursor-pointer",
        "hover:bg-surface-sunken/50 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary",
        isActive && "bg-accent-subtle/40 border-l-2 border-l-primary"
      )}
    >
      <div className="flex items-start gap-2.5">
        {/* Thumbnail */}
        <div className="w-10 h-10 rounded-md bg-surface-sunken overflow-hidden flex-shrink-0 flex items-center justify-center">
          {job.thumbnail ? (
            <img
              src={job.thumbnail}
              alt=""
              className="w-full h-full object-cover"
            />
          ) : (
            <FileImage className="w-4 h-4 text-on-surface-faint" />
          )}
        </div>

        <div className="flex-1 min-w-0">
          {/* File name */}
          <p className="text-sm font-medium text-on-surface truncate">
            {job.fileName}
          </p>
          <p className="text-xs text-on-surface-faint font-mono">
            {(job.fileSize / 1024).toFixed(0)} KB
          </p>

          {/* Status */}
          {job.status === "complete" && job.result && (
            <div className="flex items-center gap-1.5 mt-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-success" />
              <span className="text-xs text-success font-medium">
                {job.result.face_count} face{job.result.face_count !== 1 ? "s" : ""} analyzed
              </span>
            </div>
          )}

          {job.status === "error" && (
            <div className="mt-1.5">
              <div className="flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 text-danger" />
                <span className="text-xs text-danger font-medium">
                  {job.error?.message ?? "Analysis failed"}
                </span>
              </div>
              {job.error?.retryable && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    retryJob(job.id);
                  }}
                  className="mt-1 flex items-center gap-1 text-xs text-primary hover:text-primary-hovered font-medium"
                >
                  <RefreshCw className="w-3 h-3" />
                  Retry
                </button>
              )}
              {job.error?.partialResult && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setActiveJob(job.id);
                  }}
                  className="mt-0.5 text-xs text-on-surface-muted hover:text-on-surface underline"
                >
                  View partial results
                </button>
              )}
            </div>
          )}

          {job.status !== "complete" && job.status !== "error" && (
            <div className="mt-2">
              <ProgressSteps currentStep={job.currentStep} />
              <EtaDisplay eta={job.eta} queuePosition={job.queuePosition} />
            </div>
          )}
        </div>

        {/* Remove button */}
        {(job.status === "complete" || job.status === "error") && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              removeJob(job.id);
            }}
            className="p-1 text-on-surface-faint hover:text-on-surface rounded"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}
