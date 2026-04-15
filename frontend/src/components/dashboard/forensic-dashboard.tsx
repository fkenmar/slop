import type { AnalysisJob } from "@/types/api";
import { ConfidenceGauge } from "./confidence-gauge";
import { UncannyMetricsPanel } from "./uncanny-metrics";
import { AiReasoning } from "./ai-reasoning";
import { SideBySideView } from "./side-by-side-view";
import { SegmentationMask } from "./segmentation-mask";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { useQueueStore } from "@/stores/queue-store";
import { RotateCcw } from "lucide-react";

interface ForensicDashboardProps {
  job: AnalysisJob;
}

export function ForensicDashboard({ job }: ForensicDashboardProps) {
  const result = job.result ?? job.error?.partialResult;
  const [activeFaceIdx, setActiveFaceIdx] = useState(0);
  const { setActiveJob, removeJob } = useQueueStore();

  const handleNewAnalysis = () => {
    removeJob(job.id);
    setActiveJob(null);
  };

  if (!result) {
    return (
      <div className="flex items-center justify-center h-64 text-on-surface-faint text-sm">
        No results available
      </div>
    );
  }

  const face = result.faces[activeFaceIdx];
  if (!face) return null;

  return (
    <div className="space-y-4 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-on-surface">
            Forensic Report
          </h2>
          <p className="text-sm text-on-surface-muted">
            {job.fileName}
            {job.sha256 && (
              <span className="ml-2 font-mono text-xs text-on-surface-faint">
                SHA-256: {job.sha256.slice(0, 12)}...
              </span>
            )}
          </p>
        </div>
        <button
          onClick={handleNewAnalysis}
          className="flex items-center gap-1.5 px-3 py-2 text-sm font-medium text-on-surface-muted hover:text-on-surface bg-surface-elevated border border-border rounded-lg hover:bg-surface-sunken transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          New Analysis
        </button>

        {/* Face selector tabs */}
        {result.faces.length > 1 && (
          <div className="flex gap-1 bg-surface-sunken rounded-lg p-0.5">
            {result.faces.map((f, i) => (
              <button
                key={i}
                onClick={() => setActiveFaceIdx(i)}
                className={cn(
                  "px-3 py-1.5 text-xs font-medium rounded-md transition-colors",
                  i === activeFaceIdx
                    ? "bg-surface-elevated text-on-surface shadow-sm"
                    : "text-on-surface-muted hover:text-on-surface"
                )}
              >
                Face {i + 1}
                <span
                  className={cn(
                    "ml-1.5 inline-block w-2 h-2 rounded-full",
                    f.label === "Deepfake" ? "bg-danger" : "bg-success"
                  )}
                />
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Side-by-side image view */}
      {job.thumbnail && (
        <SideBySideView imageUrl={job.thumbnail} faces={result.faces} />
      )}

      {/* Confidence + Metrics grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ConfidenceGauge confidence={face.confidence} label={face.label} />
        <UncannyMetricsPanel uncanny={face.uncanny} />
      </div>

      {/* XAI Attention Map */}
      <SegmentationMask gradcam={face.gradcam} originalImage={job.thumbnail} />

      {/* AI Reasoning */}
      <AiReasoning face={face} />
    </div>
  );
}
