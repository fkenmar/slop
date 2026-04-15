import type { FaceResult } from "@/types/api";
import { generateReasoning } from "@/lib/reasoning";
import { FileText } from "lucide-react";

interface AiReasoningProps {
  face: FaceResult;
}

export function AiReasoning({ face }: AiReasoningProps) {
  const reasoning = generateReasoning(face);

  return (
    <div className="bg-surface-elevated border border-border rounded-xl p-5">
      <div className="flex items-center gap-2 mb-3">
        <FileText className="w-4 h-4 text-on-surface-muted" strokeWidth={1.5} />
        <h3 className="text-xs font-semibold text-on-surface-muted uppercase tracking-wider">
          Analysis Reasoning
        </h3>
      </div>
      <pre className="text-sm text-on-surface leading-relaxed whitespace-pre-wrap font-sans">
        {reasoning}
      </pre>
    </div>
  );
}
