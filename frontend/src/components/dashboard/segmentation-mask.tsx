import { useState } from "react";
import { Layers } from "lucide-react";
import { cn } from "@/lib/utils";

interface SegmentationMaskProps {
  gradcam: string | null;
  originalImage: string | null;
}

export function SegmentationMask({ gradcam, originalImage }: SegmentationMaskProps) {
  const [opacity, setOpacity] = useState(50);

  if (!gradcam) {
    return (
      <div className="bg-surface-elevated border border-border-muted rounded-xl p-5">
        <div className="flex items-center gap-2 mb-2">
          <Layers className="w-4 h-4 text-on-surface-faint" strokeWidth={1.5} />
          <h3 className="text-xs font-semibold text-on-surface-muted uppercase tracking-wider">
            XAI Segmentation Mask
          </h3>
        </div>
        <div className="bg-surface-sunken rounded-lg p-8 text-center">
          <p className="text-sm text-on-surface-faint">
            GradCAM data not available for this analysis.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-surface-elevated border border-border rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-on-surface-muted" strokeWidth={1.5} />
          <h3 className="text-xs font-semibold text-on-surface-muted uppercase tracking-wider">
            XAI Attention Map (GradCAM)
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-on-surface-faint">Overlay</span>
          <input
            type="range"
            min={0}
            max={100}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="w-24 h-1.5 accent-primary"
          />
          <span className="text-xs text-on-surface-faint font-mono w-8">{opacity}%</span>
        </div>
      </div>

      <div className="relative bg-surface-sunken rounded-lg overflow-hidden">
        {originalImage && (
          <img
            src={originalImage}
            alt="Original"
            className="w-full h-auto object-contain"
          />
        )}
        <img
          src={`data:image/png;base64,${gradcam}`}
          alt="GradCAM attention heatmap"
          className="absolute inset-0 w-full h-full object-contain mix-blend-normal"
          style={{ opacity: opacity / 100 }}
        />
      </div>

      <div className="flex items-center justify-between mt-3">
        <div className="flex items-center gap-4 text-xs text-on-surface-faint">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-sm bg-blue-600" /> Low attention
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-sm bg-green-500" /> Medium
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-sm bg-red-600" /> High attention
          </span>
        </div>
        <p className="text-xs text-on-surface-faint">
          Regions the model focused on for its decision
        </p>
      </div>
    </div>
  );
}
