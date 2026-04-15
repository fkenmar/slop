import { useRef, useEffect } from "react";
import type { FaceResult } from "@/types/api";

interface SideBySideViewProps {
  imageUrl: string;
  faces: FaceResult[];
}

export function SideBySideView({ imageUrl, faces }: SideBySideViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const draw = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);

      // Draw bounding boxes
      faces.forEach((face) => {
        if (!face.bbox) return;
        const [x1, y1, x2, y2] = face.bbox;
        const isDeepfake = face.label === "Deepfake";

        ctx.strokeStyle = isDeepfake ? "#b91c1c" : "#15803d";
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 4]);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.setLineDash([]);

        // Label background
        const labelText = `${face.label === "Deepfake" ? "FAKE" : "REAL"} ${face.confidence}%`;
        ctx.font = "bold 14px 'Segoe UI', sans-serif";
        const metrics = ctx.measureText(labelText);
        const labelH = 22;
        const labelY = y1 - labelH - 4;

        ctx.fillStyle = isDeepfake
          ? "rgba(185, 28, 28, 0.85)"
          : "rgba(21, 128, 61, 0.85)";
        ctx.fillRect(x1, labelY, metrics.width + 12, labelH);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(labelText, x1 + 6, labelY + 16);
      });
    };

    if (img.complete) {
      draw();
    } else {
      img.onload = draw;
    }
  }, [imageUrl, faces]);

  return (
    <div className="bg-surface-elevated border border-border rounded-xl p-5">
      <h3 className="text-xs font-semibold text-on-surface-muted uppercase tracking-wider mb-3">
        Image Analysis
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Original */}
        <div>
          <p className="text-xs text-on-surface-faint mb-2 font-medium">Original</p>
          <div className="bg-surface-sunken rounded-lg overflow-hidden">
            <img
              src={imageUrl}
              alt="Original upload"
              className="w-full h-auto object-contain"
            />
          </div>
        </div>

        {/* Annotated */}
        <div>
          <p className="text-xs text-on-surface-faint mb-2 font-medium">
            Annotated ({faces.length} face{faces.length !== 1 ? "s" : ""})
          </p>
          <div className="bg-surface-sunken rounded-lg overflow-hidden relative">
            <img
              ref={imgRef}
              src={imageUrl}
              alt=""
              className="hidden"
              crossOrigin="anonymous"
            />
            <canvas
              ref={canvasRef}
              className="w-full h-auto"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
