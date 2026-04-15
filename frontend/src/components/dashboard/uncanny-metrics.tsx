import { cn } from "@/lib/utils";
import type { UncannyMetrics } from "@/types/api";
import { METRIC_EXPLANATIONS } from "@/lib/reasoning";

interface UncannyMetricsProps {
  uncanny: UncannyMetrics;
}

function gradeClass(val: number): { bg: string; label: string; pattern: string } {
  if (val >= 70) return { bg: "bg-success", label: "Normal", pattern: "" };
  if (val >= 40) return { bg: "bg-warning", label: "Suspect", pattern: "pattern-diagonal" };
  return { bg: "bg-danger", label: "Flagged", pattern: "pattern-crosshatch" };
}

export function UncannyMetricsPanel({ uncanny }: UncannyMetricsProps) {
  const entries = Object.entries(METRIC_EXPLANATIONS) as Array<
    [keyof UncannyMetrics, (typeof METRIC_EXPLANATIONS)[keyof UncannyMetrics]]
  >;

  return (
    <div className="bg-surface-elevated border border-border rounded-xl p-5">
      <h3 className="text-xs font-semibold text-on-surface-muted uppercase tracking-wider mb-4">
        Forensic Heuristics
      </h3>

      {/* SVG pattern definitions */}
      <svg className="absolute w-0 h-0" aria-hidden="true">
        <defs>
          <pattern
            id="pat-diagonal"
            patternUnits="userSpaceOnUse"
            width="6"
            height="6"
            patternTransform="rotate(45)"
          >
            <line x1="0" y1="0" x2="0" y2="6" stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" />
          </pattern>
          <pattern
            id="pat-crosshatch"
            patternUnits="userSpaceOnUse"
            width="6"
            height="6"
          >
            <line x1="0" y1="0" x2="6" y2="6" stroke="rgba(255,255,255,0.4)" strokeWidth="1" />
            <line x1="6" y1="0" x2="0" y2="6" stroke="rgba(255,255,255,0.4)" strokeWidth="1" />
          </pattern>
        </defs>
      </svg>

      <div className="space-y-3.5">
        {entries.map(([key, info]) => {
          const val = uncanny[key];
          if (val === null || val === undefined) return null;
          const grade = gradeClass(val);

          return (
            <div key={key}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm text-on-surface">{info.label}</span>
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      "text-xs font-medium px-1.5 py-0.5 rounded",
                      val >= 70 && "bg-success-subtle text-success",
                      val >= 40 && val < 70 && "bg-warning-subtle text-warning",
                      val < 40 && "bg-danger-subtle text-danger"
                    )}
                  >
                    {grade.label}
                  </span>
                  <span className="text-sm font-mono text-on-surface-muted w-10 text-right">
                    {val}%
                  </span>
                </div>
              </div>

              {/* Bar with accessible pattern overlay */}
              <div className="relative h-2.5 bg-surface-sunken rounded-full overflow-hidden">
                <div
                  className={cn("h-full rounded-full transition-all duration-500", grade.bg)}
                  style={{ width: `${val}%` }}
                />
                {/* Pattern overlay for accessibility */}
                {grade.pattern && (
                  <svg
                    className="absolute inset-0 w-full h-full"
                    style={{ clipPath: `inset(0 ${100 - val}% 0 0)` }}
                  >
                    <rect
                      width="100%"
                      height="100%"
                      fill={`url(#${grade.pattern === "pattern-diagonal" ? "pat-diagonal" : "pat-crosshatch"})`}
                    />
                  </svg>
                )}
              </div>

              <p className="text-xs text-on-surface-faint mt-0.5">
                {val < 50 ? info.lowExplanation : info.highExplanation}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
