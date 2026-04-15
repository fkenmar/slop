import { cn } from "@/lib/utils";

interface HashVisualizerProps {
  hash: string | null;
  isHashing: boolean;
  progress: number;
}

export function HashVisualizer({ hash, isHashing, progress }: HashVisualizerProps) {
  if (!isHashing && !hash) return null;

  const displayHash = hash ?? "0".repeat(64);
  const revealedChars = isHashing
    ? Math.floor((progress / 100) * 64)
    : 64;

  return (
    <div className="bg-surface-sunken border border-border-muted rounded-lg p-3">
      <div className="flex items-center gap-2 mb-1.5">
        <div
          className={cn(
            "w-2 h-2 rounded-full",
            isHashing ? "bg-primary animate-pulse" : "bg-success"
          )}
        />
        <span className="text-xs font-medium text-on-surface-muted uppercase tracking-wider">
          SHA-256 {isHashing ? "Computing..." : "Verified"}
        </span>
      </div>
      <p className="font-mono text-xs text-on-surface break-all leading-relaxed">
        <span className="text-on-surface">
          {displayHash.slice(0, revealedChars)}
        </span>
        <span className="text-on-surface-faint/30">
          {displayHash.slice(revealedChars)}
        </span>
      </p>
    </div>
  );
}
