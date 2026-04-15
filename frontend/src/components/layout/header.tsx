import { Shield } from "lucide-react";

export function Header() {
  return (
    <header className="border-b border-border bg-surface-elevated px-6 py-4 flex items-center gap-3">
      <Shield className="w-6 h-6 text-primary" strokeWidth={1.5} />
      <div>
        <h1 className="text-lg font-semibold tracking-tight text-on-surface">
          Deepfake Detector
        </h1>
        <p className="text-xs text-on-surface-muted">
          Forensic media analysis
        </p>
      </div>
    </header>
  );
}
