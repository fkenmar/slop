interface EtaDisplayProps {
  eta: { min: number; max: number } | null;
  queuePosition: number;
}

export function EtaDisplay({ eta, queuePosition }: EtaDisplayProps) {
  if (!eta) return null;

  return (
    <div className="text-xs text-on-surface-faint mt-1 space-y-0.5">
      {queuePosition > 1 && (
        <p>
          Queue position: <span className="font-mono">#{queuePosition}</span>
        </p>
      )}
      <p>
        ETA:{" "}
        <span className="font-mono">
          ~{eta.min}-{eta.max}s
        </span>
      </p>
    </div>
  );
}
