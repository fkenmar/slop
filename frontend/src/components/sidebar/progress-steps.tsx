import { cn } from "@/lib/utils";
import { Check, Loader2 } from "lucide-react";

const STEPS = [
  "Hashing",
  "Uploading",
  "Uploading",
  "Face detection",
  "Model inference",
  "Complete",
];

interface ProgressStepsProps {
  currentStep: number;
}

export function ProgressSteps({ currentStep }: ProgressStepsProps) {
  // Deduplicate display steps
  const displaySteps = [
    { label: "Hash verification", step: 0 },
    { label: "File upload", step: 2 },
    { label: "Face detection", step: 3 },
    { label: "Model inference", step: 4 },
  ];

  return (
    <div className="flex flex-col gap-1">
      {displaySteps.map(({ label, step }) => {
        const done = currentStep > step;
        const active = currentStep === step || (step === 2 && currentStep === 1);

        return (
          <div key={step} className="flex items-center gap-2 py-0.5">
            <div
              className={cn(
                "w-4 h-4 rounded-full flex items-center justify-center flex-shrink-0",
                done && "bg-success text-on-success",
                active && "bg-primary text-on-primary",
                !done && !active && "bg-surface-sunken text-on-surface-faint"
              )}
            >
              {done ? (
                <Check className="w-2.5 h-2.5" strokeWidth={3} />
              ) : active ? (
                <Loader2 className="w-2.5 h-2.5 animate-spin" strokeWidth={3} />
              ) : (
                <span className="w-1.5 h-1.5 rounded-full bg-current opacity-40" />
              )}
            </div>
            <span
              className={cn(
                "text-xs",
                done && "text-on-surface-muted",
                active && "text-on-surface font-medium",
                !done && !active && "text-on-surface-faint"
              )}
            >
              {label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export { STEPS };
