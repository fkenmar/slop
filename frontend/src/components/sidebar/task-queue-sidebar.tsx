import { useQueueStore } from "@/stores/queue-store";
import { QueueItem } from "./queue-item";
import { ListChecks } from "lucide-react";

export function TaskQueueSidebar() {
  const { jobs, activeJobId } = useQueueStore();

  const pending = jobs.filter(
    (j) => j.status !== "complete" && j.status !== "error"
  ).length;

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border-muted">
        <div className="flex items-center gap-2">
          <ListChecks className="w-4 h-4 text-on-surface-muted" strokeWidth={1.5} />
          <h2 className="text-sm font-semibold text-on-surface">
            Analysis Queue
          </h2>
          {pending > 0 && (
            <span className="ml-auto text-xs font-mono bg-primary text-on-primary px-1.5 py-0.5 rounded">
              {pending}
            </span>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {jobs.length === 0 ? (
          <div className="p-6 text-center">
            <p className="text-sm text-on-surface-faint">
              No files analyzed yet
            </p>
            <p className="text-xs text-on-surface-faint mt-1">
              Upload an image to begin
            </p>
          </div>
        ) : (
          jobs.map((job) => (
            <QueueItem
              key={job.id}
              job={job}
              isActive={job.id === activeJobId}
            />
          ))
        )}
      </div>
    </div>
  );
}
