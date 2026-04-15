import { create } from "zustand";
import type { AnalysisJob, PredictResponse } from "@/types/api";
import { sha256 } from "@/lib/hash";
import { analyzeFaces } from "@/lib/api";

interface QueueState {
  jobs: AnalysisJob[];
  activeJobId: string | null;
  addJob: (file: File, thumbnail: string) => void;
  setActiveJob: (id: string | null) => void;
  retryJob: (id: string) => void;
  removeJob: (id: string) => void;
}

let jobCounter = 0;

function createJobId(): string {
  return `job-${Date.now()}-${++jobCounter}`;
}

export const useQueueStore = create<QueueState>((set, get) => ({
  jobs: [],
  activeJobId: null,

  setActiveJob: (id) => set({ activeJobId: id }),

  removeJob: (id) =>
    set((s) => ({ jobs: s.jobs.filter((j) => j.id !== id) })),

  addJob: (file, thumbnail) => {
    const id = createJobId();
    const job: AnalysisJob = {
      id,
      fileName: file.name,
      fileSize: file.size,
      sha256: null,
      thumbnail,
      status: "hashing",
      currentStep: 0,
      uploadProgress: 0,
      queuePosition: get().jobs.filter((j) => j.status !== "complete" && j.status !== "error").length + 1,
      eta: { min: 5, max: 15 },
      result: null,
      error: null,
      createdAt: Date.now(),
    };

    set((s) => ({ jobs: [job, ...s.jobs], activeJobId: id }));
    processJob(id, file, set, get);
  },

  retryJob: (id) => {
    const job = get().jobs.find((j) => j.id === id);
    if (!job || !job.error?.retryable) return;

    // We need the original file — store it in a WeakMap
    const file = fileCache.get(id);
    if (!file) return;

    update(set, id, {
      status: "uploading",
      currentStep: 2,
      error: null,
      uploadProgress: 0,
      eta: { min: 3, max: 10 },
    });

    runUploadAndAnalyze(id, file, set, get);
  },
}));

// File cache to support retries
const fileCache = new Map<string, File>();

function update(
  set: (fn: (s: QueueState) => Partial<QueueState>) => void,
  id: string,
  patch: Partial<AnalysisJob>
) {
  set((s) => ({
    jobs: s.jobs.map((j) => (j.id === id ? { ...j, ...patch } : j)),
  }));
}

async function processJob(
  id: string,
  file: File,
  set: (fn: (s: QueueState) => Partial<QueueState>) => void,
  get: () => QueueState
) {
  fileCache.set(id, file);

  // Step 1: Hash
  try {
    const hash = await sha256(file, (_partial, pct) => {
      update(set, id, { uploadProgress: pct });
    });
    update(set, id, { sha256: hash, status: "uploading", currentStep: 1, uploadProgress: 0 });
  } catch {
    update(set, id, {
      status: "error",
      error: { message: "Failed to compute file hash", retryable: true },
    });
    return;
  }

  await runUploadAndAnalyze(id, file, set, get);
}

async function runUploadAndAnalyze(
  id: string,
  file: File,
  set: (fn: (s: QueueState) => Partial<QueueState>) => void,
  _get: () => QueueState
) {
  // Step 2: Upload + analyze
  update(set, id, { status: "uploading", currentStep: 2, eta: { min: 3, max: 12 } });

  try {
    // Simulate step transitions during server-side processing
    let uploadDone = false;

    const result: PredictResponse = await analyzeFaces(file, (pct) => {
      update(set, id, { uploadProgress: pct });
      if (pct >= 100 && !uploadDone) {
        uploadDone = true;
        update(set, id, { status: "detecting", currentStep: 3, eta: { min: 2, max: 8 } });

        // Transition to "analyzing" after a delay
        setTimeout(() => {
          const job = _get().jobs.find((j) => j.id === id);
          if (job && job.status === "detecting") {
            update(set, id, { status: "analyzing", currentStep: 4, eta: { min: 1, max: 4 } });
          }
        }, 2000);
      }
    });

    update(set, id, {
      status: "complete",
      currentStep: 5,
      result,
      eta: null,
      uploadProgress: 100,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    const isTimeout = message.includes("timed out");
    const isNetwork = message.includes("Network");

    update(set, id, {
      status: "error",
      error: {
        message: isTimeout
          ? "Analysis timed out — the server may be under heavy load"
          : message,
        retryable: isTimeout || isNetwork,
      },
    });
  }
}
