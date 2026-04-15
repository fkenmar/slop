export interface UncannyMetrics {
  symmetry: number | null;
  eye_consistency: number | null;
  texture: number | null;
  edge_natural: number | null;
  color_consistency: number | null;
  noise_natural: number | null;
}

export interface FaceResult {
  label: "Realism" | "Deepfake";
  confidence: number;
  uncanny: UncannyMetrics;
  bbox: [number, number, number, number] | null;
  gradcam: string | null;
}

export interface PredictResponse {
  faces: FaceResult[];
  face_count: number;
  face_detected: boolean;
}

export type JobStatus =
  | "queued"
  | "hashing"
  | "uploading"
  | "detecting"
  | "analyzing"
  | "complete"
  | "error";

export interface AnalysisJob {
  id: string;
  fileName: string;
  fileSize: number;
  sha256: string | null;
  thumbnail: string | null;
  status: JobStatus;
  currentStep: number;
  uploadProgress: number;
  queuePosition: number;
  eta: { min: number; max: number } | null;
  result: PredictResponse | null;
  error: {
    message: string;
    retryable: boolean;
    partialResult?: PredictResponse;
  } | null;
  createdAt: number;
}
