import type { PredictResponse } from "@/types/api";

export async function analyzeFaces(
  file: File,
  onUploadProgress?: (pct: number) => void
): Promise<PredictResponse> {
  const form = new FormData();
  form.append("image", file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/predict");

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        onUploadProgress?.(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      console.log("XHR response:", xhr.status, xhr.responseText.slice(0, 200));
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error("Invalid response from server"));
        }
      } else {
        reject(new Error(`Server error (${xhr.status})`));
      }
    };

    xhr.onerror = () => {
      console.error("XHR error event fired");
      reject(new Error("Network error — check your connection"));
    };
    xhr.ontimeout = () => reject(new Error("Request timed out"));
    xhr.timeout = 120_000;

    console.log("Sending request to /predict...");
    xhr.send(form);
  });
}
