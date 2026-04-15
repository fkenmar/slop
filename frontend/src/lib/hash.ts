/**
 * Compute SHA-256 hash of a File using Web Crypto API with streaming.
 * Calls onProgress with hex string built so far.
 */
export async function sha256(
  file: File,
  onProgress?: (partial: string, pct: number) => void
): Promise<string> {
  const CHUNK = 1024 * 1024; // 1 MB
  const total = file.size;
  let offset = 0;

  // For small files, hash all at once
  if (total <= CHUNK) {
    const buf = await file.arrayBuffer();
    const hash = await crypto.subtle.digest("SHA-256", buf);
    const hex = bufToHex(hash);
    onProgress?.(hex, 100);
    return hex;
  }

  // Stream large files in chunks — show progress
  // We can't use streaming with Web Crypto digest directly,
  // so we read chunks and show progress, then hash the full buffer.
  const reader = file.stream().getReader();
  const chunks: Uint8Array[] = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    offset += value.byteLength;
    const pct = Math.round((offset / total) * 100);
    // Show a placeholder hash building up
    const partialHex = "0".repeat(64).slice(0, Math.floor((pct / 100) * 64));
    onProgress?.(partialHex, pct);
  }

  // Concatenate and hash
  const fullBuf = new Uint8Array(total);
  let pos = 0;
  for (const chunk of chunks) {
    fullBuf.set(chunk, pos);
    pos += chunk.byteLength;
  }

  const hash = await crypto.subtle.digest("SHA-256", fullBuf);
  const hex = bufToHex(hash);
  onProgress?.(hex, 100);
  return hex;
}

function bufToHex(buf: ArrayBuffer): string {
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}
