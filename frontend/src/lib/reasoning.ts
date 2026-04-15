import type { UncannyMetrics, FaceResult } from "@/types/api";

const METRIC_EXPLANATIONS: Record<
  keyof UncannyMetrics,
  { label: string; lowExplanation: string; highExplanation: string }
> = {
  symmetry: {
    label: "Facial Symmetry",
    lowExplanation:
      "Facial landmarks show notable asymmetry, which can indicate manipulation artifacts in the face generation process.",
    highExplanation:
      "Facial landmarks are well-balanced, consistent with natural facial proportions.",
  },
  eye_consistency: {
    label: "Eye Reflections",
    lowExplanation:
      "Eye reflections are inconsistent between left and right eyes — a common artifact in face-swap and GAN-generated deepfakes.",
    highExplanation:
      "Eye reflections are consistent across both eyes, matching expected real-world lighting.",
  },
  texture: {
    label: "Skin Texture",
    lowExplanation:
      "Skin lacks natural micro-texture in the frequency domain. AI-generated faces often produce unnaturally smooth skin.",
    highExplanation:
      "Skin texture shows natural high-frequency detail consistent with real camera sensor capture.",
  },
  edge_natural: {
    label: "Edge Naturalness",
    lowExplanation:
      "Face boundaries show abrupt edge transitions, suggesting blending artifacts from face insertion or compositing.",
    highExplanation:
      "Face boundary edges transition naturally into the surrounding image.",
  },
  color_consistency: {
    label: "Lighting Consistency",
    lowExplanation:
      "Lighting differs notably between face halves, which may indicate composited faces with mismatched source lighting.",
    highExplanation:
      "Lighting is uniform across the face, consistent with a single illumination source.",
  },
  noise_natural: {
    label: "Noise Pattern",
    lowExplanation:
      "The face region lacks natural camera sensor noise. AI-generated images typically have uniform or absent noise patterns.",
    highExplanation:
      "Natural sensor noise is present with expected non-uniform distribution across the face.",
  },
};

export function generateReasoning(face: FaceResult): string {
  const isDeepfake = face.label === "Deepfake";
  const { uncanny } = face;

  const lines: string[] = [];

  lines.push(
    isDeepfake
      ? `The model classified this face as a likely deepfake with ${face.confidence}% confidence.`
      : `The model classified this face as likely authentic with ${face.confidence}% confidence.`
  );

  // Find notable metrics
  const flagged: string[] = [];
  const clean: string[] = [];

  for (const [key, info] of Object.entries(METRIC_EXPLANATIONS)) {
    const val = uncanny[key as keyof UncannyMetrics];
    if (val === null || val === undefined) continue;

    if (val < 50) {
      flagged.push(`${info.label} (${val}%): ${info.lowExplanation}`);
    } else if (val >= 70) {
      clean.push(info.label);
    }
  }

  if (flagged.length > 0) {
    lines.push("");
    lines.push("Flagged indicators:");
    flagged.forEach((f) => lines.push(`  - ${f}`));
  }

  if (clean.length > 0 && !isDeepfake) {
    lines.push("");
    lines.push(
      `Passing indicators: ${clean.join(", ")} all scored within normal ranges.`
    );
  }

  if (isDeepfake && flagged.length === 0) {
    lines.push("");
    lines.push(
      "No individual heuristic was strongly flagged, but the neural network detected subtle patterns in the combined feature space that indicate manipulation."
    );
  }

  return lines.join("\n");
}

export { METRIC_EXPLANATIONS };
