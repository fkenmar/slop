import { AlertTriangle } from "lucide-react";

export function ExifWarning() {
  return (
    <div className="flex items-start gap-2.5 p-3 bg-warning-subtle border border-warning/20 rounded-lg">
      <AlertTriangle className="w-4 h-4 text-warning flex-shrink-0 mt-0.5" />
      <div>
        <p className="text-xs font-medium text-warning">
          Metadata advisory
        </p>
        <p className="text-xs text-warning/80 mt-0.5 leading-relaxed">
          Downloading or screenshotting images from social media strips EXIF
          metadata (camera model, GPS, timestamps) that aids forensic analysis.
          For best results, use the original file from the source device.
        </p>
      </div>
    </div>
  );
}
