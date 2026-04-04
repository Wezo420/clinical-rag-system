"use client";

import { AlertTriangle } from "lucide-react";

export function DisclaimerBanner() {
  return (
    <div className="flex items-start gap-3 p-4 rounded-xl border border-amber-400/20 bg-amber-400/5 animate-fade-in">
      <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
      <div className="text-sm">
        <p className="font-semibold text-amber-300 mb-0.5">Research Prototype — Not for Clinical Use</p>
        <p className="text-amber-200/60 leading-relaxed text-xs">
          This system generates AI-assisted hypotheses grounded in medical literature for{" "}
          <strong>research and educational purposes only</strong>. It is not a medical device,
          does not provide diagnoses, and must never be used to guide clinical decisions.
          Always consult a qualified healthcare professional.
        </p>
      </div>
    </div>
  );
}
