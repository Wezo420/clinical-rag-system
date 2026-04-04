"use client";

import { useState } from "react";
import { ClinicalInputForm } from "@/components/ClinicalInputForm";
import { ResultsDashboard } from "@/components/ResultsDashboard";
import { Header } from "@/components/Header";
import { DisclaimerBanner } from "@/components/DisclaimerBanner";
import type { AnalysisResult } from "@/api/client";

export default function HomePage() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-8 space-y-8">
        <DisclaimerBanner />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          {/* Input Panel */}
          <div className="space-y-6">
            <div className="animate-slide-up">
              <h2 className="font-display text-2xl font-bold text-white mb-1">
                Clinical Case Input
              </h2>
              <p className="text-sm text-blue-300/70">
                Provide symptoms, notes, and optional imaging for AI-assisted hypothesis generation.
              </p>
            </div>

            <ClinicalInputForm
              onResult={setResult}
              onAnalyzing={setIsAnalyzing}
            />
          </div>

          {/* Results Panel */}
          <div className="animate-fade-in" style={{ animationDelay: "0.1s" }}>
            <ResultsDashboard
              result={result}
              isLoading={isAnalyzing}
            />
          </div>
        </div>
      </main>

      <footer className="border-t border-white/5 py-4 px-8 text-center text-xs text-blue-300/40">
        ClinicalRAG v1.0 — Research prototype. Powered by Groq + FAISS + CLIP.
        Not a medical device. Not for clinical use.
      </footer>
    </div>
  );
}
