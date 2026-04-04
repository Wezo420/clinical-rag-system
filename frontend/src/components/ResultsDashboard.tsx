"use client";

import {
  AlertTriangle, BookOpen, BarChart2, CheckCircle2,
  ChevronDown, ChevronUp, ExternalLink, Shield, XCircle,
} from "lucide-react";
import { useState } from "react";
import type { AnalysisResult, ConditionHypothesis, MedicalEvidence, SafetyFlag } from "@/api/client";
import { ConfidenceChart } from "./ConfidenceChart";

interface Props {
  result: AnalysisResult | null;
  isLoading: boolean;
}

export function ResultsDashboard({ result, isLoading }: Props) {
  if (isLoading) return <LoadingSkeleton />;
  if (!result) return <EmptyState />;

  return (
    <div className="space-y-5 animate-fade-in">
      {result.safety_flags.length > 0 && <SafetyFlagsPanel flags={result.safety_flags} />}

      <div className="disclaimer-banner flex gap-2 items-start">
        <Shield className="w-4 h-4 flex-shrink-0 mt-0.5 text-rose-400" />
        <p>{result.disclaimer}</p>
      </div>

      {/* Summary */}
      <div className="card p-5 space-y-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <h3 className="font-display text-lg font-bold text-white">Summary</h3>
          <div className="flex items-center gap-2">
            <div className={`badge ${result.confidence_level === "high" ? "badge-high" : result.confidence_level === "medium" ? "badge-medium" : "badge-low"}`}>
              {result.confidence_level} confidence
            </div>
            <span className="text-xs text-blue-300/40 font-mono">{(result.confidence_overall * 100).toFixed(0)}%</span>
          </div>
        </div>
        <p className="text-sm text-blue-100/80 leading-relaxed">{result.summary}</p>
        <div className="flex flex-wrap gap-x-3 gap-y-1 pt-1 text-xs text-blue-300/50">
          <span>{result.model_used}</span>
          <span>·</span>
          <span>{result.retrieval_count} sources</span>
          <span>·</span>
          <span>{result.processing_time_ms}ms</span>
          {result.structured_data_used && <><span>·</span><span className="text-clinical-400">📊 labs used</span></>}
          {result.image_analysis_summary && <><span>·</span><span className="text-clinical-400">🖼 image analyzed</span></>}
        </div>
      </div>

      {result.condition_hypotheses.length > 1 && (
        <ConfidenceChart hypotheses={result.condition_hypotheses} />
      )}

      {result.condition_hypotheses.length > 0 && (
        <div className="card p-5 space-y-4">
          <h3 className="font-display text-lg font-bold text-white flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-clinical-400 animate-pulse-slow" />
            Possible Hypotheses
          </h3>
          <div className="space-y-3 stagger">
            {result.condition_hypotheses.map((h, i) => <HypothesisCard key={i} hypothesis={h} index={i} />)}
          </div>
        </div>
      )}

      {result.differential_reasoning && (
        <div className="card p-5 space-y-3">
          <h3 className="font-display text-lg font-bold text-white">Differential Reasoning</h3>
          <p className="text-sm text-blue-100/80 leading-relaxed whitespace-pre-wrap">{result.differential_reasoning}</p>
        </div>
      )}

      {result.evidence.length > 0 && <EvidencePanel evidence={result.evidence} />}

      {result.image_analysis_summary && (
        <div className="card p-4 space-y-2">
          <h3 className="text-sm font-semibold text-clinical-300 uppercase tracking-wider">🖼 Image Analysis</h3>
          <p className="text-sm text-blue-100/70">{result.image_analysis_summary}</p>
        </div>
      )}

      <p className="text-xs text-blue-300/20 text-right font-mono">ID: {result.result_id}</p>
    </div>
  );
}

function HypothesisCard({ hypothesis, index }: { hypothesis: ConditionHypothesis; index: number }) {
  const [expanded, setExpanded] = useState(index === 0);
  const pct = Math.round(hypothesis.confidence * 100);
  const fillColor = hypothesis.confidence_level === "high" ? "#10b981" : hypothesis.confidence_level === "medium" ? "#f59e0b" : "#f43f5e";

  return (
    <div className={`rounded-lg border transition-all animate-slide-up ${expanded ? "border-clinical-500/30 bg-clinical-500/5" : "border-white/5 hover:border-white/10"}`}>
      <button type="button" onClick={() => setExpanded(!expanded)} className="w-full flex items-center gap-3 p-3.5 text-left">
        <span className="w-6 h-6 rounded-full bg-clinical-500/20 text-clinical-300 text-xs font-bold flex items-center justify-center flex-shrink-0">{index + 1}</span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-white">{hypothesis.condition}</p>
          {hypothesis.icd10_code && <p className="text-xs text-blue-300/40 font-mono">{hypothesis.icd10_code}</p>}
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <div className={`badge ${hypothesis.confidence_level === "high" ? "badge-high" : hypothesis.confidence_level === "medium" ? "badge-medium" : "badge-low"}`}>{pct}%</div>
          {expanded ? <ChevronUp className="w-4 h-4 text-blue-300/40" /> : <ChevronDown className="w-4 h-4 text-blue-300/40" />}
        </div>
      </button>
      <div className="px-3.5 pb-2">
        <div className="confidence-bar"><div className="confidence-fill" style={{ width: `${pct}%`, background: fillColor }} /></div>
      </div>
      {expanded && (
        <div className="px-4 pb-4 pt-2 space-y-3 border-t border-white/5 mt-1">
          {hypothesis.supporting_factors.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-emerald-400 mb-1.5 flex items-center gap-1"><CheckCircle2 className="w-3 h-3" /> Supporting</p>
              <ul className="space-y-0.5">{hypothesis.supporting_factors.map((f, i) => <li key={i} className="text-xs text-blue-100/70 flex gap-1.5"><span className="text-emerald-500 mt-0.5">+</span>{f}</li>)}</ul>
            </div>
          )}
          {hypothesis.against_factors.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-rose-400 mb-1.5 flex items-center gap-1"><XCircle className="w-3 h-3" /> Against</p>
              <ul className="space-y-0.5">{hypothesis.against_factors.map((f, i) => <li key={i} className="text-xs text-blue-100/70 flex gap-1.5"><span className="text-rose-500 mt-0.5">−</span>{f}</li>)}</ul>
            </div>
          )}
          {hypothesis.recommended_workup.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-clinical-300 mb-1.5">Workup</p>
              <div className="flex flex-wrap gap-1.5">{hypothesis.recommended_workup.map((w, i) => <span key={i} className="text-xs px-2 py-0.5 rounded-full bg-clinical-500/15 text-clinical-300 border border-clinical-500/20">{w}</span>)}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function EvidencePanel({ evidence }: { evidence: MedicalEvidence[] }) {
  const [expanded, setExpanded] = useState(false);
  const visible = expanded ? evidence : evidence.slice(0, 3);
  return (
    <div className="card p-5 space-y-4">
      <h3 className="font-display text-lg font-bold text-white flex items-center gap-2">
        <BookOpen className="w-5 h-5 text-clinical-400" />Evidence ({evidence.length} sources)
      </h3>
      <div className="space-y-3">
        {visible.map((ev) => (
          <div key={ev.source_id} className="p-3.5 rounded-lg bg-clinical-950/40 border border-white/5 hover:border-white/10 transition-colors">
            <div className="flex items-start justify-between gap-2 mb-1.5">
              <p className="text-sm font-medium text-white leading-snug">{ev.title}</p>
              <div className="flex items-center gap-1.5 flex-shrink-0">
                <span className="text-xs px-1.5 py-0.5 rounded bg-clinical-500/20 text-clinical-300 font-mono">{(ev.relevance_score * 100).toFixed(0)}%</span>
                {ev.url && <a href={ev.url} target="_blank" rel="noopener noreferrer" className="text-blue-300/40 hover:text-clinical-400 transition-colors"><ExternalLink className="w-3.5 h-3.5" /></a>}
              </div>
            </div>
            <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-blue-300/50 mb-2">
              {ev.authors?.length && <span>{ev.authors.slice(0, 2).join(", ")}{ev.authors.length > 2 ? " et al." : ""}</span>}
              {ev.journal && <span className="italic">{ev.journal}</span>}
              {ev.year && <span>{ev.year}</span>}
              {ev.pmid && <span className="font-mono">PMID: {ev.pmid}</span>}
            </div>
            <p className="text-xs text-blue-100/60 leading-relaxed line-clamp-3">{ev.excerpt}</p>
          </div>
        ))}
      </div>
      {evidence.length > 3 && (
        <button type="button" onClick={() => setExpanded(!expanded)} className="w-full text-center text-xs text-clinical-400 hover:text-clinical-300 py-1 transition-colors">
          {expanded ? "Show less ↑" : `Show ${evidence.length - 3} more ↓`}
        </button>
      )}
    </div>
  );
}

function SafetyFlagsPanel({ flags }: { flags: SafetyFlag[] }) {
  const sorted = [...flags].sort((a, b) => ({ critical: 0, warning: 1, info: 2 }[a.severity as "critical"|"warning"|"info"] ?? 2) - ({ critical: 0, warning: 1, info: 2 }[b.severity as "critical"|"warning"|"info"] ?? 2));
  const styles = { critical: "bg-rose-500/10 border-rose-500/25 text-rose-300", warning: "bg-amber-500/10 border-amber-500/25 text-amber-200/80", info: "bg-clinical-500/10 border-clinical-500/25 text-clinical-200/80" };
  const iconStyles = { critical: "text-rose-400", warning: "text-amber-400", info: "text-clinical-400" };
  return (
    <div className="space-y-2">
      {sorted.map((f, i) => (
        <div key={i} className={`flex gap-2.5 p-3.5 rounded-lg border ${styles[f.severity as keyof typeof styles] ?? "bg-white/5 border-white/10 text-white/70"}`}>
          <AlertTriangle className={`w-4 h-4 flex-shrink-0 mt-0.5 ${iconStyles[f.severity as keyof typeof iconStyles] ?? "text-white/40"}`} />
          <p className="text-sm">{f.message}</p>
        </div>
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="card p-10 text-center">
      <div className="w-16 h-16 rounded-2xl bg-clinical-500/10 flex items-center justify-center mx-auto mb-5">
        <BarChart2 className="w-8 h-8 text-clinical-500/50" />
      </div>
      <h3 className="font-display text-lg font-semibold text-white mb-2">Ready for Analysis</h3>
      <p className="text-sm text-blue-300/50 max-w-xs mx-auto leading-relaxed">Enter clinical notes and optionally upload images. The AI will retrieve relevant literature and generate grounded hypotheses.</p>
      <div className="mt-6 grid grid-cols-3 gap-3 text-center text-xs text-blue-300/40">
        {[["📝", "Clinical Text"], ["🖼", "Medical Images"], ["📊", "Lab Values"]].map(([icon, label]) => (
          <div key={label} className="p-3 rounded-lg bg-white/3 border border-white/5">
            <p className="text-lg mb-1">{icon}</p><p>{label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-5 animate-fade-in">
      <div className="disclaimer-banner flex gap-2 items-center">
        <div className="skeleton w-4 h-4 rounded" /><div className="skeleton h-4 w-48 rounded" />
      </div>
      <div className="card p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="skeleton h-6 w-32 rounded" /><div className="skeleton h-6 w-24 rounded-full" />
        </div>
        <div className="space-y-2">
          <div className="skeleton h-4 w-full rounded" />
          <div className="skeleton h-4 w-4/5 rounded" />
          <div className="skeleton h-4 w-3/5 rounded" />
        </div>
      </div>
      <div className="card p-5 space-y-4">
        <div className="skeleton h-6 w-48 rounded" />
        {[1, 2, 3].map((i) => (
          <div key={i} className="rounded-lg border border-white/5 p-3.5 space-y-2">
            <div className="flex gap-3 items-center">
              <div className="skeleton w-6 h-6 rounded-full" />
              <div className="flex-1 space-y-1"><div className="skeleton h-4 w-3/4 rounded" /></div>
              <div className="skeleton h-5 w-16 rounded-full" />
            </div>
            <div className="skeleton h-1.5 w-full rounded" />
          </div>
        ))}
      </div>
    </div>
  );
}
