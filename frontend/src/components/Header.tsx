"use client";

import Link from "next/link";
import { Activity, Brain, Shield } from "lucide-react";

export function Header() {
  return (
    <header className="border-b border-white/5 bg-clinical-950/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-clinical-500 to-clinical-400 flex items-center justify-center shadow-lg group-hover:shadow-clinical-500/30 transition-shadow">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="font-display text-lg font-bold text-white tracking-tight">
              Clinical<span className="text-clinical-400">RAG</span>
            </span>
            <p className="text-[10px] text-blue-300/50 leading-none -mt-0.5">
              Research Tool
            </p>
          </div>
        </Link>

        {/* Status indicators */}
        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-1.5 text-xs text-emerald-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span>API Active</span>
          </div>

          <div className="flex items-center gap-1.5 text-xs text-amber-400/80 px-3 py-1.5 rounded-full border border-amber-400/20 bg-amber-400/5">
            <Shield className="w-3 h-3" />
            <span className="hidden sm:inline">Research Only</span>
          </div>

          <div className="hidden md:flex items-center gap-1.5 text-xs text-blue-300/60">
            <Activity className="w-3.5 h-3.5" />
            <span>Groq + FAISS</span>
          </div>
        </div>
      </div>
    </header>
  );
}
