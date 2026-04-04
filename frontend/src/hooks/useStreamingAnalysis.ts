/**
 * useStreamingAnalysis hook
 * Handles SSE-based streaming responses from the backend.
 */

import { useState, useCallback, useRef } from "react";
import { streamAnalysis } from "@/api/client";
import type { ClinicalCaseRequest } from "@/api/client";

interface StreamState {
  isStreaming: boolean;
  chunks: string[];
  fullText: string;
  error: string | null;
}

export function useStreamingAnalysis() {
  const [state, setState] = useState<StreamState>({
    isStreaming: false,
    chunks: [],
    fullText: "",
    error: null,
  });

  const abortRef = useRef<boolean>(false);

  const startStream = useCallback(
    async (payload: ClinicalCaseRequest) => {
      abortRef.current = false;
      setState({ isStreaming: true, chunks: [], fullText: "", error: null });

      await streamAnalysis(
        { ...payload, stream: true },
        (chunk) => {
          if (abortRef.current) return;
          setState((prev) => ({
            ...prev,
            chunks: [...prev.chunks, chunk],
            fullText: prev.fullText + chunk,
          }));
        },
        () => {
          setState((prev) => ({ ...prev, isStreaming: false }));
        },
        (err) => {
          setState((prev) => ({
            ...prev,
            isStreaming: false,
            error: err.message,
          }));
        }
      );
    },
    []
  );

  const stop = useCallback(() => {
    abortRef.current = true;
    setState((prev) => ({ ...prev, isStreaming: false }));
  }, []);

  const reset = useCallback(() => {
    abortRef.current = true;
    setState({ isStreaming: false, chunks: [], fullText: "", error: null });
  }, []);

  return { ...state, startStream, stop, reset };
}
