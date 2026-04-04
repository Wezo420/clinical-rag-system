/**
 * Clinical RAG API Client
 * Typed Axios wrapper for all backend endpoints.
 */

import axios, { AxiosInstance, AxiosError } from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface LabValue {
  name: string;
  value: number;
  unit: string;
  reference_range?: string;
  is_abnormal?: boolean;
}

export interface VitalSign {
  parameter: string;
  value: number;
  unit: string;
}

export interface StructuredData {
  lab_values?: LabValue[];
  vitals?: VitalSign[];
  age?: number;
  sex?: "male" | "female" | "other" | "unknown";
  medications?: string[];
  allergies?: string[];
}

export interface ClinicalCaseRequest {
  clinical_text: string;
  image_ids?: string[];
  structured_data?: StructuredData;
  modality?: string;
  stream?: boolean;
}

export interface ConditionHypothesis {
  condition: string;
  icd10_code?: string;
  confidence: number;
  confidence_level: "low" | "medium" | "high";
  supporting_factors: string[];
  against_factors: string[];
  recommended_workup: string[];
}

export interface MedicalEvidence {
  source_id: string;
  title: string;
  authors?: string[];
  journal?: string;
  year?: number;
  pmid?: string;
  doi?: string;
  excerpt: string;
  relevance_score: number;
  url?: string;
}

export interface SafetyFlag {
  flag_type: string;
  message: string;
  severity: "info" | "warning" | "critical";
}

export interface AnalysisResult {
  result_id: string;
  case_id?: string;
  status: "pending" | "processing" | "completed" | "failed";
  disclaimer: string;
  summary: string;
  condition_hypotheses: ConditionHypothesis[];
  differential_reasoning: string;
  evidence: MedicalEvidence[];
  confidence_overall: number;
  confidence_level: "low" | "medium" | "high";
  safety_flags: SafetyFlag[];
  model_used: string;
  retrieval_count: number;
  processing_time_ms: number;
  timestamp: string;
  image_analysis_summary?: string;
  structured_data_used: boolean;
}

export interface ImageUploadResponse {
  image_id: string;
  filename: string;
  modality: string;
  size_bytes: number;
  embedding_status: string;
  message: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface ApiError {
  detail: string;
  status: number;
}

// ─── Client ───────────────────────────────────────────────────────────────────

function createApiClient(): AxiosInstance {
  const client = axios.create({
    baseURL: API_BASE,
    timeout: 120_000,
    headers: { "Content-Type": "application/json" },
  });

  // Attach JWT on every request
  client.interceptors.request.use((config) => {
    if (typeof window !== "undefined") {
      const token = localStorage.getItem("access_token");
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    return config;
  });

  // Global error handler
  client.interceptors.response.use(
    (res) => res,
    (err: AxiosError) => {
      if (err.response?.status === 401) {
        if (typeof window !== "undefined") {
          localStorage.removeItem("access_token");
          window.dispatchEvent(new Event("auth:logout"));
        }
      }
      return Promise.reject(err);
    }
  );

  return client;
}

const api = createApiClient();

// ─── API Functions ────────────────────────────────────────────────────────────

export const clinicalApi = {
  /** Analyze a clinical case (non-streaming) */
  analyzeCase: async (payload: ClinicalCaseRequest): Promise<AnalysisResult> => {
    const { data } = await api.post<AnalysisResult>("/analyze-case", payload);
    return data;
  },

  /** Upload a medical image */
  uploadImage: async (
    file: File,
    modality: string = "other",
    description?: string
  ): Promise<ImageUploadResponse> => {
    const form = new FormData();
    form.append("file", file);
    form.append("modality", modality);
    if (description) form.append("description", description);

    const { data } = await api.post<ImageUploadResponse>("/upload-image", form, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 30_000,
    });
    return data;
  },

  /** Get stored result by ID */
  getResult: async (resultId: string): Promise<AnalysisResult> => {
    const { data } = await api.get<AnalysisResult>(`/results/${resultId}`);
    return data;
  },

  /** Health check */
  health: async (): Promise<{ status: string }> => {
    const { data } = await api.get("/health");
    return data;
  },

  /** Login */
  login: async (username: string, password: string): Promise<TokenResponse> => {
    const { data } = await api.post<TokenResponse>("/auth/login", { username, password });
    return data;
  },

  /** Register */
  register: async (payload: {
    username: string;
    email: string;
    password: string;
    full_name?: string;
  }) => {
    const { data } = await api.post("/auth/register", payload);
    return data;
  },
};

/**
 * Streaming analysis using Server-Sent Events.
 * Calls onChunk for each streamed text chunk, onDone when finished.
 */
export async function streamAnalysis(
  payload: ClinicalCaseRequest,
  onChunk: (chunk: string) => void,
  onDone: () => void,
  onError: (err: Error) => void
): Promise<void> {
  const token = typeof window !== "undefined" ? localStorage.getItem("access_token") : null;

  try {
    const response = await fetch(`${API_BASE}/analyze-case`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ ...payload, stream: true }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error("No response body");
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value, { stream: true });
      const lines = text.split("\n");
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const chunk = line.slice(6);
          if (chunk === "[DONE]") {
            onDone();
            return;
          }
          onChunk(chunk);
        }
      }
    }
    onDone();
  } catch (err) {
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}
