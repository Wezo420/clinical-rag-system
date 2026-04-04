"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { ConditionHypothesis } from "@/api/client";

interface Props {
  hypotheses: ConditionHypothesis[];
}

const CONFIDENCE_COLORS: Record<string, string> = {
  high: "#10b981",
  medium: "#f59e0b",
  low: "#f43f5e",
};

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ value: number; payload: ConditionHypothesis }>;
  label?: string;
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  const h = payload[0].payload;
  return (
    <div className="card p-3 text-xs space-y-1 min-w-[180px]">
      <p className="font-semibold text-white text-sm">{h.condition}</p>
      {h.icd10_code && (
        <p className="font-mono text-blue-300/50">{h.icd10_code}</p>
      )}
      <p className="text-blue-200/70">
        Confidence:{" "}
        <span
          style={{ color: CONFIDENCE_COLORS[h.confidence_level] }}
          className="font-bold"
        >
          {(h.confidence * 100).toFixed(0)}%
        </span>
      </p>
      {h.supporting_factors.length > 0 && (
        <p className="text-emerald-400/80">
          + {h.supporting_factors[0]}
          {h.supporting_factors.length > 1 &&
            ` (+${h.supporting_factors.length - 1} more)`}
        </p>
      )}
    </div>
  );
}

export function ConfidenceChart({ hypotheses }: Props) {
  if (!hypotheses.length) return null;

  const data = hypotheses.slice(0, 6).map((h) => ({
    ...h,
    name:
      h.condition.length > 22
        ? h.condition.slice(0, 20) + "…"
        : h.condition,
    value: Math.round(h.confidence * 100),
  }));

  return (
    <div className="card p-5 space-y-3">
      <h3 className="font-display text-base font-bold text-white">
        Hypothesis Confidence
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 16, bottom: 0, left: 8 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(124,196,251,0.07)"
            horizontal={false}
          />
          <XAxis
            type="number"
            domain={[0, 100]}
            tick={{ fill: "rgba(124,196,251,0.4)", fontSize: 11 }}
            tickFormatter={(v) => `${v}%`}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: "rgba(232,244,254,0.7)", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={130}
          />
          <Tooltip
            content={<CustomTooltip />}
            cursor={{ fill: "rgba(124,196,251,0.05)" }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={18}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={CONFIDENCE_COLORS[entry.confidence_level]}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-4 text-xs text-blue-300/50">
        {Object.entries(CONFIDENCE_COLORS).map(([level, color]) => (
          <span key={level} className="flex items-center gap-1.5">
            <span
              className="w-2.5 h-2.5 rounded-sm"
              style={{ background: color }}
            />
            {level}
          </span>
        ))}
      </div>
    </div>
  );
}
