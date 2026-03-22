"use client";

import { useState } from "react";
import type { DOSRequest } from "@/types";

interface Props {
  onCompute: (params: DOSRequest) => void;
  loading: boolean;
}

export default function ParameterForm({ onCompute, loading }: Props) {
  const [params, setParams] = useState<DOSRequest>({
    t_a: 0.0285,
    t_b: 0.075,
    t_nnn: 0.0,
    v_a: 0.0,
    v_b: 0.0,
    v_c: 0.0,
    d: 0.133,
    bins: 800,
    energy_range: [-0.15, 0.25],
    sigma: 5.0,
  });

  const handleChange = (key: keyof DOSRequest, value: string) => {
    const num = parseFloat(value);
    if (isNaN(num)) return;
    setParams((prev) => ({ ...prev, [key]: num }));
  };

  const handleRangeChange = (idx: 0 | 1, value: string) => {
    const num = parseFloat(value);
    if (isNaN(num)) return;
    setParams((prev) => {
      const range = [...(prev.energy_range ?? [-0.15, 0.25])] as [number, number];
      range[idx] = num;
      return { ...prev, energy_range: range };
    });
  };

  return (
    <div className="bg-white rounded-xl shadow p-6">
      <h2 className="text-lg font-semibold mb-4">Lattice Parameters</h2>
      <div className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
        {(
          [
            ["t_a", "Hopping t_a"],
            ["t_b", "Hopping t_b"],
            ["t_nnn", "NNN hopping t_nnn"],
            ["d", "NN distance d"],
            ["v_a", "On-site v_a"],
            ["v_b", "On-site v_b"],
            ["v_c", "On-site v_c"],
            ["sigma", "Gaussian σ (bins)"],
            ["bins", "Histogram bins"],
          ] as [keyof DOSRequest, string][]
        ).map(([key, label]) => (
          <label key={key} className="flex flex-col gap-1">
            <span className="text-gray-600 font-medium">{label}</span>
            <input
              type="number"
              step="any"
              value={params[key] as number}
              onChange={(e) => handleChange(key, e.target.value)}
              className="border rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-400"
            />
          </label>
        ))}
        <label className="flex flex-col gap-1">
          <span className="text-gray-600 font-medium">Energy min (eV)</span>
          <input
            type="number"
            step="any"
            value={params.energy_range?.[0] ?? -0.15}
            onChange={(e) => handleRangeChange(0, e.target.value)}
            className="border rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-gray-600 font-medium">Energy max (eV)</span>
          <input
            type="number"
            step="any"
            value={params.energy_range?.[1] ?? 0.25}
            onChange={(e) => handleRangeChange(1, e.target.value)}
            className="border rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
        </label>
      </div>
      <button
        onClick={() => onCompute(params)}
        disabled={loading}
        className="mt-5 w-full bg-blue-600 text-white rounded-lg py-2 font-semibold
                   hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
      >
        {loading ? "Computing…" : "Compute DOS"}
      </button>
    </div>
  );
}
