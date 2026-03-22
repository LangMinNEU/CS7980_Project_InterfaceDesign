"use client";

import type { JobStatus, RefinedResult } from "@/types";

interface Props {
  jobStatus: JobStatus | null;
  refinedResults: RefinedResult[] | null;
  onRunRefinement: (candidates: number[][]) => void;
  refining: boolean;
}

export default function ResultsTable({
  jobStatus,
  refinedResults,
  onRunRefinement,
  refining,
}: Props) {
  const candidates = jobStatus?.top_candidates ?? [];
  const trainObj = jobStatus?.train_obj ?? [];
  const trainX = jobStatus?.train_x ?? [];

  // Map candidate -> best objective value from train data
  const getCandidateLoss = (candidate: number[]): number | null => {
    for (let i = 0; i < trainX.length; i++) {
      const p = trainX[i];
      if (
        Math.abs(p[0] - candidate[0]) < 1e-6 &&
        Math.abs(p[1] - candidate[1]) < 1e-6
      ) {
        return -trainObj[i];  // convert to positive loss
      }
    }
    return null;
  };

  if (candidates.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 bg-white rounded-xl shadow text-gray-400">
        Top candidates will appear here after optimization completes.
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow p-6 space-y-4">
      <h2 className="text-lg font-semibold">Top Candidates from BO</h2>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b text-gray-500">
              <th className="text-left py-2 pr-4">Rank</th>
              <th className="text-left py-2 pr-4">t_a</th>
              <th className="text-left py-2 pr-4">t_b</th>
              <th className="text-left py-2 pr-4">BO Loss</th>
              {refinedResults && (
                <>
                  <th className="text-left py-2 pr-4">Refined t_a</th>
                  <th className="text-left py-2 pr-4">Refined t_b</th>
                  <th className="text-left py-2">Refined Loss</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {candidates.map((cand, i) => {
              const boLoss = getCandidateLoss(cand);
              const refined = refinedResults?.[i];
              return (
                <tr key={i} className="border-b last:border-0">
                  <td className="py-2 pr-4 font-medium text-gray-700">{i + 1}</td>
                  <td className="py-2 pr-4 font-mono">{cand[0].toFixed(5)}</td>
                  <td className="py-2 pr-4 font-mono">{cand[1].toFixed(5)}</td>
                  <td className="py-2 pr-4 font-mono text-gray-500">
                    {boLoss != null ? boLoss.toFixed(5) : "—"}
                  </td>
                  {refinedResults && (
                    <>
                      <td className="py-2 pr-4 font-mono">
                        {refined ? refined.x[0].toFixed(5) : "—"}
                      </td>
                      <td className="py-2 pr-4 font-mono">
                        {refined ? refined.x[1].toFixed(5) : "—"}
                      </td>
                      <td className="py-2 font-mono text-blue-700 font-semibold">
                        {refined ? refined.loss.toFixed(5) : "—"}
                      </td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {!refinedResults && (
        <div className="space-y-2">
          <button
            onClick={() => onRunRefinement(candidates)}
            disabled={refining || jobStatus?.status !== "complete"}
            className="mt-2 bg-purple-600 text-white rounded-lg px-5 py-2 font-semibold
                       hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {refining ? "Refining…" : "Run Local Refinement (COBYLA)"}
          </button>
          {refining && (
            <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
              <div className="h-2 w-2/5 rounded-full bg-purple-500 animate-indeterminate" />
            </div>
          )}
          {refining && (
            <p className="text-xs text-gray-400">Running COBYLA refinement on {candidates.length} candidates…</p>
          )}
        </div>
      )}
    </div>
  );
}
