"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import ParameterForm from "./components/ParameterForm";
import DOSPlot from "./components/DOSPlot";
import BOProgress from "./components/BOProgress";
import ResultsTable from "./components/ResultsTable";
import {
  computeDOS,
  startOptimization,
  pollJob,
  startLocalRefinement,
  pollRefinementJob,
} from "@/lib/api";
import type {
  DOSRequest,
  DOSResponse,
  JobStatus,
  RefinedResult,
  TargetDOS,
} from "@/types";

// ── Optimization config defaults ─────────────────────────────────────────────
const DEFAULT_N_INITIAL = 10;
const DEFAULT_N_BATCH = 15;
const DEFAULT_BATCH_SIZE = 5;

export default function Home() {
  // Section 1 — Target DOS
  const [targetDOSParams, setTargetDOSParams] = useState<DOSResponse | null>(null);
  const [dosLoading, setDOSLoading] = useState(false);
  const [dosError, setDosError] = useState<string | null>(null);

  // Section 2 — Optimization config
  const [nInitial, setNInitial] = useState(DEFAULT_N_INITIAL);
  const [nBatch, setNBatch] = useState(DEFAULT_N_BATCH);
  const [batchSize, setBatchSize] = useState(DEFAULT_BATCH_SIZE);
  const [usePeakLoss, setUsePeakLoss] = useState(false);
  const [boLoading, setBoLoading] = useState(false);
  const [boError, setBoError] = useState<string | null>(null);

  // Section 3 — BO progress
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const refinePollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Section 4 — Refinement & final results
  const [refinedResults, setRefinedResults] = useState<RefinedResult[] | null>(null);
  const [refining, setRefining] = useState(false);
  const [refineError, setRefineError] = useState<string | null>(null);

  // ── Helpers ──────────────────────────────────────────────────────────────
  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      stopPolling();
      if (refinePollRef.current) clearInterval(refinePollRef.current);
    };
  }, [stopPolling]);

  // ── Section 1: Compute target DOS ────────────────────────────────────────
  async function handleComputeDOS(params: DOSRequest) {
    setDOSLoading(true);
    setDosError(null);
    try {
      const result = await computeDOS(params);
      setTargetDOSParams(result);
    } catch (err) {
      setDosError(String(err));
    } finally {
      setDOSLoading(false);
    }
  }

  // ── Section 2: Start optimization ────────────────────────────────────────
  async function handleStartOptimization() {
    if (!targetDOSParams) {
      setBoError("Please compute a target DOS first.");
      return;
    }
    setBoError(null);
    setJobStatus(null);
    setRefinedResults(null);
    setBoLoading(true);
    stopPolling();

    const targetDOS: TargetDOS = {
      dos_counts: targetDOSParams.dos_counts,
      bin_edges: targetDOSParams.bin_edges,
    };

    try {
      const { job_id } = await startOptimization({
        n_initial: nInitial,
        n_batch: nBatch,
        batch_size: batchSize,
        target_dos: targetDOS,
        bounds: { t_a: [-0.5, 0.5], t_b: [-0.5, 0.5] },
        use_peak_loss: usePeakLoss,
      });
      setCurrentJobId(job_id);
      setBoLoading(false);

      // Start polling every 2 s
      pollRef.current = setInterval(async () => {
        try {
          const status = await pollJob(job_id);
          setJobStatus(status);
          if (status.status !== "running") {
            stopPolling();
          }
        } catch {
          // transient network errors — keep polling
        }
      }, 2000);
    } catch (err) {
      setBoError(String(err));
      setBoLoading(false);
    }
  }

  // ── Section 4: Local refinement ──────────────────────────────────────────
  async function handleRunRefinement(candidates: number[][]) {
    if (!targetDOSParams) return;
    setRefining(true);
    setRefineError(null);
    if (refinePollRef.current) clearInterval(refinePollRef.current);
    try {
      const { job_id } = await startLocalRefinement({
        candidates,
        target_dos: {
          dos_counts: targetDOSParams.dos_counts,
          bin_edges: targetDOSParams.bin_edges,
        },
      });
      refinePollRef.current = setInterval(async () => {
        try {
          const status = await pollRefinementJob(job_id);
          if (status.status === "complete" && status.results) {
            setRefinedResults(status.results);
            setRefining(false);
            clearInterval(refinePollRef.current!);
            refinePollRef.current = null;
          } else if (status.status === "failed") {
            setRefineError(status.error ?? "Refinement failed");
            setRefining(false);
            clearInterval(refinePollRef.current!);
            refinePollRef.current = null;
          }
        } catch {
          // transient network errors — keep polling
        }
      }, 2000);
    } catch (err) {
      setRefineError(String(err));
      setRefining(false);
    }
  }

  return (
    <main className="max-w-7xl mx-auto px-4 py-8 space-y-10">
      <header>
        <h1 className="text-2xl font-bold">Kagome Lattice Bayesian Optimization</h1>
        <p className="mt-1 text-gray-500 text-sm">
          Find tight-binding parameters (t_a, t_b) that reproduce a target DOS.
        </p>
      </header>

      {/* ── Section 1: Target DOS ─────────────────────────────────────────── */}
      <section>
        <h2 className="text-xl font-semibold mb-3">1 — Target DOS</h2>
        <div className="grid md:grid-cols-[5fr_7fr] gap-6">
          <ParameterForm onCompute={handleComputeDOS} loading={dosLoading} />
          <div className="flex flex-col gap-3">
            {dosError && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
                {dosError}
              </div>
            )}
            {targetDOSParams && (
              <p className="text-sm text-gray-500">
                Integral: {targetDOSParams.integral.toFixed(4)}
              </p>
            )}
            <DOSPlot
              curves={
                targetDOSParams
                  ? [
                      {
                        dos_counts: targetDOSParams.dos_counts,
                        bin_edges: targetDOSParams.bin_edges,
                        label: "Target DOS",
                        color: "#2563eb",
                        dash: "solid",
                      },
                    ]
                  : []
              }
              title="Target DOS Preview"
              height={420}
            />
          </div>
        </div>
      </section>

      {/* ── Section 2: Optimization config ──────────────────────────────── */}
      <section>
        <h2 className="text-xl font-semibold mb-3">2 — Optimization Configuration</h2>
        <div className="bg-white rounded-xl shadow p-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-5 text-sm">
            <label className="flex flex-col gap-1">
              <span className="text-gray-600 font-medium">N Initial points</span>
              <input
                type="number"
                min={1}
                max={100}
                value={nInitial}
                onChange={(e) => setNInitial(parseInt(e.target.value) || 1)}
                className="border rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-400"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-gray-600 font-medium">N Batch iterations</span>
              <input
                type="number"
                min={1}
                max={100}
                value={nBatch}
                onChange={(e) => setNBatch(parseInt(e.target.value) || 1)}
                className="border rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-400"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-gray-600 font-medium">Batch size</span>
              <input
                type="number"
                min={1}
                max={20}
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                className="border rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-400"
              />
            </label>
            <label className="flex items-center gap-2 pt-5 cursor-pointer">
              <input
                type="checkbox"
                checked={usePeakLoss}
                onChange={(e) => setUsePeakLoss(e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-gray-600 font-medium text-sm">
                Use peak loss (×10)
              </span>
            </label>
          </div>
          {boError && (
            <div className="mt-3 bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
              {boError}
            </div>
          )}
          <button
            onClick={handleStartOptimization}
            disabled={boLoading || !targetDOSParams || jobStatus?.status === "running"}
            className="mt-5 bg-green-600 text-white rounded-lg px-6 py-2 font-semibold
                       hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {boLoading
              ? "Starting…"
              : jobStatus?.status === "running"
              ? "Optimization running…"
              : "Run Optimization"}
          </button>
          {!targetDOSParams && (
            <p className="mt-2 text-xs text-gray-400">
              Compute a target DOS first (Section 1).
            </p>
          )}
        </div>
      </section>

      {/* ── Section 3: Live BO progress ──────────────────────────────────── */}
      {(jobStatus || boLoading) && (
        <section>
          <h2 className="text-xl font-semibold mb-3">3 — BO Progress</h2>
          <BOProgress jobStatus={jobStatus} />
        </section>
      )}

      {/* ── Section 4: Results ───────────────────────────────────────────── */}
      {jobStatus && (jobStatus.top_candidates?.length ?? 0) > 0 && (
        <section>
          <h2 className="text-xl font-semibold mb-3">4 — Results</h2>
          <div className="space-y-5">
            {refineError && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
                {refineError}
              </div>
            )}
            <ResultsTable
              jobStatus={jobStatus}
              refinedResults={refinedResults}
              onRunRefinement={handleRunRefinement}
              refining={refining}
            />
            {refinedResults && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {refinedResults.map((r, i) => (
                  <DOSPlot
                    key={i}
                    curves={[
                      {
                        dos_counts: targetDOSParams!.dos_counts,
                        bin_edges: targetDOSParams!.bin_edges,
                        label: "Target DOS",
                        color: "#2563eb",
                        dash: "solid",
                      },
                      {
                        dos_counts: r.dos_counts,
                        bin_edges: r.dos_bin_edges,
                        label: `Refined rank ${i + 1}`,
                        color: ["#ef4444", "#f97316", "#eab308", "#ec4899", "#84cc16"][i] ?? "#ef4444",
                        dash: "solid",
                      },
                    ]}
                    title={`Rank ${i + 1} — Target vs Refined`}
                    xLabel="Energy (eV)"
                    yLabel="DOS (a.u.)"
                  />
                ))}
              </div>
            )}
          </div>
        </section>
      )}
    </main>
  );
}
