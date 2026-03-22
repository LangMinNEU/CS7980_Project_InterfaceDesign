"use client";

import dynamic from "next/dynamic";
import type { PlotParams } from "react-plotly.js";
import type { JobStatus } from "@/types";

const Plot = dynamic<PlotParams>(() => import("react-plotly.js"), { ssr: false });

interface Props {
  jobStatus: JobStatus | null;
}

export default function BOProgress({ jobStatus }: Props) {
  if (!jobStatus) {
    return (
      <div className="flex items-center justify-center h-40 bg-white rounded-xl shadow text-gray-400">
        Start an optimization run to see progress.
      </div>
    );
  }

  const { status, iteration, total_iterations, best_loss, train_x, train_obj } =
    jobStatus;

  // Build loss-per-iteration trace from accumulated train_obj
  const iterLabels = train_obj
    ? train_obj.map((_, i) => i + 1)
    : [];
  const bestLossByPoint = train_obj
    ? train_obj.map((v) => -v)  // convert back to positive loss
    : [];

  // Running best
  const runningBest: number[] = [];
  let best = Infinity;
  for (const v of bestLossByPoint) {
    if (v < best) best = v;
    runningBest.push(best);
  }

  const progress =
    iteration != null && total_iterations
      ? Math.round((iteration / total_iterations) * 100)
      : 0;

  return (
    <div className="space-y-4">
      {/* Status bar */}
      <div className="bg-white rounded-xl shadow p-4 flex items-center gap-4">
        <div
          className={`px-3 py-1 rounded-full text-sm font-semibold ${
            status === "running"
              ? "bg-yellow-100 text-yellow-800"
              : status === "complete"
              ? "bg-green-100 text-green-800"
              : "bg-red-100 text-red-800"
          }`}
        >
          {status.toUpperCase()}
        </div>
        {iteration != null && (
          <span className="text-sm text-gray-600">
            Iteration {iteration} / {total_iterations}
          </span>
        )}
        {best_loss != null && (
          <span className="text-sm text-gray-600">
            Best loss: <span className="font-mono">{best_loss.toFixed(5)}</span>
          </span>
        )}
        {status === "running" && (
          <div className="flex-1 bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
        {jobStatus.error && (
          <span className="text-sm text-red-600">{jobStatus.error}</span>
        )}
      </div>

      {/* Running best loss chart + 2-D scatter side by side */}
      {runningBest.length > 0 && train_x && train_obj && train_x.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white rounded-xl shadow p-4">
            <Plot
              data={[
                {
                  x: iterLabels,
                  y: runningBest,
                  type: "scatter",
                  mode: "lines",
                  name: "Running best loss",
                  line: { color: "#2563eb", width: 2 },
                },
              ]}
              layout={{
                title: { text: "BO Convergence", font: { size: 14 } },
                xaxis: { title: { text: "Evaluation #" } },
                yaxis: { title: { text: "Best loss (Wasserstein)" } },
                margin: { t: 36, b: 50, l: 60, r: 20 },
                height: 420,
                autosize: true,
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: "100%" }}
            />
          </div>
          <div className="bg-white rounded-xl shadow p-4">
            <Plot
              data={[
                {
                  x: train_x.map((p) => p[0]),
                  y: train_x.map((p) => p[1]),
                  mode: "markers",
                  type: "scatter",
                  marker: {
                    color: train_obj.map((v) => -v),
                    colorscale: "Viridis",
                    reversescale: true,
                    size: 7,
                    colorbar: { title: { text: "Loss" }, thickness: 14 },
                  },
                  name: "Evaluated points",
                },
              ]}
              layout={{
                title: { text: "Parameter Space Exploration", font: { size: 14 } },
                xaxis: { title: { text: "t_a" } },
                yaxis: { title: { text: "t_b" } },
                margin: { t: 36, b: 50, l: 55, r: 60 },
                height: 420,
                autosize: true,
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: "100%" }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
