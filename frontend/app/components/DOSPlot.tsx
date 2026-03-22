"use client";

import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface DOSCurve {
  dos_counts: number[];
  bin_edges: number[];
  label: string;
  color: string;
  dash?: "dash" | "solid" | "dot";
}

interface Props {
  curves: DOSCurve[];
  title?: string;
  xLabel?: string;
  yLabel?: string;
}

function binCenters(edges: number[]): number[] {
  return edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
}

export default function DOSPlot({
  curves,
  title = "Density of States",
  xLabel = "Energy (eV)",
  yLabel = "DOS (a.u.)",
}: Props) {
  if (curves.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-white rounded-xl shadow text-gray-400">
        No DOS data yet.
      </div>
    );
  }

  const data = curves.map((c) => ({
    x: binCenters(c.bin_edges),
    y: c.dos_counts,
    type: "scatter" as const,
    mode: "lines" as const,
    name: c.label,
    line: { color: c.color, dash: c.dash ?? "solid", width: 2 },
  }));

  return (
    <div className="bg-white rounded-xl shadow p-4">
      <Plot
        data={data}
        layout={{
          title: { text: title, font: { size: 15 } },
          xaxis: { title: xLabel },
          yaxis: { title: yLabel },
          margin: { t: 40, b: 50, l: 55, r: 20 },
          legend: { x: 0.01, y: 0.99 },
          height: 320,
          autosize: true,
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: "100%" }}
      />
    </div>
  );
}
