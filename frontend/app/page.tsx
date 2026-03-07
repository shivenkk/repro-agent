"use client";

import { useState } from "react";

const API_URL = "http://localhost:8000";

interface HyperParam {
  value: string | number;
  confidence: number;
  source: string;
}

interface Comparison {
  metric: string;
  claimed: number;
}

interface Report {
  run_id: string;
  status: string;
  paper: {
    title: string;
    authors: string[];
    url: string;
    arxiv_id: string;
  };
  score: number;
  verdict: string;
  extraction_confidence: number;
  hyperparameters: Record<string, HyperParam>;
  claimed_results: Record<string, number>;
  comparisons: Comparison[];
  missing_details: string[];
  failure_reasons: string[];
  analysis: string;
  recommendations: string[];
  generated_code: string;
  progress: string[];
}

type Stage = "idle" | "loading" | "done" | "error";

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 80 ? "bg-emerald-500" : pct >= 50 ? "bg-amber-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-2 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} rounded-full transition-all`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-zinc-500">{pct}%</span>
    </div>
  );
}

function ScoreRing({ score, verdict }: { score: number; verdict: string }) {
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  const color =
    score >= 75
      ? "text-emerald-500"
      : score >= 45
      ? "text-amber-500"
      : "text-red-500";
  const verdictLabel = verdict.replace(/_/g, " ").toUpperCase();

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-36 h-36">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
          <circle
            cx="60"
            cy="60"
            r={radius}
            fill="none"
            stroke="#27272a"
            strokeWidth="8"
          />
          <circle
            cx="60"
            cy="60"
            r={radius}
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className={`${color} transition-all duration-1000`}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-3xl font-bold ${color}`}>{score}</span>
          <span className="text-xs text-zinc-500">/100</span>
        </div>
      </div>
      <span
        className={`text-sm font-semibold tracking-wide ${color}`}
      >
        {verdictLabel}
      </span>
    </div>
  );
}

export default function Home() {
  const [url, setUrl] = useState("");
  const [stage, setStage] = useState<Stage>("idle");
  const [progress, setProgress] = useState<string[]>([]);
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState("");
  const [showCode, setShowCode] = useState(false);

  async function handleAnalyze() {
    if (!url.trim()) return;
    setStage("loading");
    setProgress([]);
    setReport(null);
    setError("");
    setShowCode(false);

    try {
      // Start analysis
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paper_url: url.trim() }),
      });

      if (!res.ok) throw new Error("API request failed");

      const data = await res.json();
      const runId = data.run_id;

      // Fetch full results
      const resultRes = await fetch(`${API_URL}/results/${runId}`);
      const resultData = await resultRes.json();

      if (resultData.error) {
        setError(resultData.error);
        setStage("error");
        return;
      }

      setReport(resultData);
      setProgress(resultData.progress || []);
      setStage("done");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      setStage("error");
    }
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800">
        <div className="max-w-6xl mx-auto px-6 py-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src="/favicon.ico" alt="ReproAgent" className="w-8 h-8 rounded-lg" />
            <h1 className="text-lg font-semibold tracking-tight">ReproAgent</h1>
          </div>
          <span className="text-xs text-zinc-600">
            ML Paper Reproducibility Analysis
          </span>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-10">
        {/* Input Section */}
        <div className="mb-10">
          <label className="block text-sm text-zinc-400 mb-2">
            arXiv Paper URL
          </label>
          <div className="flex gap-3">
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://arxiv.org/abs/1512.03385"
              className="flex-1 px-4 py-3 bg-zinc-900 border border-zinc-800 rounded-lg text-sm focus:outline-none focus:border-emerald-600 placeholder:text-zinc-600"
              onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}
            />
            <button
              onClick={handleAnalyze}
              disabled={stage === "loading" || !url.trim()}
              className="px-6 py-3 bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-800 disabled:text-zinc-600 rounded-lg text-sm font-medium transition-colors"
            >
              {stage === "loading" ? "Analyzing..." : "Analyze"}
            </button>
          </div>
        </div>

        {/* Loading */}
        {stage === "loading" && (
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-4 h-4 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-zinc-400">
                Running pipeline...
              </span>
            </div>
            <div className="space-y-1">
              {progress.map((msg, i) => (
                <p key={i} className="text-xs text-zinc-600 font-mono">
                  {msg}
                </p>
              ))}
            </div>
          </div>
        )}

        {/* Error */}
        {stage === "error" && (
          <div className="bg-red-950/30 border border-red-900/50 rounded-lg p-6">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}

        {/* Results */}
        {stage === "done" && report && (
          <div className="space-y-6">
            {/* Paper Info + Score */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
              <div className="flex items-start justify-between gap-8">
                <div className="flex-1">
                  <h2 className="text-xl font-semibold mb-1">
                    {report.paper.title}
                  </h2>
                  <p className="text-sm text-zinc-500 mb-3">
                    {report.paper.authors.slice(0, 3).join(", ")}
                    {report.paper.authors.length > 3 && " et al."}
                  </p>
                  <a
                    href={report.paper.url}
                    target="_blank"
                    className="text-xs text-emerald-500 hover:text-emerald-400"
                  >
                    {report.paper.url}
                  </a>
                </div>
                <ScoreRing score={report.score} verdict={report.verdict} />
              </div>
            </div>

            {/* Extraction Confidence */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
              <h3 className="text-sm font-semibold text-zinc-300 mb-4">
                Extracted Hyperparameters
              </h3>
              <div className="space-y-3">
                {Object.entries(report.hyperparameters).map(
                  ([name, param]) => (
                    <div
                      key={name}
                      className="flex items-center justify-between"
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-sm text-zinc-400 w-32">
                          {name.replace(/_/g, " ")}
                        </span>
                        <span className="text-sm font-mono text-zinc-200">
                          {String(param.value)}
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        <ConfidenceBar value={param.confidence} />
                        {param.source && (
                          <span className="text-xs text-zinc-600">
                            {param.source}
                          </span>
                        )}
                      </div>
                    </div>
                  )
                )}
              </div>
            </div>

            {/* Claimed Results */}
            {report.comparisons.length > 0 && (
              <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
                <h3 className="text-sm font-semibold text-zinc-300 mb-4">
                  Claimed Results
                </h3>
                <div className="space-y-3">
                  {report.comparisons.map((c, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between py-2 border-b border-zinc-800 last:border-0"
                    >
                      <span className="text-sm text-zinc-400">
                        {c.metric}
                      </span>
                      <span className="text-sm font-mono text-zinc-200">
                        {c.claimed.toFixed(4)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Missing Details */}
            {report.missing_details.length > 0 && (
              <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
                <h3 className="text-sm font-semibold text-zinc-300 mb-3">
                  Missing from Paper
                </h3>
                <div className="flex flex-wrap gap-2">
                  {report.missing_details.map((item, i) => (
                    <span
                      key={i}
                      className="text-xs px-3 py-1 bg-amber-950/50 text-amber-400 rounded-full"
                    >
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Analysis */}
            {report.analysis && (
              <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
                <h3 className="text-sm font-semibold text-zinc-300 mb-3">
                  Analysis
                </h3>
                <p className="text-sm text-zinc-400 leading-relaxed">
                  {report.analysis}
                </p>
              </div>
            )}

            {/* Generated Code */}
            {report.generated_code && (
              <div className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden">
                <button
                  onClick={() => setShowCode(!showCode)}
                  className="w-full px-6 py-4 flex items-center justify-between hover:bg-zinc-800/50 transition-colors"
                >
                  <h3 className="text-sm font-semibold text-zinc-300">
                    Generated Code ({report.generated_code.length} chars)
                  </h3>
                  <span className="text-xs text-zinc-500">
                    {showCode ? "Hide" : "Show"}
                  </span>
                </button>
                {showCode && (
                  <div className="px-6 pb-6">
                    <pre className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 overflow-x-auto text-xs text-zinc-300 font-mono leading-relaxed">
                      {report.generated_code}
                    </pre>
                    <button
                      onClick={() =>
                        navigator.clipboard.writeText(report.generated_code)
                      }
                      className="mt-3 px-4 py-2 text-xs bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors"
                    >
                      Copy to clipboard
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Pipeline Progress */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
              <h3 className="text-sm font-semibold text-zinc-300 mb-3">
                Pipeline Log
              </h3>
              <div className="space-y-1">
                {report.progress.map((msg, i) => (
                  <p key={i} className="text-xs text-zinc-500 font-mono">
                    {msg}
                  </p>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {stage === "idle" && (
          <div className="text-center py-20">
            <div className="w-16 h-16 rounded-2xl bg-zinc-900 border border-zinc-800 flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">📄</span>
            </div>
            <h2 className="text-lg font-semibold text-zinc-300 mb-2">
              Analyze any ML paper
            </h2>
            <p className="text-sm text-zinc-600 max-w-md mx-auto">
              Paste an arXiv URL to extract methodology, generate reproducible
              code, and get a reproducibility score.
            </p>
            <div className="mt-6 flex flex-wrap justify-center gap-2">
              {[
                { label: "ResNet", url: "https://arxiv.org/abs/1512.03385" },
                { label: "Attention Is All You Need", url: "https://arxiv.org/abs/1706.03762" },
                { label: "BERT", url: "https://arxiv.org/abs/1810.04805" },
              ].map((paper) => (
                <button
                  key={paper.url}
                  onClick={() => setUrl(paper.url)}
                  className="px-3 py-1.5 text-xs bg-zinc-900 border border-zinc-800 rounded-lg hover:border-zinc-700 text-zinc-500 hover:text-zinc-300 transition-colors"
                >
                  {paper.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}