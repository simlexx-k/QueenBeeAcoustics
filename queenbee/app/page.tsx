/* eslint-disable @next/next/no-img-element */
"use client";

import { useEffect, useMemo, useState } from "react";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type ModelResponse = {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  stress_probability?: number;
  stress_label?: string;
};

type Status = "idle" | "uploading" | "success" | "error";

const percent = (value: number) =>
  new Intl.NumberFormat("en-US", {
    style: "percent",
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value);

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<ModelResponse | null>(null);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  const orderedProbabilities = useMemo(() => {
    if (!prediction) return [];
    return Object.entries(prediction.probabilities).sort(
      (a, b) => b[1] - a[1],
    );
  }, [prediction]);

  const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    setPrediction(null);
    setError(null);
    setSelectedFile(file ?? null);
  };

  const onSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedFile) {
      setError("Upload a WAV file first.");
      return;
    }

    setStatus("uploading");
    setError(null);
    setPrediction(null);

    try {
      const data = new FormData();
      data.append("file", selectedFile);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: data,
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const message = payload?.detail ?? response.statusText;
        throw new Error(message);
      }

      const payload = (await response.json()) as ModelResponse;
      setPrediction(payload);
      setStatus("success");
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Unexpected error");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-amber-50 via-white to-amber-100 py-12">
      <main className="mx-auto flex w-full max-w-5xl flex-col gap-10 px-4">
        <header className="rounded-3xl bg-white/90 p-6 text-center shadow-xl shadow-amber-100 backdrop-blur">
          <h1 className="text-3xl font-semibold text-slate-900 sm:text-4xl">
            QueenBee Machine
          </h1>
          <p className="mx-auto mt-2 max-w-2xl text-base text-slate-600">
            Drop a hive WAV recording and get an immediate breakdown of queen
            presence vs. absence vs. external noise from the FastAPI model.
          </p>
        </header>

        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-lg">
          <div className="max-w-3xl">
            <form className="flex flex-col gap-4" onSubmit={onSubmit}>
              <label
                htmlFor="audio"
                className="text-sm font-medium text-slate-700"
              >
                Hive audio (.wav)
              </label>
              <input
                id="audio"
                type="file"
                accept="audio/wav"
                onChange={onFileChange}
                className="w-full cursor-pointer rounded-2xl border border-dashed border-amber-300 bg-amber-50 px-4 py-6 text-center text-sm text-amber-800 hover:border-amber-400"
              />

              {previewUrl ? (
                <div className="rounded-2xl border border-slate-200 bg-gradient-to-br from-slate-50 to-white p-4 shadow-inner">
                  <p className="text-xs uppercase tracking-wide text-slate-500">
                    Preview
                  </p>
                  <p className="truncate text-sm font-medium text-slate-900">
                    {selectedFile?.name}
                  </p>
                  <audio
                    controls
                    src={previewUrl}
                    className="mt-3 w-full"
                    preload="metadata"
                  />
                </div>
              ) : (
                <p className="rounded-2xl border border-slate-200 bg-white px-4 py-5 text-sm text-slate-500 shadow-inner">
                  Drop a WAV file or tap to browse your device.
                </p>
              )}

              <button
                type="submit"
                disabled={status === "uploading"}
                className="mt-2 inline-flex items-center justify-center gap-2 rounded-2xl bg-amber-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-amber-500 disabled:cursor-not-allowed disabled:bg-amber-300"
              >
                {status === "uploading" ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                    Predicting…
                  </>
                ) : (
                  "Send to model"
                )}
              </button>
            </form>

            {error ? (
              <p className="mt-4 rounded-2xl bg-rose-50 px-4 py-3 text-sm text-rose-700">
                {error}
              </p>
            ) : null}
            {status === "success" ? (
              <p className="mt-4 rounded-2xl bg-emerald-50 px-4 py-3 text-sm text-emerald-600">
                Prediction complete in real-time.
              </p>
            ) : null}
          </div>
        </section>

        {prediction ? (
          <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-xl">
            <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  QueenBee Machine verdict
                </p>
                <h2 className="mt-2 text-3xl font-semibold text-slate-900">
                  {prediction.label}
                </h2>
                <p className="text-sm text-slate-500">
                  Confidence {percent(prediction.confidence)}
                </p>
                {typeof prediction.stress_probability === "number" ? (
                  <p className="text-sm text-slate-500">
                    Hive stress:{" "}
                    <span className="font-semibold text-slate-900">
                      {prediction.stress_label ?? "n/a"}
                    </span>{" "}
                    ({percent(prediction.stress_probability)})
                  </p>
                ) : null}
              </div>
              <div className="rounded-2xl bg-gradient-to-r from-amber-500 to-amber-400 px-6 py-3 text-sm font-semibold text-white shadow-lg">
                Live inference
              </div>
            </div>

            <div className="mt-8 grid gap-4 lg:grid-cols-3">
              {orderedProbabilities.map(([label, value]) => (
                <div
                  key={label}
                  className="rounded-2xl border border-slate-100 bg-gradient-to-br from-slate-50 to-white p-4 shadow-sm"
                >
                  <p className="text-xs uppercase tracking-wide text-slate-500">
                    {label}
                  </p>
                  <p className="text-2xl font-semibold text-slate-900">
                    {percent(value)}
                  </p>
                  <div className="mt-2 h-2 rounded-full bg-slate-200">
                    <span
                      className="block h-full rounded-full bg-amber-500 transition-all"
                      style={{ width: `${value * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>
        ) : (
          <section className="rounded-3xl border border-dashed border-slate-200 p-6 text-sm text-slate-500">
            <p>
              Upload hive audio to let QueenBee Machine evaluate the queen
              status and external noise risk.
            </p>
          </section>
        )}
      </main>
    </div>
  );
}
