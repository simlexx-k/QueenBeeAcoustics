"use client";

import { useEffect, useMemo, useState } from "react";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type Ward = {
  id: string;
  name: string;
  subcounty: string;
  latitude: number;
  longitude: number;
};

type ModelResponse = {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  stress_probability?: number;
  stress_label?: string;
  ward?: Ward;
};

type ClimateRecord = {
  date: string;
  temp_max?: number | null;
  temp_min?: number | null;
  temp_mean?: number | null;
  humidity_mean?: number | null;
  rainfall_mm?: number | null;
  wind_speed_max?: number | null;
  cloud_cover_percent?: number | null;
  ndvi_mean?: number | null;
};

type ClimateSeriesResponse = {
  count: number;
  records: ClimateRecord[];
};

type YieldForecastRecord = {
  date: string;
  predicted_yield_kg: number;
};

type YieldForecastSummary = {
  start_date?: string;
  end_date?: string;
  mean_kg_per_day?: number;
  total_kg?: number;
};

type YieldForecastResponse = {
  summary: YieldForecastSummary;
  records: YieldForecastRecord[];
};

type Status = "idle" | "uploading" | "success" | "error";

const percent = (value: number) =>
  new Intl.NumberFormat("en-US", {
    style: "percent",
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value);

const climateNumber = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 1,
});
const kgNumber = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 2,
});
const shortDateFormatter = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
});
const monthFormatter = new Intl.DateTimeFormat("en-US", {
  month: "long",
  year: "numeric",
});
const longDateFormatter = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
});

const formatShortDate = (value: string) => shortDateFormatter.format(new Date(value));
const formatMonth = (value: string) => monthFormatter.format(new Date(value));
const formatLongDate = (value: string) => longDateFormatter.format(new Date(value));
const formatValue = (value?: number | null, suffix = "") => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return `${climateNumber.format(value)}${suffix}`;
};
const formatKg = (value?: number | null) => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return `${kgNumber.format(value)} kg`;
};

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<ModelResponse | null>(null);
  const [wards, setWards] = useState<Ward[]>([]);
  const [selectedWardId, setSelectedWardId] = useState<string | null>(null);
  const [isWardLoading, setIsWardLoading] = useState(true);
  const [climateDaily, setClimateDaily] = useState<ClimateRecord[]>([]);
  const [climateMonthly, setClimateMonthly] = useState<ClimateRecord[]>([]);
  const [yieldForecast, setYieldForecast] = useState<YieldForecastResponse | null>(
    null,
  );
  const [contextError, setContextError] = useState<string | null>(null);
  const [isContextLoading, setIsContextLoading] = useState(true);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  useEffect(() => {
    let cancelled = false;
    const loadWards = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/locations/wards`);
        if (!response.ok) {
          throw new Error("Unable to load wards list.");
        }
        const payload = (await response.json()) as Ward[];
        if (!cancelled) {
          setWards(payload);
          setSelectedWardId((current) => current ?? payload[0]?.id ?? null);
        }
      } catch (err) {
        if (!cancelled) {
          setContextError(
            err instanceof Error ? err.message : "Failed to load wards.",
          );
        }
      } finally {
        if (!cancelled) {
          setIsWardLoading(false);
        }
      }
    };
    loadWards();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedWardId) {
      return;
    }
    let cancelled = false;
    const describeError = (err: unknown) =>
      err instanceof Error ? err.message : "Unable to load dataset.";

    const fetchDataset = async <T,>(
      path: string,
      label: string,
      handler: (data: T) => void,
    ) => {
      try {
        const response = await fetch(path);
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload?.detail ?? response.statusText);
        }
        const payload = (await response.json()) as T;
        if (!cancelled) {
          handler(payload);
        }
      } catch (err) {
        if (!cancelled) {
          setContextError((prev) => prev ?? `${label}: ${describeError(err)}`);
        }
      }
    };

    const loadContext = async () => {
      setIsContextLoading(true);
      setContextError(null);
      const wardParam = `ward_id=${encodeURIComponent(selectedWardId)}`;
      await Promise.all([
        fetchDataset<ClimateSeriesResponse>(
          `${API_BASE_URL}/climate/daily?limit=14&${wardParam}`,
          "Ward climate",
          (data) => setClimateDaily(data.records ?? []),
        ),
        fetchDataset<ClimateSeriesResponse>(
          `${API_BASE_URL}/climate/monthly?months=12&${wardParam}`,
          "Monthly climate",
          (data) => setClimateMonthly(data.records ?? []),
        ),
        fetchDataset<YieldForecastResponse>(
          `${API_BASE_URL}/yield/forecast?limit=120&${wardParam}`,
          "Yield forecast",
          (data) => setYieldForecast(data),
        ),
      ]);
      if (!cancelled) {
        setIsContextLoading(false);
      }
    };

    loadContext();
    return () => {
      cancelled = true;
    };
  }, [selectedWardId]);

  const orderedProbabilities = useMemo(() => {
    if (!prediction) return [];
    return Object.entries(prediction.probabilities).sort(
      (a, b) => b[1] - a[1],
    );
  }, [prediction]);
  const recentDailyClimate = useMemo(
    () => [...climateDaily].reverse(),
    [climateDaily],
  );
  const recentMonthlyClimate = useMemo(
    () => [...climateMonthly].reverse(),
    [climateMonthly],
  );
  const yieldSummary = yieldForecast?.summary;
  const hasClimateData =
    recentDailyClimate.length > 0 || recentMonthlyClimate.length > 0;
  const yieldRangeLabel =
    yieldSummary?.start_date && yieldSummary?.end_date
      ? `${formatLongDate(yieldSummary.start_date)} – ${formatLongDate(
          yieldSummary.end_date,
        )}`
      : null;
  const selectedWard = useMemo(
    () => wards.find((ward) => ward.id === selectedWardId) ?? null,
    [wards, selectedWardId],
  );
  const wardLabel = selectedWard
    ? `${selectedWard.name} · ${selectedWard.subcounty}`
    : "Select a ward";
  const climateSummary = useMemo(() => {
    if (!recentDailyClimate.length) {
      return {
        rainfall14: null,
        tempMean14: null,
        humidityMean14: null,
        ndviLatest: null,
        ndviLabel: "No data",
        latestDailyDate: null,
        latestMonthly: null as ClimateRecord | null,
      };
    }
    const rainfall14 = recentDailyClimate.reduce((acc, record) => {
      const value =
        typeof record.rainfall_mm === "number" && Number.isFinite(record.rainfall_mm)
          ? record.rainfall_mm
          : 0;
      return acc + value;
    }, 0);
    const numericMean = (values: (number | undefined | null)[]) => {
      const filtered = values.filter(
        (value): value is number => typeof value === "number" && Number.isFinite(value),
      );
      if (!filtered.length) return null;
      return filtered.reduce((acc, value) => acc + value, 0) / filtered.length;
    };
    const tempMean14 = numericMean(recentDailyClimate.map((record) => record.temp_mean));
    const humidityMean14 = numericMean(
      recentDailyClimate.map((record) => record.humidity_mean),
    );
    const ndviCandidate = recentDailyClimate.find(
      (record) => typeof record.ndvi_mean === "number" && Number.isFinite(record.ndvi_mean),
    );
    const ndviLatest = ndviCandidate?.ndvi_mean ?? null;
    let ndviLabel = "No forage data";
    if (typeof ndviLatest === "number") {
      if (ndviLatest >= 0.6) {
        ndviLabel = "Lush forage";
      } else if (ndviLatest >= 0.4) {
        ndviLabel = "Moderate forage";
      } else {
        ndviLabel = "Sparse forage";
      }
    }
    const latestMonthly = recentMonthlyClimate.length ? recentMonthlyClimate[0] : null;
    return {
      rainfall14,
      tempMean14,
      humidityMean14,
      ndviLatest,
      ndviLabel,
      latestDailyDate: recentDailyClimate[0]?.date ?? null,
      latestMonthly,
    };
  }, [recentDailyClimate, recentMonthlyClimate]);

  const yieldHighlights = useMemo(() => {
    if (!yieldForecast) return [];
    const sample = yieldForecast.records.slice(-30);
    return [...sample]
      .sort((a, b) => b.predicted_yield_kg - a.predicted_yield_kg)
      .slice(0, 3);
  }, [yieldForecast]);

  const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    setPrediction(null);
    setError(null);
    setSelectedFile(file ?? null);
  };
  const onWardChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedWardId(event.target.value || null);
  };

  const onSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedWardId) {
      setError("Select your ward to localize predictions.");
      return;
    }
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
      data.append("ward_id", selectedWardId);

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
            BeeUnity Field Console
          </h1>
          <p className="mx-auto mt-2 max-w-2xl text-base text-slate-600">
            BeeUnity blends hive acoustics with ward-level climate + NDVI context so
            Makueni County beekeepers can safeguard queen vitality, boost hive
            occupancy, and plan harvests proactively.
          </p>
          <p className="mt-3 inline-flex items-center justify-center rounded-full bg-amber-50 px-4 py-1 text-xs font-semibold uppercase tracking-wide text-amber-700">
            {isWardLoading ? "Loading wards…" : wardLabel}
          </p>
        </header>

        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-lg">
          <div className="max-w-3xl">
            <form className="flex flex-col gap-4" onSubmit={onSubmit}>
              <label className="text-sm font-medium text-slate-700" htmlFor="ward">
                Anchor predictions to your ward
              </label>
              <select
                id="ward"
                name="ward"
                disabled={isWardLoading}
                value={selectedWardId ?? ""}
                onChange={onWardChange}
                className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-900 focus:border-emerald-400 focus:outline-none"
              >
                {isWardLoading ? (
                  <option value="">Loading wards...</option>
                ) : (
                  <>
                    <option value="">Select ward</option>
                    {wards.map((ward) => (
                      <option key={ward.id} value={ward.id}>
                        {ward.name} · {ward.subcounty}
                      </option>
                    ))}
                  </>
                )}
              </select>
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
                className="w-full cursor-pointer rounded-2xl border border-dashed border-emerald-300 bg-emerald-50 px-4 py-6 text-center text-sm text-emerald-800 hover:border-emerald-400"
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
                  Drop a WAV file or tap to browse your device to start a BeeUnity scan.
                </p>
              )}

              <button
                type="submit"
                disabled={status === "uploading"}
                className="mt-2 inline-flex items-center justify-center gap-2 rounded-2xl bg-emerald-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-emerald-300"
              >
                {status === "uploading" ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                    Predicting…
                  </>
                ) : (
                  "Run BeeUnity scan"
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
                  BeeUnity verdict
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
              Upload hive audio so BeeUnity can evaluate queen stability, acoustic
              disturbances, and align the verdict with your ward’s conditions.
            </p>
          </section>
        )}

        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-lg">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Ward climate pulse
              </p>
              <h2 className="text-2xl font-semibold text-slate-900">
                Weather + NDVI context for occupancy
              </h2>
              <p className="text-sm text-slate-500">
                Latest ward observations powering BeeUnity’s occupancy, stress, and yield insights.
              </p>
            </div>
            {contextError ? (
              <p className="rounded-2xl bg-rose-50 px-3 py-2 text-xs font-medium text-rose-600">
                {contextError}
              </p>
            ) : null}
          </div>

          {isContextLoading ? (
            <p className="mt-4 text-sm text-slate-500">
              Loading climate intelligence…
            </p>
          ) : hasClimateData ? (
            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <div className="rounded-2xl border border-slate-100 bg-gradient-to-br from-amber-50 to-white p-4 shadow-inner">
                <p className="text-xs font-semibold uppercase tracking-wide text-amber-700">
                  Rainfall (past 14 days)
                </p>
                <p className="mt-2 text-3xl font-semibold text-slate-900">
                  {formatValue(climateSummary.rainfall14, " mm")}
                </p>
                <p className="text-xs text-amber-700">
                  Last reading {climateSummary.latestDailyDate
                    ? formatShortDate(climateSummary.latestDailyDate)
                    : "n/a"}
                </p>
              </div>
              <div className="rounded-2xl border border-slate-100 bg-gradient-to-br from-emerald-50 to-white p-4 shadow-inner">
                <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700">
                  Average temperature
                </p>
                <p className="mt-2 text-3xl font-semibold text-slate-900">
                  {formatValue(climateSummary.tempMean14, "°C")}
                </p>
                <p className="text-xs text-emerald-700">
                  Humidity {formatValue(climateSummary.humidityMean14, "%")} avg
                </p>
              </div>
              <div className="rounded-2xl border border-slate-100 bg-gradient-to-br from-lime-50 to-white p-4 shadow-inner">
                <p className="text-xs font-semibold uppercase tracking-wide text-lime-700">
                  NDVI forage index
                </p>
                <p className="mt-2 text-3xl font-semibold text-slate-900">
                  {formatValue(climateSummary.ndviLatest)}
                </p>
                <p className="text-xs text-lime-700">{climateSummary.ndviLabel}</p>
              </div>
              {climateSummary.latestMonthly ? (
                <div className="rounded-2xl border border-slate-100 bg-gradient-to-br from-slate-50 to-white p-4 shadow-inner">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Latest monthly summary ({formatMonth(climateSummary.latestMonthly.date)})
                  </p>
                  <p className="mt-2 text-2xl font-semibold text-slate-900">
                    {formatValue(climateSummary.latestMonthly.rainfall_mm, " mm rain")}
                  </p>
                  <p className="text-xs text-slate-500">
                    NDVI {formatValue(climateSummary.latestMonthly.ndvi_mean)} · Avg temp{" "}
                    {formatValue(climateSummary.latestMonthly.temp_mean, "°C")}
                  </p>
                </div>
              ) : null}
            </div>
          ) : (
            <p className="mt-4 text-sm text-slate-500">
              Populate <code>content/main-data/makueni_weather_ndvi_2008_2025.csv</code>{" "}
              via the Kaggle notebook to unlock climate intelligence.
            </p>
          )}
        </section>

        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-lg">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Honey yield outlook
              </p>
              <h2 className="text-2xl font-semibold text-slate-900">
                Climate-driven production forecast
              </h2>
              <p className="text-sm text-slate-500">
                Würzburg-pretrained boosting translates the weather + NDVI
                context into kg/day expectations.
              </p>
            </div>
          </div>

          {isContextLoading ? (
            <p className="mt-4 text-sm text-slate-500">Loading yield outlook…</p>
          ) : yieldForecast ? (
            <>
              <div className="mt-6 grid gap-4 sm:grid-cols-2">
                <div className="rounded-2xl bg-emerald-50 p-4 text-slate-900">
                  <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700">
                    Average kg/day
                  </p>
                  <p className="mt-2 text-3xl font-semibold">
                    {formatKg(yieldSummary?.mean_kg_per_day)}
                  </p>
                  <p className="text-xs text-emerald-700">
                    Across {yieldForecast.records.length} forecast days
                  </p>
                </div>
                <div className="rounded-2xl bg-amber-50 p-4 text-slate-900">
                  <p className="text-xs font-semibold uppercase tracking-wide text-amber-700">
                    Total projected yield
                  </p>
                  <p className="mt-2 text-3xl font-semibold">
                    {formatKg(yieldSummary?.total_kg)}
                  </p>
                  <p className="text-xs text-amber-700">
                    {yieldRangeLabel ?? "Run the climate forecasting block"}
                  </p>
                </div>
              </div>

              <div className="mt-8 rounded-2xl border border-slate-100 bg-white p-4">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Top harvest windows (next 30 days)
                </p>
                {yieldHighlights.length ? (
                  <ul className="mt-3 space-y-2 text-sm text-slate-700">
                    {yieldHighlights.map((record) => (
                      <li
                        key={`harvest-${record.date}`}
                        className="flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50 px-3 py-2"
                      >
                        <span className="font-medium text-slate-900">
                          {formatShortDate(record.date)}
                        </span>
                        <span className="text-amber-700">
                          {formatKg(record.predicted_yield_kg)}
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-2 text-sm text-slate-500">
                    Forecast ready but no standout harvest windows in the selected range.
                  </p>
                )}
              </div>
            </>
          ) : (
            <p className="mt-4 text-sm text-slate-500">
              Populate <code>content/main-data/makueni_climate_yield_forecast.csv</code>{" "}
              to expose the kg/day outlook via the API.
            </p>
          )}
        </section>
      </main>
    </div>
  );
}
