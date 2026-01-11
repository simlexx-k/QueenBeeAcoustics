BeeUnity’s field console is the frontend companion to the FastAPI service, offering a polished upload experience and live visualization of the CNN probabilities plus ward-level climate/yield intelligence.

## Prerequisites

1. Train/tune the Keras model inside the root notebook and export it to `content/beehive_audio/7/Dataset/queenbee_final_tuned_model.h5`.
2. Start the backend API from the repository root:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Environment

Set the target API host with:

```bash
export NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"
```

If unset, the frontend defaults to `http://localhost:8000`.

## Development

```
npm install
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000) and upload a WAV file. The UI will preview the audio, call `/predict`, and display the top probabilities plus the detected class.

If the backend has `hive_unified_model.pkl` + `hive_weather_acoustic.*` available, each response also includes `stress_probability` / `stress_label`, which the UI surfaces under the prediction header.

Every upload flow starts with selecting a ward from Makueni County. The UI calls `/locations/wards`, lets the user choose the ward (instead of typing lat/long), and automatically injects that `ward_id` into `/predict`. The ward selection also drives the contextual panels.

The dashboard calls `/climate/daily`, `/climate/monthly`, and `/yield/forecast` with the chosen `ward_id` to render weather/NDVI summaries plus the kg/day outlook. Run the Kaggle notebook sections that export `content/main-data/makueni_weather_ndvi_2008_2025.csv` and `content/main-data/makueni_climate_yield_forecast.csv` so these panels have data; otherwise the UI explains how to populate the required CSVs.

## Production build

```
npm run build
npm run start
```

Ensure the FastAPI server is reachable from the deployment; adjust `NEXT_PUBLIC_API_BASE_URL` accordingly (e.g., `https://api.example.com`).
