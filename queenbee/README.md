QueenBee Machine is the frontend companion to the FastAPI service, offering a polished upload experience and live visualization of the CNN probabilities.

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

## Production build

```
npm run build
npm run start
```

Ensure the FastAPI server is reachable from the deployment; adjust `NEXT_PUBLIC_API_BASE_URL` accordingly (e.g., `https://api.example.com`).
