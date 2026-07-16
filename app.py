import io
import os
import subprocess
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

app = FastAPI(title="EZ DB's Audio Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MIN_HZ = 10
MAX_HZ = 80
SAMPLE_RATE = 44100
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_ANALYSIS_SECONDS = 90
FFT_SIZE = 65536


@app.get("/")
def root():
    return {"service": "EZ DB's Audio Analyzer API", "status": "ok"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "analysis_band_hz": [MIN_HZ, MAX_HZ],
        "fft_bin_resolution_hz": round(SAMPLE_RATE / FFT_SIZE, 3),
    }


def decode_audio_to_mono_pcm(file_bytes: bytes) -> np.ndarray:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-map", "0:a:0",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-t", str(MAX_ANALYSIS_SECONDS),
        "-f", "s16le", "pipe:1",
    ]
    process = subprocess.run(command, input=file_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if process.returncode != 0 or not process.stdout:
        error_message = process.stderr.decode("utf-8", errors="ignore").strip()
        raise ValueError(error_message or "Could not decode that audio file. Try a standard MP3, WAV, M4A, or AAC file.")

    audio = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float64) / 32768.0
    if audio.size < FFT_SIZE:
        raise ValueError("That audio file is too short for a reliable low-frequency analysis.")
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


def make_integer_hz_graph(freqs: np.ndarray, db_values: np.ndarray) -> List[dict]:
    graph_points = []
    for hz in range(MIN_HZ, MAX_HZ + 1):
        db = float(np.interp(hz, freqs, db_values))
        graph_points.append({"hz": hz, "db": round(db, 2)})
    return graph_points


def average_band(graph_points: List[dict], start_hz: int, end_hz: int) -> float:
    values = [p["db"] for p in graph_points if start_hz <= p["hz"] <= end_hz]
    return round(float(np.mean(values)), 2) if values else -120.0


def analyze_audio(file_bytes: bytes) -> dict:
    audio = decode_audio_to_mono_pcm(file_bytes)
    sos = butter(4, [MIN_HZ / (SAMPLE_RATE / 2), MAX_HZ / (SAMPLE_RATE / 2)], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, audio)

    freqs, power = welch(
        filtered, fs=SAMPLE_RATE, window="hann",
        nperseg=FFT_SIZE, noverlap=FFT_SIZE // 2,
        detrend="constant", scaling="spectrum", average="mean",
    )
    db_values = 10 * np.log10(np.maximum(power, 1e-18))
    db_values -= np.max(db_values)

    band_mask = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
    band_freqs = freqs[band_mask]
    band_db = db_values[band_mask]

    if band_freqs.size == 0:
        raise ValueError("No usable frequency data was found between 10 Hz and 80 Hz.")

    strongest_index = int(np.argmax(band_db))
    strongest_peak_hz = round(float(band_freqs[strongest_index]), 1)

    raw_peaks, _ = find_peaks(band_db, prominence=2.0, distance=max(1, int(2 / (SAMPLE_RATE / FFT_SIZE))))
    detected_peaks = []
    for index in raw_peaks:
        detected_peaks.append({"hz": round(float(band_freqs[index]), 1), "db": round(float(band_db[index]), 2)})
    detected_peaks.sort(key=lambda item: item["db"], reverse=True)

    final_peaks = []
    for peak in detected_peaks:
        if all(abs(peak["hz"] - existing["hz"]) >= 2.0 for existing in final_peaks):
            final_peaks.append(peak)
        if len(final_peaks) >= 5:
            break

    if not final_peaks:
        final_peaks = [{"hz": strongest_peak_hz, "db": round(float(band_db[strongest_index]), 2)}]

    graph_points = make_integer_hz_graph(band_freqs, band_db)

    return {
        "strongest_peak_hz": strongest_peak_hz,
        "peak_list_10_80": [peak["hz"] for peak in final_peaks],
        "peaks_detail_10_80": final_peaks,
        "graph_points_10_80": graph_points,
        "bands": {
            "low": average_band(graph_points, 10, 24),
            "mid": average_band(graph_points, 25, 44),
            "upper": average_band(graph_points, 45, 80),
        },
        "analysis": {
            "range_hz": [MIN_HZ, MAX_HZ],
            "sample_rate_hz": SAMPLE_RATE,
            "fft_size": FFT_SIZE,
            "bin_resolution_hz": round(SAMPLE_RATE / FFT_SIZE, 3),
        },
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Choose an audio file first.")
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="The selected file is empty.")
        if len(file_bytes) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="That file is too large. Please use an audio file under 25 MB.")
        return analyze_audio(file_bytes)
    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Analysis took too long. Try a shorter audio file.")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        print(f"Unexpected analysis error: {error}")
        raise HTTPException(status_code=500, detail="The analyzer could not process that file. Try another audio file.")
