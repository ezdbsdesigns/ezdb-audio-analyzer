# EZ DB's Audio Analyzer API

FastAPI backend that analyzes uploaded audio and returns the strongest peak
and 10–80 Hz spectrum for the EZ DB's Designs bass frequency analyzer.

Endpoints:
- `GET  /api/health` — service status
- `POST /analyze` — multipart form-data with `file` field, returns JSON with
  `strongest_peak_hz`, `peak_list_10_80`, `graph_points_10_80`, `bands`.
