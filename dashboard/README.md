# FusionTrace dashboard

Local-only web page for the trained window-level model. Upload a clip and get
the verdict, a timeline of the fused / video / audio heads over the whole clip,
the flagged segments with their modality, and thumbnails as evidence.

```powershell
python dashboard\app.py                                   # results/runs/latest/best.pt
python dashboard\app.py --checkpoint results\runs\<run>\best.pt --port 8501
```

Then open <http://127.0.0.1:8501>. FFmpeg must be on `PATH` (or `pip install imageio-ffmpeg`).

JSON API for other front ends (CORS enabled):

| Method | Path                     | Body / result                                   |
| ------ | ------------------------ | ----------------------------------------------- |
| POST   | `/api/upload`            | multipart field `video` -> `{"run_id": ...}`    |
| GET    | `/api/status/<run_id>`   | `{status, stage, progress, result?}`            |
| GET    | `/runs/<run_id>/<file>`  | thumbnails, `timeline.png`, `result.json`       |

`result` is the dictionary produced by `fusiontrace.inference.VideoAnalyzer.analyze`
(label, video_score, segments, per-window probabilities, curves).
