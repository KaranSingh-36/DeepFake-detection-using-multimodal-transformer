import cgi
import json
import mimetypes
import os
import subprocess
import sys
import threading
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "checkpoints" / "multimodal_25000_2epoch_epoch_2.pth"
RUNS_DIR = Path(__file__).resolve().parent / "runtime"
RUNS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
SAMPLE_RATE = 16000
AUDIO_SECONDS = 4.0
NUM_FRAMES = 8
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

JOBS = {}
JOBS_LOCK = threading.Lock()
MODEL = None
MODEL_LOCK = threading.Lock()


def update_job(run_id, **values):
    with JOBS_LOCK:
        if run_id in JOBS:
            JOBS[run_id].update(values)


def load_model(run_id):
    global MODEL
    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL
        update_job(run_id, stage="Loading the multimodal model", progress=12)
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
        sys.path.insert(0, str(PROJECT_ROOT))
        from models.multimodal_model import MultimodalCrossAttentionModel

        model = MultimodalCrossAttentionModel().to(DEVICE)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        MODEL = model
        return MODEL


def read_frames(video_path, run_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the uploaded video.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    duration = total_frames / fps if fps > 0 else 0
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("The uploaded video contains no readable frames.")

    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=np.int64)
    frames = []
    thumbnails = []
    target = 0
    index = 0
    while target < len(indices):
        if not cap.grab():
            break
        if index == int(indices[target]):
            ok, frame = cap.retrieve()
            if not ok:
                if not frames:
                    cap.release()
                    raise RuntimeError("A sampled video frame could not be decoded.")
                frame = frames[-1].copy()
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
                frames.append(resized.astype(np.float32) / 255.0)
            thumb_path = run_dir / f"frame_{target + 1}.jpg"
            cv2.imwrite(str(thumb_path), frame if ok else cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
            thumbnails.append(f"/runs/{run_dir.name}/{thumb_path.name}")
            target += 1
        index += 1
        if index >= total_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError("No video frames could be decoded.")
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1].copy())
    array = np.transpose(np.stack(frames[:NUM_FRAMES]), (0, 3, 1, 2))
    tensor = torch.from_numpy(array.copy()).float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std, thumbnails, duration


def read_audio(video_path):
    command = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-t", f"{AUDIO_SECONDS:.3f}", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "pipe:1",
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required for audio extraction but was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"FFmpeg could not extract audio{': ' + detail if detail else '.'}") from exc
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    expected = int(SAMPLE_RATE * AUDIO_SECONDS)
    if audio.size == 0:
        raise RuntimeError("The uploaded video has no readable audio track.")
    if audio.size < expected:
        audio = np.pad(audio, (0, expected - audio.size))
    return torch.from_numpy(audio[:expected]).float()


def process_video(run_id, video_path, original_name):
    try:
        run_dir = video_path.parent
        update_job(run_id, stage="Sampling video frames", progress=28)
        frames, thumbnails, duration = read_frames(video_path, run_dir)
        update_job(run_id, stage="Extracting the first 4 seconds of audio", progress=45, frames=thumbnails)
        audio = read_audio(video_path)
        update_job(run_id, stage="Running visual and audio encoders", progress=62)
        model = load_model(run_id)
        with torch.inference_mode():
            output = model(
                frames.unsqueeze(0).to(DEVICE),
                audio.unsqueeze(0).to(DEVICE),
            )
            probabilities = torch.softmax(output, dim=1)[0].cpu().tolist()
        predicted_index = int(np.argmax(probabilities))
        labels = ["REAL", "DEEPFAKE"]
        confidence = probabilities[predicted_index] * 100
        update_job(
            run_id,
            status="complete",
            stage="Detection complete",
            progress=100,
            result={
                "file": original_name,
                "label": labels[predicted_index],
                "confidence": round(confidence, 2),
                "real_probability": round(probabilities[0] * 100, 2),
                "deepfake_probability": round(probabilities[1] * 100, 2),
                "duration": round(duration, 2),
                "frames_processed": NUM_FRAMES,
                "audio_seconds_processed": AUDIO_SECONDS,
                "device": str(DEVICE),
                "checkpoint_test_accuracy": 97.65,
            },
        )
    except Exception as exc:
        traceback.print_exc()
        update_job(run_id, status="error", stage="Processing failed", error=str(exc), progress=100)


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeepGuard | Multimodal Detection Dashboard</title>
<style>
:root{--bg:#07111f;--panel:#0d1b2e;--line:#203653;--text:#e7effa;--muted:#8da4be;--cyan:#35d6e8;--green:#49df9a;--red:#ff6c8f;--amber:#f5c96a}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#153456 0,#07111f 42%);color:var(--text);font:15px Inter,Segoe UI,Arial,sans-serif;min-height:100vh}
.wrap{max-width:1180px;margin:0 auto;padding:34px 22px 60px}.header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:26px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:2px;text-transform:uppercase;font-weight:700}.title{font-size:36px;margin:8px 0}.sub{color:var(--muted);max-width:680px}.badge{border:1px solid #28506d;border-radius:22px;padding:9px 14px;color:var(--green);font-size:13px}
.grid{display:grid;grid-template-columns:1.05fr 1.5fr;gap:18px}.card{background:rgba(13,27,46,.9);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 16px 40px #0002}.card h2{font-size:17px;margin:0 0 16px}.drop{border:1px dashed #3b6b8b;border-radius:12px;padding:34px 20px;text-align:center;cursor:pointer;background:#10243a}.drop:hover{border-color:var(--cyan);background:#132b45}.drop strong{display:block;font-size:17px;margin-bottom:7px}.drop span{color:var(--muted);font-size:13px}.drop input{display:none}.file{margin:14px 0;color:var(--cyan);word-break:break-all}.button{width:100%;border:0;border-radius:10px;padding:13px;background:linear-gradient(90deg,#27c7de,#3be2a0);color:#04121c;font-weight:800;cursor:pointer}.button:disabled{opacity:.4;cursor:not-allowed}
.pipeline{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0 20px}.step{border:1px solid var(--line);border-radius:10px;padding:12px 8px;text-align:center;color:var(--muted);font-size:12px}.step .dot{height:9px;width:9px;border-radius:50%;background:#35506d;margin:0 auto 8px}.step.active{border-color:var(--cyan);color:var(--text)}.step.active .dot{background:var(--cyan);box-shadow:0 0 12px var(--cyan)}.step.done{border-color:#287653;color:var(--green)}.step.done .dot{background:var(--green)}.progress{height:9px;background:#081524;border-radius:10px;overflow:hidden}.bar{height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--green));transition:width .4s}.status{color:var(--muted);margin:11px 0 0}.result{margin-top:18px;border-radius:12px;padding:20px;background:#0a1727;border:1px solid var(--line);display:none}.result.show{display:block}.label{font-size:32px;font-weight:800;margin:4px 0 14px}.real{color:var(--green)}.fake{color:var(--red)}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.metric{background:#102238;border-radius:9px;padding:12px}.metric small{display:block;color:var(--muted);margin-bottom:5px}.metric b{font-size:19px}.prob{margin-top:15px}.probrow{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin:7px 0 4px}.probbar{height:7px;background:#1a2d42;border-radius:6px;overflow:hidden}.probbar i{display:block;height:100%;background:var(--green)}.probbar i.red{background:var(--red)}.preview{margin-top:18px;display:none}.preview video{width:100%;border-radius:10px;max-height:280px;background:#02070d}.frames{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}.frames img{width:72px;height:54px;object-fit:cover;border-radius:5px;border:1px solid var(--line)}.note{font-size:12px;color:var(--muted);line-height:1.5;margin-top:14px}.error{color:var(--red)}
@media(max-width:800px){.header{display:block}.badge{display:inline-block;margin-top:15px}.grid{grid-template-columns:1fr}.title{font-size:29px}}
</style>
</head>
<body><main class="wrap">
<header class="header"><div><div class="eyebrow">Multimodal transformer laboratory</div><h1 class="title">DeepGuard detection dashboard</h1><div class="sub">Upload a video and watch the same visual + audio pipeline used by the 25K-sample, 2-epoch checkpoint.</div></div><div class="badge">● Local inference</div></header>
<section class="grid">
<div class="card"><h2>1. Upload media</h2><label class="drop" for="file"><strong>Drop a video here</strong><span>or click to browse · MP4, AVI, MOV, MKV · max 500 MB</span><input id="file" type="file" accept="video/*"></label><div id="filename" class="file">No video selected</div><button id="start" class="button" disabled>Analyze video</button><p class="note">The checkpoint-compatible pipeline samples 8 frames and extracts the first 4 seconds of mono 16 kHz audio.</p></div>
<div class="card"><h2>2. Live processing trace</h2><div class="pipeline"><div id="s1" class="step"><div class="dot"></div>Sample 8 frames</div><div id="s2" class="step"><div class="dot"></div>Extract 4s audio</div><div id="s3" class="step"><div class="dot"></div>Encode modalities</div><div id="s4" class="step"><div class="dot"></div>Classify</div></div><div class="progress"><div id="bar" class="bar"></div></div><div id="status" class="status">Waiting for a video.</div><div id="result" class="result"><div class="eyebrow">Final prediction</div><div id="label" class="label"></div><div class="metrics"><div class="metric"><small>Detection confidence</small><b id="confidence">—</b></div><div class="metric"><small>Video duration</small><b id="duration">—</b></div><div class="metric"><small>Checkpoint test accuracy</small><b>97.65%</b></div></div><div class="prob"><div class="probrow"><span>REAL</span><span id="real">—</span></div><div class="probbar"><i id="realbar"></i></div><div class="probrow"><span>DEEPFAKE</span><span id="fake">—</span></div><div class="probbar"><i id="fakebar" class="red"></i></div></div><p id="processed" class="note"></p><p class="note">Confidence is this video's predicted-class probability. The 97.65% figure is the saved checkpoint's test-set accuracy, not a guarantee for an individual upload.</p></div></div>
</section><section id="preview" class="card preview"><h2>3. Visual evidence sampled from the upload</h2><video id="video" controls></video><div id="frames" class="frames"></div></section>
</main>
<script>
const file=document.querySelector('#file'), start=document.querySelector('#start'), nameEl=document.querySelector('#filename'), statusEl=document.querySelector('#status'), bar=document.querySelector('#bar'), result=document.querySelector('#result');
let selected=null, objectUrl=null;
file.onchange=()=>{selected=file.files[0]||null;nameEl.textContent=selected?selected.name:'No video selected';start.disabled=!selected;if(selected){objectUrl&&URL.revokeObjectURL(objectUrl);objectUrl=URL.createObjectURL(selected);document.querySelector('#video').src=objectUrl;document.querySelector('#preview').style.display='block';}};
function stages(p){['s1','s2','s3','s4'].forEach((id,i)=>{const el=document.querySelector('#'+id);el.className='step '+(p>=([28,45,62,100][i])?'done':p>=[0,28,45,62][i]?'active':'');});}
function render(data){bar.style.width=data.progress+'%';statusEl.textContent=data.stage||'';stages(data.progress);if(data.frames){document.querySelector('#frames').innerHTML=data.frames.map(x=>`<img src="${x}" alt="representative decoded frame">`).join('');}if(data.status==='error'){statusEl.className='status error';statusEl.textContent=data.error;start.disabled=false;return;}if(data.status==='complete'){const r=data.result;const label=document.querySelector('#label');label.textContent=r.label;label.className='label '+(r.label==='REAL'?'real':'fake');document.querySelector('#confidence').textContent=r.confidence+'%';document.querySelector('#duration').textContent=r.duration+'s';document.querySelector('#real').textContent=r.real_probability+'%';document.querySelector('#fake').textContent=r.deepfake_probability+'%';document.querySelector('#realbar').style.width=r.real_probability+'%';document.querySelector('#fakebar').style.width=r.deepfake_probability+'%';document.querySelector('#processed').textContent=`Processed ${r.frames_processed} video frames and ${r.audio_seconds_processed}s of audio on ${r.device}.`;result.classList.add('show');start.disabled=false;}}
async function poll(run){const timer=setInterval(async()=>{const data=await fetch('/api/status/'+run).then(r=>r.json());render(data);if(data.status==='complete'||data.status==='error')clearInterval(timer);},500);}
start.onclick=async()=>{if(!selected)return;start.disabled=true;result.classList.remove('show');statusEl.className='status';statusEl.textContent='Uploading video…';bar.style.width='3%';const fd=new FormData();fd.append('video',selected);const response=await fetch('/api/upload',{method:'POST',body:fd});const data=await response.json();if(!response.ok){render({status:'error',error:data.error,progress:100});return;}poll(data.run_id);};
</script></body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            data = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path.startswith("/api/status/"):
            run_id = parsed.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(run_id)
            if job is None:
                self.send_json({"error": "Run not found."}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json(job)
            return
        if parsed.path.startswith("/runs/"):
            relative = unquote(parsed.path[len("/runs/"):]).replace("/", os.sep)
            candidate = (RUNS_DIR / relative).resolve()
            if RUNS_DIR.resolve() not in candidate.parents or not candidate.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
            data = candidate.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path != "/api/upload":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                self.send_json({"error": "Upload must be between 1 byte and 500 MB."}, HTTPStatus.BAD_REQUEST)
                return
            form = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"}
            )
            item = form["video"] if "video" in form else None
            if item is None or not getattr(item, "filename", ""):
                self.send_json({"error": "Choose a video file first."}, HTTPStatus.BAD_REQUEST)
                return
            suffix = Path(item.filename).suffix.lower() or ".mp4"
            if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}:
                self.send_json({"error": "Unsupported video type."}, HTTPStatus.BAD_REQUEST)
                return
            run_id = uuid.uuid4().hex
            run_dir = RUNS_DIR / run_id
            run_dir.mkdir()
            video_path = run_dir / f"upload{suffix}"
            with video_path.open("wb") as target:
                target.write(item.file.read())
            with JOBS_LOCK:
                JOBS[run_id] = {"status": "running", "stage": "Upload received", "progress": 8}
            threading.Thread(
                target=process_video,
                args=(run_id, video_path, Path(item.filename).name),
                daemon=True,
            ).start()
            self.send_json({"run_id": run_id}, HTTPStatus.ACCEPTED)
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format, *args):
        print(f"[dashboard] {self.address_string()} - {format % args}")


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", "8501"))
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Dashboard running at http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler).serve_forever()
