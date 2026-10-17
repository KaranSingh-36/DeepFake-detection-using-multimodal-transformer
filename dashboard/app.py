"""FusionTrace local dashboard: upload a video, see where it is fake.

    python dashboard/app.py                               # uses results/runs/latest/best.pt
    python dashboard/app.py --checkpoint results/runs/<run>/best.pt --port 8501

Standard library HTTP server, no extra dependencies beyond the pipeline.
JSON API (also usable from the React front end in src/):
    POST /api/upload   multipart field "video"       -> {"run_id": ...}
    GET  /api/status/<run_id>                        -> job state, result when complete
    GET  /runs/<run_id>/<file>                       -> thumbnails / timeline PNG
"""

import argparse
import json
import mimetypes
import os
import re
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusiontrace.inference import VideoAnalyzer, plot_timeline  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent / "runtime"
RUNS_DIR.mkdir(exist_ok=True)
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
ALLOWED = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}

JOBS = {}
JOBS_LOCK = threading.Lock()
ANALYZER = None
ANALYZER_LOCK = threading.Lock()
CHECKPOINT = PROJECT_ROOT / "results" / "runs" / "latest" / "best.pt"


def update_job(run_id, **values):
    with JOBS_LOCK:
        if run_id in JOBS:
            JOBS[run_id].update(values)


def get_analyzer(run_id=None):
    global ANALYZER
    with ANALYZER_LOCK:
        if ANALYZER is None:
            if run_id:
                update_job(run_id, stage="Loading the model (first request only)", progress=10)
            if not CHECKPOINT.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}. Train first or pass --checkpoint.")
            ANALYZER = VideoAnalyzer(str(CHECKPOINT))
        return ANALYZER


def save_thumbnails(result, run_dir):
    frames = result.pop("frames")
    times = result.pop("frame_times")
    strip = []
    for index in np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int):
        name = f"strip_{index}.jpg"
        cv2.imwrite(str(run_dir / name), cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR))
        strip.append({"time": times[index], "url": f"/runs/{run_dir.name}/{name}"})
    for n, seg in enumerate(result["segments"]):
        centre = (seg["start"] + seg["end"]) / 2.0
        index = int(np.argmin(np.abs(np.asarray(times) - centre)))
        name = f"segment_{n}.jpg"
        cv2.imwrite(str(run_dir / name), cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR))
        seg["thumbnail"] = f"/runs/{run_dir.name}/{name}"
    result["frame_strip"] = strip


def process_video(run_id, video_path, original_name):
    try:
        run_dir = video_path.parent
        analyzer = get_analyzer(run_id)
        update_job(run_id, stage="Decoding frames, cropping faces, extracting audio", progress=30)
        result = analyzer.analyze(str(video_path), keep_frames=True)
        update_job(run_id, stage="Rendering evidence", progress=90)
        save_thumbnails(result, run_dir)
        plot_timeline(result, run_dir / "timeline.png")
        result["timeline_png"] = f"/runs/{run_dir.name}/timeline.png"
        result["file"] = original_name
        result["checkpoint"] = str(CHECKPOINT)
        result["checkpoint_epoch"] = analyzer.checkpoint.get("epoch")
        dev = analyzer.checkpoint.get("metrics", {}).get("video_level", {})
        result["checkpoint_dev_auc"] = dev.get("auc")
        with open(run_dir / "result.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        update_job(run_id, status="complete", stage="Analysis complete", progress=100, result=result)
    except Exception as exc:
        traceback.print_exc()
        update_job(run_id, status="error", stage="Processing failed", error=str(exc), progress=100)


def parse_multipart(body, content_type):
    """Minimal multipart/form-data parser -> {field: (filename, bytes)}."""
    match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not match:
        raise ValueError("Missing multipart boundary.")
    boundary = ("--" + match.group(1)).encode()
    fields = {}
    for part in body.split(boundary)[1:]:
        if part.strip() in (b"", b"--"):
            continue
        header_blob, _, data = part.partition(b"\r\n\r\n")
        headers = header_blob.decode("utf-8", errors="ignore")
        name = re.search(r'name="([^"]*)"', headers)
        filename = re.search(r'filename="([^"]*)"', headers)
        if name:
            fields[name.group(1)] = (filename.group(1) if filename else None, data[:-2] if data.endswith(b"\r\n") else data)
    return fields


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FusionTrace | where is this video fake?</title>
<style>
:root{--bg:#07111f;--panel:#0d1b2e;--line:#203653;--text:#e7effa;--muted:#8da4be;--cyan:#35d6e8;--green:#49df9a;--red:#ff6c8f;--amber:#f5c96a}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#153456 0,#07111f 42%);color:var(--text);font:15px Inter,Segoe UI,Arial,sans-serif;min-height:100vh}
.wrap{max-width:1180px;margin:0 auto;padding:30px 22px 60px}.header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:22px}
.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:2px;text-transform:uppercase;font-weight:700}.title{font-size:34px;margin:8px 0}.sub{color:var(--muted);max-width:720px}
.badge{border:1px solid #28506d;border-radius:22px;padding:9px 14px;color:var(--green);font-size:13px}
.grid{display:grid;grid-template-columns:1fr 1.4fr;gap:18px}.card{background:rgba(13,27,46,.9);border:1px solid var(--line);border-radius:16px;padding:20px}.card h2{font-size:17px;margin:0 0 14px}
.drop{border:1px dashed #3b6b8b;border-radius:12px;padding:30px 20px;text-align:center;cursor:pointer;background:#10243a}.drop:hover{border-color:var(--cyan)}.drop strong{display:block;font-size:17px;margin-bottom:7px}.drop span{color:var(--muted);font-size:13px}.drop input{display:none}
.file{margin:14px 0;color:var(--cyan);word-break:break-all}.button{width:100%;border:0;border-radius:10px;padding:13px;background:linear-gradient(90deg,#27c7de,#3be2a0);color:#04121c;font-weight:800;cursor:pointer}.button:disabled{opacity:.4;cursor:not-allowed}
.progress{height:9px;background:#081524;border-radius:10px;overflow:hidden;margin-top:14px}.bar{height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--green));transition:width .4s}.status{color:var(--muted);margin:11px 0 0}
video{width:100%;border-radius:10px;max-height:300px;background:#02070d;margin-top:14px}
.result{display:none}.result.show{display:block}.label{font-size:32px;font-weight:800;margin:4px 0 10px}.real{color:var(--green)}.fake{color:var(--red)}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-bottom:14px}.metric{background:#102238;border-radius:9px;padding:12px}.metric small{display:block;color:var(--muted);margin-bottom:5px}.metric b{font-size:18px}
canvas{width:100%;height:170px;background:#081524;border-radius:10px;display:block}
.legend{display:flex;gap:16px;font-size:12px;color:var(--muted);margin:8px 0 14px}.legend i{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:1px}
td img{height:44px;border-radius:5px;vertical-align:middle;margin-right:8px}.jump{border:1px solid var(--line);background:#102238;color:var(--cyan);border-radius:6px;padding:4px 10px;cursor:pointer}
.strip{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}.strip img{width:70px;height:70px;object-fit:cover;border-radius:6px;border:1px solid var(--line)}
.note{font-size:12px;color:var(--muted);line-height:1.5;margin-top:14px}.error{color:var(--red)}
@media(max-width:860px){.grid{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main class="wrap">
<header class="header"><div><div class="eyebrow">Multimodal transformer · window-level localisation</div><h1 class="title">FusionTrace detection dashboard</h1>
<div class="sub">Every 1.6 s window of the whole clip is scored by the visual head, the audio head and the fused cross-attention head. The timeline shows <em>where</em> the model thinks the clip is manipulated and by which modality.</div></div><div class="badge">● Local inference</div></header>
<section class="grid">
<div class="card"><h2>1. Upload media</h2><label class="drop" for="file"><strong>Drop a video here</strong><span>or click to browse · MP4, MOV, AVI, MKV, WEBM · max 500 MB</span><input id="file" type="file" accept="video/*"></label>
<div id="filename" class="file">No video selected</div><button id="start" class="button" disabled>Analyze video</button>
<div class="progress"><div id="bar" class="bar"></div></div><div id="status" class="status">Waiting for a video.</div>
<video id="video" controls style="display:none"></video>
<p class="note">Frames are face-cropped and encoded with ViT at 5 fps; the full audio track is encoded with Wav2Vec2. Nothing leaves this machine.</p></div>
<div class="card"><h2>2. Result and timeline</h2>
<div id="empty" class="note">The timeline appears here after analysis.</div>
<div id="result" class="result">
<div class="eyebrow">Clip verdict</div><div id="label" class="label"></div>
<div class="metrics"><div class="metric"><small>Clip score</small><b id="score">—</b></div><div class="metric"><small>Flagged time</small><b id="fakesec">—</b></div><div class="metric"><small>Video head max</small><b id="vmax">—</b></div><div class="metric"><small>Audio head max</small><b id="amax">—</b></div></div>
<canvas id="timeline" width="1100" height="340"></canvas>
<div class="legend"><span><i style="background:#ff6c8f"></i>fused</span><span><i style="background:#35d6e8"></i>video head</span><span><i style="background:#49df9a"></i>audio head</span><span><i style="background:#f5c96a;height:1px"></i>segment threshold</span></div>
<table><thead><tr><th>Segment</th><th>Score</th><th>Modality</th><th></th></tr></thead><tbody id="segments"></tbody></table>
<div id="strip" class="strip"></div>
<p id="meta" class="note"></p>
</div></div>
</section></main>
<script>
const $=s=>document.querySelector(s);const file=$('#file'),start=$('#start'),nameEl=$('#filename'),statusEl=$('#status'),bar=$('#bar'),result=$('#result'),video=$('#video');
let selected=null,objectUrl=null,last=null;
file.onchange=()=>{selected=file.files[0]||null;nameEl.textContent=selected?selected.name:'No video selected';start.disabled=!selected;if(selected){objectUrl&&URL.revokeObjectURL(objectUrl);objectUrl=URL.createObjectURL(selected);video.src=objectUrl;video.style.display='block';}};
function drawTimeline(r){const c=$('#timeline'),ctx=c.getContext('2d'),W=c.width,H=c.height,L=44,R=12,T=14,B=34;ctx.clearRect(0,0,W,H);ctx.fillStyle='#081524';ctx.fillRect(0,0,W,H);
const x=t=>L+(W-L-R)*t/r.duration,y=p=>T+(H-T-B)*(1-p);
ctx.fillStyle='rgba(255,108,143,.18)';r.segments.forEach(s=>ctx.fillRect(x(s.start),T,x(s.end)-x(s.start),H-T-B));
ctx.strokeStyle='#203653';ctx.lineWidth=1;[0,.25,.5,.75,1].forEach(p=>{ctx.beginPath();ctx.moveTo(L,y(p));ctx.lineTo(W-R,y(p));ctx.stroke();ctx.fillStyle='#8da4be';ctx.font='12px sans-serif';ctx.fillText(p.toFixed(2),4,y(p)+4);});
ctx.setLineDash([6,5]);ctx.strokeStyle='#f5c96a';ctx.beginPath();ctx.moveTo(L,y(r.step_threshold));ctx.lineTo(W-R,y(r.step_threshold));ctx.stroke();ctx.setLineDash([]);
const line=(arr,col,w)=>{if(!arr)return;ctx.strokeStyle=col;ctx.lineWidth=w;ctx.beginPath();arr.forEach((p,i)=>{const px=x(r.curve.times[i]),py=y(p);i?ctx.lineTo(px,py):ctx.moveTo(px,py)});ctx.stroke();};
line(r.curve.video,'#35d6e8',1.5);line(r.curve.audio,'#49df9a',1.5);line(r.curve.fused,'#ff6c8f',3);
ctx.fillStyle='#8da4be';const step=r.duration>20?5:r.duration>8?2:1;for(let t=0;t<=r.duration;t+=step){ctx.fillText(t+'s',x(t)-8,H-12);}
if(video.duration){ctx.strokeStyle='#e7effa';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x(video.currentTime),T);ctx.lineTo(x(video.currentTime),H-B);ctx.stroke();}}
$('#timeline').onclick=e=>{if(!last)return;const c=$('#timeline'),rect=c.getBoundingClientRect(),px=(e.clientX-rect.left)*c.width/rect.width,L=44,R=12;video.currentTime=Math.max(0,Math.min(last.duration,(px-L)/(c.width-L-R)*last.duration));};
video.ontimeupdate=()=>{if(last)drawTimeline(last);};
function render(data){bar.style.width=data.progress+'%';statusEl.textContent=data.stage||'';if(data.status==='error'){statusEl.className='status error';statusEl.textContent=data.error;start.disabled=false;return;}
if(data.status==='complete'){const r=data.result;last=r;$('#empty').style.display='none';result.classList.add('show');const label=$('#label');label.textContent=r.label;label.className='label '+(r.label==='REAL'?'real':'fake');
$('#score').textContent=r.video_score.toFixed(3)+' / thr '+r.threshold.toFixed(2);$('#fakesec').textContent=r.fake_seconds+' s of '+r.duration+' s';$('#vmax').textContent=r.max_video_head==null?'n/a':r.max_video_head.toFixed(3);$('#amax').textContent=r.max_audio_head==null?'n/a':r.max_audio_head.toFixed(3);
drawTimeline(r);const tb=$('#segments');tb.innerHTML=r.segments.length?r.segments.map((s,i)=>`<tr><td>${s.thumbnail?`<img src="${s.thumbnail}">`:''}${s.start.toFixed(2)}s – ${s.end.toFixed(2)}s</td><td>${s.score.toFixed(3)}</td><td>${s.modality}</td><td><button class="jump" data-t="${s.start}">▶ jump</button></td></tr>`).join(''):'<tr><td colspan="4" style="color:#8da4be">No segment above the threshold.</td></tr>';
tb.querySelectorAll('.jump').forEach(b=>b.onclick=()=>{video.currentTime=parseFloat(b.dataset.t);video.play();});
$('#strip').innerHTML=(r.frame_strip||[]).map(f=>`<img title="${f.time}s" src="${f.url}">`).join('');
$('#meta').textContent=`${r.windows.length} windows of ${r.window_seconds}s (stride ${r.stride_seconds}s) scored in ${r.elapsed_seconds}s on ${r.device}. Audio track: ${r.has_audio?'yes':'none (silence used)'}. Checkpoint epoch ${r.checkpoint_epoch}, dev AUC ${r.checkpoint_dev_auc==null?'n/a':r.checkpoint_dev_auc.toFixed(4)}. Scores are model probabilities, not guarantees.`;
start.disabled=false;}}
async function poll(run){const timer=setInterval(async()=>{const data=await fetch('/api/status/'+run).then(r=>r.json());render(data);if(data.status==='complete'||data.status==='error')clearInterval(timer);},600);}
start.onclick=async()=>{if(!selected)return;start.disabled=true;result.classList.remove('show');$('#empty').style.display='block';statusEl.className='status';statusEl.textContent='Uploading video…';bar.style.width='3%';const fd=new FormData();fd.append('video',selected);const response=await fetch('/api/upload',{method:'POST',body:fd});const data=await response.json();if(!response.ok){render({status:'error',error:data.error,progress:100});return;}poll(data.run_id);};
</script></body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

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
            self.send_json(job if job is not None else {"error": "Run not found."},
                           HTTPStatus.OK if job is not None else HTTPStatus.NOT_FOUND)
            return
        if parsed.path.startswith("/runs/"):
            relative = unquote(parsed.path[len("/runs/"):]).replace("/", os.sep)
            candidate = (RUNS_DIR / relative).resolve()
            if RUNS_DIR.resolve() not in candidate.parents or not candidate.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = candidate.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(str(candidate))[0] or "application/octet-stream")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if urlparse(self.path).path != "/api/upload":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                self.send_json({"error": "Upload must be between 1 byte and 500 MB."}, HTTPStatus.BAD_REQUEST)
                return
            fields = parse_multipart(self.rfile.read(length), self.headers.get("Content-Type", ""))
            filename, data = fields.get("video", (None, b""))
            if not filename or not data:
                self.send_json({"error": "Choose a video file first."}, HTTPStatus.BAD_REQUEST)
                return
            suffix = Path(filename).suffix.lower() or ".mp4"
            if suffix not in ALLOWED:
                self.send_json({"error": f"Unsupported video type {suffix}."}, HTTPStatus.BAD_REQUEST)
                return
            run_id = uuid.uuid4().hex
            run_dir = RUNS_DIR / run_id
            run_dir.mkdir()
            video_path = run_dir / f"upload{suffix}"
            video_path.write_bytes(data)
            with JOBS_LOCK:
                JOBS[run_id] = {"status": "running", "stage": "Upload received", "progress": 8}
            threading.Thread(target=process_video, args=(run_id, video_path, Path(filename).name), daemon=True).start()
            self.send_json({"run_id": run_id}, HTTPStatus.ACCEPTED)
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format, *args):
        print(f"[dashboard] {self.address_string()} - {format % args}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT", "8501")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--preload", action="store_true", help="load the model before serving")
    args = parser.parse_args()
    CHECKPOINT = Path(args.checkpoint)
    print(f"Checkpoint: {CHECKPOINT}")
    if args.preload:
        get_analyzer()
    print(f"Dashboard running at http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), DashboardHandler).serve_forever()
