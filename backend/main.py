from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
import cv2
import numpy as np
import os, uuid, json, subprocess
from pathlib import Path

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

jobs = {}

# ──────────────────────────────────────────────
# ffmpeg helpers
# ──────────────────────────────────────────────
def has_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False

def extract_audio(src: str, dst: str) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-vn", "-acodec", "copy", dst],
        capture_output=True
    )
    return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0

def mux(video: str, audio: str, out: str) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-i", audio,
         "-c:v", "libx264", "-preset", "fast", "-crf", "22",
         "-c:a", "aac", "-b:a", "192k", "-shortest", out],
        capture_output=True
    )
    return r.returncode == 0

def encode_mp4(src: str, out: str) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src,
         "-c:v", "libx264", "-preset", "fast", "-crf", "22",
         "-c:a", "aac", out],
        capture_output=True
    )
    return r.returncode == 0

# ──────────────────────────────────────────────
# Watermark detection (auto)
# ──────────────────────────────────────────────
def build_auto_mask(video_path: str, n=12):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release(); return None

    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    accum = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        thr = lap.mean() + 1.8 * lap.std()
        m = (lap > thr).astype(np.float32)
        accum = m if accum is None else accum + m

    cap.release()
    if accum is None: return None

    consistency = accum / len(indices)
    mask = (consistency > 0.55).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k)
    return mask

# ──────────────────────────────────────────────
# Manual region → mask
# ──────────────────────────────────────────────
def regions_to_mask(regions: list, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for r in regions:
        x1 = int(r["x"] * w);  y1 = int(r["y"] * h)
        x2 = int((r["x"] + r["w"]) * w)
        y2 = int((r["y"] + r["h"]) * h)
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        mask[y1:y2, x1:x2] = 255
    return mask

# ──────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────
def process_video(job_id: str, src: str, regions: list, auto: bool):
    try:
        ffmpeg = has_ffmpeg()
        jobs[job_id] = {"status": "starting", "progress": 2}

        # 1. Extract audio
        audio_path = str(UPLOAD_DIR / f"{job_id}.aac")
        got_audio = False
        if ffmpeg:
            jobs[job_id] = {"status": "audio_extract", "progress": 5}
            got_audio = extract_audio(src, audio_path)

        # 2. Open video
        cap = cv2.VideoCapture(src)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total == 0:
            jobs[job_id] = {"status": "error", "message": "Video read nahi hua"}
            return

        # 3. Build mask
        mask = None
        if regions:
            jobs[job_id] = {"status": "building_mask", "progress": 10}
            mask = regions_to_mask(regions, H, W)
        elif auto:
            jobs[job_id] = {"status": "detecting", "progress": 10}
            mask = build_auto_mask(src)

        if mask is None or mask.sum() == 0:
            jobs[job_id] = {"status": "error", "message": "Koi watermark region nahi mila. Manual select karo."}
            return

        # 4. Process frames → temp raw video (no audio)
        temp_vid = str(OUTPUT_DIR / f"{job_id}_raw.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = cv2.VideoWriter(temp_vid, fourcc, fps, (W, H))

        cap = cv2.VideoCapture(src)
        done = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.inpaint(frame, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            out_vid.write(frame)
            done += 1
            if done % 20 == 0:
                jobs[job_id] = {
                    "status": "processing",
                    "progress": int(15 + done / total * 72),
                    "frame": done, "total": total
                }
        cap.release()
        out_vid.release()

        # 5. Mux audio back
        final = str(OUTPUT_DIR / f"{job_id}_final.mp4")
        jobs[job_id] = {"status": "muxing", "progress": 90}

        if ffmpeg:
            if got_audio:
                ok = mux(temp_vid, audio_path, final)
                if not ok:
                    encode_mp4(temp_vid, final)   # fallback without audio
            else:
                encode_mp4(temp_vid, final)
        else:
            os.rename(temp_vid, final)

        # Cleanup
        for f in [temp_vid, audio_path]:
            if os.path.exists(f): os.remove(f)

        jobs[job_id] = {"status": "completed", "progress": 100}

    except Exception as e:
        jobs[job_id] = {"status": "error", "message": str(e)}

# ──────────────────────────────────────────────
# API endpoints
# ──────────────────────────────────────────────
@app.post("/upload")
async def upload(
    bg: BackgroundTasks,
    file: UploadFile = File(...),
    regions: str = Form(default="[]"),
    auto_detect: str = Form(default="true"),
):
    job_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".mp4"
    src = str(UPLOAD_DIR / f"{job_id}_src{ext}")

    with open(src, "wb") as f:
        f.write(await file.read())

    parsed = json.loads(regions)
    do_auto = auto_detect.lower() == "true" and not parsed

    jobs[job_id] = {"status": "queued", "progress": 0}
    bg.add_task(process_video, job_id, src, parsed, do_auto)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    return jobs.get(job_id, {"status": "not_found"})


@app.get("/download/{job_id}")
def download(job_id: str):
    path = OUTPUT_DIR / f"{job_id}_final.mp4"
    if not path.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return FileResponse(str(path), media_type="video/mp4",
                        filename="watermark_removed.mp4")


@app.get("/thumbnail/{job_id}")
def thumbnail(job_id: str):
    files = list(UPLOAD_DIR.glob(f"{job_id}_src*"))
    if not files:
        return JSONResponse({"error": "not found"}, status_code=404)
    cap = cv2.VideoCapture(str(files[0]))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return JSONResponse({"error": "frame error"}, status_code=500)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/")
def root():
    return {"status": "ok", "ffmpeg": has_ffmpeg()}
