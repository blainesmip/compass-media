"""Compass Culture TikTok / Reel cut (v3, 13 Nov 2026).
- Full moment, not a 15s snippet: pass --ss/--t to cover the whole verse/moment and end on a natural stop.
- No cover-ups: no blur bands, no inpainting. Burned-in text gets cropped out or the clip gets a cleaner source.
- Quality: one encode from the best source, Lanczos scaling, native frame rate, slow preset, CRF 16, AAC 256k.
- Branding: small Compass star top-right; small @thecompassculture + credit low down where the credit sits.
- Headline centred, but moved up or down when it would sit over a face (YuNet, sampled across the clip).
"""
import argparse, json, subprocess, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

p = argparse.ArgumentParser()
p.add_argument("video"); p.add_argument("out")
p.add_argument("--lines", nargs="+", required=True)
p.add_argument("--credit", default=None)
p.add_argument("--ss", type=float, default=0); p.add_argument("--t", type=float, default=None)
p.add_argument("--crop", default=None, help="pre-crop source w:h:x:y (in source pixels)")
p.add_argument("--font", default="RobotoCond-800i.ttf")
p.add_argument("--logo", default="compass_white.png")
p.add_argument("--face_model", default="yunet.onnx")
p.add_argument("--fs", type=int, default=69)
p.add_argument("--top", type=int, default=None, help="force cap top of first line (skips face check)")
p.add_argument("--handle_y", type=int, default=1690)
p.add_argument("--logo_h", type=int, default=96)
p.add_argument("--fade", type=float, default=0.4, help="audio fade out at the end, seconds")
p.add_argument("--overlay", default="tt_overlay.png")
p.add_argument("--box_fs", type=int, default=88)
a = p.parse_args(); W, H = 1080, 1920
if len(a.lines) > 3: raise SystemExit("max 3 headline lines")

def probe(path):
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate:format=duration",
                                   "-of", "json", path], capture_output=True, text=True, check=True).stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    return int(v["width"]), int(v["height"]), v["r_frame_rate"], float(j["format"]["duration"])

pre = f"crop={a.crop}," if a.crop else ""
vf = f"{pre}scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1"
sw, sh, fps, dur = probe(a.video)
t = a.t or (dur - a.ss)
cw, ch = (int(x) for x in a.crop.split(":")[:2]) if a.crop else (sw, sh)
if max(W / cw, H / ch) > 1.05:
    print(f"WARNING: source is being upscaled {max(W / cw, H / ch):.2f}x. Find a higher-res source before posting.")

hf = ImageFont.truetype(a.font, a.fs); cap = -hf.getbbox("H", anchor="ls")[1]; pitch = round(a.fs * 1.11)
bf = ImageFont.truetype("anton.ttf", a.box_fs); BPAD, BRH, BGAP = 24, round(a.box_fs*1.2), 10
block_h = len(a.lines) * BRH + (len(a.lines) - 1) * BGAP

def face_boxes():
    import cv2
    if not os.path.exists(a.face_model): return []
    det = cv2.FaceDetectorYN.create(a.face_model, "", (W, H), 0.6)
    boxes = []
    for k in range(10):
        ts = a.ss + t * (k + 0.5) / 10
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{ts:.2f}", "-i", a.video, "-frames:v", "1", "-vf", vf,
                              "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
        if len(raw) != W * H * 3: continue
        fr = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
        _, f = det.detect(fr)
        if f is not None:
            for x, y, w, h, *_ in f: boxes.append((x, y - 0.25 * h, x + w, y + 1.1 * h))  # pad for hair/chin
    return boxes

def overlaps(top, boxes):
    y0, y1 = top - 20, top + block_h + 20
    widest = max(bf.getlength(l) for l in a.lines) + 2 * BPAD; x0, x1 = (W - widest) / 2, (W + widest) / 2
    return sum(1 for b in boxes if b[1] < y1 and b[3] > y0 and b[0] < x1 and b[2] > x0)

if a.top is None:
    boxes = face_boxes(); centre = round(H / 2 - block_h / 2)
    # candidates: centre first, then moving away in 40px steps, kept clear of TikTok's top bar and caption area
    cands = [centre] + [centre + s * d for d in range(40, 700, 40) for s in (-1, 1)]
    cands = [c for c in cands if 300 <= c <= a.handle_y - 120 - block_h]
    a.top = min(cands, key=lambda c: (overlaps(c, boxes), abs(c - centre)))
    print(f"headline top {a.top} (centre {centre}), face hits {overlaps(a.top, boxes)} of {len(boxes)} samples")

def text_layer(items):
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ds = ImageDraw.Draw(shadow)
    txt = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dt = ImageDraw.Draw(txt)
    for s, f, y, fill in items:
        x = (W - f.getlength(s)) / 2
        ds.text((x + 2, y + 2), s, font=f, fill=(0, 0, 0, 120), anchor="ls")
        dt.text((x, y), s, font=f, fill=fill, anchor="ls")
    return shadow.filter(ImageFilter.GaussianBlur(1.5)), txt

items = []
box = Image.new("RGBA", (W, H), (0, 0, 0, 0)); db = ImageDraw.Draw(box)
MINT = (12, 247, 192)
for i, line in enumerate(a.lines):
    wd = bf.getlength(line) + 2 * BPAD
    if wd > W - 80: raise SystemExit(f"line too wide: {line}")
    y = a.top + i * (BRH + BGAP); x0 = (W - wd) / 2
    db.rectangle([x0, y, x0 + wd, y + BRH], fill=((255, 255, 255) if i % 2 == 0 else MINT) + (255,))
    db.text((W / 2, y + BRH / 2 + 2), line, font=bf, fill=(0, 0, 0, 255), anchor="mm")
items.append(("@thecompassculture", ImageFont.truetype(a.font, 30), a.handle_y, (255, 255, 255, 215)))
if a.credit:
    items.append((a.credit, ImageFont.truetype(a.font, 24), a.handle_y + 36, (240, 240, 240, 200)))
ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
ov.alpha_composite(box)
lg = Image.open(a.logo).convert("RGBA"); lw = int(lg.width * a.logo_h / lg.height)
lg = lg.resize((lw, a.logo_h), Image.LANCZOS); lg.putalpha(lg.getchannel("A").point(lambda v: int(v * 0.7)))
ov.alpha_composite(lg, (W - 60 - lw, 230))
s_, t_ = text_layer(items); ov.alpha_composite(s_); ov.alpha_composite(t_); ov.save(a.overlay)

af = f"afade=t=out:st={max(t - a.fade, 0):.2f}:d={a.fade}"
cmd = ["ffmpeg", "-y", "-ss", f"{a.ss}", "-t", f"{t:.2f}", "-i", a.video, "-i", a.overlay,
       "-filter_complex", f"[0:v]{vf}[b];[b][1:v]overlay=0:0,format=yuv420p[v];[0:a]{af}[au]",
       "-map", "[v]", "-map", "[au]", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high",
       "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", a.out]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode: print(r.stderr[-2000:]); raise SystemExit(1)
print("ok", a.out, f"{t:.1f}s from source {sw}x{sh} @ {fps}")
