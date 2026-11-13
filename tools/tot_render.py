"""Compass This-or-That v2 renderer (house style locked 13 Nov 2026). Usage: python3 -I render2.py spec.json
Spec: src, words (whisper json of src), out, title [row1, row2], pairs [[a,b],...] per question,
items [[start, end, role(Q/A/P), "CAPTION TOKENS"], ...]; mark the winning token with * (loser drops there).
Grade B-rich LUT, two-shot on questions, punch-in on the guest for answers, mint artist names,
zoom-out Anton title, bracket icons, winner slides to centre at the end. One encode."""
import json, subprocess, wave, sys, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SPEC = json.load(open(sys.argv[1]))
SRC = SPEC["src"]; LUT = "B-rich.cube"; FPS = 60000 / 1001; W, H = 1080, 1920
MINT = (12, 247, 192)
NAMEW = {"J._COLE", "DRAKE", "KENDRICK", "KANYE", "WEST", "TRAVIS", "SCOTT", "PLAYBOI", "CARTI", "FRANK", "OCEAN", "WEEKND", "TYLER",
         "CREATOR", "BRENT", "FAIYAZ", "BRYSON", "TILLER", "TORY", "LANEZ", "DON", "TOLIVER", "YOUNG", "THUG", "STEVE", "LACEY"}
words = json.load(open(SPEC["words"]))
det = cv2.FaceDetectorYN.create("yunet.onnx", "", (540, 960), 0.55)

def grab(t):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", SRC, "-frames:v", "1", "-vf",
                          f"format=rgb48le,lut3d={LUT}:interp=tetrahedral,format=rgb24", "-f", "rawvideo", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3)

def people(t):
    fr = grab(t); small = cv2.resize(fr, (540, 960))
    _, f = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if f is None: return None, []
    out = []
    for r in f:
        x, y, w, h = [v * 2 for v in r[:4]]
        hat = fr[int(max(0, y - 0.9 * h)):int(max(1, y - 0.05 * h)), int(max(0, x)):int(min(W, x + w))].astype(np.float32)
        red = float((hat[..., 0] - (hat[..., 1] + hat[..., 2]) / 2).mean()) if hat.size else 0
        out.append(dict(cx=x + w / 2, cy=y + h / 2, w=w, red=red))
    if len(out) == 1: return (out[0], []) if out[0]["red"] > 35 else (None, out)
    cino = max(out, key=lambda o: o["red"]); others = [o for o in out if o is not cino]
    guest = min(others, key=lambda o: abs(o["cx"] - cino["cx"]))
    return cino, [guest]

# ---- audio (for trimming) ----
w = wave.open(SPEC["wav"]); A = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).astype(np.float32) / 32768
SR = 48000; mono = A.mean(1)
def trim(s, e, lo, hi):
    a0, a1 = max(lo, s - 0.12), min(hi, e + 0.12)
    seg = mono[int(a0 * SR):int(a1 * SR)]; fr = int(0.01 * SR)
    db = np.array([20 * np.log10(np.sqrt((seg[i:i + fr] ** 2).mean()) + 1e-6) for i in range(0, len(seg) - fr, fr)])
    on = np.where(db > db.max() - 26)[0]
    if not len(on): return s, e
    return max(a0, a0 + on[0] * 0.01 - 0.05), min(a1, a0 + (on[-1] + 1) * 0.01 + 0.10)

# ---- timeline ----
raw_items = SPEC["items"]; timeline = []; t_out = 0.0; last_view = None
for k, (s, e, role, cap) in enumerate(raw_items):
    lo = raw_items[k - 1][1] if k else 0; hi = raw_items[k + 1][0] if k + 1 < len(raw_items) else e + 0.3
    s, e = trim(s, e, max(lo, s - 0.12), min(hi, e + 0.12))
    n = int(round((e - s) * FPS)); d = n / FPS
    cino, guests = people(s + d / 2)
    if role == "A" and guests: view = (guests[0]["cx"], guests[0]["cy"])
    elif role == "A" and not guests and last_view: view = last_view
    elif cino is not None and guests: view = ((cino["cx"] + guests[0]["cx"]) / 2, (cino["cy"] + guests[0]["cy"]) / 2)
    elif guests: view = (guests[0]["cx"], guests[0]["cy"])
    else: view = last_view or (W / 2, 820)
    if role == "A": last_view = view
    z0, z1 = (1.15, 1.18) if role != "A" else ((1.40, 1.56) if d > 1.6 else (1.45, 1.50))
    toks = cap.split(); win = [i for i, t in enumerate(toks) if t.endswith("*") or t.endswith("*?")]
    toks = [t.replace("*", "") for t in toks]
    ws = [x for x in words if s - 0.05 <= x[0] < e]
    if len(ws) >= len(toks): starts = [ws[int(i * len(ws) / len(toks))][0] for i in range(len(toks))]
    else:
        sp0 = ws[0][0] if ws else s + 0.05; sp1 = (ws[-1][1] if ws else e) - 0.1
        starts = list(np.linspace(max(s, sp0), max(s, sp1), len(toks)))
    starts = [max(s, x) - s + t_out for x in starts]; starts[0] = t_out
    timeline.append(dict(s=s, n=n, d=d, t0=t_out, role=role, cx=view[0], cy=view[1], z0=z0, z1=z1, toks=toks, starts=starts,
                         win=starts[win[0]] if win else None))
    t_out += d
TOTAL = t_out; print(f"total {TOTAL:.2f}s", flush=True)

# ---- bracket ----
q_start = []; prev = None
for it in timeline:
    if it["role"] == "Q" and prev != "Q": q_start.append(it["t0"])
    prev = it["role"]
q_win = [None] * len(q_start); qi = -1; prev = None
for it in timeline:
    if it["role"] == "Q" and prev != "Q": qi += 1
    prev = it["role"]
    if it["role"] == "A" and it["win"] is not None and qi >= 0 and q_win[qi] is None: q_win[qi] = it["win"]
PAIRS = SPEC["pairs"]; FINAL = SPEC["final_winner"]
entries = [dict(n=PAIRS[0][0], sl="L", tin=0.0, tout=None), dict(n=PAIRS[0][1], sl="R", tin=0.08, tout=None)]
def alive(e, t): return e["tout"] is None or e["tout"] > t
for q, (a, b) in enumerate(PAIRS):
    if q > 0:
        tq = q_start[q]
        for n in (a, b):
            if any(e["n"] == n and alive(e, tq) for e in entries): continue
            used = {e["sl"] for e in entries if alive(e, tq) and e["tin"] <= tq + 0.001}
            free = [sl for sl in "LR" if sl not in used]
            if not free:
                for e in entries:
                    if alive(e, tq) and e["n"] not in (a, b): e["tout"] = tq
                used = {e["sl"] for e in entries if alive(e, tq)}; free = [sl for sl in "LR" if sl not in used]
            entries.append(dict(n=n, sl=free[0], tin=tq, tout=None))
    nxt = PAIRS[q + 1] if q + 1 < len(PAIRS) else None
    if nxt: loser = a if a not in nxt else (b if b not in nxt else None)
    else: loser = b if a == FINAL else a
    if loser is None: continue
    drop = q_win[q] if q_win[q] is not None else (q_start[q + 1] - 0.05 if nxt else TOTAL - 0.5)
    for e in entries:
        if e["n"] == loser and e["tout"] is None: e["tout"] = drop
FINAL_OUT = max(e["tout"] for e in entries if e["tout"] is not None)
for e in entries: print("  icon", e, flush=True)

def load_icon(name):
    im = Image.open(f"icons/{name}.png").convert("RGBA"); im = im.crop(im.getbbox())
    s = min(460 / im.width, 620 / im.height); return im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
ICON = {e["n"]: load_icon(e["n"]) for e in entries}
fcap = ImageFont.truetype("RC800.ttf", 120); fanton = ImageFont.truetype("anton.ttf", 130)

def ease_out_back(x, c1=1.70158 * 1.2): c3 = c1 + 1; return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def pop_scale(t, t_in, t_out):
    if t < t_in: return 0
    if t_out is not None and t >= t_out: x = (t - t_out) / 0.18; return 0 if x >= 1 else (1 - x) ** 2
    x = min(1, (t - t_in) / 0.30); return max(0, ease_out_back(x))

def make_title():
    rows = SPEC["title"]
    f = fanton; pad, rh, gap = 28, 154, 12
    while max(f.getlength(r) for r in rows) + 2 * pad > 1000: f = ImageFont.truetype("anton.ttf", f.size - 4)
    wmax = max(f.getlength(r) for r in rows) + 2 * pad
    L = Image.new("RGBA", (int(wmax), rh * 2 + gap), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    for i, (t, bg) in enumerate(zip(rows, [(255, 255, 255), MINT])):
        wd = f.getlength(t) + 2 * pad; y = i * (rh + gap); x0 = (wmax - wd) / 2
        d.rectangle([x0, y, x0 + wd, y + rh], fill=bg + (255,)); d.text((wmax / 2, y + rh / 2 + 2), t, font=f, fill=(0, 0, 0, 255), anchor="mm")
    return L
TITLE = make_title(); T_HOLD, T_ZIP = 0.6, 0.26; TOP = 215; HH = TITLE.height
BIG = (1.0, TOP + HH / 2); SMALL = (0.72, TOP + HH * 0.72 / 2)
def title_state(t):
    if t <= T_HOLD: return BIG
    x = min(1, (t - T_HOLD) / T_ZIP); x = 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
    return (BIG[0] + (SMALL[0] - BIG[0]) * x, BIG[1] + (SMALL[1] - BIG[1]) * x)
title_cache = {}
def title_layer(t):
    sc, cy = title_state(t); key = (round(sc, 3), round(cy))
    if key not in title_cache:
        im = TITLE.resize((int(TITLE.width * sc), int(TITLE.height * sc)), Image.LANCZOS)
        L = Image.new("RGBA", (W, H), (0, 0, 0, 0)); sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        pos = (int((W - im.width) / 2), int(cy - im.height / 2))
        blk = Image.new("RGBA", im.size, (0, 0, 0, 120)); blk.putalpha(im.getchannel("A").point(lambda v: v * 120 // 255))
        sh.alpha_composite(blk, (pos[0] + 6, pos[1] + 8)); L = Image.alpha_composite(L, sh.filter(ImageFilter.GaussianBlur(10)))
        L.alpha_composite(im, pos); title_cache[key] = L
    return title_cache[key]

cap_cache = {}
def is_name(toks, i):
    t = toks[i].rstrip("?!.,")
    if t == "THE": return i + 1 < len(toks) and toks[i + 1].rstrip("?!.,") in ("WEEKND", "CREATOR")
    return t in NAMEW
def cap_layer(it, i):
    key = (id(it), i)
    if key in cap_cache: return cap_cache[key]
    tok = it["toks"][i]; word = tok.replace("_", " "); col = MINT if is_name(it["toks"], i) else (255, 255, 255)
    f = fcap
    while f.getlength(word) > 1000: f = ImageFont.truetype("RC800.ttf", f.size - 6)
    L = Image.new("RGBA", (W, 260), (0, 0, 0, 0)); x = (W - f.getlength(word)) / 2
    s = Image.new("RGBA", (W, 260), (0, 0, 0, 0)); ImageDraw.Draw(s).text((x + 4, 134), word, font=f, fill=(0, 0, 0, 200), anchor="lm", stroke_width=6, stroke_fill=(0, 0, 0, 200))
    L = Image.alpha_composite(s.filter(ImageFilter.GaussianBlur(8)), L); ImageDraw.Draw(L).text((x, 130), word, font=f, fill=col + (255,), anchor="lm")
    cap_cache[key] = L; return L

SLOTX = {"L": 290, "R": 790}
def overlay(t):
    ov = title_layer(t).copy()
    def put(icon, cx, sc):
        if sc <= 0.01: return
        im = icon.resize((max(1, int(icon.width * sc)), max(1, int(icon.height * sc))), Image.LANCZOS)
        top = 1020 + 64 + (icon.height - im.height) // 2
        ov.alpha_composite(im, (int(cx - im.width / 2), int(top)))
    for e in entries:
        n, t_in, t_o = e["n"], e["tin"], e["tout"]
        if t < t_in - 0.01 or (t_o is not None and t > t_o + 0.2): continue
        sc = pop_scale(t, t_in, t_o); cx = SLOTX[e["sl"]]
        if n == FINAL and t_o is None and t > FINAL_OUT:
            x = min(1, (t - FINAL_OUT - 0.1) / 0.35); x = 0 if x < 0 else 1 - (1 - x) ** 3
            cx = cx + (540 - cx) * x; sc *= 1 + 0.12 * x
        put(ICON[n], cx, sc)
    for it in timeline:
        if it["t0"] <= t < it["t0"] + it["d"]:
            ks = [i for i, st in enumerate(it["starts"]) if st <= t]
            if ks: ov.alpha_composite(cap_layer(it, ks[-1]), (0, 960 - 130))
    return np.asarray(ov)

# ---- audio ----
parts = []
for k, it in enumerate(timeline):
    a = A[int(it["s"] * SR): int(it["s"] * SR) + int(round(it["d"] * SR))].copy()
    f = int(0.008 * SR); r = np.linspace(0, 1, f)[:, None]; a[:f] *= r; a[-f:] *= r[::-1]
    if k == len(timeline) - 1: g = int(0.12 * SR); a[-g:] *= np.linspace(1, 0, g)[:, None]
    parts.append(a)
aud = np.concatenate(parts)
def plop(f0, f1, amp, dur=0.06):
    tt = np.arange(int(dur * SR)) / SR; fr = f0 * (f1 / f0) ** (tt / dur); ph = 2 * np.pi * np.cumsum(fr) / SR
    env = np.minimum(1, tt / 0.0015) * np.exp(-tt / 0.035); return (np.sin(ph) * env * amp).astype(np.float32)
pin, pout = plop(620, 170, 0.16), plop(420, 140, 0.11)
def add(p, t):
    i = int(t * SR); j = min(len(aud), i + len(p))
    if i < len(aud): aud[i:j] += p[:j - i, None]
for e in entries:
    add(pin, e["tin"] + 0.02)
    if e["tout"] is not None: add(pout, e["tout"])
wo = wave.open("tmp_raw.wav", "wb"); wo.setnchannels(2); wo.setsampwidth(2); wo.setframerate(SR)
wo.writeframes((np.clip(aud, -1, 1) * 32767).astype(np.int16).tobytes()); wo.close()
af = "highpass=f=80,acompressor=threshold=-22dB:ratio=3:attack=5:release=90:makeup=3dB"
m = subprocess.run(["ffmpeg", "-i", "tmp_raw.wav", "-af", af + ",loudnorm=I=-10:TP=-1.0:LRA=7:print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
j = json.loads(m[m.rindex("{"):m.rindex("}") + 1])
ln = f"loudnorm=I=-10:TP=-1.0:LRA=7:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true"
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", "tmp_raw.wav", "-af", af + "," + ln + ",alimiter=limit=0.89:level=false", "-ar", "48000", "tmp_audio.wav"], check=True)

# ---- video ----
enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "60000/1001", "-i", "-",
                        "-i", "tmp_audio.wav", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high",
                        "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                        "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", SPEC["out"]], stdin=subprocess.PIPE)
fi = 0
for it in timeline:
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{it['s']:.3f}", "-i", SRC, "-frames:v", str(it["n"]), "-vf",
                            f"format=rgb48le,lut3d={LUT}:interp=tetrahedral,format=rgb24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    for k in range(it["n"]):
        buf = dec.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        x = k / max(1, it["n"] - 1); x = x * x * (3 - 2 * x)
        z = it["z0"] + (it["z1"] - it["z0"]) * x; cw, ch = W / z, H / z
        fy = 0.36 if it["role"] == "A" else 0.40
        x0 = float(np.clip(it["cx"] - cw / 2, 0, W - cw)); y0 = float(np.clip(it["cy"] - fy * ch, 0, H - ch))
        img = cv2.warpAffine(fr, np.array([[z, 0, -x0 * z], [0, z, -y0 * z]], np.float32), (W, H), flags=cv2.INTER_LANCZOS4)
        bl = cv2.GaussianBlur(img, (0, 0), 1.0); img = cv2.addWeighted(img, 1.35, bl, -0.35, 0)
        ov = overlay(fi / FPS); a = ov[..., 3:4].astype(np.float32) / 255
        enc.stdin.write((img * (1 - a) + ov[..., :3] * a).astype(np.uint8).tobytes()); fi += 1
    dec.stdout.close(); dec.wait()
enc.stdin.close(); enc.wait(); print("ok", SPEC["out"], fi, flush=True)
