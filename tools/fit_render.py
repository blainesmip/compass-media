"""Compass FIT CHECK / BRAND BREAKDOWN renderer v1 (Nov 2026). Usage: python3 -I fit_render.py spec.json
Spec: src (vertical or rotated source), wav (48k stereo of src), words (json [[start,end,word],...] of src), out,
title [row1,row2], items [[start, end, "CAPTION TOKENS"], ...] (source seconds, kept in order),
tags [[source_time, "logo.jpg", "Display Name", "@handle", verified_bool], ...]  -> Instagram-style profile tag pops up when the brand is said.
Brand tokens to colour mint: spec "mint_words". Grade B LUT, house captions (one word, centre), Anton title top, one encode."""
import json, subprocess, wave, sys, difflib, re, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SPEC = json.load(open(sys.argv[1]))
SRC = SPEC["src"]; LUT = SPEC.get("lut", "B-rich.cube"); FPS = 60000 / 1001; W, H = 1080, 1920; SR = 48000
MINT = (12, 247, 192)
MINTW = set(SPEC.get("mint_words", []))
words = SPEC.get("words_data") or json.load(open(SPEC["words"]))
VF = f"scale={W}:{H}:flags=lanczos,format=rgb48le,lut3d={LUT}:interp=tetrahedral,format=rgb24"

w = wave.open(SPEC["wav"]); A = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).astype(np.float32) / 32768
mono = A.mean(1)

def env_db(a0, a1):
    seg = mono[int(a0 * SR):int(a1 * SR)]; fr = int(0.01 * SR)
    return np.array([20 * np.log10(np.sqrt((seg[i:i + fr] ** 2).mean()) + 1e-6) for i in range(0, max(0, len(seg) - fr), fr)])

def split_item(s, e):
    """Remove inner silences > 0.40 s (keep 0.10 s either side). Returns list of (s, e)."""
    db = env_db(s, e)
    if not len(db): return [(s, e)]
    thr = max(db.max() - 30, np.percentile(db, 20) + 4)
    quiet = db < thr; out = []; cur = s; i = 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]: j += 1
            if (j - i) * 0.01 > 0.40 and i > 0 and j < len(quiet):
                out.append((cur, s + i * 0.01 + 0.10)); cur = s + j * 0.01 - 0.10
            i = j
        else: i += 1
    out.append((cur, e)); return [(a, b) for a, b in out if b - a > 0.2]

def norm(t): return re.sub(r"[^a-z0-9]", "", t.lower())

# ---- timeline ----
segs = []; t_out = 0.0
ZP = []; LIFT = []
for k, row in enumerate(SPEC["items"]):
    s, e, cap = row[:3]; ZP.append(row[3] if len(row) > 3 else k % 2); NOSPLIT = row[4] if len(row) > 4 else False; LIFT.append(row[5] if len(row) > 5 else 0)
    raw_t = cap.split(); toks = []; BRK = set()
    for t_ in raw_t:
        if t_ == "|": BRK.add(len(toks))
        else: toks.append(t_)
    ws = [x for x in words if s - 0.05 <= x[0] < e]
    # align tokens to heard words
    a = [norm(x[2]) for x in ws]; b = [norm(t) for t in toks]
    st = [None] * len(toks); en = [None] * len(toks)
    for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for q in range(blk.size): st[blk.b + q] = ws[blk.a + q][0]; en[blk.b + q] = ws[blk.a + q][1]
    known = [i for i, v in enumerate(st) if v is not None]
    if not known: st = list(np.linspace(s + 0.05, e - 0.2, len(toks)))
    else:
        for i in range(len(toks)):
            if st[i] is None:
                lo = max([j for j in known if j < i], default=None); hi = min([j for j in known if j > i], default=None)
                if lo is None: st[i] = max(s, st[hi] - 0.18 * (hi - i))
                elif hi is None: st[i] = st[lo] + 0.2 * (i - lo)
                else: st[i] = st[lo] + (st[hi] - st[lo]) * (i - lo) / (hi - lo)
    for i in range(1, len(st)): st[i] = max(st[i], st[i - 1] + 0.1)
    pieces = [(s, e)] if NOSPLIT else split_item(s, e)
    for pi, (ps, pe) in enumerate(pieces):
        n = int(round((pe - ps) * FPS)); d = n / FPS
        last = (k == len(SPEC["items"]) - 1 and pi == len(pieces) - 1)
        segs.append(dict(s=ps, e=pe, n=n, d=d, t0=t_out, item=k, last=last))
        t_out += d
    en = [min(e, y if y is not None else x + 0.3) for x, y in zip(st, en)]
    SPEC["items"][k] = dict(s=s, e=e, toks=toks, st=st, en=en, brk=BRK)
TOTAL = t_out; print(f"total {TOTAL:.2f}s, {len(segs)} segments", flush=True)

def src2out(ts):
    for g in segs:
        if g["s"] - 0.001 <= ts < g["e"]: return g["t0"] + ts - g["s"]
    nxt = [g for g in segs if g["s"] > ts]
    return nxt[0]["t0"] if nxt else None

# captions on output timeline
CAPS = []
for k, it in enumerate(SPEC["items"]):
    for i, (tok, ts) in enumerate(zip(it["toks"], it["st"])):
        to = src2out(max(ts, it["s"]))
        if to is not None: CAPS.append((to, tok))
CAPS.sort()
# group single words into short phrases (screenshot style): <=3 words / 18 chars, break on pauses or item change
item_of = []; TEND = {}
for k, it in enumerate(SPEC["items"]):
    for ii, (tok, ts) in enumerate(zip(it["toks"], it["st"])):
        to = src2out(max(ts, it["s"]))
        if to is not None: item_of.append((to, tok, k, ii in it["brk"])); TEND[(k, to)] = src2out(min(it["en"][ii], it["e"] - 0.01)) or to + 0.3
item_of.sort()
PHR = []
MAXW, MAXC = SPEC.get("cap_words", 4), SPEC.get("cap_chars", 24)
from itertools import groupby
for k, grp in groupby(item_of, key=lambda x: x[2]):
    grp = list(grp); chunks = []; cur = []; forced = any(x[3] for x in grp)
    for i, (to, tok, _, brk) in enumerate(grp):
        if forced:
            if brk and cur: chunks.append(cur); cur = []
            cur.append((to, tok)); continue
        gap = to - grp[i - 1][0] if i else 0
        glue = cur and cur[-1][1].endswith("*") and tok.endswith("*")
        if cur and not glue and (len(cur) >= MAXW or len(" ".join([c[1] for c in cur] + [tok])) > MAXC or (gap > 0.5 and len(cur) >= 2)):
            chunks.append(cur); cur = []
        cur.append((to, tok))
    if cur: chunks.append(cur)
    if forced:
        for c in chunks: PHR.append(dict(t=c[0][0], w=[x[1] for x in c], k=k, e=TEND.get((k, c[-1][0]), c[-1][0] + 0.3)))
        continue
    # no orphans: fold single-word chunks into a neighbour
    i = 0
    while i < len(chunks):
        if len(chunks) > 1 and len(chunks[i]) == 1:
            j = i - 1 if i > 0 else i + 1
            a, b = (chunks[j], chunks[i]) if j < i else (chunks[i], chunks[j])
            merged = a + b
            if len(" ".join(c[1] for c in merged)) <= MAXC + 6:
                chunks[min(i, j)] = merged; del chunks[max(i, j)]; i = max(0, i - 1); continue
        i += 1
    for c in chunks: PHR.append(dict(t=c[0][0], w=[x[1] for x in c], k=k, e=TEND.get((k, c[-1][0]), c[-1][0] + 0.3)))
item_end = {}
for g in segs: item_end[g["item"]] = g["t0"] + g["d"]
CAP_END = []
for k, it in enumerate(SPEC["items"]):
    CAP_END.append(item_end[k])

# tags on output timeline
TAGS = []
for tg_ in SPEC.get("tags", []):
    ts, logo, name, handle, ver = tg_[:5]; anc = tg_[5] if len(tg_) > 5 else None; dur = tg_[6] if len(tg_) > 6 and tg_[6] else SPEC.get("tag_dur", 2.4); side = tg_[7] if len(tg_) > 7 else None; fol = tg_[8] if len(tg_) > 8 else None; ty = tg_[9] if len(tg_) > 9 else None
    to = src2out(ts)
    if to is not None: TAGS.append(dict(t=to - 0.05, logo=logo, name=name, handle=handle, ver=ver, anchor=anc, dur=dur, side=side, follow=fol, y=ty))
TAGS.sort(key=lambda x: x["t"])
ST = lambda a, b: a.get("y") and b.get("y")  # stacked pair (fixed y): shown together
for i in range(1, len(TAGS)):
    if not ST(TAGS[i], TAGS[i - 1]): TAGS[i]["t"] = max(TAGS[i]["t"], TAGS[i - 1]["t"] + 1.3)
for i, tg in enumerate(TAGS):
    nx_ = [u["t"] for u in TAGS[i + 1:] if not ST(u, tg)]; nxt = nx_[0] if nx_ else 1e9
    tg["out"] = min(tg["t"] + tg["dur"], nxt - 0.05, TOTAL - 0.2)
for tg in TAGS:
    print("  tag", round(tg["t"], 2), round(tg["out"], 2), tg["handle"], flush=True)
    if tg["out"] - tg["t"] < 1.5: print(f"  WARNING TAG TOO SHORT TO READ: {tg['handle']} on screen {tg['out'] - tg['t']:.2f}s (needs 1.5 s+; Blaines 18 Nov)", flush=True)

fcap = ImageFont.truetype("RC800.ttf", 120); fanton = ImageFont.truetype("anton.ttf", 130)
fname = ImageFont.truetype("RC800.ttf", 46); fhandle = ImageFont.truetype("RobotoCond-600.ttf", 38) if False else ImageFont.truetype("RC800.ttf", 36)

def ease_out_back(x, c1=1.70158 * 1.2): c3 = c1 + 1; return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2

# ---- title (house style) ----
def hanger(sz, col=(0, 0, 0, 255)):
    S = sz * 4; im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im); w_ = int(S * 0.075)
    cx = S / 2; top = S * 0.12
    d.arc([cx - S * 0.11, top, cx + S * 0.11, top + S * 0.22], 180, 450, fill=col, width=w_)   # hook
    d.line([(cx, top + S * 0.22), (cx, top + S * 0.32)], fill=col, width=w_)
    d.line([(cx, top + S * 0.32), (S * 0.06, S * 0.78)], fill=col, width=w_, joint="curve")
    d.line([(cx, top + S * 0.32), (S * 0.94, S * 0.78)], fill=col, width=w_, joint="curve")
    d.line([(S * 0.06, S * 0.78), (S * 0.94, S * 0.78)], fill=col, width=w_)
    return im.resize((sz, sz), Image.LANCZOS)
def make_box_rows(rows, icon=None):
    f = fanton; pad, rh, gap = 28, 154, 12
    iw = 140 if icon else 0
    while max(f.getlength(r) + (iw if i == 0 else 0) for i, r in enumerate(rows)) + 2 * pad > 1000: f = ImageFont.truetype("anton.ttf", f.size - 4)
    widths = [f.getlength(r) + 2 * pad + (iw if i == 0 else 0) for i, r in enumerate(rows)]; wmax = max(widths)
    L = Image.new("RGBA", (int(wmax), rh * len(rows) + gap * (len(rows) - 1)), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    BGS = SPEC.get("box_bgs", [(255, 255, 255), MINT, (255, 255, 255)]); FGS = SPEC.get("box_fgs", [(0, 0, 0)] * 3)
    for i, t in enumerate(rows):
        bg = tuple(BGS[i]); fg = tuple(FGS[i])
        wd = widths[i]; y = i * (rh + gap); x0 = (wmax - wd) / 2
        d.rectangle([x0, y, x0 + wd, y + rh], fill=bg + (255,))
        if i == 0 and icon:
            d.text((x0 + pad + (wd - 2 * pad - iw) / 2, y + rh / 2 + 2), t, font=f, fill=fg + (255,), anchor="mm")
            L.alpha_composite(hanger(104, fg + (255,)), (int(x0 + wd - pad - iw + 30), int(y + (rh - 104) / 2)))
        else: d.text((x0 + wd / 2, y + rh / 2 + 2), t, font=f, fill=fg + (255,), anchor="mm")
    return L
def make_title():
    if SPEC.get("hook_icon"): return make_box_rows(SPEC["title"], SPEC["hook_icon"])
    rows = SPEC["title"]; f = fanton; pad, rh, gap = 28, 154, 12
    while max(f.getlength(r) for r in rows) + 2 * pad > 1000: f = ImageFont.truetype("anton.ttf", f.size - 4)
    wmax = max(f.getlength(r) for r in rows) + 2 * pad
    L = Image.new("RGBA", (int(wmax), rh * 2 + gap), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    for i, (t, bg) in enumerate(zip(rows, [(255, 255, 255), MINT])):
        wd = f.getlength(t) + 2 * pad; y = i * (rh + gap); x0 = (wmax - wd) / 2
        d.rectangle([x0, y, x0 + wd, y + rh], fill=bg + (255,)); d.text((wmax / 2, y + rh / 2 + 2), t, font=f, fill=(0, 0, 0, 255), anchor="mm")
    return L
TITLE = make_title()
if SPEC.get("hook_scale"): TITLE = TITLE.resize((int(TITLE.width * SPEC["hook_scale"]), int(TITLE.height * SPEC["hook_scale"])), Image.LANCZOS)
T_HOLD, T_ZIP = 0.6, 0.26; HH = TITLE.height; TOP = int(SPEC["hook_cy"] - HH / 2) if SPEC.get("hook_cy") else 215
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

# ---- captions (house style) ----
CAP_Y = SPEC.get("cap_y", 1390)
fphr = ImageFont.truetype("RC800.ttf", SPEC.get("cap_size", 92))
phr_cache = {}
def phr_layer(words_):
    key = tuple(words_)
    if key in phr_cache: return phr_cache[key]
    mset = {norm(m) for m in MINTW}
    parts = [(w_.replace("_", " ").replace("*", "").lower() if SPEC.get("lowercase", True) else w_.replace("*", ""),
              MINT if (w_.endswith("*") or norm(w_) in mset) else (255, 255, 255)) for w_ in words_]
    f = fphr; sp = f.getlength(" ")
    while sum(f.getlength(t) for t, _ in parts) + sp * (len(parts) - 1) > 1000: f = ImageFont.truetype("RC800.ttf", f.size - 4); sp = f.getlength(" ")
    tw = sum(f.getlength(t) for t, _ in parts) + sp * (len(parts) - 1)
    L = Image.new("RGBA", (W, 240), (0, 0, 0, 0)); sh = Image.new("RGBA", (W, 240), (0, 0, 0, 0))
    ds = ImageDraw.Draw(sh); d = ImageDraw.Draw(L); x = (W - tw) / 2
    for t_, c_ in parts:
        d.text((x, 120), t_, font=f, fill=c_ + (255,), anchor="lm", stroke_width=SPEC.get("cap_stroke", 0), stroke_fill=(0, 0, 0, 255)); x += f.getlength(t_) + sp
    phr_cache[key] = L; return L

cap_cache = {}
def cap_layer(tok):
    if tok in cap_cache: return cap_cache[tok]
    clean = tok.replace("_", " ").replace("*", "")
    col = MINT if tok.endswith("*") or norm(clean) in {norm(m) for m in MINTW} else (255, 255, 255)
    f = fcap
    while f.getlength(clean) > 1000: f = ImageFont.truetype("RC800.ttf", f.size - 6)
    L = Image.new("RGBA", (W, 260), (0, 0, 0, 0)); x = (W - f.getlength(clean)) / 2
    s = Image.new("RGBA", (W, 260), (0, 0, 0, 0)); ImageDraw.Draw(s).text((x + 4, 134), clean, font=f, fill=(0, 0, 0, 200), anchor="lm", stroke_width=6, stroke_fill=(0, 0, 0, 200))
    L = Image.alpha_composite(s.filter(ImageFilter.GaussianBlur(8)), L); ImageDraw.Draw(L).text((x, 130), clean, font=f, fill=col + (255,), anchor="lm")
    cap_cache[tok] = L; return L

# ---- Instagram-style profile tag ----
def verified_badge(sz):
    S = sz * 4; im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    cx = cy = S / 2; pts = []
    for i in range(24):
        r = S * (0.5 if i % 2 == 0 else 0.42); a = np.pi * 2 * i / 24
        pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    d.polygon(pts, fill=(0, 149, 246, 255))
    d.line([(S * 0.30, S * 0.52), (S * 0.45, S * 0.66), (S * 0.71, S * 0.38)], fill=(255, 255, 255, 255), width=int(S * 0.09), joint="curve")
    return im.resize((sz, sz), Image.LANCZOS)
VB = verified_badge(40)

FBTN = ImageFont.truetype("RC800.ttf", 38)
def make_tag(logo, name, handle, ver, follow=None):
    AV = 104; padl, padr, hgt = 14, 40, 132
    if follow: handle = handle + "  ·  " + follow
    nw = fname.getlength(name) + (52 if ver else 0); hw = fhandle.getlength(handle)
    BTW = int(FBTN.getlength("Follow") + 56) if follow else 0
    wd = int(padl + AV + 22 + max(nw, hw) + padr + (BTW + 10 if follow else 0))
    S = 3  # supersample
    card = Image.new("RGBA", (wd * S, hgt * S), (0, 0, 0, 0)); d = ImageDraw.Draw(card)
    d.rounded_rectangle([0, 0, wd * S - 1, hgt * S - 1], radius=hgt * S // 2, fill=(255, 255, 255, 250))
    card = card.resize((wd, hgt), Image.LANCZOS)
    av = Image.open(logo).convert("RGB"); m = min(av.size); av = av.crop(((av.width - m) // 2, (av.height - m) // 2, (av.width + m) // 2, (av.height + m) // 2)).resize((AV * S, AV * S), Image.LANCZOS)
    mask = Image.new("L", (AV * S, AV * S), 0); ImageDraw.Draw(mask).ellipse([0, 0, AV * S - 1, AV * S - 1], fill=255)
    ring = Image.new("RGBA", (AV * S, AV * S), (0, 0, 0, 0)); ImageDraw.Draw(ring).ellipse([0, 0, AV * S - 1, AV * S - 1], outline=(219, 219, 219, 255), width=2 * S)
    avr = Image.new("RGBA", (AV * S, AV * S), (0, 0, 0, 0)); avr.paste(av, (0, 0), mask); avr.alpha_composite(ring)
    avr = avr.resize((AV, AV), Image.LANCZOS)
    card.alpha_composite(avr, (padl, (hgt - AV) // 2))
    d = ImageDraw.Draw(card); tx = padl + AV + 22
    d.text((tx, 64), name, font=fname, fill=(10, 10, 10, 255), anchor="ls")
    if ver: card.alpha_composite(VB, (int(tx + fname.getlength(name) + 10), 26))
    d.text((tx, 106), handle, font=fhandle, fill=(115, 115, 115, 255), anchor="ls")
    if follow:
        bx1 = wd - 30; bx0 = bx1 - BTW; by0, by1 = (hgt - 70) // 2, (hgt + 70) // 2
        d.rounded_rectangle([bx0, by0, bx1, by1], radius=18, fill=(0, 149, 246, 255))
        d.text(((bx0 + bx1) / 2, (by0 + by1) / 2 + 1), "Follow", font=FBTN, fill=(255, 255, 255, 255), anchor="mm")
    # soft shadow
    L = Image.new("RGBA", (wd + 80, hgt + 80), (0, 0, 0, 0))
    sh = Image.new("RGBA", L.size, (0, 0, 0, 0)); ImageDraw.Draw(sh).rounded_rectangle([40, 48, 40 + wd, 48 + hgt], radius=hgt // 2, fill=(0, 0, 0, 110))
    L = Image.alpha_composite(L, sh.filter(ImageFilter.GaussianBlur(16))); L.alpha_composite(card, (40, 40)); sc_ = SPEC.get("tag_scale", 1.0); return L.resize((int(L.width * sc_), int(L.height * sc_)), Image.LANCZOS)
for tg in TAGS: tg["img"] = make_tag(tg["logo"], tg["name"], tg["handle"], tg["ver"], tg.get("follow"))
TAG_CY = 1330

def tag_state(t, tg):
    if t < tg["t"] or t > tg["out"] + 0.2: return 0, 0
    if t < tg["t"] + 0.32: x = (t - tg["t"]) / 0.32; return max(0, ease_out_back(x)), min(1, x * 3)
    if t > tg["out"]: x = (t - tg["out"]) / 0.2; return 1 - 0.25 * x, 1 - x
    return 1, 1

HOOK = SPEC.get("hook_secs")
CTA = None; CTA_T = 1e9
if SPEC.get("cta"):
    c_txt, c_src = SPEC["cta"]
    CTA = make_box_rows([c_txt])
    sc_ = SPEC.get("hook_scale", 1.0); CTA = CTA.resize((int(CTA.width * sc_), int(CTA.height * sc_)), Image.LANCZOS)
    CTA_T = src2out(c_src) or (TOTAL - 1.5)
EMPTY = Image.new("RGBA", (W, H), (0, 0, 0, 0))
HOOK_FN = None
if SPEC.get("hook_anim"):
    _src = open(SPEC.get("hook_anim_file", "hook_anims.py")).read().split("V={")[0]
    _ns = {"HOOK_WORDS": SPEC.get("hook_words", ["RATE", "THE", "FIT"]), "HOOK_ICON": SPEC.get("hook_icon_on", True), "HOLD": SPEC.get("hook_hold", 1.6), "HOOK_ROWS": SPEC.get("hook_rows"), "ROW_SCALE": SPEC.get("hook_row_scale"), "HOOK_CY": SPEC.get("hook_cy_px", 860)}; exec(_src, _ns); HOOK_FN = _ns[SPEC["hook_anim"]]
def hook_layer(t):
    if HOOK_FN is not None:
        L = HOOK_FN(t)
        if CTA is not None and t >= CTA_T: L.alpha_composite(CTA, (int((W - CTA.width) / 2), int(H / 2 - CTA.height / 2)))
        return L
    if not SPEC.get("title"): return EMPTY.copy()
    if HOOK is None: return title_layer(t).copy()
    if t >= HOOK: L = EMPTY.copy()
    else: L = title_layer(min(t, T_HOLD)).copy()  # stays put, hard cut out
    if CTA is not None and t >= CTA_T: L.alpha_composite(CTA, (int((W - CTA.width) / 2), int(H / 2 - CTA.height / 2)))
    return L

# --- tag tracking (KLT on a downscaled grey frame) ---
TRK = {}
def track(tg, t, frame):
    if tg.get("anchor") is None or frame is None: return None
    g = cv2.cvtColor(cv2.resize(frame, (270, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    st = TRK.get(id(tg))
    if st is None:
        if tg.get("side") == "face":
            det = cv2.FaceDetectorYN.create("yunet.onnx", "", (540, 960), 0.5)
            fs_ = det.detect(cv2.cvtColor(cv2.resize(frame, (540, 960)), cv2.COLOR_RGB2BGR))[1]
            if fs_ is not None:
                ax0, ay0 = tg["anchor"]
                r = min(fs_, key=lambda r: abs(r[0] * 2 + r[2] - ax0) + abs(r[1] * 2 + r[3] - ay0))
                tg["anchor"] = [float(r[0] * 2 + r[2] * 2), float(r[1] * 2 + r[3])]  # right edge of the face, face centre height
        p = np.array([[[tg["anchor"][0] / 4, tg["anchor"][1] / 4]]], np.float32)
        TRK[id(tg)] = st = dict(g=g, p=p, sm=np.array(tg["anchor"], np.float32), lost=False); return st["sm"]
    if not st["lost"]:
        p1, ok, err = cv2.calcOpticalFlowPyrLK(st["g"], g, st["p"], None, winSize=(31, 31), maxLevel=3)
        if ok is not None and ok[0][0] == 1 and err[0][0] < 25:
            st["p"] = p1; raw = p1[0][0] * 4
            raw = np.clip(raw, [60, 400], [1020, 1700]); tgt = st["sm"] * 0.9 + raw * 0.1
            step = tgt - st["sm"]; n_ = float(np.hypot(*step)); mx = SPEC.get("tag_max_px", 6.0)
            st["sm"] = st["sm"] + (step * (mx / n_) if n_ > mx else step)
        else: st["lost"] = True
    st["g"] = g; return st["sm"]


_FDET = None
def faces_in(frame):
    global _FDET
    if frame is None: return []
    if _FDET is None: _FDET = cv2.FaceDetectorYN.create("yunet.onnx", "", (540, 960), 0.6)
    sm = cv2.resize(frame, (540, 960)); _, f = _FDET.detect(cv2.cvtColor(sm, cv2.COLOR_RGB2BGR))
    if f is None: return []
    k = W / 540.0
    return [(r[0] * k, r[1] * k, r[2] * k, r[3] * k) for r in f]
def avoid_faces(tg, t, frame, cx, cy, w, h):
    """Place each tag ONCE when it first appears, then lock it there (no jumping).
    Face tags sit just under that person's face; other tags stay in the middle unless that covers a face,
    in which case they drop just under the face."""
    if tg.get("_lock") is not None: return tg["_lock"]
    faces = faces_in(frame) if frame is not None else []
    m = 30; lo_y, hi_y = h / 2 + 380, CAP_Y - 190 - h / 2
    def box(f): fx, fy, fw, fh = f; return fx - fw * 0.15, fy - fh * 0.35, fx + fw * 1.15, fy + fh * 1.1
    def hit(x, y):
        for f in faces:
            fx0, fy0, fx1, fy1 = box(f)
            if x + w / 2 > fx0 - m and x - w / 2 < fx1 + m and y + h / 2 > fy0 - m and y - h / 2 < fy1 + m: return f
        return None
    if tg.get("side") == "face" and faces:
        ax = tg["anchor"][0] if tg.get("anchor") else W / 2
        f = min(faces, key=lambda f: abs(f[0] + f[2] / 2 - ax)); fx0, fy0, fx1, fy1 = box(f)
        cx, cy = (fx0 + fx1) / 2, fy1 + m + h / 2
    else:
        f = hit(cx, cy)
        if f is not None:
            fx0, fy0, fx1, fy1 = box(f); cy = fy1 + m + h / 2
    cx = min(max(cx, w / 2 + 10), W - w / 2 - 10); cy = min(max(cy, lo_y), hi_y)
    tg["_lock"] = (cx, cy); return cx, cy


_LG = Image.open("compass_white.png").convert("RGBA"); _LW = int(_LG.width * 96 / _LG.height)
_LG = _LG.resize((_LW, 96), Image.LANCZOS); _LG.putalpha(_LG.getchannel("A").point(lambda v: int(v * 0.7)))

def overlay(t, frame=None):
    ov = hook_layer(t)
    for tg in TAGS:
        sc, al = tag_state(t, tg)
        if t >= tg["t"] - 0.01 and t <= tg["out"] + 0.2: pos = track(tg, t, frame)
        else: pos = None
        if sc <= 0.02 or al <= 0: continue
        im = tg["img"]
        if abs(sc - 1) > 0.002: im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.LANCZOS)
        if al < 1: im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * al)))
        if pos is None:
            cx, cy = W / 2, SPEC.get("tag_default_y", 1150)
        else:
            ax, ay = float(pos[0]), float(pos[1])
            # dot on the item, pill just above it (or below if the item sits low)
            off = im.height / 2 + 34
            cy = ay - off if ay > 700 and ay < 1250 else (ay + off if ay >= 1250 else ay + off)
            cx = ax
            if tg.get("side") == "face": cx, cy = ax + im.width / 2 + 24, ay
            elif tg.get("side") == "right": cx, cy = ax + im.width / 2 + 70, ay
            elif tg.get("side") == "left": cx, cy = ax - im.width / 2 - 70, ay
        cx = min(max(cx, im.width / 2 + 10), W - im.width / 2 - 10); cy = min(max(cy, im.height / 2 + 380), CAP_Y - 190 - im.height / 2)
        if tg.get("y"): cx, cy = W / 2, tg["y"]
        else: cx, cy = avoid_faces(tg, t, frame, cx, cy, im.width, im.height)
        ov.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))
    ph = [p_ for p_ in PHR if p_["t"] <= t]
    if ph:
        p_ = ph[-1]; it_end = item_end[p_["k"]]
        nx = [q["t"] for q in PHR if q["t"] > p_["t"]]; nx = nx[0] if nx else 1e9
        hold = min(it_end, nx if nx - p_["e"] < 0.5 else p_["e"] + 0.3)
        if t < hold: ov.alpha_composite(phr_layer(p_["w"]), (0, CAP_Y - 120))
    ov.alpha_composite(_LG, (W - 60 - _LW, 230))
    return np.asarray(ov)

# ---- audio ----
parts = []
for g in segs:
    a = A[int(g["s"] * SR): int(g["s"] * SR) + int(round(g["d"] * SR))].copy()
    f = int(0.008 * SR); r = np.linspace(0, 1, f)[:, None]; a[:f] *= r; a[-f:] *= r[::-1]
    if g["last"]: q = int(0.12 * SR); a[-q:] *= np.linspace(1, 0, q)[:, None]
    parts.append(a)
aud = np.concatenate(parts)
def plop(f0, f1, amp, dur=0.06):
    tt = np.arange(int(dur * SR)) / SR; fr = f0 * (f1 / f0) ** (tt / dur); ph = 2 * np.pi * np.cumsum(fr) / SR
    env = np.minimum(1, tt / 0.0015) * np.exp(-tt / 0.035); return (np.sin(ph) * env * amp).astype(np.float32)
pin = plop(620, 170, 0.14)
for tg in TAGS:
    i = int((tg["t"] + 0.02) * SR); j = min(len(aud), i + len(pin))
    if i < len(aud): aud[i:j] += pin[:j - i, None]
TMPR, TMPA = SPEC["out"] + ".raw.wav", SPEC["out"] + ".aud.wav"
wo = wave.open(TMPR, "wb"); wo.setnchannels(2); wo.setsampwidth(2); wo.setframerate(SR)
wo.writeframes((np.clip(aud, -1, 1) * 32767).astype(np.int16).tobytes()); wo.close()
af = "highpass=f=80,acompressor=threshold=-22dB:ratio=3:attack=5:release=90:makeup=3dB"
m = subprocess.run(["ffmpeg", "-i", TMPR, "-af", af + ",loudnorm=I=-10:TP=-1.0:LRA=7:print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
j = json.loads(m[m.rindex("{"):m.rindex("}") + 1])
ln = f"loudnorm=I=-10:TP=-1.0:LRA=7:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true"
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", TMPR, "-af", af + "," + ln + ",alimiter=limit=0.89:level=false", "-ar", "48000", TMPA], check=True)

PREVIEW = bool(SPEC.get("preview")); RW, RH = (540, 960) if PREVIEW else (W, H)
if PREVIEW:
    _w = Image.new("RGBA", (300, 64), (0, 0, 0, 0)); _d = ImageDraw.Draw(_w)
    _d.rounded_rectangle([12, 12, 288, 56], radius=10, fill=(0, 0, 0, 160)); _d.text((150, 34), "PREVIEW · LOW RES", font=ImageFont.truetype("RC800.ttf", 24), fill=(255, 255, 255, 255), anchor="mm")
    WM = np.asarray(_w)
# ---- b-roll: their reels full-screen, or product photos as pop-in cards over a blurred shot ----
BROLL = []
for b in SPEC.get("broll", []):
    t0_, t1_ = src2out(b["s"]), src2out(b["e"] - 0.001)
    if t0_ is None or t1_ is None: continue
    d = dict(b); d["t0"], d["t1"] = t0_ - 0.6 / FPS, t1_ + 0.001  # start on the cut frame itself (no 1-frame flash of the new camera shot)
    if d["type"] == "cards": d["ims"] = [Image.open(f).convert("RGB") for f in d["files"]]
    BROLL.append(d); print("  broll", d["type"], round(t0_, 2), round(t1_, 2), d.get("file", d.get("files")), flush=True)
_bdec = {}
def _video_frame(b, t):
    key = id(b)
    if key not in _bdec:
        vf = f"scale={RW}:{RH}:force_original_aspect_ratio=increase:flags=lanczos,crop={RW}:{RH},fps=60000/1001,format=rgb24"
        _bdec[key] = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{b.get('in', 0):.3f}", "-i", b["file"], "-an", "-vf", vf, "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
        b["_last"] = None
    buf = _bdec[key].stdout.read(RW * RH * 3)
    if len(buf) == RW * RH * 3: b["_last"] = np.frombuffer(buf, np.uint8).reshape(RH, RW, 3)
    return b["_last"]
def ease_out_back2(x, c1=1.70158): c3 = c1 + 1; return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def _card(im, wmax, hmax):
    sc = min(wmax / im.width, hmax / im.height); r = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS).convert("RGBA")
    m = Image.new("L", r.size, 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, r.width - 1, r.height - 1], radius=int(28 * RW / W), fill=255); r.putalpha(m)
    return r
def _cards_frame(b, t, cam):
    base = cv2.GaussianBlur(cv2.resize(cam, (RW // 4, RH // 4)), (0, 0), 6); base = cv2.resize(base, (RW, RH)).astype(np.float32) * 0.5
    canvas = Image.fromarray(base.astype(np.uint8)).convert("RGBA"); n = len(b["ims"])
    if "_cards" not in b: b["_cards"] = [_card(im, RW * (0.62 if n > 1 else 0.8), RH * 0.5) for im in b["ims"]]
    for i, cd in enumerate(b["_cards"]):
        tin = b["t0"] + 0.12 + i * 0.35; x = (t - tin) / 0.32
        if x <= 0: continue
        sc = max(0.01, ease_out_back2(min(1, x))); ang = ((-6, 5, -3)[i % 3]) if n > 1 else 0
        im = cd.resize((max(1, int(cd.width * sc)), max(1, int(cd.height * sc))), Image.BICUBIC).rotate(ang, resample=Image.BICUBIC, expand=True)
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0)); sh.putalpha(im.getchannel("A").point(lambda v: v * 120 // 255)); sh = sh.filter(ImageFilter.GaussianBlur(10))
        cx = RW / 2 + ((i - (n - 1) / 2) * RW * 0.16 if n > 1 else 0); cy = RH * 0.45 + ((i - (n - 1) / 2) * RH * 0.05 if n > 1 else 0)
        canvas.alpha_composite(sh, (int(cx - im.width / 2 + 6), int(cy - im.height / 2 + 10))); canvas.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))
    return np.asarray(canvas.convert("RGB"))
def broll_frame(t, cam):
    for b in BROLL:
        if b["t0"] <= t < b["t1"]:
            return _video_frame(b, t) if b["type"] == "video" else _cards_frame(b, t, cam)
    return None

def VFG(g):
    v = VF.replace(f"scale={W}:{H}", f"scale={RW}:{RH}")
    st = LIFT[g["item"]]
    if st:  # exposure lift in S-Log3 space before the grade: ~0.077 of full scale per stop
        o = int(st * 0.077 * 65535); v = v.replace("format=rgb48le,", f"format=rgb48le,lutrgb=r=val+{o}:g=val+{o}:b=val+{o},", 1)
    return v

FX = SPEC.get("fx")
if FX:
    yy, xx = np.mgrid[0:RH, 0:RW]; dd = np.sqrt(((xx - RW / 2) / (RW / 2)) ** 2 + ((yy - RH * 0.45) / (RH / 2)) ** 2)
    VIG = (1 - FX.get("vignette", 0.42) * np.clip(dd - 0.35, 0, 1) ** 1.6).astype(np.float32)[..., None]; del yy, xx, dd
    RNG = np.random.default_rng(7)
def post_fx(img):
    x = img.astype(np.float32) / 255
    sm = cv2.resize(x, (RW // 4, RH // 4), interpolation=cv2.INTER_AREA)
    L = (sm * [0.2126, 0.7152, 0.0722]).sum(-1, keepdims=True); thr = FX.get("bloom_thr", 0.7)
    m = np.clip((L - thr) / (1 - thr), 0, 1) * sm
    bloom = cv2.resize(cv2.GaussianBlur(m, (0, 0), 30 / 4 * RW / W), (RW, RH), interpolation=cv2.INTER_LINEAR)
    x = x + bloom * FX.get("bloom", 0.22)
    x = x * VIG
    n = RNG.normal(0, 1, (RH // 2, RW // 2)).astype(np.float32); n = cv2.resize(n, (RW, RH))[..., None]
    Lx = (x * [0.2126, 0.7152, 0.0722]).sum(-1, keepdims=True)
    x = x + n * FX.get("grain", 0.035) * (0.4 + 0.6 * (1 - np.abs(Lx - 0.5) * 2))
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)

# ---- video: handheld footage, gentle alternating push so jump cuts read as intentional ----
enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{RW}x{RH}", "-r", "60000/1001", "-i", "-",
                        "-i", TMPA, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast" if PREVIEW else SPEC.get("preset", "slow"), "-crf", "22" if PREVIEW else "16", "-profile:v", "high",
                        "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                        "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", SPEC["out"]], stdin=subprocess.PIPE)
fi = 0
for gi, g in enumerate(segs):
    z0, z1 = ((1.00, 1.03) if ZP[g["item"]] == 0 else (1.08, 1.11))
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{g['s']:.3f}", "-i", SRC, "-frames:v", str(g["n"]), "-vf", VFG(g), "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    for k in range(g["n"]):
        buf = dec.stdout.read(RW * RH * 3)
        if len(buf) < RW * RH * 3: break
        fr = np.frombuffer(buf, np.uint8).reshape(RH, RW, 3)
        x = k / max(1, g["n"] - 1); x = x * x * (3 - 2 * x); z = z0 + (z1 - z0) * x
        x0 = (RW - RW / z) / 2; y0 = (RH - RH / z) * 0.40
        img = cv2.warpAffine(fr, np.array([[z, 0, -x0 * z], [0, z, -y0 * z]], np.float32), (RW, RH), flags=cv2.INTER_LANCZOS4) if z > 1.0005 else fr
        bl = cv2.GaussianBlur(img, (0, 0), 1.0); img = cv2.addWeighted(img, 1.35, bl, -0.35, 0)
        bfr = broll_frame(fi / FPS, img) if BROLL else None
        if bfr is not None: img = bfr.copy()
        if FX: img = post_fx(img)
        ov = overlay(fi / FPS, img)
        if PREVIEW: ov = cv2.resize(ov, (RW, RH), interpolation=cv2.INTER_AREA); ov[:WM.shape[0], :WM.shape[1]] = np.maximum(ov[:WM.shape[0], :WM.shape[1]], WM) if False else ov[:WM.shape[0], :WM.shape[1]]
        if PREVIEW:
            a2 = WM[..., 3:4].astype(np.float32) / 255; reg = ov[:WM.shape[0], :WM.shape[1]].astype(np.float32)
            reg[..., :3] = reg[..., :3] * (1 - a2) + WM[..., :3] * a2; reg[..., 3:4] = np.maximum(reg[..., 3:4], WM[..., 3:4]); ov[:WM.shape[0], :WM.shape[1]] = reg.astype(np.uint8)
        a = ov[..., 3:4].astype(np.float32) / 255
        enc.stdin.write((img * (1 - a) + ov[..., :3] * a).astype(np.uint8).tobytes()); fi += 1
    dec.stdout.close(); dec.wait()
enc.stdin.close(); enc.wait(); print("ok", SPEC["out"], fi, flush=True)
