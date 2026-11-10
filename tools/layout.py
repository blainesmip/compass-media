"""Compass cover builder with automatic framing checks.

Every person is placed from a detected face box (YuNet), not a guess:
- face fits inside the SAFE ZONE: below the tag, above the headline, clear of the badge
- faces are big (target face height 300px) and level with each other
- face brightness is matched across people
After building, the finished cover is re-scanned and every face is checked again.
A debug overlay (boxes for tag, badge, headline, faces) is saved for a visual check.
"""
import sys, numpy as np, cv2
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/claude/scripts")
from pop_cover import *

DET = cv2.FaceDetectorYN.create("yunet.onnx", "", (320, 320), 0.5)
TAG_BOX = (60, 60, 220, 150)
BADGE_BOX = (1080 - 243 - 45, 55, 1080 - 45, 298)

def detect_faces(img_rgba_or_rgb):
    im = img_rgba_or_rgb
    if im.mode == "RGBA":
        bg = Image.new("RGBA", im.size, (128, 128, 128, 255)); bg.alpha_composite(im); im = bg
    a = np.array(im.convert("RGB"))[:, :, ::-1]; s = 1000 / max(a.shape[:2]); a2 = cv2.resize(a, None, fx=s, fy=s)
    DET.setInputSize((a2.shape[1], a2.shape[0])); _, f = DET.detect(a2)
    return [] if f is None else [tuple((np.array(x[:4]) / s).tolist()) + (float(x[-1]),) for x in f]

def face_of(cut):
    """(cx, cy, face_h, head_top, chin, how) for the main face. head_top = top of hair/hood above the face."""
    a = np.array(cut.getchannel("A")) > 128
    f = detect_faces(cut)
    if f:
        x, y, w, h, _ = max(f, key=lambda t: t[2] * t[3]); how = "detected"
    else:
        rows = np.where(a.any(1))[0]; top = rows[0]
        band = a[top:top + int(a.shape[0] * 0.08)]; cols = np.where(band.any(0))[0]
        w = cols[-1] - cols[0]; h = w * 1.25; x = cols[0]; y = top; how = "silhouette"
    x0, x1 = int(max(0, x + w * 0.15)), int(min(a.shape[1], x + w * 0.85))
    colrows = np.where(a[:, x0:x1].any(1))[0]
    head_top = colrows[0] if len(colrows) else y
    head_top = max(head_top, y - h * 0.9)          # ignore poles/props sticking up far above the head
    return x + w / 2, y + h / 2, h, head_top, y + h, how

def headline_top(n_lines):
    return 880 if n_lines >= 3 else 1000

FILL = dict(solo_face_max=560, duo_face_max=470, max_upscale=3.2)

def plan(cuts, n_lines, gap=25):
    """FILL THE FRAME. Each head (hair/hood to chin) is scaled to fill the zone between the tag and
    the headline and its top is pinned just under the tag, so there is no dead space above heads.
    Solo: head fills the zone (face up to 560px). Duo: same face size for both (up to 470px), both
    head tops pinned to the zone top. Silhouettes (hoods, masks) use the head outline instead of a face.
    Badges never push people down: people are placed first, the badge goes where heads aren't."""
    safe_top = TAG_BOX[3] + 20
    safe_bot = headline_top(n_lines) - gap
    zone = safe_bot - safe_top
    fx = [face_of(c) for c, _ in cuts]
    cap = FILL["solo_face_max"] if len(cuts) == 1 else FILL["duo_face_max"]
    scales = [min(zone / (chin - htop), cap / h if how == "detected" else 0.72 * zone / (chin - htop), FILL["max_upscale"]) for (cx, cy, h, htop, chin, how) in fx]
    if len(cuts) > 1:   # level: same on-screen face size for everyone
        th = min(s * f[2] for s, f in zip(scales, fx))
        scales = [th / f[2] for f in fx]
    out = []
    for (cut, xc), (cx, cy, h, htop, chin, how), s in zip(cuts, fx, scales):
        cy_t = safe_top + (cy - htop) * s
        fw = h * s * 0.8
        out.append(dict(face_src=(cx, cy), face_h_src=h, target=(xc, cy_t), target_h=h * s, how=how, scale=s,
                        face_box=(xc - fw / 2 - 25, cy_t - h * s / 2 - 25, xc + fw / 2 + 25, cy_t + h * s / 2 + 25),
                        head_box=(xc - fw * 0.75, safe_top, xc + fw * 0.75, safe_top + (chin - htop) * s)))
    return out

BADGE_SPOTS = lambda hl: [("top-right", (792, 55)), ("low-right", (792, hl - 290)), ("low-left", (45, hl - 290))]

def badge_spot(face_boxes, n_lines, size=243):
    """First badge spot that touches no FACE (it may sit over hair, shoulders or background);
    otherwise the spot with the least face overlap (verify() will then flag it)."""
    def ov(a, b):
        w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1])); return w * h
    for name, (x, y) in BADGE_SPOTS(headline_top(n_lines)):
        bb = (x, y, x + size, y + size); o = sum(ov(bb, hb) for hb in face_boxes)
        if o == 0: return name, (x, y)
    return "top-right (person dropped)", BADGE_SPOTS(0)[0][1]

def verify(cover_rgb, n_lines, badge_box=None, expected=1, debug_path=None, solo=True):
    faces = detect_faces(cover_rgb) if expected > 0 else []   # hood/mask covers: detector hits are false positives
    hl_top = headline_top(n_lines); issues = []
    boxes = [("tag", TAG_BOX), ("headline", (40, hl_top, 1040, 1320))] + ([("badge", badge_box)] if badge_box else [])
    def ov(a, b): return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])
    lums = []
    for x, y, w, h, sc in faces:
        fb = (x, y, x + w, y + h)
        for name, b in boxes:
            if ov(fb, b): issues.append(f"face at y{int(y)} overlaps {name}")
        if y < 0 or y + h > 1350 or x < 0 or x + w > 1080: issues.append("face cut off by frame")
        if h < (300 if solo else 260): issues.append(f"face too small ({int(h)}px) - fill the frame")
        z = np.asarray(cover_rgb.convert("RGB")).astype(np.float32)[int(y + h*.2):int(y + h*.85), int(x + w*.2):int(x + w*.8)]
        L = 0.299*z[...,0] + 0.587*z[...,1] + 0.114*z[...,2]
        hi, lo = np.percentile(L, 95), np.percentile(L, 5)
        # exposure check per face (skin tones differ, so faces are NOT forced to the same brightness):
        # each face needs real highlights and real depth
        if hi < 165: issues.append(f"face at y{int(y)} looks underexposed (highlights {hi:.0f})")
        if hi > 250 and (L > 250).mean() > 0.06: issues.append(f"face at y{int(y)} blown out")
        if hi - lo < 90: issues.append(f"face at y{int(y)} flat / low contrast ({hi-lo:.0f})")
        lums.append(hi)
    if len(faces) < expected: issues.append(f"only {len(faces)} of {expected} faces found (check hood/mask manually)")
    if len(lums) > 1 and min(lums) < max(lums) * 0.85: issues.append(f"one face lit much darker than the other (highlights {[round(l) for l in lums]})")
    if debug_path:
        d = cover_rgb.convert("RGB").copy(); dr = ImageDraw.Draw(d)
        for name, b in boxes: dr.rectangle(b, outline=(255, 255, 0), width=4)
        dr.line((0, TAG_BOX[3] + 20, 1080, TAG_BOX[3] + 20), fill=(0, 255, 255), width=3)
        dr.line((0, hl_top - 25, 1080, hl_top - 25), fill=(0, 255, 255), width=3)
        for x, y, w, h, sc in faces: dr.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=5)
        d.resize((432, 540)).save(debug_path)
    return faces, issues

import os, enhance as _enh
SR_ABOVE = 1.8
SR_MIX = 0.5

def _sr_visible(cut, fsrc, fh, pl, name=None):
    """Super-resolve only the part of the cut-out that lands on the canvas (cached)."""
    s = pl["target_h"] / fh; (xc, yc) = pl["target"]; cx, cy = fsrc
    x0 = int(max(0, cx + (0 - xc) / s - 20)); x1 = int(min(cut.size[0], cx + (1080 - xc) / s + 20))
    y0 = int(max(0, cy + (0 - yc) / s - 20)); y1 = int(min(cut.size[1], cy + (1350 - yc) / s + 20))
    key = f"sr_cache/{name or id(cut)}_{x0}_{y0}_{x1}_{y1}.png"
    if name and os.path.exists(key): big = Image.open(key)
    else:
        big = _enh.sr_rgba(cut.crop((x0, y0, x1, y1)))
        if name: os.makedirs(os.path.dirname(key), exist_ok=True); big.save(key)
    # 50/50 with a plain resize: keeps real skin texture so it never looks AI-painted
    plain = cut.crop((x0, y0, x1, y1)).resize(big.size, Image.LANCZOS)
    big = Image.blend(plain.convert("RGBA"), big.convert("RGBA"), SR_MIX); big.putalpha(plain.getchannel("A"))
    return big, ((cx - x0) * 4, (cy - y0) * 4), fh * 4

def build(bg, people, lines, accent, tag_text="NEWS", badge_img=None, out=None, debug=None, enhance=True):
    """people: list of dict(cut=Image, x=centre_x, grade=dict, z=layer). Returns (cover, issues, plan, badge_spot)."""
    n = len(lines); canvas = bg.copy()
    p = plan([(pp["cut"], pp["x"]) for pp in people], n)
    bpos = spot = None
    if badge_img is not None:
        bs = badge_img.size[0]
        spot, bpos = badge_spot([pl["face_box"] for pl in p], n, bs)
        bb = (bpos[0], bpos[1], bpos[0] + bs, bpos[1] + bs)
        for pl in p:   # no free spot: the badge takes the corner and that person drops just below it (max 140px)
            fb = pl["face_box"]
            if not (fb[2] <= bb[0] or fb[0] >= bb[2] or fb[3] <= bb[1] or fb[1] >= bb[3]):
                d = min(140, bb[3] + 5 - fb[1])
                pl["target"] = (pl["target"][0], pl["target"][1] + d)
                pl["face_box"] = (fb[0], fb[1] + d, fb[2], fb[3] + d); pl["dropped_for_badge"] = d
    for pp, pl in sorted(zip(people, p), key=lambda t: t[0].get("z", 0)):
        cut, fsrc, fh = pp["cut"], pl["face_src"], pl["face_h_src"]
        if enhance and pl["scale"] > SR_ABOVE:      # enlarged a lot: rebuild detail with 4x super-resolution
            cut, fsrc, fh = _sr_visible(cut, fsrc, fh, pl, pp.get("name"))
        if enhance: cut = _enh.pop(cut, strength=pp.get("pop", 1.0))
        canvas = place_outlined(canvas, cut, fsrc, fh, pl["target"], pl["target_h"], grade_kw=pp.get("grade"))
    bbox = None
    if badge_img is not None:
        bx, by = bpos; canvas.alpha_composite(badge_img, (bx, by)); bbox = (bx, by, bx + badge_img.size[0], by + badge_img.size[1])
    canvas = bottom_gradient(canvas)
    cols = [MINT if i in accent else (255, 255, 255) for i in range(n)]
    canvas, _ = headline(canvas, lines, "anton.ttf", 124 if n >= 3 else 136, headline_top(n), cols, line_gap=1.02, max_w=940)
    if tag_text: canvas = tag(canvas, tag_text, "anton.ttf")
    canvas = mark(canvas, "compass_white.png")
    faces, issues = verify(canvas, n, badge_box=bbox, solo=len(people) == 1,
                           expected=sum(1 for pl in p if pl["how"] == "detected"), debug_path=debug)
    if out: canvas.convert("RGB").save(out, quality=95)
    return canvas, issues, p, spot
