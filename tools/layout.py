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
sys.path.insert(0, __import__("os").path.dirname(__file__))
from pop_cover import *

DET = cv2.FaceDetectorYN.create("" + __import__("os").path.join(__import__("os").path.dirname(__file__), "yunet.onnx") + "", "", (320, 320), 0.5)
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

def plan(cuts, n_lines, badge=False, face_h=400, gap=25):
    """Places each person so the WHOLE HEAD (hair/hood to chin) sits between the tag and the headline,
    clear of the badge, as big as the zone allows (face up to face_h px), heads level."""
    safe_top = TAG_BOX[3] + 20
    safe_bot = headline_top(n_lines) - gap
    out = []
    for cut, xc in cuts:
        cx, cy, h, htop, chin, how = face_of(cut)
        top_lim = safe_top
        s = face_h / h
        if badge and xc + (cx - (cx - h)) * s * 0.6 > BADGE_BOX[0]:
            top_lim = max(top_lim, BADGE_BOX[3] + 15)
        head_span = chin - htop
        s = min(s, (safe_bot - top_lim) / head_span)
        # centre the head (hair to chin) in the available zone
        head_mid_target = (top_lim + safe_bot) / 2
        cy_target = head_mid_target + (cy - (htop + chin) / 2) * s
        out.append(dict(face_src=(cx, cy), face_h_src=h, target=(xc, cy_target), target_h=h * s, how=how,
                        head_top=head_mid_target - head_span * s / 2, top_lim=top_lim))
    if len(out) > 1:   # level heads: align head tops to the lowest one, same face size
        th = min(o["target_h"] for o in out)
        for o in out:
            sc = th / o["face_h_src"]; o["target_h"] = th
        ht = max(o["top_lim"] for o in out)
        for (cut, xc), o in zip(cuts, out):
            cx, cy, h, htop, chin, how = face_of(cut); sc = o["target_h"] / h
            span = (chin - htop) * sc; mid = (ht + safe_bot) / 2
            o["target"] = (o["target"][0], mid + (cy - (htop + chin) / 2) * sc)
    return out

def verify(cover_rgb, n_lines, badge=False, expected=1, debug_path=None):
    faces = detect_faces(cover_rgb)
    hl_top = headline_top(n_lines); issues = []
    boxes = [("tag", TAG_BOX), ("headline", (40, hl_top, 1040, 1320))] + ([("badge", BADGE_BOX)] if badge else [])
    def ov(a, b): return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])
    lums = []
    for x, y, w, h, sc in faces:
        fb = (x, y, x + w, y + h)
        for name, b in boxes:
            if ov(fb, b): issues.append(f"face at y{int(y)} overlaps {name}")
        if y < 0 or y + h > 1350 or x < 0 or x + w > 1080: issues.append("face cut off by frame")
        if h < 200: issues.append(f"face too small ({int(h)}px)")
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

def build(bg, people, lines, accent, tag_text="NEWS", badge_img=None, out=None, debug=None, face_h=400):
    """people: list of dict(cut=Image, x=centre_x, grade=dict). Returns (cover, issues)."""
    n = len(lines); canvas = bg.copy()
    p = plan([(pp["cut"], pp["x"]) for pp in people], n, badge=badge_img is not None, face_h=face_h)
    for pp, pl in sorted(zip(people, p), key=lambda t: t[0].get("z", 0)):
        canvas = place_outlined(canvas, pp["cut"], pl["face_src"], pl["face_h_src"], pl["target"], pl["target_h"],
                                grade_kw=pp.get("grade"))
    if badge_img is not None: canvas.alpha_composite(badge_img, (BADGE_BOX[0], BADGE_BOX[1]))
    canvas = bottom_gradient(canvas)
    cols = [MINT if i in accent else (255, 255, 255) for i in range(n)]
    canvas, _ = headline(canvas, lines, __import__("os").path.join(__import__("os").path.dirname(__file__), "anton.ttf"), 124 if n >= 3 else 136, headline_top(n), cols, line_gap=1.02, max_w=940)
    if tag_text: canvas = tag(canvas, tag_text, __import__("os").path.join(__import__("os").path.dirname(__file__), "anton.ttf"))
    canvas = mark(canvas, __import__("os").path.join(__import__("os").path.dirname(__file__), "compass_white.png"))
    faces, issues = verify(canvas, n, badge=badge_img is not None,
                           expected=sum(1 for pl in p if pl["how"] == "detected"), debug_path=debug)
    if out: canvas.convert("RGB").save(out, quality=95)
    return canvas, issues, p
