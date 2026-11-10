"""Compass cover 'pop' compositor: cut-out subjects on a clean dark/neon background,
graded for contrast + depth, soft rim glow, placed so faces sit between the tag (~y150)
and the divider star (~y745-860)."""
import numpy as np, cv2
from PIL import Image, ImageFilter, ImageEnhance, ImageChops

W, H = 1080, 1350

def neon_bg(top=(8, 8, 30), mid=(58, 14, 92), glow=(255, 70, 140), glow_y=0.62, glow_r=0.55):
    y = np.linspace(0, 1, H)[:, None]; x = np.linspace(0, 1, W)[None, :]
    base = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        base[..., c] = top[c] + (mid[c] - top[c]) * np.clip(y * 1.3, 0, 1)
    d = np.sqrt(((x - 0.5) * 1.1) ** 2 + ((y - glow_y) * 1.6) ** 2)
    g = np.clip(1 - d / glow_r, 0, 1) ** 2
    for c in range(3):
        base[..., c] = base[..., c] * (1 - 0.6 * g) + glow[c] * 0.6 * g
    vig = np.clip(1 - (np.sqrt((x - 0.5) ** 2 + (y - 0.45) ** 2) - 0.45) * 1.2, 0.55, 1)
    return Image.fromarray(np.clip(base * vig[..., None], 0, 255).astype(np.uint8)).convert("RGBA")

def grade(rgba, gamma=0.85, contrast=1.18, sat=1.08, sharpen=70):
    a = rgba.getchannel("A"); rgb = rgba.convert("RGB")
    arr = np.asarray(rgb).astype(np.float32) / 255
    arr = arr ** gamma                                   # lift mids
    arr = 0.5 + (arr - 0.5) * contrast                   # contrast around mid grey
    arr = np.clip(arr, 0.02, 0.985)                      # keep detail in blacks/whites
    rgb = Image.fromarray((arr * 255).astype(np.uint8))
    rgb = ImageEnhance.Color(rgb).enhance(sat)
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=2.2, percent=sharpen, threshold=2))
    out = rgb.convert("RGBA"); out.putalpha(a); return out

def place(canvas, cut, face_xy, face_h, target_xy, target_face_h, glow=(255, 225, 240), glow_alpha=0.55):
    s = target_face_h / face_h
    cut = cut.resize((round(cut.width * s), round(cut.height * s)), Image.LANCZOS)
    ox = round(target_xy[0] - face_xy[0] * s); oy = round(target_xy[1] - face_xy[1] * s)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0)); layer.paste(cut, (ox, oy), cut)
    a = layer.getchannel("A")
    # soft drop shadow then rim glow, both behind the subject
    sh = a.filter(ImageFilter.GaussianBlur(18)).point(lambda v: int(v * 0.55))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 255)); shadow.putalpha(sh)
    canvas.alpha_composite(shadow, (8, 14))
    ring = a.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(10))
    ring = ImageChops.subtract(ring, a.filter(ImageFilter.GaussianBlur(2)))
    ring = ring.point(lambda v: int(min(255, v * 1.6) * glow_alpha))
    rim = Image.new("RGBA", (W, H), glow + (255,)); rim.putalpha(ring)
    canvas.alpha_composite(rim)
    canvas.alpha_composite(layer)
    return canvas

def hard_alpha(a, thr=110, feather=1.2):
    a = a.point(lambda v: 255 if v >= thr else 0)
    return a.filter(ImageFilter.GaussianBlur(feather))

def place_outlined(canvas, cut, face_xy, face_h, target_xy, target_face_h, stroke=10, stroke_col=(255, 255, 255),
                   shadow=True, grade_kw=None):
    s = target_face_h / face_h
    cut = cut.resize((round(cut.width * s), round(cut.height * s)), Image.LANCZOS)
    cut = grade(cut, **(grade_kw or {}))
    a = hard_alpha(cut.getchannel("A")); cut.putalpha(a)
    ox = round(target_xy[0] - face_xy[0] * s); oy = round(target_xy[1] - face_xy[1] * s)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0)); layer.paste(cut, (ox, oy), cut)
    la = layer.getchannel("A")
    if shadow:
        sh = la.filter(ImageFilter.GaussianBlur(22)).point(lambda v: int(v * 0.6))
        sl = Image.new("RGBA", (W, H), (0, 0, 0, 255)); sl.putalpha(sh); canvas.alpha_composite(sl, (10, 16))
    if stroke:
        ring = la.point(lambda v: 255 if v > 40 else 0).filter(ImageFilter.MaxFilter(stroke * 2 + 1))
        ring = ring.filter(ImageFilter.GaussianBlur(0.8))
        st = Image.new("RGBA", (W, H), stroke_col + (255,)); st.putalpha(ring); canvas.alpha_composite(st)
    canvas.alpha_composite(layer)
    return canvas

def badge(src_img, box, size=250, ring=9, ring_col=(255, 255, 255)):
    im = src_img.crop(box).resize((size, size), Image.LANCZOS).convert("RGBA")
    m = Image.new("L", (size, size), 0); from PIL import ImageDraw
    ImageDraw.Draw(m).ellipse((0, 0, size - 1, size - 1), fill=255)
    im.putalpha(m)
    out = Image.new("RGBA", (size + ring * 2, size + ring * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(out); d.ellipse((0, 0, size + ring * 2 - 1, size + ring * 2 - 1), fill=ring_col + (255,))
    out.alpha_composite(im, (ring, ring))
    return out

def face_lum(img, box):
    z = np.asarray(img.convert("RGB")).astype(np.float32)[box[1]:box[3], box[0]:box[2]]
    return float((0.299 * z[..., 0] + 0.587 * z[..., 1] + 0.114 * z[..., 2]).mean())

def headline(canvas, lines, font_path, size, top, colours, line_gap=0.98, shadow=True, max_w=1000):
    """lines: list of strings; colours: list of (r,g,b) per line. Anton matches the Canva template font."""
    from PIL import ImageDraw, ImageFont
    f = ImageFont.truetype(font_path, size)
    while max(f.getlength(l) for l in lines) > max_w: size -= 4; f = ImageFont.truetype(font_path, size)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0)); sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer); ds = ImageDraw.Draw(sh)
    asc = -f.getbbox("H", anchor="ls")[1]; y = top + asc
    for l, col in zip(lines, colours):
        x = (W - f.getlength(l)) / 2
        ds.text((x + 4, y + 6), l, font=f, fill=(0, 0, 0, 170), anchor="ls")
        d.text((x, y), l, font=f, fill=col + (255,), anchor="ls")
        y += round(size * line_gap)
    if shadow: canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(6)))
    canvas.alpha_composite(layer)
    return canvas, size

MINT = (12, 247, 192)

def tag(canvas, text, font_path, xy=(60, 60), size=64, pad=(16, 10)):
    from PIL import ImageDraw, ImageFont
    f = ImageFont.truetype(font_path, size); w = f.getlength(text); bb = f.getbbox(text, anchor="ls")
    h = bb[3] - bb[1]; bw = int(w + pad[0] * 2); bh = int(h + pad[1] * 2)
    d = ImageDraw.Draw(canvas); x, y = xy
    d.rectangle((x, y, x + bw, y + bh), fill=MINT + (255,))
    d.text((x + pad[0], y + pad[1] - bb[1]), text, font=f, fill=(0, 0, 0, 255), anchor="ls")
    return canvas

def bottom_gradient(canvas, start=620, strength=235):
    y = np.arange(H); a = np.clip((y - start) / (H - start + 30), 0, 1) ** 1.3 * strength
    g = Image.new("RGBA", (W, H), (0, 0, 0, 255)); g.putalpha(Image.fromarray(np.repeat(a[:, None], W, 1).astype(np.uint8)))
    canvas.alpha_composite(g); return canvas

def mark(canvas, logo_path, w=62, opacity=0.8, bottom=28):
    lg = Image.open(logo_path).convert("RGBA"); lh = int(lg.height * w / lg.width); lg = lg.resize((w, lh), Image.LANCZOS)
    lg.putalpha(lg.getchannel("A").point(lambda v: int(v * opacity)))
    canvas.alpha_composite(lg, ((W - w) // 2, H - lh - bottom)); return canvas

def bg_from(src, blur=18, bright=0.55, sat=1.2, tint=None, crop=None):
    im = Image.open(src).convert("RGB")
    if crop: im = im.crop(crop)
    r = max(W / im.width, H / im.height); im = im.resize((int(im.width * r) + 1, int(im.height * r) + 1), Image.LANCZOS)
    x0 = (im.width - W) // 2; y0 = (im.height - H) // 2; im = im.crop((x0, y0, x0 + W, y0 + H))
    im = im.filter(ImageFilter.GaussianBlur(blur)); im = ImageEnhance.Brightness(im).enhance(bright); im = ImageEnhance.Color(im).enhance(sat)
    if tint:
        t = Image.new("RGB", (W, H), tint); im = Image.blend(im, t, 0.25)
    # vignette
    yy, xx = np.mgrid[0:H, 0:W]; v = np.clip(1.15 - np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H * 0.45) / H) ** 2) * 1.1, 0.45, 1)
    a = np.asarray(im).astype(np.float32) * v[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")

def stat_badge(big, small, font_path, size=225, ring=9):
    from PIL import ImageDraw, ImageFont
    out = Image.new("RGBA", (size + ring * 2, size + ring * 2), (0, 0, 0, 0)); d = ImageDraw.Draw(out)
    d.ellipse((0, 0, size + ring * 2 - 1, size + ring * 2 - 1), fill=(255, 255, 255, 255))
    d.ellipse((ring, ring, ring + size - 1, ring + size - 1), fill=(10, 12, 20, 255))
    fb = ImageFont.truetype(font_path, int(size * 0.5)); fs = ImageFont.truetype(font_path, int(size * 0.13))
    c = (size + ring * 2) / 2
    d.text((c, c + size * 0.08), big, font=fb, fill=MINT + (255,), anchor="ms")
    d.text((c, c + size * 0.27), small, font=fs, fill=(255, 255, 255, 255), anchor="ms")
    return out
