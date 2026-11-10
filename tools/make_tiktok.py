import argparse, subprocess
from PIL import Image, ImageDraw, ImageFont, ImageFilter
p = argparse.ArgumentParser()
p.add_argument("video"); p.add_argument("out")
p.add_argument("--lines", nargs="+", required=True)
p.add_argument("--credit", default=None)
p.add_argument("--ss", type=float, default=0); p.add_argument("--t", type=float, default=None)
p.add_argument("--crop", default=None, help="pre-crop source w:h:x:y")
p.add_argument("--blurband", default=None, help="y0:y1 in output px")
p.add_argument("--font", default="RobotoCond-800i.ttf")
p.add_argument("--logo", default="compass_white.png")
p.add_argument("--fs", type=int, default=69)
p.add_argument("--top", type=int, default=None, help="cap top of first headline line (default: block centred on the frame)")
p.add_argument("--handle_y", type=int, default=1480, help="centre of handle + star, just above TikTok's caption area")
p.add_argument("--overlay", default="tt_overlay.png")
a = p.parse_args(); W, H = 1080, 1920
if len(a.lines) > 3: raise SystemExit("max 3 headline lines")
def text_layer(items):
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ds = ImageDraw.Draw(shadow)
    txt = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dt = ImageDraw.Draw(txt)
    for s, f, y, fill in items:
        x = (W - f.getlength(s)) / 2
        ds.text((x + 2, y + 2), s, font=f, fill=(0, 0, 0, 120), anchor="ls")
        dt.text((x, y), s, font=f, fill=fill, anchor="ls")
    return shadow.filter(ImageFilter.GaussianBlur(1.5)), txt
hf = ImageFont.truetype(a.font, a.fs); cap = -hf.getbbox("H", anchor="ls")[1]; pitch = round(a.fs * 1.11)
if a.top is None: a.top = round(H / 2 - ((len(a.lines) - 1) * pitch + cap) / 2)
items = []; base = a.top + cap
for i, line in enumerate(a.lines):
    if hf.getlength(line) > W - 120: raise SystemExit(f"line too wide: {line}")
    items.append((line, hf, base + i * pitch, (255, 255, 255, 255)))
last_base = base + (len(a.lines) - 1) * pitch
handle = "@thecompassculture"; fs_h = 40
while ImageFont.truetype(a.font, fs_h).getlength(handle) < 288: fs_h += 1
hfont = ImageFont.truetype(a.font, fs_h); handle_c = a.handle_y
items.append((handle, hfont, handle_c - hfont.getbbox("h", anchor="ls")[1] / 2, (255, 255, 255, 235)))
if a.credit:
    cfs = 26
    while ImageFont.truetype(a.font, cfs).getlength(a.credit) < 0.85 * 288 and cfs < 30: cfs += 1
    items.append((a.credit, ImageFont.truetype(a.font, cfs), handle_c + 220, (240, 240, 240, 245)))
ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
lg = Image.open(a.logo).convert("RGBA"); lh = 176; lw = int(lg.width * lh / lg.height)
lg = lg.resize((lw, lh), Image.LANCZOS); lg.putalpha(lg.getchannel("A").point(lambda v: int(v * 0.42)))
ov.alpha_composite(lg, ((W - lw) // 2, int(handle_c - lh / 2)))
sh, tx = text_layer(items); ov.alpha_composite(sh); ov.alpha_composite(tx); ov.save(a.overlay)
pre = f"crop={a.crop}," if a.crop else ""
chain = f"[0:v]{pre}scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[base]"
inputs = ["-ss", f"{a.ss}"] + (["-t", f"{a.t}"] if a.t else []) + ["-i", a.video, "-i", a.overlay]
if a.blurband:
    y0, y1 = map(int, a.blurband.split(":")); bh = y1 - y0
    m = Image.new("L", (W, bh), 0); md = ImageDraw.Draw(m)
    for y in range(bh): md.line([(0, y), (W, y)], fill=int(255 * max(0, min(1, y / 100, (bh - 1 - y) / 100))))
    m.save("tt_bandmask.png"); inputs += ["-loop", "1", "-i", "tt_bandmask.png"]
    chain += (f";[base]split[b1][b2];[b2]crop={W}:{bh}:0:{y0},gblur=sigma=50,eq=brightness=-0.16[bl];"
              f"[bl][2:v]alphamerge[blm];[b1][blm]overlay=0:{y0}[base]")
cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", chain + ";[base][1:v]overlay=0:0[v]", "-map", "[v]", "-map", "0:a?",
       "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-crf", "19", "-r", "30", "-c:a", "aac", "-b:a", "192k",
       "-shortest"] + (["-t", f"{a.t}"] if a.t else []) + ["-movflags", "+faststart", a.out]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode: print(r.stderr[-2000:]); raise SystemExit(1)
print("ok", a.out)
