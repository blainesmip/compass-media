import argparse, subprocess, json
from PIL import Image
p = argparse.ArgumentParser()
p.add_argument("video"); p.add_argument("out")
p.add_argument("--logo", required=True)
p.add_argument("--logo_w", type=int, default=130)
p.add_argument("--opacity", type=float, default=0.5)
p.add_argument("--t", type=float, default=None, help="trim length; default = audio duration")
p.add_argument("--overlay", default="slide2_overlay.png")
a = p.parse_args()
W, H = 1080, 1350
info = json.loads(subprocess.run(["ffprobe","-v","error","-show_entries","stream=codec_type,width,height,duration","-of","json",a.video],capture_output=True,text=True,check=True).stdout)
v = next(s for s in info["streams"] if s["codec_type"]=="video")
au = next((s for s in info["streams"] if s["codec_type"]=="audio"), None)
vw, vh = int(v["width"]), int(v["height"])
fh = min(int(round(W*vh/vw/2))*2, H); top = (H-fh)//2
t = a.t or (float(au["duration"]) if au and au.get("duration") else None)
lg = Image.open(a.logo).convert("RGBA"); lh = int(lg.height*a.logo_w/lg.width)
lg = lg.resize((a.logo_w, lh), Image.LANCZOS); lg.putalpha(lg.getchannel("A").point(lambda x: int(x*a.opacity)))
ov = Image.new("RGBA",(W,H),(0,0,0,0)); mid = top+fh+(H-(top+fh))//2
ov.alpha_composite(lg, ((W-a.logo_w)//2, mid-lh//2)); ov.save(a.overlay)
fc = (f"[0:v]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[bg];"
      f"[b]scale={W}:{fh}[fg];[bg][fg]overlay=0:{top}[v1];[v1][1:v]overlay=0:0[v]")
cmd = ["ffmpeg","-y","-i",a.video,"-i",a.overlay,"-filter_complex",fc,"-map","[v]","-map","0:a?","-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p","-crf","20","-r","30","-c:a","aac","-movflags","+faststart"]
if t: cmd += ["-t", f"{t:.2f}"]
subprocess.run(cmd+[a.out], check=True, capture_output=True); print("ok", a.out)
