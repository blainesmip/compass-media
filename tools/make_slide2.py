"""Instagram video slide for landscape sources (v2, 13 Nov 2026): whole frame fitted to 1080 wide on 1080x1350,
background = the same clip scaled to fill (no blur, no darkening). No logo. One high-quality encode, native fps."""
import argparse, subprocess, json
p = argparse.ArgumentParser()
p.add_argument("video"); p.add_argument("out")
p.add_argument("--ss", type=float, default=0)
p.add_argument("--t", type=float, default=None, help="length; default = rest of the audio")
a = p.parse_args(); W, H = 1080, 1350
info = json.loads(subprocess.run(["ffprobe","-v","error","-show_entries","stream=codec_type,width,height,duration","-of","json",a.video],capture_output=True,text=True,check=True).stdout)
v = next(s for s in info["streams"] if s["codec_type"]=="video")
au = next((s for s in info["streams"] if s["codec_type"]=="audio"), None)
vw, vh = int(v["width"]), int(v["height"])
fh = min(int(round(W*vh/vw/2))*2, H); top = (H-fh)//2
t = a.t or ((float(au["duration"]) - a.ss) if au and au.get("duration") else None)
fc = (f"[0:v]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1[bg];"
      f"[b]scale={W}:{fh}:flags=lanczos[fg];[bg][fg]overlay=0:{top},format=yuv420p[v]")
cmd = ["ffmpeg","-y","-ss",f"{a.ss}"] + (["-t",f"{t:.2f}"] if t else []) + ["-i",a.video,"-filter_complex",fc,"-map","[v]","-map","0:a?",
       "-c:v","libx264","-preset","slow","-crf","16","-profile:v","high","-c:a","aac","-b:a","256k","-movflags","+faststart",a.out]
subprocess.run(cmd, check=True, capture_output=True); print("ok", a.out)
