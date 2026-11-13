"""Instagram video slide, 4:5 fill (v2, 13 Nov 2026): no logo (the cover and our page name already brand it),
one high-quality encode from the best source, native frame rate, Lanczos scaling."""
import argparse, subprocess
p = argparse.ArgumentParser(); p.add_argument("video"); p.add_argument("out")
p.add_argument("--crop", default=None, help="pre-crop source w:h:x:y")
p.add_argument("--ss", type=float, default=0); p.add_argument("--t", type=float, default=None)
a = p.parse_args(); W, H = 1080, 1350
pre = f"crop={a.crop}," if a.crop else ""
vf = f"{pre}scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1,format=yuv420p"
cmd = ["ffmpeg", "-y", "-ss", f"{a.ss}"] + (["-t", f"{a.t}"] if a.t else []) + ["-i", a.video, "-vf", vf,
       "-map", "0:v", "-map", "0:a?", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high",
       "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", a.out]
subprocess.run(cmd, check=True, capture_output=True); print("ok", a.out)
