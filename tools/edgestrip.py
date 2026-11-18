"""python3 edgestrip.py render.log out.mp4 sheet.jpg : 14-frame strips across every B-roll entry/exit (catches 1-frame flashes)."""
import sys, re, subprocess
log, mp4, out = sys.argv[1:4]
ts = [float(x) for m in re.findall(r"broll video ([\d.]+) ([\d.]+)", open(log).read()) for x in m]
ims = []
for i, t in enumerate(ts):
    f = f"/tmp/edge{i}.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0, t - 0.12):.3f}", "-i", mp4, "-frames:v", "14", "-vf", "scale=110:-2,tile=14x1", "-update", "1", f])
    ims += ["-i", f]
subprocess.run(["ffmpeg", "-v", "error", "-y", *ims, "-filter_complex", f"vstack=inputs={len(ts)}", out])
print("strips at", ts)
