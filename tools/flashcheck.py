"""python3 flashcheck.py out.mp4 : finds hard cuts in the finished video (isolated spikes in frame difference,
not motion) and flags any two cuts within 0.3 s = a flash / one-frame glitch (Blaines, 18 Nov)."""
import sys, subprocess, numpy as np
f = sys.argv[1]
fps = eval(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", f]).decode().split()[0])
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-vf", "scale=72:128,format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
a = np.frombuffer(raw, np.uint8).reshape(-1, 128, 72).astype(np.float32)
# histogram distance is robust to motion; pixel diff catches same-palette cuts
h = np.stack([np.histogram(x, 32, (0, 255))[0] for x in a]).astype(np.float32); h /= h.sum(1, keepdims=True)
dh = np.abs(np.diff(h, axis=0)).sum(1); dp = np.abs(np.diff(a, axis=0)).mean((1, 2))
cuts = []
for i in range(len(dp)):
    lo, hi = max(0, i - 8), min(len(dp), i + 9)
    loc = np.median(np.r_[dp[lo:i], dp[i + 1:hi]]) if hi - lo > 1 else 0
    if dp[i] > 12 and dp[i] > 3 * max(loc, 2) and dh[i] > 0.25: cuts.append((i + 1) / fps)
# runs of 3+ spikes on consecutive frames are fast motion (hood going up, hand waves), not cuts
grp, cur = [], []
for c in cuts:
    if cur and c - cur[-1] > 2.6 / fps: grp.append(cur); cur = []
    cur.append(c)
if cur: grp.append(cur)
cuts = [x for g in grp if len(g) < 3 for x in g]
cuts = sorted(set(cuts))
print("cuts:", [round(c, 2) for c in cuts])
bad = [(round(x, 2), round(y, 2)) for x, y in zip(cuts, cuts[1:]) if y - x < 0.3]
print(("FLASH " + str(bad)) if bad else "no flashes")
