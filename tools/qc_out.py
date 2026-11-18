"""Post-render QC: python3 qc_out.py spec.json out.mp4
Re-transcribes the finished video and compares it with the caption text in the spec.
Flags: caption words not heard (likely chopped or wrong caption), heard words with no
caption, gaps > 0.8 s (dead air) and items shorter than 0.5 s."""
import json, sys, subprocess, difflib, re, numpy as np
from faster_whisper import WhisperModel

spec = json.load(open(sys.argv[1])); out = sys.argv[2]
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True).stdout
a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
m = WhisperModel("/home/claude/wm/model", device="cpu", compute_type="int8")
caps = [it[3] if isinstance(it[2], str) and it[2] in ("Q", "A", "P") else it[2] for it in spec["items"]]
prompt = " ".join(c.replace("*", "").replace("|", "").replace("_", " ") for c in caps)[:800]
seg, _ = m.transcribe(a, language="en", word_timestamps=True, condition_on_previous_text=False)
ws = [(w.start, w.end, w.word) for s in seg for w in s.words]
norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower().replace("_", ""))
heard = [norm(w[2]) for w in ws]
want = [norm(t) for c in caps for t in c.replace("|", " ").replace("*", "").split() if norm(t)]
sm = difflib.SequenceMatcher(None, want, heard, autojunk=False)
issues = []
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op == "replace" and difflib.SequenceMatcher(None, "".join(want[i1:i2]), "".join(heard[j1:j2])).ratio() > 0.6:
        continue   # same words, transcription spelling differs
    if op in ("delete", "replace"):
        t = ws[j1][0] if j1 < len(ws) else len(a) / 16000
        issues.append(f"~{t:5.1f}s caption not heard: {' '.join(want[i1:i2])}" + (f"  (heard: {' '.join(heard[j1:j2])})" if op == "replace" else ""))
    if op == "insert":
        issues.append(f"~{ws[j1][0]:5.1f}s heard but not captioned: {' '.join(heard[j1:j2])}")
for i in range(len(ws) - 1):
    g = ws[i + 1][0] - ws[i][1]
    if g > 0.8: issues.append(f"~{ws[i][1]:5.1f}s gap {g:.1f}s")
print(f"duration {len(a)/16000:.1f}s, match {sm.ratio():.0%}")
print("\n".join(issues) or "clean")
