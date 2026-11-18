"""Waveform cut checker for Compass interview edits.
Usage: python3 cutcheck.py spec.json a16.wav [--fix]
For every item edge, looks at the 10 ms RMS envelope of the source audio:
- an END is bad if sound is still loud at the cut (a word is being chopped)
- a START is bad if sound is already loud at the cut (the word's attack is missing)
With --fix it moves each bad edge to the nearest quiet point (max 0.6 s, never past
a neighbouring item) and rewrites the spec. Prints a report either way."""
import json, sys, wave, numpy as np

def env(wav):
    w = wave.open(wav); sr = w.getframerate(); ch = w.getnchannels()
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    if ch > 1: a = a.reshape(-1, ch).mean(1)
    hop = sr // 100; n = len(a) // hop
    return 20 * np.log10(np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-6)

def floor_at(db, t, span=3.0):
    k = int(t * 100); seg = db[max(0, k - int(span * 100)):k + int(span * 100)]
    return np.percentile(seg, 20)

def loud(db, t, fl, w=0.06):
    k = int(t * 100); seg = db[max(0, k - int(w * 100) // 2):k + int(w * 100) // 2 + 1]
    return seg.max() > fl

def quiet_run(db, k, fl, n=6):
    return k + n < len(db) and (db[k:k + n] < fl).all()

def check(items, db, fix=False, maxs=0.6, words=None):
    rep = []
    for i, it in enumerate(items):
        s, e = it[0], it[1]
        lo = items[i - 1][1] if i else 0
        hi = items[i + 1][0] if i + 1 < len(items) else e + maxs
        body = db[int(s * 100):int(e * 100)]
        sp = np.percentile(body, 90) if len(body) else -20
        fl = max(sp - 9, floor_at(db, (s + e) / 2) + 10)   # "speech-loud" threshold
        qfl = fl - 3
        cont_after = i + 1 < len(items) and abs(items[i + 1][0] - e) < 0.02
        cont_before = i > 0 and abs(items[i - 1][1] - s) < 0.02
        if not cont_after and loud(db, e, fl):
            nw = min([w[0] for w in (words or []) if w[0] > e - 0.05 and w[1] > e + 0.05] or [e + maxs])  # next word not in this item
            k = int(e * 100); lim = int(min(e + maxs, hi - 0.02, nw + 0.25) * 100)  # whisper word edges drift ~0.1-0.2 s (18 Nov: 'morning', 'this' chopped)
            while k < lim and not quiet_run(db, k, qfl): k += 1
            if k >= lim:   # no silence: cut at the energy dip between this word and the next
                a_, b_ = int(e * 100), max(int(e * 100) + 1, lim)
                sm = np.convolve(db[a_:b_ + 1], np.ones(3) / 3, "same"); k = a_ + int(np.argmin(sm)); ne = round(k / 100, 2)
                if sm.min() > qfl - 6: rep.append(f"item {i} END {e:.2f} WARNING no real dip found: check by ear/slice transcription")
                rep.append(f"item {i} END {e:.2f} no silence before next word -> energy dip {ne:.2f}")
                if fix: it[1] = ne
                continue_end = True
            else: continue_end = False
            ne = round(k / 100 + 0.04, 2)
            if not continue_end:
                rep.append(f"item {i} END {e:.2f} cuts into sound -> {ne:.2f}")
                if fix: it[1] = ne
        if not cont_before and loud(db, s, fl):
            pw = max([w[1] for w in (words or []) if w[1] < s + 0.05 and w[0] < s - 0.05] or [s - maxs])  # previous word not in this item
            k = int(s * 100); lim = int(max(s - maxs, lo + 0.02, pw - 0.25) * 100)
            while k > lim and not quiet_run(db, k - 6, qfl): k -= 1
            if k <= lim:
                a_, b_ = min(lim, int(s * 100) - 1), int(s * 100)
                sm = np.convolve(db[a_:b_ + 1], np.ones(3) / 3, "same"); k = a_ + int(np.argmin(sm)); ns = round(k / 100, 2)
                if sm.min() > qfl - 6: rep.append(f"item {i} START {s:.2f} WARNING no real dip found: check by ear/slice transcription")
                rep.append(f"item {i} START {s:.2f} no silence after previous word -> energy dip {ns:.2f}")
                if fix: it[0] = ns
                continue
            ns = round(k / 100 - 0.04, 2)
            rep.append(f"item {i} START {s:.2f} starts mid-sound -> {ns:.2f}")
            if fix: it[0] = ns
    # verify: every non-joined edge must now be quiet relative to its item, else flag for a manual slice check
    for i, it in enumerate(items):
        s_, e_ = it[0], it[1]; body = db[int(s_ * 100):int(e_ * 100)]
        if not len(body): continue
        fl = max(np.percentile(body, 90) - 9, floor_at(db, (s_ + e_) / 2) + 10)
        ja = i + 1 < len(items) and abs(items[i + 1][0] - e_) < 0.02; jb = i > 0 and abs(items[i - 1][1] - s_) < 0.02
        if not ja and loud(db, e_, fl): rep.append(f"UNRESOLVED item {i} END {e_:.2f} still in a word: slice-transcribe e-2..e and e-2..e+0.6 and pick the gap by hand")
        if not jb and loud(db, s_, fl): rep.append(f"UNRESOLVED item {i} START {s_:.2f} still in a word: slice-transcribe and pick by hand")
    return rep

def reel_cuts(f):
    import subprocess
    fps = eval(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", f]).decode().split()[0])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-vf", "scale=96:170,format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, 170, 96).astype(np.float32); d = np.abs(np.diff(a, axis=0)).mean((1, 2))
    return [(i + 1) / fps for i in np.where(d > max(18, np.median(d) * 4))[0]], len(a) / fps

def broll_edges(spec, base, rep):
    """B-roll must not start or end within 0.25 s of a hard cut inside the source reel (one-frame glitch / flash, Blaines 18 Nov)."""
    import os
    for b in spec.get("broll", []):
        if b.get("type") != "video": continue
        cuts, L = reel_cuts(os.path.join(base, b["file"])); dur = b["e"] - b["s"]; i0 = b.get("in", 0); i1 = i0 + dur
        bad = [c for c in cuts if (0 < c - i0 < 0.25) or (0 <= i1 - c < 0.25)]  # starting just before a reel cut, or ending just after one, flashes a few frames of another shot
        if i1 > L: rep.append(f"broll {b['file']} runs past the reel end ({i1:.2f} > {L:.2f})")
        if bad:
            ok = [c + 0.02 for c in cuts if c + 0.02 + dur <= L and all(not (0 <= (c + 0.02 + dur) - x < 0.25) for x in cuts)]
            rep.append(f"broll {b['file']} in {i0:.2f}-{i1:.2f} sits next to reel cut(s) {[round(c, 2) for c in bad]} -> clean in-points (start on a cut): {[round(x, 2) for x in ok]}")

if __name__ == "__main__":
    spec = json.load(open(sys.argv[1])); db = env(sys.argv[2]); fix = "--fix" in sys.argv
    import os; wp = os.path.join(os.path.dirname(sys.argv[1]), spec.get("words", "words.json"))
    words = json.load(open(wp)) if os.path.exists(wp) else None
    rep = check(spec["items"], db, fix, words=words)
    # B-roll rule (Blaines 18 Nov): never cut to camera then switch to B-roll <1 s later; start the B-roll on the cut
    for b in spec.get("broll", []):
        if b.get("type") != "video": continue
        st = [it[0] for it in spec["items"] if 0 < b["s"] - it[0] < 1.0]
        if st:
            d = b["s"] - st[-1]
            rep.append(f"broll {b['file']} starts {d:.2f}s after a cut -> move to {st[-1]:.2f}")
            if fix: b["s"] = st[-1]; b["in"] = round(max(0, b.get("in", 0) - d), 2)
        # B-roll must not end just before a cut (flash back to camera for <2 s): run it to the cut
        en = [it[1] for it in spec["items"] if it[0] < b["e"] and 0 < it[1] - b["e"] < 2.0]
        if en:
            rep.append(f"broll {b['file']} ends {en[0]-b['e']:.2f}s before a cut -> extend to {en[0]:.2f} (check reel length)")
            if fix: b["e"] = en[0]
    broll_edges(spec, os.path.dirname(os.path.abspath(sys.argv[1])), rep)
    print("\n".join(rep) or "all cuts clean")
    if fix: json.dump(spec, open(sys.argv[1], "w"))
