"""preflight.py spec.json [--tot|--fit]   (run in the video's work dir, needs a16.wav)

Catches, BEFORE rendering, the mistakes Blaines kept correcting (Nov 2026):
  1. CHOPPED  - last word of an item is cut (word-end drift: "jacket", "ever", "Shimmy")
  2. RUNS-ON  - the same speaker keeps talking right after the cut (cut mid-thought: Hudsons, Idi, Yoshi "skater flex")
  3. LEAD-IN  - speech right before an item start (missing setup: "this chain is from my brother")
  4. DROPPED  - kept-out stretches that contain a question or an answer (Ivan's J. Cole Q, Brandon's Thug Q)
  5. NAMES    - (TOT) the artists heard in a question don't match the bracket pair; winners don't chain
  6. ENDING   - final item: too little room after the last word, or the interviewer starts a new question
  7. FIT      - CTA on screen too long, tags too short/late/near the end, hook overlap
Prints a short report; anything flagged gets an ear-probe (edgeprobe.py) before render.
"""
import sys, json, re, wave, numpy as np
sys.path.insert(0, '/home/claude')
from snap import load
from faster_whisper import WhisperModel

spec_p = sys.argv[1]; mode = '--tot' if '--tot' in sys.argv else ('--fit' if '--fit' in sys.argv else None)
S = json.load(open(spec_p)); items = S['items']
if mode is None: mode = '--tot' if (items and items[0][2] in ('Q', 'A', 'P')) else '--fit'
TOT = mode == '--tot'
db = load('a16.wav')
w = wave.open('a16.wav'); SR = w.getframerate(); A = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
DUR = len(A) / SR
M = WhisperModel('/home/claude/wm/model', device='cpu', compute_type='int8')
ART = ("Drake, Kanye, Kendrick, J. Cole, Lil Wayne, 50 Cent, Eminem, Tyler the Creator, Travis Scott, Playboi Carti, "
       "Don Toliver, Loe Shimmy, NBA YoungBoy, Young Thug, Lil Durk, Gunna, The Weeknd, Michael Jackson, Brent Faiyaz, "
       "Bryson Tiller, Tory Lanez, Central Cee, Stormzy, Dave, J Hus, Headie One, Skepta, Destroy Lonely")
PROMPT = ("This or that. " + ART) if TOT else ""
nm = lambda t: re.sub(r'[^a-z0-9]', '', t.lower())

def tr(s, e):
    s = max(0, s); e = min(DUR, e)
    if e - s < 0.15: return []
    seg, _ = M.transcribe(A[int(s * SR):int(e * SR)], language='en', beam_size=5, word_timestamps=True,
                          initial_prompt=PROMPT, condition_on_previous_text=False)
    return [(s + x.start, s + x.end, x.word.strip()) for g in seg for x in g.words]

def loud(t0, t1, thr):
    """fraction of 10 ms frames in [t0,t1] above thr dB"""
    a, b = int(t0 * 100), int(t1 * 100)
    seg = db[max(0, a):max(a + 1, b)]
    return float((seg > thr).mean()) if len(seg) else 0.0

def lvl(s, e):
    seg = db[int(s * 100):int(e * 100)]
    return float(np.percentile(seg, 80)) if len(seg) else -60

flags = []
def flag(kind, i, msg): flags.append((kind, i, msg)); print(f'  {kind:8s} item {i:2d}: {msg}')

ALIAS = {'jacob': 'cole', 'jcole': 'cole', 'draco': 'drake', 'doug': 'durk', 'duck': 'durk', 'dunk': 'durk', 'yantuck': 'thug',
         'tug': 'thug', 'lushimi': 'shimmy', 'lohshimi': 'shimmy', 'shemi': 'shimmy', 'shimmies': 'shimmy', 'ghana': 'gunna',
         'kaniye': 'kanye', 'kanya': 'kanye', 'yanboi': 'youngboy', 'oconnor': 'kanye', 'talady': 'tyler', 'dantolliver': 'toliver',
         'dontoliver': 'toliver', 'oliver': 'toliver', 'tellever': 'toliver', 'tellover': 'toliver', '57': '50', 'fifty': '50'}
KEYS = {'drake': 'drake', 'kanye': 'kanye', 'kendrick': 'kendrick', 'cole': 'j-cole', 'wayne': 'lil-wayne', '50': '50-cent',
        'cent': '50-cent', 'eminem': 'eminem', 'tyler': 'tyler-the-creator', 'creator': 'tyler-the-creator', 'travis': 'travis-scott',
        'carti': 'playboi-carti', 'toliver': 'don-toliver', 'shimmy': 'loe-shimmy', 'youngboy': 'nba-youngboy', 'thug': 'young-thug',
        'durk': 'lil-durk', 'gunna': 'gunna', 'weeknd': 'the-weeknd', 'jackson': 'michael-jackson', 'faiyaz': 'brent-faiyaz',
        'tiller': 'bryson-tiller', 'lanez': 'tory-lanez', 'cee': 'central-cee', 'stormzy': 'stormzy', 'dave': 'dave', 'hus': 'j-hus',
        'headie': 'headie-one', 'skepta': 'skepta', 'lonely': 'destroy-lonely', 'mj': 'michael-jackson'}
def artists(words):
    out = []
    for w_ in words:
        k = nm(w_)
        for al, rep in ALIAS.items():
            if len(al) > 2 and al in k: k = k.replace(al, rep)
        k = ALIAS.get(k, k)
        for key, art in KEYS.items():
            if key in k and (not out or out[-1] != art): out.append(art); break
    return out

print(f'== preflight {spec_p} ({"This or That" if TOT else "fit/brand"}), {len(items)} items')
for i, it in enumerate(items):
    s, e = it[0], it[1]
    cap = it[3] if TOT else it[2]
    nxt = items[i + 1][0] if i + 1 < len(items) else None
    prv = items[i - 1][1] if i else None
    joined_after = nxt is not None and abs(nxt - e) < 0.02
    joined_before = prv is not None and abs(prv - s) < 0.02
    toks = [nm(t.replace('*', '')) for t in cap.split()]; toks = [t for t in toks if t]
    ref = lvl(s, e)
    # 1/2. end of item
    if not joined_after:
        inside = tr(e - 1.6, e); ext = tr(e - 1.6, e + 0.9)
        last = toks[-1] if toks else ''
        ins_n = [nm(x[2]) for x in inside]; ext_n = [nm(x[2]) for x in ext]
        if last and last not in ins_n[-3:] and last in ext_n:
            flag('CHOPPED', i, f'last caption word "{last}" only heard past the cut ({" ".join(x[2] for x in ext)})')
        after = [x for x in ext if x[0] >= e - 0.02]
        tail_loud = loud(e, e + 0.35, ref - 12)
        if after and after[0][0] < e + 0.45:
            flag('RUNS-ON', i, f'speech {after[0][0]-e:+.2f}s after cut: "{" ".join(x[2] for x in after)}" -> keep it, or confirm it is the interviewer')
        elif tail_loud > 0.5:
            flag('RUNS-ON', i, f'audio still loud right after the cut ({tail_loud:.0%} of next 0.35 s) -> ear-probe')
    # 3. start of item
    if not joined_before:
        pre = tr(s - 1.2, s + 0.4)
        before = [x for x in pre if x[1] > s - 0.6 and x[0] < s - 0.04]
        if before:
            flag('LEAD-IN', i, f'speech just before start: "{" ".join(x[2] for x in before)}" (does the line start earlier?)')

# 4. dropped stretches
spans = sorted((it[0], it[1]) for it in items)
gaps = [(a[1], b[0]) for a, b in zip(spans, spans[1:]) if b[0] - a[1] > 0.6]
for g0, g1 in gaps:
    ws = tr(g0, g1); txt = ' '.join(x[2] for x in ws)
    if not ws: continue
    arts = artists([x[2] for x in ws]) if TOT else []
    q_like = TOT and len(arts) >= 2 and re.search(r'\bor\b', txt.lower())
    a_like = TOT and len(arts) >= 1
    tag = 'DROPPED' if (q_like or a_like or len(ws) >= 4) else 'gap'
    print(f'  {tag:8s} {g0:6.2f}-{g1:6.2f}: "{txt}"' + (f'  artists={arts}' if arts else '') +
          ('  <- a QUESTION was cut: its answer must go too (or keep both)' if q_like else ''))
    if tag == 'DROPPED': flags.append(('DROPPED', -1, txt))

# 5. names vs bracket (TOT)
if TOT:
    pairs = S.get('pairs', []); qs = [k for k, it in enumerate(items) if it[2] == 'Q']
    # merge split questions (consecutive Q rows)
    groups = []
    for k in qs:
        if groups and groups[-1][-1] == k - 1: groups[-1].append(k)
        else: groups.append([k])
    if len(groups) != len(pairs): flag('NAMES', -1, f'{len(groups)} questions but {len(pairs)} bracket pairs')
    for gi, (grp, pr) in enumerate(zip(groups, pairs)):
        s, e = items[grp[0]][0], items[grp[-1]][1]
        heard = artists([x[2] for x in tr(s, e)])
        capd = artists(' '.join(items[k][3] for k in grp).split())
        if sorted(set(capd)) != sorted(set(pr)) and set(capd) - {'lil-durk', 'destroy-lonely'}:
            flag('NAMES', grp[0], f'caption artists {capd} != pair {pr}')
        extra = [a for a in heard if a not in pr]
        if extra: flag('NAMES', grp[0], f'heard extra artist(s) {extra} in this question ({heard}) -> two questions merged, or a re-ask? check by ear')
        miss = set(pr) - set(heard)
        if miss: flag('NAMES', grp[0], f'pair {pr} but heard {heard} in "{" ".join(items[k][3] for k in grp)}" -> check by ear')
        if gi and pr[0] not in pairs[gi - 1]: flag('NAMES', grp[0], f'bracket breaks: {pr} after {pairs[gi-1]}')
        ans = [k for k in range(grp[-1] + 1, len(items)) if items[k][2] == 'A'][:1]
        if ans:
            win = [t for t in items[ans[0]][3].split() if '*' in t]
            wa = artists([t.replace('*', '') for t in win])
            nxt_pair = pairs[gi + 1] if gi + 1 < len(pairs) else [S.get('final_winner')]
            if wa and wa[0] not in nxt_pair: flag('NAMES', ans[0], f'answer winner {wa[0]} is not carried into the next pair {nxt_pair}')

# 6. ending
le = items[-1][1]; after = tr(le - 1.4, le + 1.2)
post = [x for x in after if x[0] >= le - 0.02]
ref_l = lvl(items[-1][0], le); fr = db[int((le - 1.5) * 100):int(le * 100)]
hot = np.where(fr > ref_l - 15)[0]
room = (len(fr) - 1 - hot[-1]) / 100 if len(hot) else 1.5
if room < 0.25: flag('ENDING', len(items) - 1, f'only {room:.2f}s of quiet before the end - the last word may be clipped; leave ~0.5 s of room')
HARD_END = [x for x in after if x[0] < le - 0.02 and x[1] > le + 0.05]
if HARD_END: flag('ENDING', len(items) - 1, f'CLIPPED: "{HARD_END[0][2]}" runs {HARD_END[0][1]-le:+.2f}s past the end')
if post and post[0][0] < le + 0.15: HARD_END.append(post[0])
if post and post[0][0] < le + 0.4:
    flag('ENDING', len(items) - 1, f'next speech starts {post[0][0]-le:+.2f}s after the end: "{" ".join(x[2] for x in post)}" (new question? cut before it)')

# 7. fit/brand overlays
if not TOT:
    out_len = sum(it[1] - it[0] for it in items)
    cta = S.get('cta')
    if cta:
        on = sum(max(0, it[1] - max(it[0], cta[1])) for it in items)
        if on > 2.0: flag('FIT', -1, f'CTA on screen {on:.1f}s of speech (keep it to the last ~1.5 s)')
    for tg in S.get('tags', []):
        t, dur = tg[0], tg[6]
        if dur < 2.0: flag('FIT', -1, f'tag {tg[3]} only {dur}s (aim 2-2.5 s)')
        ends_in = sum(max(0, it[1] - max(it[0], t)) for it in items)
        if ends_in < 3.0: flag('FIT', -1, f'tag {tg[3]} appears in the last {ends_in:.1f}s - move it to where the item is first shown')


# 25 Nov 2026: "fit" / "quickly" cut mid-word on the All Talk fit check OPENING item - only the final end was hard-checked.
# Waveform check on EVERY cut edge (start and end of each non-joined item): the 10 ms envelope at the cut must be
# quiet relative to that item's speech, and no transcribed word may straddle it. Any failure blocks the render.
EDGE_BAD = []
for i, it in enumerate(items):
    s, e = it[0], it[1]; ref = lvl(s, e)
    nxt = items[i + 1][0] if i + 1 < len(items) else None
    prv = items[i - 1][1] if i else None
    for edge, kind, joined in ((e, 'end', nxt is not None and abs(nxt - e) < 0.02), (s, 'start', prv is not None and abs(prv - s) < 0.02)):
        if joined: continue
        win = db[int((edge - 0.04) * 100):int((edge + 0.04) * 100)]
        hot = len(win) and float(win.max()) > ref - 14
        ws = tr(edge - 1.0, edge + 1.0)
        strad = [x for x in ws if x[0] < edge - 0.03 and x[1] > edge + 0.03]
        strad = [x for x in strad if len(win) and float(win.max()) > ref - 18]
        if hot or strad:
            # suggest the nearest quiet point (60 ms below ref-18) moving outward from the item
            sug = None
            rng = np.arange(edge, edge + 0.8, 0.01) if kind == 'end' else np.arange(edge, edge - 0.8, -0.01)
            for t in rng:
                if float(db[int((t - 0.03) * 100):int((t + 0.03) * 100)].max()) < ref - 18: sug = round(float(t), 2); break
            msg = f'{kind} {edge:.2f} cuts through audio (peak {float(win.max()) if len(win) else 0:.0f} dB vs speech {ref:.0f})' + \
                  (f', word "{strad[0][2]}" {strad[0][0]:.2f}-{strad[0][1]:.2f}' if strad else '') + (f' -> try {sug}' if sug else '')
            flag('EDGE', i, msg); EDGE_BAD.append((i, kind, edge, sug))
if EDGE_BAD and not S.get('edges_ok'):
    print('BLOCKED: a cut lands mid-word. Move it to the suggested quiet point (or ear-probe and set "edges_ok": true)')
    sys.exit(1)

print(f'== {len(flags)} flag(s)' + ('' if flags else ' - clean'))

# 23 Nov 2026: "backgrounds" was clipped on Effing Problem after an ENDING flag was waved off.
# A word running past the end, or speech within 0.15 s of it, now blocks the render unless an ear-probe
# confirmed it is safe and the spec says "ending_ok": true.
if HARD_END and not S.get('ending_ok'):
    print('BLOCKED: last word runs past the end (or speech right after it). Extend the end or ear-probe and set "ending_ok": true')
    sys.exit(1)
