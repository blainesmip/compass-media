"""python3 edgeprobe.py a16.wav t [t ...] : envelope (10 ms dB) around each edge + slice transcripts ending/starting there."""
import sys, numpy as np, wave
sys.path.insert(0, '/home/claude'); from snap import load
from faster_whisper import WhisperModel
wav = sys.argv[1]; ts = [float(x) for x in sys.argv[2:]]
db = load(wav); w = wave.open(wav); sr = w.getframerate(); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
m = WhisperModel('/home/claude/wm/model', device='cpu', compute_type='int8')
def tr(s, e):
    seg, _ = m.transcribe(a[int(s * sr):int(e * sr)], language='en', word_timestamps=True)
    return ' | '.join(f'{x.word.strip()} {s + x.start:.2f}-{s + x.end:.2f}' for g in seg for x in g.words)
for t in ts:
    k = int(t * 100)
    print(f'== {t}: env {t-0.4:.2f}..{t+0.6:.2f}:', ' '.join(str(int(v)) for v in db[k - 40:k + 60]))
    print('   to edge :', tr(t - 2.0, t)); print('   +0.6    :', tr(t - 2.0, t + 0.6)); print('   from    :', tr(t, t + 2.0))
