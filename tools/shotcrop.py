import subprocess, numpy as np, cv2, sys
src, ss, T = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
SW, SH, CW = 3840, 2160, 1215
cuts = [0] + [float(x) for x in open('cuts.txt').read().split()] + [T]
det = cv2.FaceDetectorYN.create('yunet.onnx', '', (960, 540), 0.6)
parts = []; centre = (SW - CW) // 2
for a, b in zip(cuts, cuts[1:]):
    xs = []
    for k in range(5):
        t = ss + a + (b - a) * (k + 0.5) / 5
        raw = subprocess.run(['ffmpeg','-v','error','-ss',f'{t:.2f}','-i',src,'-frames:v','1','-vf','scale=960:540','-f','rawvideo','-pix_fmt','bgr24','-'],capture_output=True).stdout
        fr = np.frombuffer(raw, np.uint8).reshape(540, 960, 3)
        _, f = det.detect(fr)
        if f is not None:
            big = max(f, key=lambda r: r[2]*r[3]); xs.append((big[0] + big[2]/2) * 4)
    x = centre
    if xs:
        fx = float(np.median(xs))
        if not (centre + 0.2*CW < fx < centre + 0.8*CW):   # only shift if clearly out of the safe middle
            x = int(np.clip(fx - CW/2, 0, SW - CW))
    print(f'{a:6.2f}-{b:6.2f} faces {len(xs)} x {x}' + ('' if x == centre else ' (shifted)'))
    parts.append((a, b, x))
expr = str(parts[-1][2])
for a, b, x in reversed(parts[:-1]): expr = f"if(lt(t\\,{b:.3f})\\,{x}\\,{expr})"
open('cropx.txt', 'w').write(expr)
