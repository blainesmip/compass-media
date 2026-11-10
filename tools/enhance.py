"""Face enhancement for Compass covers.

1. sr(img): 4x super-resolution with Upscayl's 'high-fidelity' model (ncnn, CPU), tiled.
   Used only when a person is enlarged a lot (low-res source); it rebuilds detail
   instead of blowing up blur.
2. pop(rgba): the 'thumbnail face' look without making it cartoonish:
   - face-local contrast (CLAHE on lightness, feathered to the face only)
   - dodge & burn: lifts the highlights already on the face (forehead, nose bridge,
     cheekbones) and deepens the shadows, so the face has shape
   - eye pop: brighter, crisper eyes using the detector's eye landmarks
   - fine sharpening + a touch of warmth/vibrance on skin
   Everything is a lift of what is already in the photo: no AI face replacement.
"""
import numpy as np, cv2, ncnn
from PIL import Image

MODEL = "/home/claude/models/upscayl/resources/models/high-fidelity-4x"
_net = None
def _get_net():
    global _net
    if _net is None:
        _net = ncnn.Net(); _net.opt.use_vulkan_compute = False; _net.opt.num_threads = 2
        _net.load_param(MODEL + ".param"); _net.load_model(MODEL + ".bin")
    return _net

def sr(img, tile=192, pad=12):
    """4x upscale of an RGB PIL image."""
    net = _get_net(); a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    H, W = a.shape[:2]; out = np.zeros((H * 4, W * 4, 3), np.float32)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            y0, x0 = max(0, y - pad), max(0, x - pad); y1, x1 = min(H, y + tile + pad), min(W, x + tile + pad)
            t = np.ascontiguousarray(a[y0:y1, x0:x1].transpose(2, 0, 1))
            ex = net.create_extractor(); ex.input("data", ncnn.Mat(t))
            _, o = ex.extract("output"); o = np.array(o).transpose(1, 2, 0)
            oy, ox = (y - y0) * 4, (x - x0) * 4; th, tw = min(tile, H - y) * 4, min(tile, W - x) * 4
            out[y * 4:y * 4 + th, x * 4:x * 4 + tw] = o[oy:oy + th, ox:ox + tw]
    return Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8))

def sr_rgba(rgba):
    rgb = sr(rgba.convert("RGB")); a = rgba.getchannel("A").resize(rgb.size, Image.LANCZOS)
    rgb.putalpha(a); return rgb

def _faces(rgb):
    det = cv2.FaceDetectorYN.create("yunet.onnx", "", (320, 320), 0.5)
    a = rgb[:, :, ::-1]; s = 1000 / max(a.shape[:2]); a2 = cv2.resize(a, None, fx=s, fy=s)
    det.setInputSize((a2.shape[1], a2.shape[0])); _, f = det.detect(a2)
    return [] if f is None else [x[:14] / s for x in f]

def pop(rgba, strength=1.0):
    """Thumbnail face pop on a cut-out (RGBA). Returns RGBA."""
    rgb = np.asarray(rgba.convert("RGB")).copy(); alpha = rgba.getchannel("A")
    faces = _faces(rgb)
    if not faces: return rgba
    H, W = rgb.shape[:2]; k = strength
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); L = lab[..., 0]
    out_L = L.copy(); mask_all = np.zeros((H, W), np.float32)
    for f in faces:
        x, y, w, h = f[:4]; cx, cy = x + w / 2, y + h / 2
        m = np.zeros((H, W), np.float32)
        cv2.ellipse(m, (int(cx), int(cy - h * 0.05)), (int(w * 0.85), int(h * 0.95)), 0, 0, 360, 1, -1)
        m = cv2.GaussianBlur(m, (0, 0), w * 0.3); mask_all = np.maximum(mask_all, m)
        r = max(3, int(w * 0.18))
        # local contrast
        cl = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(4, 4)).apply(L.clip(0, 255).astype(np.uint8)).astype(np.float32)
        Lc = L + (cl - L) * 0.4 * k
        # dodge & burn: amplify the face's own large-scale light shape (nose bridge, cheekbones, forehead)
        base = cv2.GaussianBlur(Lc, (0, 0), r)
        shape = base - cv2.GaussianBlur(Lc, (0, 0), r * 4)
        Lc = Lc + shape * 0.55 * k
        # micro detail (skin texture, lashes) - gentle, keeps it photographic
        Lc = Lc + (Lc - cv2.GaussianBlur(Lc, (0, 0), max(1.0, w / 180))) * 0.6 * k
        # eyes: brighter whites + crisper iris
        for ex, ey in (f[4:6], f[6:8]):
            em = np.zeros((H, W), np.float32); cv2.circle(em, (int(ex), int(ey)), int(w * 0.11), 1, -1)
            em = cv2.GaussianBlur(em, (0, 0), w * 0.04)
            Lc = Lc + em * ((Lc - cv2.GaussianBlur(Lc, (0, 0), w * 0.03)) * 0.9 + 6) * k
        # no halo: strip the broad brightness shift, keep the shape/detail changes
        d = Lc - L; d = d - cv2.GaussianBlur(d, (0, 0), w * 0.35) * 0.75
        out_L = out_L + d * m
    lab[..., 0] = out_L.clip(0, 255)
    # a touch of warmth/vibrance on the face only
    lab[..., 1] = 128 + (lab[..., 1] - 128) * (1 + 0.10 * k * mask_all)
    lab[..., 2] = 128 + (lab[..., 2] - 128) * (1 + 0.10 * k * mask_all)
    res = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    im = Image.fromarray(res); im.putalpha(alpha); return im
