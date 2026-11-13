import numpy as np, sys
N = 33
M = np.array([[1.6269474,-0.5401385,-0.0868089],[-0.1785155,1.4179409,-0.2394254],[-0.0444361,-0.1959199,1.2403560]])
def slog3_to_lin(x):
    c = x * 1023
    return np.where(c >= 171.2102946929, 10 ** ((c - 420) / 261.5) * 0.19 - 0.01, (c - 95) * 0.01125 / (171.2102946929 - 95))
def filmic(x, contrast):  # Hable-style shoulder/toe, normalised
    A,B,C,D,E,F = 0.22,0.30,0.10,0.20*contrast,0.01,0.30
    f = lambda v: ((v*(A*v+C*B)+D*E)/(v*(A*v+B)+D*F))-E/F
    return f(x) / f(11.2)
def look(rgb, exp, contrast, sat, toe, warm, cool_shadows, lift):
    lin = slog3_to_lin(rgb) @ M.T
    lin = np.clip(lin, 0, None) * 2 ** exp
    lin[...,0] *= 1 + warm; lin[...,2] *= 1 - warm
    out = filmic(lin * 2.0, contrast)
    out = np.clip(out, 0, 1) ** (1 / 2.2)
    out = out ** (1 + toe * (1 - out) ** 2)                      # deeper shadows, highlights kept
    L = (out * [0.2126, 0.7152, 0.0722]).sum(-1, keepdims=True)
    out = L + (out - L) * sat
    sh = (1 - L) ** 3
    out = out + sh * np.array([-cool_shadows * 0.5, 0, cool_shadows]) * 0.06
    out = lift + out * (1 - lift)
    return np.clip(out, 0, 1)
looks = {
 "A-clean":  dict(exp=0.6, contrast=1.0, sat=1.05, toe=0.10, warm=0.00, cool_shadows=0.0, lift=0.0),
 "B-rich":   dict(exp=0.7, contrast=1.2, sat=1.20, toe=0.35, warm=0.03, cool_shadows=0.0, lift=0.0),
 "C-night":  dict(exp=0.6, contrast=1.3, sat=1.12, toe=0.40, warm=0.05, cool_shadows=0.8, lift=0.0),
}
g = np.linspace(0, 1, N)
b, gg, r = np.meshgrid(g, g, g, indexing="ij")
rgb = np.stack([r, gg, b], -1).reshape(-1, 3)
for name, p in looks.items():
    out = look(rgb.copy(), **p)
    with open(f"{name}.cube", "w") as f:
        f.write(f"LUT_3D_SIZE {N}\n")
        for v in out: f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
print("ok")
