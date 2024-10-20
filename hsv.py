import numpy as np
import matplotlib.pyplot as plt


def rgb2hsv(rgb: np.ndarray):
    """Vectorized RGB to HSV"""

    hsv = np.zeros_like(rgb)
    h, s, v = np.split(hsv, 3, -1)
    h = h.squeeze(-1)
    s = s.squeeze(-1)
    v = v.squeeze(-1)

    rgb_amax = rgb.argmax(-1)
    rgb_max = rgb.max(-1)
    rgb_min = rgb.min(-1)

    r = (rgb_max - rgb_min).astype(np.float32)
    ok = r > 0

    # fmt: off
    m = ok & (rgb_amax==0); h[m] = 0+(rgb[m, 1] - rgb[m, 2]) / r[m]
    m = ok & (rgb_amax==1); h[m] = 2+(rgb[m, 2] - rgb[m, 0]) / r[m]
    m = ok & (rgb_amax==2); h[m] = 4+(rgb[m, 0] - rgb[m, 1]) / r[m]
    # fmt: on
    h[:] *= 60
    h[h < 0] += 360

    s[ok] = r[ok] / rgb_max[ok]
    v[:] = rgb_max

    return np.stack((h, s, v), -1)


def hsv2rgb(hsv: np.ndarray):

    h, s, v = np.split(hsv, 3, axis=-1)
    h = h.squeeze(-1)
    s = s.squeeze(-1)
    v = v.squeeze(-1)

    h = h / 60
    hi = np.floor(h)
    f = h - hi
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    w = hi % 6
    rgb = np.zeros_like(hsv)
    # fmt: off
    m = w==0; rgb[m,0],rgb[m,1],rgb[m,2] = v[m],t[m],p[m]
    m = w==1; rgb[m,0],rgb[m,1],rgb[m,2] = q[m],v[m],p[m]
    m = w==2; rgb[m,0],rgb[m,1],rgb[m,2] = p[m],v[m],t[m]
    m = w==3; rgb[m,0],rgb[m,1],rgb[m,2] = p[m],q[m],v[m]
    m = w==4; rgb[m,0],rgb[m,1],rgb[m,2] = t[m],p[m],v[m]
    m = w==5; rgb[m,0],rgb[m,1],rgb[m,2] = v[m],p[m],q[m]
    # fmt: on

    return rgb


import math

ENC_MAX_HUE = 300
ENC_UNIQUE = int(ENC_MAX_HUE / 60) * 255 + 1
ENC_BITS = math.log(ENC_UNIQUE) / math.log(2.0)
INV_DEPTH = np.nan


def encode(depth: np.ndarray, max_hue: float = ENC_MAX_HUE):
    ok = np.isfinite(depth) & (depth >= 0) & (depth <= 1.0)
    h = depth * max_hue
    s = np.ones_like(h)
    v = s
    h[~ok] = 0
    s[~ok] = 0
    v[~ok] = 0

    hsv = np.stack((h, s, v), -1)
    rgb = hsv2rgb(hsv)
    return rgb


def quantize(rgb):
    return np.round(255 * rgb).astype(np.uint8)


def decode(rgb: np.ndarray, max_hue: float = ENC_MAX_HUE, inv_depth: float = INV_DEPTH):
    if rgb.dtype == np.uint8:
        rgb = rgb / 255.0
    hsv = rgb2hsv(rgb)
    ok = (hsv[..., 1] > 0.5) & (hsv[..., 2] > 0.5) & (hsv[..., 0] <= max_hue)

    d = np.full(rgb.shape[:-1], inv_depth, dtype=np.float32)
    d[ok] = hsv[ok, 0] / max_hue
    return d


def main():
    h = np.arange(330)
    s = np.ones(330)
    v = np.ones(330)
    hsv = np.stack((h, s, v), -1)
    rgb = hsv2rgb(hsv)
    hsv2 = rgb2hsv(rgb)
    print(abs(hsv - hsv2).sum(-1).max())

    # fig, ax = plt.subplots()
    # ax.imshow(rgb[None], aspect="auto")
    # plt.show()

    d = np.linspace(0, 1.0, 1276)
    rgb_enc = encode(d)
    rgb_byte = np.round(255 * rgb_enc).astype(np.uint8)  # quantization happens here
    num_diff = (abs(np.diff(rgb_byte, axis=0)).sum(-1) > 0).sum() + 1
    print(num_diff)

    fig, ax = plt.subplots()
    ax.imshow(rgb[None], aspect="auto", extent=(0, 1, 0, 1))

    fig, ax = plt.subplots()
    dd = decode(rgb_byte / 255)
    print(abs(d - dd).max())
    plt.plot(d, dd)
    plt.show()

    rng = np.random.default_rng(123)
    d = rng.random((512, 512), dtype=np.float32)
    rgb = encode(d)
    qrgb = quantize(rgb)
    dr = decode(qrgb)
    err_abs = abs(dr - d).max()
    print(err_abs)

    print(ENC_MAX_HUE, ENC_UNIQUE, ENC_BITS)

    d = np.logspace(-5, 0, 100)
    rgb = encode(d)
    qrgb = quantize(rgb)
    dr = decode(qrgb)
    err_abs = abs(dr - d)
    print(err_abs.max(), np.argmax(err_abs))

    # print(np.argmax())

    rgb = np.array(
        [[10, 20, 50], [255, 0, 41], [201, 114, 255], [111, 63, 140]], dtype=np.uint8
    )
    # first two should be nan (value/saturation) and above ENC_MAX_HUE
    # last one and second last should be 277.1/300 ~ 0.92 (desaturated ~55%, devalued ~55%)

    d = decode(rgb)
    print(d)

    pass


if __name__ == "__main__":
    main()
