import numpy as np
import matplotlib.pyplot as plt
import huecodec as hc
import av
import io
import time
from pprint import pprint


def generate_depth_images(n: int, speed: int = 10):
    t = np.linspace(0, 1, 512)
    d_col = np.cos(2 * np.pi / 0.25 * t)
    d_row = np.cos(2 * np.pi / 0.25 * t)
    d = d_col[None, :] + d_row[:, None]
    d = (d - d.min()) * 0.5  # [0..2]

    gen = np.random.default_rng(123)

    # Delete random rectangles to mimick hard-edges
    def rr():
        x1 = gen.integers(0, d.shape[1])
        y1 = gen.integers(0, d.shape[0])
        x2 = x1 + gen.integers(d.shape[1] - x1)
        y2 = y1 + gen.integers(d.shape[0] - y1)
        return slice(y1, y2), slice(x1, x2)

    for _ in range(n):
        dmod = np.roll(d, -speed, axis=0).copy()
        dmod[*rr()] = 0
        dmod[*rr()] = 0
        dmod[*rr()] = 0
        dmod[*rr()] = 0
        yield dmod


def hue_enc_dec(gt, zrange, inv_depth):
    e = hc.depth2rgb(gt, zrange=zrange, inv_depth=inv_depth)
    d = hc.rgb2depth(e, zrange=zrange, inv_depth=inv_depth)
    return d


def av_enc_dec(gt, zrange, inv_depth, codec):
    file = io.BytesIO()

    output = av.open(file, "w", format="mp4")
    stream = output.add_stream(codec["name"], rate=1, options=codec["options"])
    stream.width = gt.shape[2]
    stream.height = gt.shape[1]
    stream.pix_fmt = codec["pix_fmt"]
    # stream.options = {"crf": "17"}

    t = time.perf_counter()

    for d in gt:
        rgb = hc.depth2rgb(d, zrange=zrange, inv_depth=inv_depth)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        packet = stream.encode(frame)
        output.mux(packet)

    packet = stream.encode(None)
    output.mux(packet)
    output.close()

    tenc = time.perf_counter() - t

    file.seek(0)
    input = av.open(file, "r")
    t = time.perf_counter()
    ds = []
    for f in input.decode(video=0):
        rgb = f.to_rgb().to_ndarray()
        d = hc.rgb2depth(rgb, zrange=zrange, inv_depth=inv_depth)
        ds.append(d)

    tdec = time.perf_counter() - t
    return np.stack(ds, 0), {
        "nbytes": file.getbuffer().nbytes,
        "tenc": tenc,
        "tdec": tdec,
    }


codecs = [
    {
        "name": "libx264",
        "options": {"qp": "0"},  # use qp for 10bit pixfmt
        "pix_fmt": "yuv444p10le",
    },
    {
        "name": "libx265",
        "options": {"qp": "0"},  # use qp for 10bit pixfmt
        "pix_fmt": "yuv444p10le",
    },
]


def analyze(gt, pred, title):
    extra = {}
    if isinstance(pred, tuple):
        pred, extra = pred

    err = abs(gt - pred)
    mse = np.square(gt - pred).mean()
    rmse = np.sqrt(mse)
    return {
        "title": title,
        "abs_err_mean": err.mean().item(),
        "abs_err_std": err.std().item(),
        "mse": mse.item(),
        "rmse": rmse.item(),
        **extra,
    }


def main():
    gt = np.stack(list(generate_depth_images(30)), 0)

    pprint(
        analyze(gt, hue_enc_dec(gt, zrange=(0.0, 2.0), inv_depth=False), "hue-linear")
    )

    pprint(
        analyze(
            gt,
            av_enc_dec(gt, zrange=(0.0, 2.0), inv_depth=False, codec=codecs[0]),
            "h264-lossless-linear",
        )
    )

    pprint(
        analyze(
            gt,
            av_enc_dec(gt, zrange=(0.0, 2.0), inv_depth=False, codec=codecs[1]),
            "h265-lossless-linear",
        )
    )


if __name__ == "__main__":
    main()
