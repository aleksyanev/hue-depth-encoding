import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import av
import io
import time
from pprint import pprint
from itertools import product

import huecodec as hc


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


def hue_enc_dec(gt, zrange, inv_depth, **kwargs):
    t = time.perf_counter()
    e = hc.depth2rgb(gt, zrange=zrange, inv_depth=inv_depth)
    tenc = time.perf_counter() - t  # not very accurate, use benchmarks
    t = time.perf_counter()
    d = hc.rgb2depth(e, zrange=zrange, inv_depth=inv_depth)
    tdec = time.perf_counter() - t
    return d, {
        "tenc": tenc,
        "tdec": tdec,
    }


def av_enc_dec(gt, zrange, inv_depth, codec):
    file = io.BytesIO()

    output = av.open(file, "w", format="mp4")
    stream = output.add_stream(codec["name"], rate=1, options=codec["options"])
    stream.width = gt.shape[2]
    stream.height = gt.shape[1]
    stream.pix_fmt = codec["pix_fmt"]

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


# codecs = [
#     {
#         "name": "libx264",
#         "options": {"qp": "0"},  # use qp for 10bit pixfmt
#         "pix_fmt": "yuv444p10le",
#     },
#     {
#         "name": "libx265",
#         "options": {"qp": "0"},  # use qp for 10bit pixfmt
#         "pix_fmt": "yuv444p10le",
#     },
# ]


def analyze(gt, pred):
    extra = {}
    if isinstance(pred, tuple):
        pred, extra = pred

    err = abs(gt - pred)
    mse = np.square(gt - pred).mean()
    rmse = np.sqrt(mse)
    return {
        "abs_err_mean": err.mean().item(),
        "abs_err_std": err.std().item(),
        "mse": mse.item(),
        "rmse": rmse.item(),
        **extra,
    }


matrix = {
    "zrange": [(0.0, 2.0), (0.0, 4.0)],
    "linear": [True],
    "codec": [
        {
            "name": "none",
            "variant": "hue-only",
        },
        {
            "name": "libx264",
            "variant": "x264-lossless",
            "options": {"qp": "0"},  # use qp instead of crf for 10bit pixfmt
            "pix_fmt": "yuv444p10le",  # use 10bit to avoid lossy conversion from rgb
        },
        {
            "name": "libx264",
            "variant": "x264-default",
            "options": None,
            "pix_fmt": "yuv444p10le",  # use 10bit to avoid lossy conversion from rgb
        },
    ],
}


def run(gt, zrange, linear, codec):
    method = hue_enc_dec if codec["variant"] == "hue-only" else av_enc_dec
    title = f'{codec["variant"]=}/{linear=}/{zrange=}'

    try:
        pred = method(gt, zrange=zrange, inv_depth=not linear, codec=codec)
        report = analyze(gt, pred)
    except Exception as e:
        print(e)
        report = {}

    report["title"] = title
    report["variant"] = codec["variant"]
    report["zrange"] = zrange
    # report['linear'] = codec["linear"]

    return report


def execute_variants(gt):
    gen = product(matrix["codec"], matrix["zrange"], matrix["linear"])
    reports = []
    for codec, zrange, linear in gen:
        reports.append(run(gt, zrange, linear, codec))
    return reports


def main():
    gt = np.stack(list(generate_depth_images(30)), 0)
    reports = execute_variants(gt)

    df = pd.DataFrame(reports)
    del df["title"]
    del df["mse"]
    del df["abs_err_std"]
    del df["abs_err_mean"]
    df = df.reindex(columns=["variant", "zrange", "rmse", "tenc", "tdec", "nbytes"])

    df["tenc"] /= len(gt)
    df["tdec"] /= len(gt)
    df["nbytes"] /= len(gt) * 1e3

    print(df)
    print()
    print(df.to_markdown())

    # pprint(
    #     analyze(gt, hue_enc_dec(gt, zrange=(0.0, 2.0), inv_depth=False), "hue-linear")
    # )

    # pprint(
    #     analyze(
    #         gt,
    #         av_enc_dec(gt, zrange=(0.0, 2.0), inv_depth=False, codec=codecs[0]),
    #         "h264-lossless-linear",
    #     )
    # )

    # pprint(
    #     analyze(
    #         gt,
    #         av_enc_dec(gt, zrange=(0.0, 2.0), inv_depth=False, codec=codecs[1]),
    #         "h265-lossless-linear",
    #     )
    # )


if __name__ == "__main__":
    main()
