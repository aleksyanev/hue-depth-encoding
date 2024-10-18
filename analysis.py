import io
import time
import traceback
from itertools import cycle, islice, product

import av
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import huecodec as hc

N_ENCDEC_FRAMES = 1000

MATRIX = {
    "zrange": [(0.0, 2.0), (0.0, 4.0)],
    "linear": [True],
    "codec": [
        {
            "variant": "hue-only",
            "name": "none",
        },
        {
            "variant": "h264-lossless-cpu",
            "name": "libx264",
            "options": {"qp": "0"},  # use qp instead of crf for 10bit pixfmt
            "pix_fmt": "yuv444p10le",  # use 10bit to avoid lossy conversion from rgb
        },
        {
            "variant": "h264-default-cpu",
            "name": "libx264",
            "options": None,
            "pix_fmt": "yuv420p",
        },
        {
            "variant": "h264-lossless-gpu",
            "name": "h264_nvenc",
            "options": {"preset": "lossless"},
            "pix_fmt": "gbrp",  # planar gbr, only way i could make this lossless
        },
        {
            "variant": "h264-default-gpu",
            "name": "h264_nvenc",
            "options": None,
            "pix_fmt": "yuv420p",
        },
    ],
}


def generate_synthetic_depth_images(n: int, speed: int = 10):
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

    # process N_ENCDEC_FRAMES by cycling batched gt
    for depth in islice(cycle(gt), N_ENCDEC_FRAMES):
        e = hc.depth2rgb(depth, zrange=zrange, inv_depth=inv_depth)
    tenc = time.perf_counter() - t  # not very accurate, use benchmarks

    t = time.perf_counter()
    for rgb in islice(cycle([e]), N_ENCDEC_FRAMES):
        d = hc.rgb2depth(rgb, zrange=zrange, inv_depth=inv_depth)
    tdec = time.perf_counter() - t

    e = hc.depth2rgb(gt, zrange=zrange, inv_depth=inv_depth)
    d = hc.rgb2depth(e, zrange=zrange, inv_depth=inv_depth)

    factor = gt.shape[0] / N_ENCDEC_FRAMES
    return d, {
        "tenc": tenc * factor,
        "tdec": tdec * factor,
        "nbytes": e.nbytes,
    }


def av_enc_dec(gt, zrange, inv_depth, codec):
    file = io.BytesIO()

    output = av.open(file, "w", format="mp4")
    stream = output.add_stream(codec["name"], rate=1, options=codec["options"])
    stream.width = gt.shape[2]
    stream.height = gt.shape[1]
    stream.pix_fmt = codec["pix_fmt"]

    # to reduce impact of overhead of the codec,
    # we virtually repeat the experiment
    t = time.perf_counter()
    for d in islice(cycle(gt), N_ENCDEC_FRAMES):
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
    for fidx, f in enumerate(input.decode(video=0)):
        rgb = f.to_rgb().to_ndarray()
        d = hc.rgb2depth(rgb, zrange=zrange, inv_depth=inv_depth)
        if fidx < gt.shape[0]:
            ds.append(d)

    tdec = time.perf_counter() - t
    factor = gt.shape[0] / N_ENCDEC_FRAMES
    return np.stack(ds, 0), {
        "nbytes": file.getbuffer().nbytes * factor,
        "tenc": tenc * factor,
        "tdec": tdec * factor,
    }


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


def run(cfg: OmegaConf, gt, zrange, linear, codec):
    method = hue_enc_dec if codec["variant"] == "hue-only" else av_enc_dec

    try:
        pred = method(gt, zrange=zrange, inv_depth=not linear, codec=codec)
        report = analyze(gt, pred)
    except Exception:
        traceback.print_exc()
        report = {}
    return report


def execute_variants(cfg: OmegaConf, gt):
    var_filter = cfg.get("variant", None)
    if isinstance(var_filter, str):
        var_filter = [var_filter]

    gen = product(MATRIX["codec"], MATRIX["zrange"], MATRIX["linear"])
    reports = []
    for codec, zrange, linear in gen:
        title = f'{codec["variant"]=}/{linear=}/{zrange=}'
        if var_filter is None or codec["variant"] in var_filter:
            print(f"running {title}")
            report = run(cfg, gt, zrange, linear, codec)
            report["variant"] = codec["variant"]
            report["zrange"] = zrange
            report["title"] = title
            reports.append(report)
        else:
            print(f"skipping {title}")
    return reports


def plot_depth(d, zrange, name):
    fig = plt.figure(figsize=plt.figaspect(1 / 2.3), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.05, 1, 1], wspace=0.1)
    ax = fig.add_subplot(gs[1])
    im = ax.imshow(d)
    ax = fig.add_subplot(gs[0])
    plt.colorbar(im, cax=ax)
    ax.yaxis.set_ticks_position("left")
    ax = fig.add_subplot(gs[2])
    im = ax.imshow(hc.depth2rgb(d, zrange=zrange))
    fig.savefig(f"tmp/{name}.png", dpi=300)
    plt.close(fig)


def main():
    cfg = OmegaConf.from_cli()
    gt = np.stack(list(generate_synthetic_depth_images(100)), 0)
    plot_depth(gt[0], (0.0, 2.0), "synthetic")

    # warmup
    hue_enc_dec(gt, (0.0, 2.0), False)

    # run variants
    reports = execute_variants(cfg, gt)

    # Format
    df = pd.DataFrame(reports)
    del df["title"]
    del df["mse"]
    del df["abs_err_std"]
    del df["abs_err_mean"]
    df = df.reindex(columns=["variant", "zrange", "rmse", "tenc", "tdec", "nbytes"])

    df["tenc"] /= len(gt) / 1e3  # msec/frame
    df["tdec"] /= len(gt) / 1e3  # msec/frame
    df["nbytes"] /= len(gt) * 1024  # kb/frame

    df = df.rename(
        columns={
            "zrange": "zrange [m]",
            "rmse": "rmse [m]",
            "tenc": "tenc [ms/img]",
            "tdec": "tdec [ms/img]",
            "nbytes": "size [kb/img]",
        }
    )

    print(df)
    print()
    print(df.to_markdown(index=False, floatfmt=("", "", ".5f", ".2f", ".2f", ".2f")))


if __name__ == "__main__":
    main()
