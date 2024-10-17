import pytest
import numpy as np

from huecodec import codec as hc


@pytest.mark.parametrize("use_lut", [True, False])
def test_enc_dec(use_lut):
    assert hc.HUE_ENCODER_MAX == 1530

    z = np.arange(0, hc.HUE_ENCODER_MAX + 1, dtype=np.uint16)
    rgb = hc.hue_encode(z)
    zr = hc.hue_decode(rgb)
    np.testing.assert_allclose(zr, z)

    r = (0.0, 2.0)
    d = np.random.rand(1024, 1024) * 2  # [0..2)
    rgb = hc.depth2rgb(d, r, inv_depth=False, use_lut=use_lut)
    dr = hc.rgb2depth(rgb, r, inv_depth=False, use_lut=use_lut)
    assert abs(dr - d).max() < 1e-3

    r = (0.1, 2.1)
    d += 0.1
    rgb = hc.depth2rgb(d, r, inv_depth=True, use_lut=use_lut)
    dr = hc.rgb2depth(rgb, r, inv_depth=True, use_lut=use_lut)

    if use_lut:
        # should be initialized
        assert hc.HUE_ENC_LUT.shape == (1531, 3)
        assert hc.HUE_DEC_LUT.shape == (256, 256, 256)


# def test_inv():

#     d = np.logspace(-2, 0, 1000)
#     rgb = hc.depth2rgb(d, (0.01, 1.1), inv_depth=True)
#     dr = hc.rgb2depth(rgb, (0.01, 1.1), inv_depth=True)

#     import matplotlib.pyplot as plt

#     plt.plot(d, abs(d - dr), label="inv")

#     # print(abs(d - dr))
#     # print(abs(d - dr) / abs(d))

#     rgb = hc.depth2rgb(d, (0.01, 1.1), inv_depth=False)
#     dr = hc.rgb2depth(rgb, (0.01, 1.1), inv_depth=False)
#     plt.plot(d, abs(d - dr), label="~inv")
#     plt.legend()
#     plt.show()
#     # print(abs(d - dr))
#     # print(abs(d - dr) / abs(d))

#     # rgb = hc.depth2rgb(d, (0.0, 1.0), inv_depth=False)
#     # dr = hc.rgb2depth(rgb, (0.0, 1.0), inv_depth=False)
#     # print(abs(d - dr))


@pytest.mark.parametrize("shape", [(1, 512), (512, 512), (10, 512, 512)])
@pytest.mark.parametrize("range", [(0.1, 1.0), (0.1, 2.0)])
@pytest.mark.parametrize("inv_depth", [True, False])
@pytest.mark.parametrize("seed", [123, 456])
def test_enc_dec_variants(seed, shape, range, inv_depth):
    rng = np.random.default_rng(seed)
    d = rng.random(shape, dtype=np.float32) * (range[1] - range[0]) + range[0]
    rgb = hc.depth2rgb(d, range, inv_depth=inv_depth)
    dr = hc.rgb2depth(rgb, range, inv_depth=inv_depth)

    err_abs = abs(dr - d)
    assert err_abs.max() < 1e-1


def test_lossy_compression():
    import av

    t = np.linspace(0, 1, 640)
    d = np.sin(2 * np.pi / 1 * t) * 0.9 + 1.0
    d = np.tile(d, (480, 1))

    # plt.imshow(d)
    # plt.show()

    def animate(d, total: int = 1000):
        for _ in range(total):
            d = np.roll(d, -1)
            yield d

    drange = (0.0, 2.0)
    din = np.stack(list(animate(d, 100)), 0)
    ded = np.stack([hc.rgb2depth(hc.depth2rgb(dd, drange), drange) for dd in din], 0)

    def d2c(gen):
        for f in gen:
            rgb = hc.depth2rgb(f, (0, 2.0))
            yield rgb

    def np2av(gen):
        for f in gen:
            yield av.VideoFrame.from_ndarray(f, format="rgb24")

    def muxit(c, frame):
        for packet in stream.encode(frame):
            c.mux(packet)

    cout = av.open("tmp/x.mp4", "w", format="mp4")
    stream = cout.add_stream(
        "libx265",
        rate=30,
        options={"x265-params": "crf=0:lossless=1:preset=veryslow:qp=0"},
    )
    stream.pix_fmt = "yuv444p10le"  # rgb->yuv420p is lossy!
    stream.width = d.shape[1]
    stream.height = d.shape[0]

    for frame in np2av(d2c(din)):
        muxit(cout, frame)
    muxit(cout, None)
    cout.close()

    container = av.open("tmp/x.mp4")

    def av2np(gen):
        for f in gen:
            f: av.VideoFrame
            rgb = f.to_rgb().to_ndarray()
            yield rgb

    def c2d(gen):
        for rgb in gen:
            d = hc.rgb2depth(rgb, drange)
            yield d

    dr = np.stack(list(c2d(av2np(container.decode(video=0)))), 0)

    assert abs(ded - din).max() < 1e-3
    assert abs(dr - din).max() < 5e-3

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 4)
    # axs[0].imshow(din[20])
    # axs[1].imshow(dr[20])
    # axs[2].imshow(abs(ded[20] - din[20]), vmin=0, vmax=0.02)
    # axs[3].imshow(abs(dr[20] - din[20]), vmin=0, vmax=0.02)
    # print(abs(ded[20] - din[20]).max())
    # print(abs(dr[20] - din[20]).max())
    # plt.show()


@pytest.mark.parametrize("shape", [(480, 640), (1080, 1920)])
def test_enc_perf(benchmark, shape):
    d = np.random.rand(*shape)

    _ = benchmark(hc.depth2rgb, d, (0.0, 1.0))


@pytest.mark.parametrize("shape", [(480, 640), (1080, 1920)])
def test_dec_perf(benchmark, shape):
    d = np.random.rand(*shape)

    e = hc.depth2rgb(d, (0.0, 1.0))
    _ = benchmark(hc.rgb2depth, e, (0.0, 1.0))
