import numpy as np
import matplotlib.pyplot as plt

import huecodec as hc

SIZE_DEFAULT = 8
SIZE_LARGE = 10
plt.rc("font", family="Roboto")  # controls default font
plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_DEFAULT)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("figure", titlesize=SIZE_LARGE)  # fontsize of the tick labels


def plot_linear_vs_disparity():

    depth = np.linspace(10, 50, 100)

    near_lin = 10
    far_lin = 50

    near_disp = 10
    far_disp = 50

    disp = np.clip(
        (1 / depth - 1 / near_disp) / (1 / far_disp - 1 / near_disp), 0.0, 1.0
    )
    lin = np.clip((depth - near_lin) / (far_lin - near_lin), 0.0, 1.0)

    z_disp = np.round(disp * hc.HUE_ENCODER_MAX).astype(np.uint16)
    z_lin = np.round(lin * hc.HUE_ENCODER_MAX).astype(np.uint16)

    e_disp = hc.hue_encode(z_disp)
    e_lin = hc.hue_encode(z_lin)

    z_disp_r = hc.hue_decode(e_disp)
    z_lin_r = hc.hue_decode(e_lin)

    depth_from_disp = 1 / (
        (z_disp_r / hc.HUE_ENCODER_MAX) * (1 / far_disp - 1 / near_disp) + 1 / near_disp
    )
    depth_from_lin = (z_lin_r / hc.HUE_ENCODER_MAX) * (far_lin - near_lin) + near_lin

    fig = plt.figure(figsize=(10, 6), layout="constrained")
    fig.suptitle(
        f"Encoder/Decoder comparison for linear/disparity variants.\nnear {near_lin} / far {far_lin}"
    )
    gs = fig.add_gridspec(3, 2, height_ratios=(3, 1, 1))

    ax = fig.add_subplot(gs[0, 0])
    # Transformed values
    ax.plot(depth, disp, label="disparity")
    ax.scatter(depth[::5], disp[::5], s=4)
    ax.plot(depth, lin, label="linear")
    ax.scatter(depth[::5], lin[::5], s=4)
    ax.set_xlim(depth.min() - 0.5, depth.max() - 0.5)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("depth")
    ax.set_ylabel("normalized depth")
    ax.set_title("Depth normalization")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    # Transformed values
    ax.plot(depth, abs(depth_from_disp - depth), label="disparity")
    ax.plot(depth, abs(depth_from_lin - depth), label="linear")
    ax.set_xlim(depth.min(), depth.min() + 40)
    ax.set_ylim(0, 0.1)
    ax.set_xlabel("depth")
    ax.set_ylabel("absolute depth error")
    ax.set_title("Encoding/Decoding error")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[1, :])
    ax.set_title("Hue encoding: linear")
    ax.imshow(e_lin.reshape(1, -1, 3), extent=(depth.min(), depth.max(), 0, 1))
    ax.set_aspect("auto")

    ax = fig.add_subplot(gs[2, :])
    ax.set_title("Hue encoding: disparity")
    ax.imshow(e_disp.reshape(1, -1, 3), extent=(depth.min(), depth.max(), 0, 1))
    ax.set_aspect("auto")
    ax.set_xlabel("depth")

    fig.savefig("etc/compare_encoding.svg")
    fig.savefig("etc/compare_encoding.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    plot_linear_vs_disparity()
