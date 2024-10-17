""" Hue-based depth to color compression codec.

This library offers an efficient encoder/decoder that converts 16-bit 
single-channel data into an 8-bit, three-channel color format. The encoding 
reduces the data to approximately 10.5 bits while preserving the key 
advantage of allowing further compression using standard lossy codecs with 
minimal compression artifacts.


Christoph Heindl, 2024/10, 
https://github.com/cheind/hue-depth-encoding

References:
        Sonoda, Tetsuri, and Anders Grunnet-Jepsen.
        "Depth image compression by colorization for Intel RealSense depth
        cameras." Intel, Rev 1.0 (2021).
"""

import numpy as np

HUE_ENCODER_MAX = 1530
HUE_ENC_LUT: np.ndarray = None
HUE_DEC_LUT: np.ndarray = None


def hue_encode(z: np.ndarray) -> np.ndarray:
    """Encode an uint16 depth map into an RGB uint8 color map.

    In production use `hue_encode_lut` which uses a pre-computed
    lookup table.

    Params:
        z: (*,) uint16 array in range [0,HUE_ENCODER_MAX]

    Returns:
        rgb: (*,3) uint8 array

    References:
        Sonoda, Tetsuri, and Anders Grunnet-Jepsen.
        "Depth image compression by colorization for Intel RealSense depth
        cameras." Intel, Rev 1.0 (2021).
    """

    # Increase depth is mapped to 6 different HUE gradients. First and last
    # bins are used to capture 0 and >MAX.
    idx = (
        np.digitize(
            z,
            [0, 1, 256, 511, 766, 1021, 1276, 1531, np.iinfo(np.uint16).max],
        )
        - 1
    )

    o = np.empty(z.shape + (3,), dtype=np.uint8)

    # fmt: off
    m = idx == 0; o[m,0], o[m,1], o[m,2] = 0, 0, 0              # 0 <= z < 1             ; black
    m = idx == 1; o[m,0], o[m,1], o[m,2] = 255, z[m] - 1, 0     # 1 <= z < 256           ; red -> yellow
    m = idx == 2; o[m,0], o[m,1], o[m,2] = 511 - z[m], 255, 0   # 256 <= z < 512         ; yellow -> green
    m = idx == 3; o[m,0], o[m,1], o[m,2] = 0, 255, z[m]-511     # 512 <= z < 766         ; green -> mint
    m = idx == 4; o[m,0], o[m,1], o[m,2] = 0, 1021-z[m], 255    # 766 <= z < 1021        ; mint -> blue
    m = idx == 5; o[m,0], o[m,1], o[m,2] = z[m]-1021, 0, 255    # 1021 <= z < 1276       ; blue -> magenta
    m = idx == 6; o[m,0], o[m,1], o[m,2] = 255, 0, 1531-z[m]    # 1276 <= z < 1531       ; magenta -> red
    m = idx == 7; o[m,0], o[m,1], o[m,2] = 255, 0, 0            # 1531 <= z < max        ; red
    # fmt:on

    return o


def hue_decode(rgb: np.ndarray) -> np.ndarray:
    """Decode a hue-encoded RGB uint8 image into a uint16 depthmap.

    In production use `hue_decode_lut` which uses a pre-computed
    lookup table.

     Params:
        rgb: (*,3) uint8 RGB array in range [0,255]

    Returns:
        z: (*,) uint16 array in range [0,HUE_ENCODER_MAX]

    References:
        Sonoda, Tetsuri, and Anders Grunnet-Jepsen.
        "Depth image compression by colorization for Intel RealSense depth
        cameras." Intel, Rev 1.0 (2021).
    """
    assert rgb.shape[-1] == 3

    r, g, b = np.split(rgb.astype(int), 3, -1)

    z = np.empty(rgb.shape[:2], dtype=np.uint16)

    not_zero = (rgb > 0).any(-1, keepdims=True)

    # The follwing conditions are overlapping
    # but since we generally use LUT, the
    # performance hit is irrelevant
    c1 = (r >= g) & (r >= b) & (g >= b)  # r largest, then g
    c2 = (r >= g) & (r >= b) & (g < b)  # r largest, then b
    c3 = (g >= r) & (g >= b) & ~(c1 | c2)  # g largest
    c4 = (b >= r) & (b >= r) & ~(c1 | c2 | c3)  # b largest

    z = (
        (g - b + 1) * (not_zero & c1)
        + (g - b + 1531) * (not_zero & c2)
        + (b - r + 511) * (not_zero & c3)
        + (r - g + 1021) * (not_zero & c4)
    )
    return z.astype(np.uint16).squeeze(-1)


def _create_enc_lut():
    z = np.arange(0, HUE_ENCODER_MAX + 1, dtype=np.uint16)
    return hue_encode(z)


def hue_encode_lut(z: np.ndarray) -> np.ndarray:
    """Encode an uint16 depth map into an RGB uint8 color map.

    Params:
        z: (*,) uint16 array in range [0,HUE_ENCODER_MAX]

    Returns:
        rgb: (*,3) uint8 array

    References:
        Sonoda, Tetsuri, and Anders Grunnet-Jepsen.
        "Depth image compression by colorization for Intel RealSense depth
        cameras." Intel, Rev 1.0 (2021).
    """
    global HUE_ENC_LUT
    if HUE_ENC_LUT is None:
        HUE_ENC_LUT = _create_enc_lut()

    return np.ascontiguousarray(HUE_ENC_LUT[z])


def _create_dec_lut():
    rgb = np.stack(
        np.meshgrid(
            np.arange(256, dtype=np.uint8),
            np.arange(256, dtype=np.uint8),
            np.arange(256, dtype=np.uint8),
            indexing="ij",
        ),
        -1,
    )

    return hue_decode(rgb)


def hue_decode_lut(rgb: np.ndarray) -> np.ndarray:
    """Decode a hue-encoded RGB uint8 image into a uint16 depthmap.

    In production use `hue_decode_lut` which uses a pre-computed
    lookup table.

     Params:
        rgb: (*,3) uint8 RGB array in range [0,255]

    Returns:
        z: (*,) uint16 array in range [0,HUE_ENCODER_MAX]

    References:
        Sonoda, Tetsuri, and Anders Grunnet-Jepsen.
        "Depth image compression by colorization for Intel RealSense depth
        cameras." Intel, Rev 1.0 (2021).
    """
    global HUE_DEC_LUT
    if HUE_DEC_LUT is None:
        HUE_DEC_LUT = _create_dec_lut()

    return np.ascontiguousarray(HUE_DEC_LUT[rgb[..., 0], rgb[..., 1], rgb[..., 2]])


def depth2rgb(
    d: np.ndarray,
    zrange: tuple[float, float],
    inv_depth: bool = False,
    use_lut: bool = True,
):
    """Compress depth to RGB

    The colorization process requires fitting a 16-bit depth map into a 10.5-bit color image.
    We limit the depth range to a subset of the 0-65535 range and re-normalize before colorization.

    With disparity encoding we actually encode 1/depth with the property that for closer depths
    the quantization is finer and coarser for larger depth values. Note that NaN and inf values
    are mapped to zrange min.

    Params:
        d: (*,) depth map
        zrange: clipping range for depth values before normalization to [0..HUE_ENCODER_MAX]
        inv_depth: colorizes 1/depth with finer quantization for closer depths
        use_lut: Wether to use fast lookup tables or compute on the fly

    Returns:
        rgb: color encoded depth map
    """

    if inv_depth:
        assert zrange[0] > 0, "zmin==0 not handled for inverse depth encoding"

        zmin = 1 / zrange[1]
        zmax = 1 / zrange[0]
        zrange = (zmin, zmax)
        with np.errstate(divide="ignore"):
            d = np.where(d > 0, 1 / d, zmax)

    d = np.clip((d - zrange[0]) / (zrange[1] - zrange[0]), 0.0, 1.0)
    z = np.round(HUE_ENCODER_MAX * d).astype(np.uint16)
    return hue_encode_lut(z) if use_lut else hue_encode(z)


def rgb2depth(
    rgb: np.ndarray,
    zrange: tuple[float, float],
    inv_depth: bool = False,
    use_lut: bool = True,
):
    """Decompress RGB to depth

    See `depth2rgb` for explanation of parameters.

    Params:
        rgb: (*,3) color map
        zrange: zrange used during compression
        inv_depth: wether depth or disparity was encoded
        use_lut: Wether to use fast lookup tables or compute on the fly.
            LUT is initialized lazily on first run, hence first run may take
            longer.

    Returns:
        rgb: color encoded depth map
    """
    z = hue_decode_lut(rgb) if use_lut else hue_decode(rgb)
    z = z.astype(np.float32)
    if inv_depth:
        assert zrange[0] > 0, "zmin==0 not handled for inverse depth encoding"
        zmin = 1 / zrange[1]
        zmax = 1 / zrange[0]
        zrange = (zmin, zmax)

    d = (z / HUE_ENCODER_MAX) * (zrange[1] - zrange[0]) + zrange[0]

    if inv_depth:
        with np.errstate(divide="ignore"):
            d = np.where(d > 0, 1 / d, zrange[1])
    return d


if __name__ == "__main__":
    print(HUE_DEC_LUT.nbytes)
