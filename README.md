# hue-encoding
This project provides efficient vectorized Python code to perform depth `<->` color encoding based on

> Sonoda, Tetsuri, and Anders Grunnet-Jepsen.
"Depth image compression by colorization for Intel RealSense depth cameras." Intel, Rev 1.0 (2021).

Here's the encoding of depth values from a moving sine wave.

https://github.com/user-attachments/assets/2814d21a-3d1b-4857-b415-dc1c5ae31460

## Installation

With Python >= 3.11 execute

```shell
git clone https://github.com/cheind/hue-depth-encoding.git

# default install requires only numpy
pip install -e .
# development install adds more packages to run tests/analysis
pip install -e '.[dev]'
# or with gpu codec support in dev mode
pip install -e '.[dev]' --no-binary=av 
```

## Properties

The encoding is designed to transform 16bit single channel images to RGB color images that can be processed by standard (lossy) image codec with minimal compression artefacts. This leads to a compression factor of up to 80x.

## Method

The compression first transforms raw depth values to the applicable encoding range [0..1530] (~10.5bit) via a normalization strategy (linear/disparity) using a user provided near/far value. Next, for each encoding value the corresponding color is computed. The color is chosen carefully to respect the aforementioned properties. Upon decoding the reverse process is computed.

### Depth transformations
The encoding allows for linear and disparity depth normalization. In linear mode, equal depth ratios are preserved in the encoding range [0..1530], whereas in disparity mode more emphasis is put on closer depth values than on larger ones, leading to more accurare depth resolution closeup.

![](etc/compare_encoding.svg)


## Implementation

This implementation is vectorized using numpy and can handle any image shapes `(*,H,W) <-> (*,H,W,3)`. For improved performance, we precompute encoder and decoder lookup tables to reduce encoding/decoding to a simple lookup. The lookup tables require ~32MB of memory. Use `use_lut=False` switch to rely on the pure vectorized implementation. See benchmarks below for effects.

## Usage

There is currently no PyPi package to install, however the code is contained in a single file for easy of distribution :)

```python
# Import
import huecodec as hc

# Random float depths
d = rng.random((5,240,320), dtype=np.float32)*0.9 + 0.1

# Encode
rgb = hc.depth2rgb(d, zrange=(0.1,1.0), inv_depth=False)
# (5,240,320,3), uint8

# Decode
depth = hc.rgb2depth(rgb, zrange=(0.1,1.0), inv_depth=False)
# (5,240,320), float32
```

## Evaluation

### Encoding/Decoding Roundtrips

The script `python analysis.py` compares encoding and decoding characteristics for different standard video codecs using hue depth encoding. The reported figures have the following meaning:

 - **rmse** [m] root mean square error per depth pixel between groundtruth and transcoded depthmaps
 - **<x** [%] percent of absolute errors less than x
 - **tenc** [milli-sec/frame] encoding time per frame
 - **tdec** [milli-sec/frame] decoding time per frame
 - **nbytes** [kb/frame] kilo-bytes per encoded frame on disk.

All tests are carried out on a 12th Gen Intel® Core™ i9-12900K × 24 with NVIDIA GeForce RTX™ 3090 Ti.

```shell
# run analysis only for specific variants
python analysis.py variant=[hue-only,h264-lossless-gpu]
```

#### Synthetic Depthmaps

Each test encodes/decodes a sequence `(100,512,512)` of np.float32 depthmaps in range `[0..2]` containing a sinusoidal pattern plus random hard depth edges. The pattern moves horizontally over time.

![](etc/synthetic.png)

The reported values are
| variant           | zrange [m]   |   rmse [m] |   <1mm [%] |   <5mm [%] |   <1cm [%] |   tenc [ms/img] |   tdec [ms/img] |   size [kb/img] |
|:------------------|:-------------|-----------:|-----------:|-----------:|-----------:|----------------:|----------------:|----------------:|
| hue-only          | (0.0, 2.0)   |    0.00033 |       1.00 |       1.00 |       1.00 |            1.90 |            1.29 |          768.00 |
| hue-only          | (0.0, 4.0)   |    0.00067 |       0.82 |       1.00 |       1.00 |            1.90 |            1.23 |          768.00 |
| h264-lossless-cpu | (0.0, 2.0)   |    0.00033 |       1.00 |       1.00 |       1.00 |            5.87 |            2.92 |           31.97 |
| h264-lossless-cpu | (0.0, 4.0)   |    0.00067 |       0.82 |       1.00 |       1.00 |            6.67 |            2.76 |           26.30 |
| h264-default-cpu  | (0.0, 2.0)   |    0.09704 |       0.29 |       0.54 |       0.73 |            4.88 |            1.87 |           11.81 |
| h264-default-cpu  | (0.0, 4.0)   |    0.12807 |       0.28 |       0.53 |       0.72 |            6.02 |            1.85 |            9.51 |
| h264-lossless-gpu | (0.0, 2.0)   |    0.00033 |       1.00 |       1.00 |       1.00 |            2.60 |            2.08 |           70.97 |
| h264-lossless-gpu | (0.0, 4.0)   |    0.00067 |       0.82 |       1.00 |       1.00 |            2.60 |            1.84 |           30.75 |
| h264-tuned-gpu    | (0.0, 2.0)   |    0.16941 |       0.84 |       0.99 |       0.99 |            2.59 |            2.02 |           29.94 |
| h264-tuned-gpu    | (0.0, 4.0)   |    0.10816 |       0.67 |       1.00 |       1.00 |            2.60 |            1.89 |           16.07 |
| h265-lossless-gpu | (0.0, 2.0)   |    0.00033 |       1.00 |       1.00 |       1.00 |            2.59 |            2.83 |           36.62 |
| h265-lossless-gpu | (0.0, 4.0)   |    0.00067 |       0.82 |       1.00 |       1.00 |            2.60 |            2.81 |           33.63 |
| h264-default-gpu  | (0.0, 2.0)   |    0.09996 |       0.29 |       0.54 |       0.73 |            2.58 |            2.63 |           18.38 |
| h264-default-gpu  | (0.0, 4.0)   |    0.12741 |       0.28 |       0.52 |       0.72 |            2.57 |            2.30 |           13.42 |

#### Real Depthmaps

Each tests encodes a sequence of `(30,600,800)` of np.float32 depthmaps taken with a RealSense 415 in range `[0..2]` containing a sitting person moving.

![](etc/real.png)

The reported values are

| variant           | zrange [m]   |   rmse [m] |   <1mm [%] |   <5mm [%] |   <1cm [%] |   tenc [ms/img] |   tdec [ms/img] |   size [kb/img] |
|:------------------|:-------------|-----------:|-----------:|-----------:|-----------:|----------------:|----------------:|----------------:|
| hue-only          | (0.0, 2.0)   |    0.00023 |       1.00 |       1.00 |       1.00 |            2.00 |            1.31 |          900.00 |
| hue-only          | (0.0, 4.0)   |    0.00047 |       0.91 |       1.00 |       1.00 |            2.06 |            1.35 |          900.00 |
| h264-lossless-cpu | (0.0, 2.0)   |    0.00023 |       1.00 |       1.00 |       1.00 |           10.64 |            4.34 |           87.96 |
| h264-lossless-cpu | (0.0, 4.0)   |    0.00047 |       0.91 |       1.00 |       1.00 |           10.48 |            3.84 |           67.15 |
| h264-default-cpu  | (0.0, 2.0)   |    0.14033 |       0.70 |       0.96 |       0.97 |           11.02 |            2.79 |           44.25 |
| h264-default-cpu  | (0.0, 4.0)   |    0.26466 |       0.64 |       0.88 |       0.95 |           10.48 |            2.54 |           37.43 |
| h264-lossless-gpu | (0.0, 2.0)   |    0.00023 |       1.00 |       1.00 |       1.00 |            2.85 |            2.66 |           92.63 |
| h264-lossless-gpu | (0.0, 4.0)   |    0.00047 |       0.91 |       1.00 |       1.00 |            2.84 |            2.60 |           76.17 |
| h264-tuned-gpu    | (0.0, 2.0)   |    0.22042 |       0.87 |       0.98 |       0.98 |            2.85 |            2.43 |           61.44 |
| h264-tuned-gpu    | (0.0, 4.0)   |    0.29478 |       0.74 |       0.97 |       0.98 |            2.83 |            2.31 |           44.42 |
| h265-lossless-gpu | (0.0, 2.0)   |    0.00023 |       1.00 |       1.00 |       1.00 |            2.84 |            5.26 |           81.33 |
| h265-lossless-gpu | (0.0, 4.0)   |    0.00047 |       0.91 |       1.00 |       1.00 |            2.84 |            4.78 |           69.25 |
| h264-default-gpu  | (0.0, 2.0)   |    0.15737 |       0.69 |       0.95 |       0.97 |            2.85 |            3.89 |           35.48 |
| h264-default-gpu  | (0.0, 4.0)   |    0.29960 |       0.64 |       0.86 |       0.95 |            2.83 |            3.74 |           33.47 |

```
python analysis.py data=path/to/npy
```

### Hue Runtime Benchmark

Here are benchmark results for encoding/decoding float32 depthmaps of various sizes with differnt characteristics. Note, this is pure depth -> color -> depth transcoding without any video codecs involved.

```
------------- benchmark: 8 tests ------------
Name (time in ms)              Mean          
---------------------------------------------
enc_perf[LUT-(640x480)]        2.1851 (1.0)    
dec_perf[LUT-(640x480)]        2.2124 (1.01)   
enc_perf[LUT-(1920x1080)]     17.9139 (8.20) 
dec_perf[LUT-(1920x1080)]     16.4741 (7.54)   
enc_perf[noLUT-(640x480)]     22.3938 (10.25)  
dec_perf[noLUT-(640x480)]      6.9320 (3.17)   
dec_perf[noLUT-(1920x1080)]   74.6038 (34.14)  
enc_perf[noLUT-(1920x1080)]  158.0871 (72.35)  
---------------------------------------------
```

```shell
# run tests and benchmarks
pytest
```

### Note on Video Codecs

When tuning the prameters for a video codec you need to take into account lossy compression that happens on  different levels of encoding:
 - **spatial/temporal** quantization based how images change over time. Usually controlled by the codec `preset/profile` (lossless, low-latency,...) and quality parameters such as `crf`, `qp`.
 - **color space** quantization due to converting RGB to target pixel format. Different codecs support different pixel color formats of which most perform a lossy compression from rgb to target space. 

Print encoder supported options and pixel formats
```shell
# print encoder options
ffmpeg -h encoder=h264_nvenc
```

See packing and compression of color formats
https://github.com/FFmpeg/FFmpeg/blob/master/libavutil/pixfmt.h


### Takeaways
Here are some take aways to consider
 - adjust zrange as tightly as possible to your use-case. This increases filesize but reduces reconstruction loss.
 - prefer loss-less codecs if affordable
 - when using lossy codecs ensure that your application does not rely on depth-edges.

The following plot shows a lossy h264 encoding on real data. Most of the errors are sub-mm, only few > 1cm. 

![](etc/h264-tuned-gpu.2.hist.png)

Turns out these errors are located on depth edges where the lossy codec starts interpolating values.
 

## Notes

The original paper referenced is potentially inaccurate in its equations. This has been noted in varios posts [#10415](https://github.com/IntelRealSense/librealsense/issues/10145),[#11187](https://github.com/IntelRealSense/librealsense/issues/11187),[#10302](https://github.com/IntelRealSense/librealsense/issues/10302).

This implementation is based on the original paper and code from
https://github.com/jdtremaine/hue-codec/.