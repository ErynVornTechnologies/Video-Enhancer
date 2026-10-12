"""Smoke tests: every engine loads and keeps brightness; an interlaced
synthetic clip is detected, deinterlaced to double rate and upscaled.
Run:  py -3.12 test_enhancer.py
"""
import os
import subprocess
import tempfile

import torch

import enhancer
from model import ENGINES, Upscaler


def test_engines():
    x = torch.rand(1, 3, 96, 128, device="cuda") * 0.6 + 0.2
    for k in ENGINES:
        up = Upscaler(k, log=lambda *_: None)
        y = up(x)
        assert y.shape[-2:] == (96 * up.scale, 128 * up.scale), k
        diff = abs(y.float().mean().item() - x.mean().item())
        assert diff < 0.05, f"{k}: brightness drift {diff:.3f}"  # old 'fast' bug: ~80
        print(f"ok  engine {k}: {up.arch} x{up.scale}, drift {diff:.4f}")


def test_pipeline():
    d = tempfile.mkdtemp()
    src, out = os.path.join(d, "i.mpg"), os.path.join(d, "o.mp4")
    subprocess.run([enhancer.FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=720x576:rate=50,interlace=scan=tff,setsar=16/15",
                    "-f", "lavfi", "-i", "sine=f=440", "-t", "2", "-c:v", "mpeg2video",
                    "-flags", "+ilme+ildct", "-top", "1", "-c:a", "mp2", src], check=True)
    r = enhancer.enhance_video(src, out, mode="hd", engine="span", log=lambda *_: None)
    i = enhancer.probe(out)
    assert (i["w"], i["h"]) == (1440, 1080), (i["w"], i["h"])  # 4:3 -> square pixels
    assert float(i["fps"]) == 50.0, i["fps"]                  # double-rate deinterlace
    assert i["audio"] == "aac" and r["frames"] >= 95
    print(f"ok  pipeline: 576i -> 1440x1080p50 + audio, {r['fps']:.1f} f/s")


if __name__ == "__main__":
    test_engines()
    test_pipeline()
    print("ALL TESTS PASSED")
