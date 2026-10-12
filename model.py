"""Super-resolution engines, loaded through spandrel.

Part of Video Enhancer (Boring Stuff series) by Erynvorn Technologies.
https://erynvorntechnologies.com | https://github.com/ErynVornTechnologies

spandrel ships verified implementations of ESRGAN/RRDBNet, SRVGGNetCompact,
SPAN, RealPLKSR and many more, so any .pth from OpenModelDB loads without
hand-written architecture code. (The previous hand-written SRVGGNetCompact
shared ONE PReLU across all 33 layers and dropped the nearest-neighbour
residual, which is why the old "Fast" engine produced blown-out garbage.)

Engines (all weights live in models/):
  vhs2hd   2xVHS2HD (RealPLKSR, x2)        experimental, slow, softens
  span     2xLiveActionV1 (SPAN, x2)       default: keeps natural texture, fast
  compact  realesr-general-x4v3 (x4)       generic fallback, fast
  x2plus   RealESRGAN_x2plus (RRDBNet, x2) legacy, strong, can look waxy
"""
import os

import torch
import torch.nn.functional as F

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

ENGINES = {
    "vhs2hd": {
        "file": "2xVHS2HD-RealPLKSR.pth",
        "label": "VHS2HD (RealPLKSR) - experimental, slow",
        "url": "https://github.com/starinspace/StarinspaceUpscale/releases/"
               "download/Models/2xVHS2HD-RealPLKSR.pth",
        "license": "CC-BY-SA-4.0, author: asterixcool",
    },
    "span": {
        "file": "2xLiveActionV1_SPAN.pth",
        "label": "Natural detail - recommended (2xLiveActionV1, SPAN)",
        "url": "https://raw.githubusercontent.com/jcj83429/upscaling/"
               "f73a3a02874360ec6ced18f8bdd8e43b5d7bba57/2xLiveActionV1_SPAN/"
               "2xLiveActionV1_SPAN_490000.pth",
        "license": "CC-BY-NC-SA-4.0 (non-commercial), author: jcj83429",
    },
    "compact": {
        "file": "realesr-general-x4v3.pth",
        "label": "General, fast (realesr-general-x4v3)",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.2.5.0/realesr-general-x4v3.pth",
        "license": "BSD-3-Clause (Real-ESRGAN)",
    },
    "x2plus": {
        "file": "RealESRGAN_x2plus.pth",
        "label": "Real-ESRGAN x2plus (legacy, heavy)",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.2.1/RealESRGAN_x2plus.pth",
        "license": "BSD-3-Clause (Real-ESRGAN)",
    },
}


def ensure_weights(engine, log=print):
    """Return the local weights path, downloading it once if missing."""
    import urllib.request
    spec = ENGINES[engine]
    path = os.path.join(MODEL_DIR, spec["file"])
    if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(MODEL_DIR, exist_ok=True)
    log(f"[model] downloading {spec['file']} ...")
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(path + ".part", path)
    return path


class Upscaler:
    """One SR network on the GPU, fp16 where supported, batched frames.

    __call__(x): x is (B, 3, H, W) float16/32 in [0, 1] on the GPU.
    Returns (B, 3, H*scale, W*scale) float16 in [0, 1].
    """

    def __init__(self, engine, device="cuda", log=print):
        from spandrel import ModelLoader
        path = ensure_weights(engine, log)
        desc = ModelLoader(device="cpu").load_from_file(path)
        self.engine = engine
        self.arch = desc.architecture.name
        self.scale = desc.scale
        self.multiple = max(1, desc.size_requirements.multiple_of)
        self.min_size = max(1, desc.size_requirements.minimum)
        self.half = bool(desc.supports_half) and device == "cuda"
        self.dtype = torch.float16 if self.half else torch.float32
        self.net = desc.model.eval().to(device=device, dtype=self.dtype)
        self.device = device
        self.tile = 0  # 0 = whole frame; set automatically on OOM
        log(f"[model] {engine}: {self.arch} x{self.scale}, "
            f"{'fp16' if self.half else 'fp32'}")

    def _run(self, x):
        h, w = x.shape[-2:]
        ph = (-h) % self.multiple + max(0, self.min_size - h)
        pw = (-w) % self.multiple + max(0, self.min_size - w)
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        y = self.net(x)
        if ph or pw:
            y = y[..., :h * self.scale, :w * self.scale]
        return y

    def _tiled(self, x, tile, pad=24):
        b, c, h, w = x.shape
        s = self.scale
        out = torch.empty((b, c, h * s, w * s), device=x.device, dtype=x.dtype)
        for y in range(0, h, tile):
            for xx in range(0, w, tile):
                y0, x0 = max(0, y - pad), max(0, xx - pad)
                y1, x1 = min(h, y + tile + pad), min(w, xx + tile + pad)
                t = self._run(x[..., y0:y1, x0:x1])
                th, tw = min(tile, h - y), min(tile, w - xx)
                out[..., y * s:(y + th) * s, xx * s:(xx + tw) * s] = \
                    t[..., (y - y0) * s:(y - y0 + th) * s, (xx - x0) * s:(xx - x0 + tw) * s]
        return out

    @torch.inference_mode()
    def __call__(self, x):
        x = x.to(self.dtype)
        while True:
            try:
                y = self._tiled(x, self.tile) if self.tile else self._run(x)
                return y.clamp_(0, 1)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                h, w = x.shape[-2:]
                self.tile = (max(h, w) // 2) if not self.tile else self.tile // 2
                if self.tile < 128:
                    raise
