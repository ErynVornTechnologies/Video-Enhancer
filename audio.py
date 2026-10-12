"""Audio restoration for old tapes (VHS / Hi8 / camcorder) - Video Enhancer.

Part of the Boring Stuff series by Erynvorn Technologies.
https://erynvorntechnologies.com | https://github.com/ErynVornTechnologies

Modes:
  off       audio copied / re-encoded untouched (previous behaviour)
  clean     classic DSP only, never changes voices: de-clip (hard-clipped
            peaks), de-click, rumble high-pass, automatic 50 Hz hum notch
            (only if hum is detected), tape-hiss reduction (afftdn), then
            level evening (gentle compressor + EBU R128 loudness + limiter)
  voice     clean + DeepFilterNet (speech denoiser, CPU, ~7x real time).
            An attenuation limit keeps some background, so music and
            ambience are reduced, not deleted.
  voice-hq  clean + ClearerVoice MossFormer2_SE_48K (GPU), blended with the
            classic result so non-speech sounds survive. Separate venv.

Everything runs in 32-bit float with 3 dB headroom: in tests, processing a
16-bit file that already peaks at -0.3 dB pushed the output into clipping.
"""
import os
import shutil
import subprocess
import time

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe" if os.path.exists(r"C:\ffmpeg\bin\ffmpeg.exe") else "ffmpeg"
NOWIN = 0x08000000 if os.name == "nt" else 0
DFN_EXE = os.path.join(APP_DIR, "tools", "deep-filter.exe")
CV_PY = os.path.join(APP_DIR, ".venv_audio", "Scripts", "python.exe")
CV_WORKER = os.path.join(APP_DIR, "cv_worker.py")
SR = 48000

MODES = {
    "off": "Off (keep original sound)",
    "clean": "Clean tape: hiss, clicks, hum, clipping, levels",
    "voice": "Clean + AI voice (DeepFilterNet)",
    "voice-hq": "Clean + AI voice HQ (ClearerVoice, GPU)",
}


def available_modes():
    m = ["off", "clean"]
    if os.path.isfile(DFN_EXE):
        m.append("voice")
    if os.path.isfile(CV_PY) and os.path.isfile(CV_WORKER):
        m.append("voice-hq")
    return m


def _run(args):
    r = subprocess.run(args, capture_output=True, text=True, errors="replace", creationflags=NOWIN)
    if r.returncode != 0:
        raise RuntimeError(f"{os.path.basename(args[0])} failed:\n{(r.stderr or r.stdout)[-1500:]}")
    return r


def _run_live(args, on_line=None, ticker=None):
    """Run a tool, feed each stdout line to on_line(line); ticker() every second.
    stderr is drained in a thread (a full pipe would otherwise deadlock)."""
    import threading
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         errors="replace", bufsize=1, creationflags=NOWIN)
    err = []
    threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True).start()
    if ticker:
        def tick():
            while p.poll() is None:
                ticker()
                time.sleep(1)
        threading.Thread(target=tick, daemon=True).start()
    for line in p.stdout:
        if on_line:
            on_line(line.strip())
    p.wait()
    time.sleep(0.1)
    if p.returncode != 0:
        raise RuntimeError(f"{os.path.basename(args[0])} failed:\n{''.join(err)[-1500:]}")


def _probe_dur(path):
    d = os.path.dirname(FFMPEG)
    ffprobe = os.path.join(d, "ffprobe.exe" if os.name == "nt" else "ffprobe") if d else "ffprobe"
    r = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, creationflags=NOWIN)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


class _Progress:
    """Maps stage progress to one overall fraction + a readable status line."""

    def __init__(self, cb, stages):
        self.cb, self.stages, self.t0 = cb, stages, time.time()

    def __call__(self, i, frac, eta=None):
        if not self.cb:
            return
        frac = max(0.0, min(1.0, frac))
        overall = (i + frac) / len(self.stages)
        el = time.time() - self.t0
        if eta is None and overall > 0.02:
            eta = el / overall * (1 - overall)
        eta_s = f"   ETA {eta / 60:.0f} min" if eta and eta > 60 else (f"   ETA {eta:.0f} s" if eta else "")
        self.cb(f"Audio {i + 1}/{len(self.stages)}: {self.stages[i]}  {frac * 100:.0f}%{eta_s}", overall)


def _ffmpeg_progress(total_s, report):
    """on_line handler for ffmpeg -progress pipe:1 output."""
    def on_line(line):
        if line.startswith(("out_time_us=", "out_time_ms=")) and total_s > 0:
            try:
                report(int(line.split("=")[1]) / 1e6 / total_s)
            except ValueError:
                pass
    return on_line


def detect_hum(path, start=0.0, seconds=60):
    """-> mains frequency (50 or 60) if a hum is clearly audible, else None."""
    cmd = [FFMPEG, "-v", "error"] + (["-ss", str(start)] if start else []) + [
        "-i", path, "-t", str(seconds), "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, creationflags=NOWIN).stdout
    a = np.frombuffer(raw, np.int16).astype(np.float32)
    if len(a) < 8000 * 5:
        return None
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    f = np.fft.rfftfreq(len(a), 1 / 8000)

    def band(c, bw=1.5):
        return spec[(f > c - bw) & (f < c + bw)].mean() + 1e-9

    best = None
    for mains in (50, 60):
        # fundamental or 2nd harmonic clearly above the neighbouring spectrum
        score = max(band(h) / ((band(h - 8) + band(h + 8)) / 2) for h in (mains, 2 * mains))
        if 20 * np.log10(score) > 6 and (best is None or score > best[1]):
            best = (mains, score)
    return best[0] if best else None


AFFTDN_DELAY = 1200  # samples at 48 kHz, measured by cross-correlation (= 25 ms)


def _pre_chain(hum, hiss_nr):
    # resample FIRST so afftdn always runs at 48 kHz, where its delay is a known
    # 1200 samples; trimming them keeps lip sync exact (measured: 0 samples)
    f = [f"aresample={SR}", "aformat=sample_fmts=flt", "volume=-3dB",   # float + headroom
         "adeclip", "adeclick", "highpass=f=40"]
    if hum:
        f += [f"bandreject=f={hum * k}:width_type=q:w=12" for k in (1, 2, 3)]
    f.append(f"afftdn=nr={hiss_nr}:nf=-45:tn=1")
    f.append(f"atrim=start_sample={AFFTDN_DELAY},asetpts=PTS-STARTPTS")
    return ",".join(f)


POST_CHAIN = ",".join([
    # gentle: a stronger setting (thr -20 dB, ratio 2.5, +2 dB makeup) lifted the
    # tape noise floor by ~3 dB in tests, cancelling the hiss reduction
    "acompressor=threshold=-14dB:ratio=2:attack=20:release=300",
    "loudnorm=I=-16:TP=-1.5:LRA=11",          # TV-friendly loudness (EBU R128 style)
    f"aresample={SR}",                         # loudnorm works at 192 kHz internally
    "alimiter=limit=0.84:level=false:latency=1",  # ceiling ~-1.5 dBFS; latency=1 removes its 5 ms delay
])


def process(src, dst, mode="clean", start=None, duration=None, log=print,
            dfn_atten_db=18, cv_mix=0.85, progress=None):
    """Restore the audio of `src` into `dst` (float WAV, 48 kHz stereo).

    start/duration trim exactly like the video so A/V stay aligned.
    Returns dst. Raises if a requested AI stage is missing or fails.
    """
    if mode not in MODES or mode == "off":
        raise ValueError(f"audio mode {mode!r}: nothing to do")
    t0 = time.time()
    tmp = dst + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    names = ["tape clean-up (DSP)"]
    if mode == "voice":
        names.append("AI voice (DeepFilterNet, CPU)")
    elif mode == "voice-hq":
        names.append("AI voice HQ (ClearerVoice, GPU)")
    names.append("levels + loudness")
    prog = _Progress(progress, names)
    total_s = duration or max(0.0, _probe_dur(src) - (start or 0.0))
    ffp = ["-progress", "pipe:1", "-nostats"]
    try:
        hum = detect_hum(src, start or 0.0)
        hiss = 20 if mode == "clean" else 6  # AI modes: lighter DSP, the model does the rest
        log(f"[audio] mode={mode}, {total_s / 60:.1f} min, "
            f"hum={'%d Hz notch' % hum if hum else 'none detected'}")
        pre = os.path.join(tmp, "pre.wav")
        trim = (["-ss", str(start)] if start else [])
        prog(0, 0)
        _run_live([FFMPEG, "-v", "error", "-y"] + trim + ["-i", src]
                  + (["-t", f"{duration:.3f}"] if duration else [])
                  + ["-vn", "-ac", "2", "-ar", str(SR), "-af", _pre_chain(hum, hiss),
                     "-c:a", "pcm_f32le"] + ffp + [pre],
                  on_line=_ffmpeg_progress(total_s, lambda f: prog(0, f)))
        stage = pre
        if mode == "voice":
            if not os.path.isfile(DFN_EXE):
                raise RuntimeError("DeepFilterNet not installed (tools/deep-filter.exe)")
            out_dir = os.path.join(tmp, "dfn")
            os.makedirs(out_dir, exist_ok=True)
            # deep-filter prints no progress: estimate from its measured speed
            # (~7.5x real time on CPU); the bar stops at 99 % until it finishes
            t_dfn, expect = time.time(), max(1.0, total_s / 7.5)
            _run_live([DFN_EXE, "-D", "-a", str(dfn_atten_db), "-o", out_dir, pre],
                      ticker=lambda: prog(1, min(0.99, (time.time() - t_dfn) / expect),
                                          max(0.0, expect - (time.time() - t_dfn))))
            stage = os.path.join(out_dir, "pre.wav")
        elif mode == "voice-hq":
            if not os.path.isfile(CV_PY):
                raise RuntimeError("ClearerVoice environment not installed (.venv_audio)")
            ai = os.path.join(tmp, "cv.wav")

            def cv_line(line):
                if line.startswith("[cv-progress]"):
                    d, t, eta = line.split()[1:4]
                    prog(1, int(d) / int(t), float(eta))
                elif line.startswith("[cv]"):
                    log(line)
            prog(1, 0)
            _run_live([CV_PY, CV_WORKER, pre, ai], on_line=cv_line)
            # blend: AI voice + a little of the classic result keeps music/ambience
            _run([FFMPEG, "-v", "error", "-y", "-i", ai, "-i", pre, "-filter_complex",
                  f"[0:a]volume={cv_mix}[a];[1:a]volume={1 - cv_mix:.2f}[b];"
                  f"[a][b]amix=inputs=2:normalize=0", "-c:a", "pcm_f32le",
                  os.path.join(tmp, "mix.wav")])
            stage = os.path.join(tmp, "mix.wav")
        # AI stages can return a few ms less (DeepFilterNet: ~30 ms), which would
        # make -shortest drop the last video frame: pad/trim to the exact length
        # target = the VIDEO length when known (pads a slightly shorter source
        # soundtrack with silence), else the source length before the afftdn trim
        n = os.path.getsize(pre) - 44  # float32 stereo WAV, 44-byte header (approx.)
        exact = duration if duration else max(0.0, n / (SR * 2 * 4) + AFFTDN_DELAY / SR)
        # apad must come BEFORE loudnorm: placed after it, it was ignored (tested)
        last = len(names) - 1
        prog(last, 0)
        _run_live([FFMPEG, "-v", "error", "-y", "-i", stage, "-af",
                   f"apad=whole_len={int(round(exact * SR))},{POST_CHAIN}", "-t", f"{exact:.4f}",
                   "-ar", str(SR), "-c:a", "pcm_f32le"] + ffp + [dst],
                  on_line=_ffmpeg_progress(exact, lambda f: prog(last, f)))
        prog(last, 1.0, 0)
        log(f"[audio] done in {time.time() - t0:.1f}s")
        return dst
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":  # quick manual test: python audio.py in.wav out.wav mode
    import sys
    process(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "clean")
