"""Video Enhancer - deinterlace + AI super-resolution for old footage.

Part of the Boring Stuff series by Erynvorn Technologies.
https://erynvorntechnologies.com | https://github.com/ErynVornTechnologies

Pipeline (one pass, nothing uncompressed touches the disk):
  ffmpeg decode -> [bwdif deinterlace, double rate] -> [crop] -> [hqdn3d]
  -> correct BT.601/709 matrix -> rgb24 pipe
  -> GPU: SR model (x2/x4, fp16, batched) -> one antialiased resize to target
     (this also fixes anamorphic PAL/NTSC pixels) -> optional detail blend
  -> rgb24 pipe -> ffmpeg encode (proper matrix + colour tags) + source audio

Modes:
  keep  - output at the source frame size (SR used as a detail restorer)
  hd    - output at --height (default 1080) with square pixels, width from DAR

Reader, GPU and writer run in separate threads so the pipes never stall
the GPU.
"""
import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
import warnings
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

from model import ENGINES, Upscaler

FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe" if os.path.exists(r"C:\ffmpeg\bin\ffmpeg.exe") else "ffmpeg"
FFPROBE = r"C:\ffmpeg\bin\ffprobe.exe" if os.path.exists(r"C:\ffmpeg\bin\ffprobe.exe") else "ffprobe"
NOWIN = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW (pyw GUI)

DENOISE = {"off": None, "light": "hqdn3d=1.5:1.2:3:3", "medium": "hqdn3d=3:2.5:6:5",
           "strong": "hqdn3d=5:4:9:8"}
warnings.filterwarnings("ignore", message="The given buffer is not writable")
# audio codecs copied as-is; anything else (FLAC/PCM from captures, MP2 from
# DVD/tape recorders) becomes AAC, because TVs often refuse FLAC/Opus in MP4
MP4_AUDIO_OK = {"aac", "mp3", "ac3", "eac3"}


def _ratio(s, default):
    try:
        n, d = (int(v) for v in re.split("[:/]", s))
        return Fraction(n, d) if n > 0 and d > 0 else default
    except Exception:
        return default


def probe(path):
    """Video stream facts needed by the pipeline."""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True, check=True, creationflags=NOWIN).stdout
    j = json.loads(out)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    fps = _ratio(v.get("avg_frame_rate", "0/0"), None) or _ratio(v.get("r_frame_rate"), Fraction(25))
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) in (90, 270):  # phone video: ffmpeg autorotates on decode
        w, h = h, w
    sar = _ratio(v.get("sample_aspect_ratio", "1:1"), Fraction(1))
    dur = float(v.get("duration") or j["format"].get("duration") or 0)
    return {
        "w": w, "h": h, "fps": fps, "sar": sar, "dur": dur,
        "field_order": v.get("field_order", "unknown"),
        "matrix": v.get("color_space", ""), "range": v.get("color_range", ""),
        "audio": a["codec_name"] if a else None,
    }


def detect_interlace(path, frames=500):
    """idet over the first frames -> 'tff' | 'bff' | None (progressive)."""
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-nostats", "-i", path, "-an", "-frames:v", str(frames),
         "-vf", "idet", "-f", "null", "-"],
        capture_output=True, text=True, creationflags=NOWIN)
    # ffmpeg may print the block more than once (filter re-init); the
    # first copy can be all zeros, so sum every block
    hits = re.findall(r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)",
                      r.stderr)
    if not hits:
        return None
    tff, bff, prog = (sum(int(h[i]) for h in hits) for i in range(3))
    if tff + bff > max(prog, 20):
        return "tff" if tff >= bff else "bff"
    return None


def plan(info, path, mode="hd", height=1080, deinterlace="auto", denoise="off",
         crop=(0, 0, 0, 0), log=print):
    """Decide filters, sizes and colour handling for one job."""
    w, h, sar = info["w"], info["h"], info["sar"]
    cl, ct, cr, cb = crop
    cw, ch = w - cl - cr, h - ct - cb
    if cw < 16 or ch < 16:
        raise ValueError("crop removes the whole frame")

    parity = None
    if deinterlace in ("auto", "on", "frame"):
        fo = info["field_order"]
        tagged = {"tt": "tff", "bt": "tff", "bb": "bff", "tb": "bff"}.get(fo)
        detected = detect_interlace(path)
        if deinterlace == "auto":
            parity = detected or tagged
            # flagged interlaced but fields look alike (film on tape):
            # bwdif at frame rate, no duplicated frames
            if parity and not detected:
                deinterlace = "frame"
        else:
            parity = detected or tagged or "tff"
        rate = "x2 rate" if deinterlace != "frame" else "frame rate"
        log(f"[scan] field_order tag={fo}, idet={detected or 'progressive'} "
            f"-> {f'deinterlace {parity}, {rate}' if parity else 'progressive, no deinterlace'}")

    fps_out = info["fps"] * (2 if parity and deinterlace != "frame" else 1)

    vf = []
    if parity:
        rate = "send_frame" if deinterlace == "frame" else "send_field"
        vf.append(f"bwdif=mode={rate}:parity={parity}:deint=all")
    if any(crop):
        vf.append(f"crop={cw}:{ch}:{cl}:{ct}")
    if DENOISE.get(denoise):
        vf.append(DENOISE[denoise])
    vf.append(f"fps={fps_out.numerator}/{fps_out.denominator}")
    tag = info["matrix"]
    if tag in ("bt709",):
        m_in = "bt709"
    elif tag in ("smpte170m", "bt470bg", "bt601", "fcc"):
        m_in = "bt601"
    else:
        m_in = "bt709" if h >= 720 else "bt601"
    r_in = "pc" if info["range"] in ("pc", "jpeg") else "tv"
    vf.append(f"scale=in_color_matrix={m_in}:in_range={r_in}:out_range=pc:"
              f"flags=bicubic+accurate_rnd+full_chroma_int,format=rgb24")

    if mode == "keep":
        out_w, out_h, out_sar = cw, ch, sar
    else:
        dar = Fraction(cw, ch) * sar
        out_h = int(height)
        out_w = int(round(out_h * dar / 2)) * 2
        out_sar = Fraction(1)
    m_out = "bt709" if out_h >= 720 else "bt601"
    return {"vf": ",".join(vf), "in_w": cw, "in_h": ch, "out_w": out_w, "out_h": out_h,
            "out_sar": out_sar, "fps": fps_out, "m_out": m_out, "parity": parity}


def _encoder_cmd(p, info, in_path, out_path, crf, encoder, audio_src=None):
    """audio_src: restored WAV from audio.process (already trimmed), or None
    to take the audio straight from in_path."""
    fps = p["fps"]
    tags = (["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
            if p["m_out"] == "bt709" else
            ["-colorspace", "smpte170m", "-color_primaries", "smpte170m",
             "-color_trc", "smpte170m"])
    sar = p["out_sar"]
    vf = (f"scale=in_range=pc:out_color_matrix={p['m_out']}:out_range=tv:"
          f"flags=bicubic+accurate_rnd,format=yuv420p,"
          f"setsar={sar.numerator}/{sar.denominator}")
    if encoder == "nvenc":
        venc = ["-c:v", "hevc_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr",
                "-cq", str(crf + 2), "-b:v", "0", "-tag:v", "hvc1"]
    else:
        venc = ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                "-tune", "film", "-profile:v", "high"]
    if info["audio"] is None:
        aenc = []
    elif audio_src:
        aenc = ["-c:a", "aac", "-b:a", "256k"]
    elif info["audio"] in MP4_AUDIO_OK:
        aenc = ["-c:a", "copy"]
    else:
        aenc = ["-c:a", "aac", "-b:a", "256k"]
    return ([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{p['out_w']}x{p['out_h']}",
             "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "-",
             "-i", audio_src or in_path, "-map", "0:v:0", "-map", "1:a:0?",
             "-vf", vf] + venc + tags + ["-color_range", "tv"] + aenc +
            ["-shortest", "-movflags", "+faststart", out_path])


# Measured peak VRAM per input pixel (fp16 where supported), e.g. SPAN on one
# 720x576 frame ~0.94 GB. Batching above 2 gave <10 % speed, so keep it small.
VRAM_PER_PX = {"span": 2.4e-6, "compact": 1.0e-6, "x2plus": 2.1e-6, "vhs2hd": 1.8e-6}


def default_batch(engine, w, h, free_gb=None):
    """Frames per GPU call, bounded by the VRAM that is actually free."""
    b = 2 if engine in ("span", "compact") else 1
    if free_gb is not None:
        per_frame = VRAM_PER_PX.get(engine, 2.5e-6) * w * h
        b = min(b, int((free_gb - 1.0) / per_frame))
    return max(1, b)


def enhance_video(in_path, out_path, mode="hd", engine="span", height=1080,
                  deinterlace="auto", denoise="off", strength=1.0,
                  crop=(0, 0, 0, 0), crf=17, encoder="x264", batch=0,
                  device="cuda", progress_cb=None, log=print, stop_event=None,
                  max_frames=None, start=None, max_seconds=None, audio_mode="off",
                  audio_progress=None):
    """Process one file. progress_cb(done, total, t0) is called ~every second."""
    torch.backends.cudnn.benchmark = True
    info = probe(in_path)
    log(f"[input] {os.path.basename(in_path)}: {info['w']}x{info['h']} "
        f"SAR {info['sar']} @ {float(info['fps']):.3f} fps, {info['dur']:.1f}s, "
        f"matrix={info['matrix'] or 'untagged'}")
    p = plan(info, in_path, mode, height, deinterlace, denoise, crop, log)
    iw, ih, ow, oh = p["in_w"], p["in_h"], p["out_w"], p["out_h"]
    log(f"[plan] {iw}x{ih} -> {ow}x{oh} @ {float(p['fps']):.3f} fps, "
        f"engine={engine}, denoise={denoise}, strength={strength}")

    if max_seconds:
        max_frames = int(max_seconds * float(p["fps"]))
    # Audio restoration runs first (before the SR model takes VRAM) and is
    # trimmed exactly like the video: same start, same length.
    audio_wav = None
    if audio_mode and audio_mode != "off" and info["audio"]:
        import audio as audio_mod
        audio_wav = out_path + ".audio.wav"
        dur = (max_frames / float(p["fps"])) if max_frames else None
        log(f"[audio] restoring the soundtrack ({audio_mode}) before the video pass...")
        audio_mod.process(in_path, audio_wav, audio_mode, start, dur, log, progress=audio_progress)
    up = Upscaler(engine, device, log)
    # Windows drivers spill to system RAM instead of raising OOM when VRAM is
    # full, which makes everything 10-100x slower, so size the batch to free VRAM.
    free = torch.cuda.mem_get_info()[0] / 2**30 if device == "cuda" else None
    B = batch or default_batch(engine, iw, ih, free)
    if free is not None and free < 2.0:
        log(f"[warn] only {free:.1f} GB VRAM free. Close other GPU apps (games, AI "
            f"tools, a second Enhancer window) or processing may slow down a lot.")
    total = int(info["dur"] * float(p["fps"])) or None
    if max_frames:
        total = min(total or max_frames, max_frames)

    dec_cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
    if start:
        dec_cmd += ["-ss", str(start)]
    dec_cmd += ["-i", in_path, "-an", "-sn", "-vf", p["vf"]]
    if max_frames:
        dec_cmd += ["-frames:v", str(max_frames)]
    dec_cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    enc_cmd = _encoder_cmd(p, info, in_path, out_path, crf, encoder, audio_wav)
    # Lip sync: the decoder's CFR rawvideo output pads a late-starting video
    # stream with duplicated frames from t=0 (verified: a grabber capture with
    # video starting 0.6 s after audio stays in sync). So the audio only needs
    # the same trim as the video, nothing more.
    if start and info["audio"] and not audio_wav:
        i = enc_cmd.index(in_path)
        enc_cmd[i - 1:i - 1] = ["-ss", str(start)]

    enc_log = open(out_path + ".ffmpeg.log", "w+b")
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, creationflags=NOWIN,
                           bufsize=iw * ih * 3 * 4)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=enc_log,
                           creationflags=NOWIN)
    q_in, q_out = queue.Queue(maxsize=3 * B + 4), queue.Queue(maxsize=3 * B + 4)
    errors = []
    fb = iw * ih * 3

    def reader():
        try:
            while True:
                buf = dec.stdout.read(fb)
                if not buf or len(buf) < fb:
                    break
                q_in.put(buf)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            q_in.put(None)

    enc_closed = threading.Event()

    def writer():
        try:
            while (item := q_out.get()) is not None:
                enc.stdin.write(item)
        except (BrokenPipeError, OSError):
            # encoder finished first (e.g. -shortest when the audio ends a
            # moment before the video): not an error by itself, the encoder's
            # exit code decides below
            enc_closed.set()
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        if enc_closed.is_set() or errors:
            while q_out.get() is not None:  # drain so the GPU loop never blocks
                pass

    threading.Thread(target=reader, daemon=True).start()
    wt = threading.Thread(target=writer, daemon=True)
    wt.start()

    done, t0, last = 0, time.time(), 0.0
    pinned = torch.empty((B, ih, iw, 3), dtype=torch.uint8).pin_memory()
    eof = False
    try:
        while not eof:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("cancelled by user")
            if errors:
                raise errors[0]
            if enc_closed.is_set():
                break
            n = 0
            while n < B:
                buf = q_in.get()
                if buf is None:
                    eof = True
                    break
                pinned[n].copy_(torch.frombuffer(buf, dtype=torch.uint8).view(ih, iw, 3))
                n += 1
            if n == 0:
                break
            x = pinned[:n].to(device, non_blocking=True)
            x = x.permute(0, 3, 1, 2).to(up.dtype).div_(255)
            y = up(x)
            if strength < 1.0:  # pull back toward a plain upscale
                base = F.interpolate(x, size=y.shape[-2:], mode="bicubic",
                                     align_corners=False).clamp_(0, 1)
                y = base.lerp_(y, strength)
            if y.shape[-2:] != (oh, ow):
                y = F.interpolate(y.float(), size=(oh, ow), mode="bicubic",
                                  antialias=True, align_corners=False)
            y = y.mul(255).round_().clamp_(0, 255).to(torch.uint8)
            y = y.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            for k in range(n):
                q_out.put(y[k].tobytes())
            done += n
            now = time.time()
            if now - last >= 1.0:
                last = now
                if progress_cb:
                    progress_cb(done, total or done, t0)
                else:
                    sp = done / max(now - t0, 1e-3)
                    eta = ((total - done) / sp / 60) if total else 0
                    print(f"\r  {done}/{total or '?'} frames  {sp:5.1f} f/s  "
                          f"ETA {eta:5.1f} min   ", end="", flush=True)
    finally:
        q_out.put(None)
        wt.join()
        try:
            enc.stdin.close()
        except OSError:
            pass
        dec.kill()
        enc.wait()
        enc_log.seek(0)
        msg = enc_log.read().decode(errors="replace").strip()
        enc_log.close()
        # hand VRAM back: the GUI process stays alive between jobs, and the
        # caching allocator would otherwise keep gigabytes reserved
        x = y = pinned = None
        del up
        if device == "cuda":
            torch.cuda.empty_cache()
        if audio_wav:
            try:
                os.remove(audio_wav)
            except OSError:
                pass
    if errors:
        raise errors[0]
    if enc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoder failed:\n{msg[-2000:]}")
    try:
        os.remove(out_path + ".ffmpeg.log")
    except OSError:
        pass
    dt = time.time() - t0
    log(f"\n[done] {done} frames in {dt/60:.1f} min ({done/max(dt,1e-3):.1f} f/s) -> {out_path}")
    return {"frames": done, "seconds": dt, "fps": done / max(dt, 1e-3)}


VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov", ".mpg", ".mpeg", ".vob", ".m2ts", ".mts", ".ts", ".dv", ".wmv")


def find_original(video_path):
    """For an Enhancer output like 'Tape_..._1080p_span.mp4', find the capture it
    came from ('Tape_....mkv' etc.) in the same folder, or None."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    base = re.sub(r"_(\d+p|enh)_[A-Za-z0-9-]+(_preview)?(_audio-[a-z-]+)?$", "", stem)
    if base == stem:
        return None
    folder = os.path.dirname(os.path.abspath(video_path))
    for ext in VIDEO_EXTS:
        cand = os.path.join(folder, base + ext)
        if os.path.isfile(cand) and os.path.abspath(cand) != os.path.abspath(video_path):
            return cand
    return None


def audio_only(video_path, out_path, audio_mode="clean", audio_from="auto", log=print,
               progress=None):
    """Restore just the soundtrack of a finished video: the picture is stream-
    copied (no re-encode, no quality loss, seconds instead of hours).

    audio_from: 'auto' = the original capture if it sits next to the file and
    has the same length (better: FLAC instead of an AAC copy), else the
    video's own audio; or an explicit path; or None for the video's own audio.
    """
    import audio as audio_mod
    info = probe(video_path)
    src = video_path
    cand = find_original(video_path) if audio_from == "auto" else audio_from
    if cand:
        o = probe(cand)
        if o["audio"] and abs(o["dur"] - info["dur"]) <= 1.0:
            src = cand
            log(f"[audio] using the original capture's sound: {os.path.basename(cand)}")
        elif audio_from not in (None, "auto"):
            raise ValueError(f"{os.path.basename(cand)} is {o['dur']:.1f}s but the video is "
                             f"{info['dur']:.1f}s: they don't belong together")
        else:
            log(f"[audio] {os.path.basename(cand)} has a different length "
                f"(preview/trimmed?), using the video's own sound")
    if src == video_path and not info["audio"]:
        raise ValueError("this video has no audio track")
    if audio_mode == "off" and src == video_path:
        raise ValueError("audio 'off' means 'take the original capture's sound', but no original "
                         "with the same length was found next to this file")
    t0 = time.time()
    wav = out_path + ".audio.wav"
    try:
        if audio_mode == "off":
            # untouched original sound, only converted to AAC (TVs reject FLAC
            # in MP4) and cut to the video length
            wav = src
            log("[audio] original sound, no processing")
        else:
            audio_mod.process(src, wav, audio_mode, None, info["dur"], log, progress=progress)
        if progress:
            progress("Muxing: picture copied untouched + new soundtrack...", 1.0)
        r = subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                            "-i", video_path, "-i", wav, "-map", "0:v:0", "-map", "1:a:0",
                            # -t, not -shortest: with a stream-copied picture
                            # -shortest cut 11 frames off a 72 s tape (tested)
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                            "-t", f"{info['dur']:.3f}", "-movflags", "+faststart", out_path],
                           capture_output=True, text=True, creationflags=NOWIN)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed:\n{r.stderr[-1500:]}")
    finally:
        if wav != src:  # never delete the source when it was used directly
            try:
                os.remove(wav)
            except OSError:
                pass
    log(f"[done] audio-only ({audio_mode}) in {time.time() - t0:.1f}s -> {out_path}")
    return out_path


def audio_only_output(video_path, audio_mode):
    stem, _ = os.path.splitext(os.path.abspath(video_path))
    return f"{stem}_audio-{'original' if audio_mode == 'off' else audio_mode}.mp4"


def default_output(in_path, mode, engine, height=1080):
    name = os.path.splitext(os.path.basename(in_path))[0]
    tag = f"{height}p" if mode == "hd" else "enh"
    return os.path.join(os.path.dirname(os.path.abspath(in_path)),
                        f"{name}_{tag}_{engine}.mp4")


def main():
    ap = argparse.ArgumentParser(
        description="Video Enhancer: deinterlace + AI upscale for VHS / old camera footage",
        epilog="Erynvorn Technologies | https://erynvorntechnologies.com")
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o")
    ap.add_argument("--mode", choices=["hd", "keep"], default="hd",
                    help="hd = upscale to --height; keep = source frame size")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--engine", choices=list(ENGINES), default="span")
    ap.add_argument("--deinterlace", choices=["auto", "on", "frame", "off"], default="auto",
                    help="auto = idet detection; on = double-rate bwdif; frame = single rate")
    ap.add_argument("--denoise", choices=list(DENOISE), default="off")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="1.0 = full model output, lower = blend toward plain upscale")
    ap.add_argument("--crop", default="0,0,0,0", help="left,top,right,bottom in source px")
    ap.add_argument("--audio", choices=["off", "clean", "voice", "voice-hq"], default="off",
                    help="clean = tape hiss/clicks/hum/clipping + levels; voice = + DeepFilterNet; "
                         "voice-hq = + ClearerVoice (GPU)")
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--encoder", choices=["x264", "nvenc"], default="x264")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--start", type=float, help="start time in seconds (preview)")
    ap.add_argument("--frames", type=int, help="process only N output frames (preview)")
    ap.add_argument("--audio-only", action="store_true",
                    help="keep the picture as-is (stream copy) and only restore the sound "
                         "with --audio (default clean)")
    ap.add_argument("--audio-from", default="auto",
                    help="with --audio-only: 'auto' (original capture next to the file if the "
                         "length matches), 'self', or a path")
    a = ap.parse_args()
    if a.audio_only:
        mode = a.audio  # "off" = original sound untouched (from --audio-from)
        src = None if a.audio_from == "self" else a.audio_from
        audio_only(a.input, a.output or audio_only_output(a.input, mode), mode, src)
        return
    out = a.output or default_output(a.input, a.mode, a.engine, a.height)
    crop = tuple(int(v) for v in a.crop.split(","))
    enhance_video(a.input, out, a.mode, a.engine, a.height, a.deinterlace, a.denoise,
                  a.strength, crop, a.crf, a.encoder, a.batch,
                  max_frames=a.frames, start=a.start, audio_mode=a.audio)


if __name__ == "__main__":
    main()
