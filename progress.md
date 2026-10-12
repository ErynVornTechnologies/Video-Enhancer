# progress.md: Video Enhancer (Boring Stuff #08)

Handover notes for any agent picking this up. Read this before you change anything.

## Status (2026-10-09)

- v2 working and tested end to end on the RTX 4070 SUPER (Windows 11, Python 3.12).
- Repo prepared here for GitHub; **not pushed yet**. Planned remote:
  `github.com/ErynVornTechnologies/Video-Enhancer` (name to be confirmed by Kamil).
- Goal (from Kamil): clean, artefact-free upscale of VHS / 480p / 720p material to 1080p
  for a 55" TV. Not a "super-duper" enhancer; reliability over hype.

## Where things live

| Path | What |
|---|---|
| `desktop_apps_prod\08_Video_Enhancer\` | **this repo (source of truth from now on)** |
| `H:\CLAUDE_CODE\Video_Enhancer\app\` | working copy where v2 was developed (uses global py -3.12) |
| `H:\CLAUDE_CODE\Video_Enhancer\diag\` | diagnostics: benchmarks, VHS simulation clips, comparison crops |
| `Products_GitHub\video_enhancer\` | **v1 by a previous agent, kept untouched on purpose. Do not edit.** |

## What was wrong with v1 (diagnosed and verified)

1. Hand-written `SRVGGNetCompact` ("Fast" engine): one PReLU instance reused for all
   33 layers (state_dict loads, last weights win) and the nearest-upsample residual
   was missing. Output mean was 82.4 vs input 0.41, so every frame was blown out.
2. No deinterlacing at all.
3. No SAR/DAR handling: anamorphic PAL/NTSC came out with the wrong shape.
4. rgb24 → yuv420p encode with the default BT.601 matrix and no tags at HD sizes, giving a colour shift.
5. `F.interpolate` downscale without `antialias=True`, causing aliasing and shimmer.
6. One frame per GPU call, synchronous pipes, encoder stderr on a PIPE (deadlock risk).

## v2 architecture

- `model.py`: `ENGINES` registry plus `Upscaler` (spandrel loader, fp16 where
  `supports_half`, padding to `size_requirements`, automatic tiling on CUDA OOM).
- `enhancer.py`: `probe` → `detect_interlace` (idet, **sums all printed blocks**,
  because ffmpeg prints a zero block first) → `plan` (filters, sizes, colour) →
  threaded pipeline reader → GPU → writer → x264/NVENC with source audio.
- Deinterlace policy (`auto`): idet detects interlacing, giving bwdif at double rate.
  If the stream is only *flagged* interlaced (e.g. film on tape), bwdif runs at frame rate.
- `gui.py`: CustomTkinter, imports torch first (EaseUS MSVCP140 DLL clash, WinError 1114).
- `INSTALL.bat`: local `.venv`; **CUDA torch must be installed before spandrel**,
  otherwise pip replaces it with a CPU-only torch from PyPI. This happened once to the
  global Python and was repaired by reinstalling `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`.

## Model evaluation (720x576 PAL 16:9 sim → 1080p, same frame)

| Engine | f/s end to end | Verdict |
|---|---|---|
| span (2xLiveActionV1 SPAN) | 22-25 | **default**; natural texture; licence CC-BY-NC-SA (non-commercial) |
| compact (realesr-general-x4v3) | 13-17 | smooth, plastic; BSD |
| x2plus (RealESRGAN) | 3-5 | waxy faces; BSD |
| vhs2hd (2xVHS2HD RealPLKSR) | ~1.5 | softer than bicubic, desaturated; spandrel says no fp16; experimental |

Caveat: tests used a **synthetic** VHS (Tears of Steel, degraded and interlaced with
ffmpeg). Real tape captures have not been tested yet; that is the next step.

## VRAM: important on Windows

- With full VRAM, the NVIDIA Windows driver **does not raise OOM**: it spills to system RAM
  ("sysmem fallback") and everything runs 10-100x slower. Seen live: an idle GUI kept
  ~3.3 GB cached and the self-test (batch 6) froze at 88 % GPU without producing frames.
- Fixes in `enhancer.py`: VRAM is released after every job (`del up` +
  `torch.cuda.empty_cache()`), and the batch is sized from `torch.cuda.mem_get_info()`
  (`VRAM_PER_PX`, measured: SPAN ~0.94 GB per 720x576 frame, compact ~0.38, x2plus ~0.84,
  vhs2hd ~0.70). Default batch is 2 for span/compact and 1 otherwise; batch 4 gave
  under 10 % speed-up.
- Never benchmark while another GPU job runs: numbers drop ~2.3x (seen 2026-10-09).

## Tests

`.venv\Scripts\python.exe test_enhancer.py`: all 4 engines (brightness drift < 0.05) plus
synthetic 576i + audio → 1440x1080p50 AAC. Last full pass 2026-10-09 in `.venv` after the VRAM/batch and branding changes: ALL TESTS PASSED, 23.7 f/s; VRAM reserved after a job = 0.00 GB. `INSTALL.bat` was tested end to end (fresh .venv, weights
download verified).

## Open items / ideas

- [ ] Test on a real VHS capture from Kamil; tune denoise defaults.
- [ ] Optional QTGMC path (VapourSynth) for top-quality deinterlacing.
- [ ] TensorRT (vs-mlrt) for speed; temporal-consistency model (e.g. RealBasicVSR) if shimmer is an issue.
- [ ] Commercial-safe default if this ever ships to clients (SPAN weights are NC).
- [ ] Push to GitHub. The LinkedIn post is deliberately **not** in the repo
      (Kamil's decision); the draft lives in `H:\CLAUDE_CODE\Video_Enhancer\LinkedIn_Post.md`.

## Audio restoration (2026-10-10, Kamil: speech + VHS hiss + tape faults + clipping)

- `audio.py`: modes off / clean / voice / voice-hq; `enhancer.py --audio` and a GUI "Audio" menu.
  Runs BEFORE the SR model loads (no VRAM held on failure), same start/length as the video,
  float WAV `<out>.audio.wav` (deleted after), then AAC 256k in the MP4.
- clean = adeclip, adeclick, highpass 40, auto hum notch (FFT, > 6 dB above neighbours;
  none on Kamil's tape), afftdn nr=20, compressor thr -14 / ratio 2, loudnorm -16 LUFS,
  limiter. A stronger compressor (-20 dB, 2.5:1, +2 dB) raised the noise floor ~3 dB, so it was dropped.
- voice = DeepFilterNet standalone `tools\deep-filter.exe` v0.5.6 (Rust, CPU, no Python deps),
  `-D -a 18` (delay compensation, 18 dB attenuation limit). It returns ~30 ms shorter, so the
  final pass pads with `apad=whole_len` placed BEFORE loudnorm (after it, apad was ignored).
- voice-hq = ClearerVoice MossFormer2_SE_48K in a separate `.venv_audio` (clearvoice pins
  numpy<2, librosa, opencv). Called via `cv_worker.py`, blended 85 % AI / 15 % classic.
  **clearvoice 0.1.2 bug**: with huggingface_hub 2.x the pointer file `last_best_checkpoint`
  is not downloaded and clearvoice silently runs on RANDOM weights. cv_worker writes the
  pointer and refuses to run without it. Weights in `models_audio\checkpoints\`.
- Kamil's tape (30 s, loudness-matched; gap = median minus 10th percentile of 50 ms RMS):
  original 13.9 dB, clean ~13-15 dB (the "noise" is mostly room ambience, not hiss),
  voice ~20 dB, voice-hq ~20 dB. No digital clipping plateaus in this tape (peaks -0.3 dBFS,
  no flat tops), so any audible distortion is analogue and can only be softened.
- E2E (10 s preview, all modes): 250 frames, AAC 48 kHz stereo, video = audio = 10.000 s.
- A/B listening files: `H:\CLAUDE_CODE\Video_Enhancer\audio_lab\AB\` (1_original … 4_voice-hq).

## Audio only (2026-10-10, Kamil: "the picture is done, fix only the sound from the original")

- `enhancer.audio_only()` + `--audio-only [--audio-from auto|self|path]` + GUI button AUDIO ONLY.
  Sound source = the original capture (found by name via `find_original`, or chosen in a dialog),
  accepted only if its length matches the video within 1 s. Picture = `-c:v copy`.
  Audio mode "off" = original sound untouched (AAC). Output `<video>_audio-<mode>.mp4`.
- Verified on the 72 s tape (1811 frames): video packets md5-identical in all modes,
  **lag vs original audio 0 ms** in off/clean/voice/voice-hq, 11-25 s per run.
- Bugs found and fixed on the way:
  * `-shortest` with a stream-copied picture cut 11 frames, so use `-t <video duration>`.
  * `afftdn` delays by exactly 1200 samples at 48 kHz (25 ms; the same at nr 6 and 20). The
    chain now resamples to 48 kHz FIRST and trims 1200 samples after afftdn.
    `alimiter` had 5 ms of lookahead delay, so `latency=1`.
  * The restored track is padded to the VIDEO length (the capture's audio is 72.15 s vs 72.44 s video).

## Integration with VHS Capture #09 (2026-10-09)

- `09_VHS_Capture` can hand recordings to this app (`enhancer.py -i file --denoise light`).
  Captures are MJPEG 640x480 full-range + FLAC in MKV, wall-clock (VFR) timestamps.
- Audio: only AAC/MP3/AC3/E-AC3 are copied; FLAC/PCM/Opus/MP2 → AAC 256k (TV compatibility).
- `BrokenPipeError` from the writer when the encoder exits first (`-shortest`) is no longer
  fatal; the encoder exit code decides.
- Lip sync verified with a synthetic clip whose video starts 0.6 s after the audio:
  flash 2.00 s, beep 2.02 s. The CFR rawvideo decoder pads the start, so **no extra
  audio offset is needed**. One was tried, measured to be wrong (0.6 s off) and reverted.

## Branding (2026-10-09)

- Logo copied from `H:\ANTIGRAVITY LAB\ERYNVORN\Website\ERYNVORN WEB\EVT logo.png` to
  `assets/evt_logo.png`; `assets/evt_mark.png` + `assets/evt_icon.ico` are the square mark
  cropped from it (window icon). Brand gold `#c9b39b`.
- GUI: logo header, window icon, footer links (website, GitHub org, LinkedIn).
  README: logo header, `docs/gui.png` screenshot, Erynvorn footer. Links also in code
  docstrings, CLI `--help` epilog and `INSTALL.bat`.
- `docs/gui.png` is generated by `H:\CLAUDE_CODE\Video_Enhancer\diag\gui_shot.py`.
