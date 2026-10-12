"""ClearerVoice speech enhancement worker (runs in its own venv).

Usage: python cv_worker.py <in.wav> <out.wav> [model]
Default model: MossFormer2_SE_48K (48 kHz speech enhancement).

Isolated on purpose: clearvoice pins numpy<2, librosa, opencv... which would
break the video venv.

Guard against a silent clearvoice failure (seen 2026-10-10 with
huggingface_hub 2.x): the 221 MB weights downloaded but the tiny pointer file
'last_best_checkpoint' did not, and clearvoice then ran on RANDOM weights,
only printing a warning (output ~25 dB quieter, pure garbage). We download the
weights ourselves, write the pointer if missing, and refuse to run without it.
"""
import os
import sys
import time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models_audio")  # clearvoice looks in ./checkpoints/<model>


def ensure_weights(model):
    ck = os.path.join(MODELS, "checkpoints", model)
    pointer = os.path.join(ck, "last_best_checkpoint")
    weights = os.path.join(ck, "last_best_checkpoint.pt")
    if not os.path.isfile(weights):
        from huggingface_hub import snapshot_download
        print(f"[cv] downloading {model} weights (one-off)...", flush=True)
        snapshot_download(repo_id=f"alibabasglab/{model}", local_dir=ck)
    if not os.path.isfile(pointer) and os.path.isfile(weights):
        with open(pointer, "w", encoding="utf-8") as f:
            f.write("last_best_checkpoint.pt\n")
    if not (os.path.isfile(pointer) and os.path.isfile(weights)):
        raise SystemExit(f"[cv] ERROR: weights for {model} missing in {ck}; refusing to run "
                         f"on random weights")


def main():
    src, dst = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    model = sys.argv[3] if len(sys.argv) > 3 else "MossFormer2_SE_48K"
    os.makedirs(MODELS, exist_ok=True)
    os.chdir(MODELS)
    ensure_weights(model)
    from clearvoice import ClearVoice
    import torch
    t0 = time.time()
    cv = ClearVoice(task="speech_enhancement", model_names=[model])
    print(f"[cv] model {model} loaded in {time.time() - t0:.1f}s, cuda={torch.cuda.is_available()}",
          flush=True)
    info = sf.info(src)
    sr, n, chans = info.samplerate, info.frames, info.channels
    # Long tapes (82 min seen) in ONE call: no progress at all and the whole
    # channel in GPU/CPU memory. Process 60 s chunks with 1 s linear crossfades
    # instead, streaming from disk, and print progress for the GUI.
    chunk, ovl = 60 * sr, 1 * sr
    step = chunk - ovl
    starts = list(range(0, max(1, n - ovl), step))
    total = len(starts) * chans
    out = np.zeros((n, chans), dtype=np.float32)
    tmp_in = f"{dst}.chunk.in.wav"
    done, t1 = 0, time.time()
    for ch in range(chans):  # the models are mono: one pass per channel
        for i, s in enumerate(starts):
            seg, _ = sf.read(src, start=s, stop=min(n, s + chunk), dtype="float32", always_2d=True)
            seg = seg[:, ch]
            sf.write(tmp_in, seg, sr, subtype="FLOAT")
            y = np.asarray(cv(input_path=tmp_in, online_write=False), dtype=np.float32).reshape(-1)
            L = len(seg)
            y = y[:L] if len(y) >= L else np.pad(y, (0, L - len(y)))
            w = np.ones(L, dtype=np.float32)
            if i > 0:                      # fade in over the overlap with the previous chunk
                k = min(ovl, L)
                w[:k] = np.linspace(0, 1, k, dtype=np.float32)
            if s + chunk < n:              # fade out where the next chunk takes over
                k = min(ovl, L)
                w[L - k:] = np.minimum(w[L - k:], np.linspace(1, 0, k, dtype=np.float32))
            out[s:s + L, ch] += y * w
            done += 1
            el = time.time() - t1
            print(f"[cv-progress] {done} {total} {el / done * (total - done):.0f}", flush=True)
    if os.path.exists(tmp_in):
        os.remove(tmp_in)
    sf.write(dst, out, sr, subtype="FLOAT")
    print(f"[cv] {n / sr / 60:.1f} min x {chans} ch in {(time.time() - t1) / 60:.1f} min", flush=True)
    print("[cv] done", flush=True)


if __name__ == "__main__":
    main()
