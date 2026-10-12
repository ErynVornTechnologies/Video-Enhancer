"""Video Enhancer - GUI. Part of the Boring Stuff series by Erynvorn Technologies.
https://erynvorntechnologies.com | https://github.com/ErynVornTechnologies

Worker thread pushes event tuples onto a queue; the Tk loop drains it.
torch is imported first (via model) so our MSVCP140 wins over shell
extensions such as EaseUS RightMenu (WinError 1114 in c10.dll otherwise).
"""
import os
import queue
import sys
import threading
import time
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: F401,E402  - must load before any native file dialog
import enhancer  # noqa: E402
from model import ENGINES  # noqa: E402

import customtkinter as ctk  # noqa: E402
from tkinter import filedialog, messagebox  # noqa: E402
from PIL import Image  # noqa: E402

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")
VIDEO_TYPES = "*.mp4 *.mkv *.avi *.mov *.mpg *.mpeg *.vob *.m2ts *.mts *.ts *.wmv *.dv *.webm"
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
GOLD = "#c9b39b"  # Erynvorn brand colour, taken from the logo
LINKS = [("erynvorntechnologies.com", "https://erynvorntechnologies.com"),
         ("GitHub", "https://github.com/ErynVornTechnologies"),
         ("LinkedIn", "https://www.linkedin.com/in/kamil-krzysztof-nagorski/")]


class EnhancerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Video Enhancer | Erynvorn Technologies")
        self.geometry("720x920")
        try:
            self.iconbitmap(os.path.join(ASSETS, "evt_icon.ico"))
        except Exception:  # noqa: BLE001 - icon is cosmetic
            pass
        self.in_path = None
        self.info = None
        self.job = None
        self.stop = threading.Event()
        self.q = queue.Queue()
        self._build()
        self.after(100, self._poll)

    # ------------------------------------------------------------------ UI
    def _row(self, parent, r, label, widget):
        ctk.CTkLabel(parent, text=label).grid(row=r, column=0, padx=(14, 8), pady=6, sticky="w")
        widget.grid(row=r, column=1, padx=8, pady=6, sticky="w")

    def _build(self):
        try:
            logo = Image.open(os.path.join(ASSETS, "evt_logo.png"))
            self._logo = ctk.CTkImage(light_image=logo, dark_image=logo, size=(240, 58))
            ctk.CTkLabel(self, text="", image=self._logo).pack(pady=(16, 6))
        except Exception:  # noqa: BLE001
            pass
        ctk.CTkLabel(self, text="Video Enhancer", font=("Roboto", 22, "bold")).pack(pady=(2, 2))
        ctk.CTkLabel(self, text="Deinterlace + AI upscale for VHS / old camera footage",
                     text_color="#8a8a8a").pack(pady=(0, 10))

        f = ctk.CTkFrame(self); f.pack(padx=18, pady=4, fill="x")
        self.lbl_file = ctk.CTkLabel(f, text="No file selected", text_color="#4ade80",
                                     wraplength=520, justify="left")
        self.lbl_file.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        ctk.CTkButton(f, text="Choose video", width=130, command=self.pick).pack(side="right", padx=10)
        self.lbl_info = ctk.CTkLabel(self, text="", text_color="#8a8a8a")
        self.lbl_info.pack()

        o = ctk.CTkFrame(self); o.pack(padx=18, pady=8, fill="x")
        self.v_engine = ctk.StringVar(value="span")
        ef = ctk.CTkFrame(o, fg_color="transparent")
        for k, spec in ENGINES.items():
            ctk.CTkRadioButton(ef, text=spec["label"], variable=self.v_engine, value=k,
                               command=self.update_hint).pack(anchor="w", pady=2)
        self._row(o, 0, "Engine:", ef)
        self.v_mode = ctk.StringVar(value="hd")
        mf = ctk.CTkFrame(o, fg_color="transparent")
        ctk.CTkRadioButton(mf, text="Upscale to", variable=self.v_mode, value="hd",
                           command=self.update_hint).pack(side="left")
        self.v_height = ctk.StringVar(value="1080")
        ctk.CTkOptionMenu(mf, values=["720", "1080", "1440", "2160"], variable=self.v_height,
                          width=80, command=lambda _: self.update_hint()).pack(side="left", padx=6)
        ctk.CTkRadioButton(mf, text="Keep source size", variable=self.v_mode, value="keep",
                           command=self.update_hint).pack(side="left", padx=12)
        self._row(o, 1, "Output:", mf)
        self.v_deint = ctk.StringVar(value="auto")
        self._row(o, 2, "Deinterlace:", ctk.CTkOptionMenu(
            o, values=["auto", "on", "frame", "off"], variable=self.v_deint, width=110))
        self.v_dn = ctk.StringVar(value="light")
        self._row(o, 3, "Denoise:", ctk.CTkOptionMenu(
            o, values=list(enhancer.DENOISE), variable=self.v_dn, width=110))
        sf = ctk.CTkFrame(o, fg_color="transparent")
        self.v_str = ctk.DoubleVar(value=1.0)
        self.lbl_str = ctk.CTkLabel(sf, text="100%", width=50)
        ctk.CTkSlider(sf, from_=0.5, to=1.0, number_of_steps=10, variable=self.v_str, width=220,
                      command=lambda v: self.lbl_str.configure(text=f"{v*100:.0f}%")).pack(side="left")
        self.lbl_str.pack(side="left", padx=6)
        self._row(o, 4, "AI strength:", sf)
        self.v_crop = ctk.StringVar(value="0,0,0,0")
        cf = ctk.CTkFrame(o, fg_color="transparent")
        ctk.CTkEntry(cf, textvariable=self.v_crop, width=110).pack(side="left")
        ctk.CTkLabel(cf, text="L,T,R,B px  (VHS: 0,0,0,8 hides head-switching noise)",
                     text_color="#8a8a8a").pack(side="left", padx=6)
        self._row(o, 5, "Crop:", cf)
        import audio as audio_mod
        self.audio_labels = {audio_mod.MODES[m]: m for m in audio_mod.available_modes()}
        self.v_audio = ctk.StringVar(value=audio_mod.MODES["off"])
        self._row(o, 6, "Audio:", ctk.CTkOptionMenu(
            o, values=list(self.audio_labels), variable=self.v_audio, width=330))
        self.lbl_hint = ctk.CTkLabel(self, text="", text_color="#facc15")
        self.lbl_hint.pack(pady=4)

        self.bar = ctk.CTkProgressBar(self, width=640); self.bar.set(0); self.bar.pack(pady=8)
        self.lbl_status = ctk.CTkLabel(self, text="Ready"); self.lbl_status.pack()
        self.txt = ctk.CTkTextbox(self, width=680, height=170, font=("Consolas", 11))
        # bottom-up: footer, then buttons, then the log takes what is left
        foot = ctk.CTkFrame(self, fg_color="transparent"); foot.pack(side="bottom", pady=(0, 10))
        ctk.CTkLabel(foot, text="Boring Stuff by Erynvorn Technologies  |",
                     text_color="#6a6a6a").pack(side="left", padx=(0, 6))
        for text, url in LINKS:
            lnk = ctk.CTkLabel(foot, text=text, text_color=GOLD, cursor="hand2")
            lnk.pack(side="left", padx=4)
            lnk.bind("<Button-1>", lambda _e, u=url: webbrowser.open(u))
        b = ctk.CTkFrame(self, fg_color="transparent"); b.pack(side="bottom", pady=10)
        self.btn_prev = ctk.CTkButton(b, text="PREVIEW 10 s", width=150, height=44,
                                      command=lambda: self.start(preview=True))
        self.btn_prev.pack(side="left", padx=6)
        self.btn_aud = ctk.CTkButton(b, text="AUDIO ONLY", width=130, height=44, fg_color="#7c6a55",
                                     hover_color="#5f5142", command=lambda: self.start(audio_only=True))
        self.btn_aud.pack(side="left", padx=6)
        self.btn_run = ctk.CTkButton(b, text="ENHANCE", width=200, height=44,
                                     font=("Roboto", 16, "bold"), fg_color="#dc2626",
                                     hover_color="#b91c1c", command=self.start)
        self.btn_run.pack(side="left", padx=6)
        self.btn_stop = ctk.CTkButton(b, text="Cancel", width=100, height=44, state="disabled",
                                      fg_color="#444", command=self.stop.set)
        self.btn_stop.pack(side="left", padx=6)
        self.txt.pack(padx=18, pady=8, fill="both", expand=True)

    # ------------------------------------------------------------ events
    def _poll(self):
        try:
            while True:
                kind, *a = self.q.get_nowait()
                if kind == "log":
                    self.txt.insert("end", a[0].strip("\n") + "\n"); self.txt.see("end")
                elif kind == "progress":
                    self.lbl_status.configure(text=a[0]); self.bar.set(a[1])
                elif kind in ("done", "error"):
                    self._idle()
                    (messagebox.showinfo if kind == "done" else messagebox.showerror)(
                        "Done" if kind == "done" else "Error", a[0])
                    if kind == "done" and a[1]:
                        os.startfile(a[1])
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _idle(self):
        self.job = None
        self.lbl_status.configure(text="Ready")
        for w in (self.btn_run, self.btn_prev, self.btn_aud):
            w.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def start_audio_only(self):
        """Keep the picture (stream copy), restore only the soundtrack."""
        mode = self.audio_labels.get(self.v_audio.get(), "off")
        # The sound should come from the ORIGINAL recording (lossless), not from
        # the AAC copy inside the enhanced file. Found by name, or picked here.
        orig = enhancer.find_original(self.in_path)
        if not orig:
            orig = filedialog.askopenfilename(
                title="Original recording to take the sound from (Cancel = use this file's own sound)",
                initialdir=os.path.dirname(self.in_path),
                filetypes=[("Video", VIDEO_TYPES), ("All files", "*.*")]) or None
        if mode == "off" and not orig:
            messagebox.showinfo("Audio only", "Audio 'Off' means: put the original recording's sound, "
                                "untouched, under this picture.\nChoose the original recording, or "
                                "pick an Audio mode such as 'Clean tape'.")
            return
        self._ao_source = orig or None
        out = enhancer.audio_only_output(self.in_path, mode)
        if os.path.exists(out) and not messagebox.askyesno("Overwrite?", f"{out}\nalready exists. Overwrite?"):
            return
        for w in (self.btn_run, self.btn_prev, self.btn_aud):
            w.configure(state="disabled")
        self.bar.set(0)
        self.lbl_status.configure(text=f"Restoring audio ({mode})... picture is copied untouched")
        self.q.put(("log", f"[job] AUDIO ONLY {os.path.basename(self.in_path)} mode={mode}"))

        def work():
            try:
                enhancer.audio_only(self.in_path, out, mode, self._ao_source,
                                    log=lambda m: self.q.put(("log", m)),
                                    progress=lambda t, f: self.q.put(("progress", t, f)))
                self.q.put(("done", f"Audio restored ({mode}), picture untouched:\n{out}", None))
            except Exception as e:  # noqa: BLE001
                self.q.put(("error", f"{e}"))
        self.job = threading.Thread(target=work, daemon=True)
        self.job.start()

    def pick(self):
        p = filedialog.askopenfilename(title="Choose a video",
                                       filetypes=[("Video", VIDEO_TYPES), ("All files", "*.*")])
        if not p:
            return
        self.in_path = p
        self.lbl_file.configure(text=p)
        try:
            self.info = enhancer.probe(p)
            i = self.info
            self.lbl_info.configure(text=f"{i['w']}x{i['h']}  SAR {i['sar']}  {float(i['fps']):.3f} fps  "
                                         f"{i['dur']/60:.1f} min  field order: {i['field_order']}")
        except Exception as e:  # noqa: BLE001
            self.info = None
            self.lbl_info.configure(text=f"probe failed: {e}")
        self.update_hint()

    def update_hint(self):
        i = self.info
        if not i:
            return
        if self.v_mode.get() == "keep":
            out = f"{i['w']}x{i['h']}"
        else:
            hh = int(self.v_height.get())
            out = f"{int(round(hh * i['w'] * i['sar'] / i['h'] / 2)) * 2}x{hh}"
        self.lbl_hint.configure(text=f"Output ~{out}   engine: {self.v_engine.get()}")

    # ------------------------------------------------------------- jobs
    def start(self, preview=False, audio_only=False):
        if self.job or not self.in_path:
            if not self.in_path:
                messagebox.showwarning("No file", "Choose a video first.")
            return
        if audio_only:
            return self.start_audio_only()
        try:
            crop = tuple(int(v) for v in self.v_crop.get().split(","))
            assert len(crop) == 4 and min(crop) >= 0
        except Exception:  # noqa: BLE001
            messagebox.showerror("Crop", "Crop must be four numbers: left,top,right,bottom")
            return
        mode, engine, height = self.v_mode.get(), self.v_engine.get(), int(self.v_height.get())
        out = enhancer.default_output(self.in_path, mode, engine, height)
        kw = dict(mode=mode, engine=engine, height=height, deinterlace=self.v_deint.get(),
                  denoise=self.v_dn.get(), strength=round(self.v_str.get(), 2), crop=crop,
                  audio_mode=self.audio_labels.get(self.v_audio.get(), "off"))
        if preview:
            out = out[:-4] + "_preview.mp4"
            dur = (self.info or {}).get("dur", 0)
            kw["start"] = max(0.0, dur / 2 - 5) if dur > 20 else 0
            kw["max_seconds"] = 10
        elif os.path.exists(out) and not messagebox.askyesno("Overwrite?", f"{out}\nalready exists. Overwrite?"):
            return
        self.stop.clear()
        for w in (self.btn_run, self.btn_prev, self.btn_aud):
            w.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.bar.set(0)
        self.q.put(("log", f"[job] {'PREVIEW ' if preview else ''}{os.path.basename(self.in_path)} {kw}"))
        self.job = threading.Thread(target=self._run, args=(out, kw, preview), daemon=True)
        self.job.start()

    def _run(self, out, kw, preview):
        def prog(done, total, t0):
            sp = done / max(time.time() - t0, 1e-3)
            eta = (total - done) / max(sp, 1e-3) / 60
            self.q.put(("progress", f"{done}/{total} frames   {sp:.1f} f/s   ETA {eta:.1f} min",
                        min(1.0, done / max(total, 1))))
        try:
            r = enhancer.enhance_video(self.in_path, out, progress_cb=prog, stop_event=self.stop,
                                       log=lambda m: self.q.put(("log", m)),
                                       audio_progress=lambda t, f: self.q.put(("progress", t, f)),
                                       **kw)
            self.q.put(("done", f"Finished: {r['frames']} frames, {r['fps']:.1f} f/s\n{out}",
                        out if preview else None))
        except Exception as e:  # noqa: BLE001
            self.q.put(("error", f"{e}"))


if __name__ == "__main__":
    EnhancerApp().mainloop()
