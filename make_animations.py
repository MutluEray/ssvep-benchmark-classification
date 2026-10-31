"""Website animations built from real data and real held-out predictions.

  speller  - "replay speller": a word is typed on the 40-target keyboard. For every character
             a real trial of that target is taken from one block, and ensemble TRCA trained on
             the OTHER five blocks of the same subject makes the decision (0.5 s of EEG).
             Errors are kept; a wrong character triggers a real backspace trial.
  anatomy  - "anatomy of one decision": one trial, the data window grows from 0.1 to 1.0 s.
             Raw EEG -> filter bank -> TRCA spatial filter vs template -> 40 target scores.

Photosensitivity: real SSVEP flicker (8-15.8 Hz) is NOT reproduced. In the speller animation
each key pulses at its stimulus frequency divided by 8 (1-2 Hz, below the WCAG 3 flashes/s limit).

Output (default results/media/): speller.mp4, speller.gif, speller_poster.png,
                                 anatomy.mp4, anatomy.gif, anatomy_poster.png
MP4 and GIF need ffmpeg (brew install ffmpeg  /  conda install -c conda-forge ffmpeg).
Without ffmpeg a GIF is written with Pillow instead.

Examples
  python3 make_animations.py --plan-only                      # print the speller sequence, no rendering
  python3 make_animations.py                                  # both animations, subject 3, "HELLO_DTU"
  python3 make_animations.py --which anatomy --char T --block 2
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

import calibrated
import config
import data
import filterbank as fb
import preprocessing as pp

# Keyboard of the benchmark (Chen et al., 2015 PNAS, Fig. 1): 5 rows x 8 columns.
# Key at (row r, column c) flickers at 8 + c + 0.2 r Hz.
KEYS = ["ABCDEFGH", "IJKLMNOP", "QRSTUVWX", "YZ012345", "6789_,.<"]   # '_' = space, '<' = backspace

# Colours (dark theme)
BG, PANEL, GRID = "#0d1321", "#151d30", "#2a3550"
TXT, MUTED = "#e8ecf4", "#8a96b0"
ACCENT, OK, BAD, CUE = "#4cc9f0", "#43d17a", "#ff5c6c", "#ff3b3b"

FPS = 30


# ============================================================================= helpers
def key_layout(stim_freqs):
    """target index -> (row, col, char) using the stimulus frequencies (robust to target order)."""
    layout = {}
    for t, f in enumerate(stim_freqs):
        base = int(np.floor(f + 1e-6))
        row, col = int(round((f - base) / 0.2)), base - 8
        layout[t] = (row, col, KEYS[row][col])
    return layout


def char_to_target(layout):
    return {ch: t for t, (_, _, ch) in layout.items()}


def check_ffmpeg():
    return shutil.which("ffmpeg") is not None


def save_animation(anim, fig, out_mp4, out_gif, gif_width=640):
    import matplotlib.animation as mpla
    if check_ffmpeg():
        anim.save(out_mp4, writer=mpla.FFMpegWriter(fps=FPS, bitrate=4000,
                                                    extra_args=["-pix_fmt", "yuv420p"]),
                  savefig_kwargs={"facecolor": BG})
        print(f"[video] {out_mp4}")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out_mp4), "-vf",
                        f"fps=15,scale={gif_width}:-1:flags=lanczos,split[s0][s1];"
                        "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
                        str(out_gif)], check=True)
        print(f"[gif]   {out_gif}")
    else:
        print("ffmpeg not found: writing GIF only (install ffmpeg for MP4)")
        anim.save(out_gif, writer=mpla.PillowWriter(fps=15), savefig_kwargs={"facecolor": BG})
        print(f"[gif]   {out_gif}")


def load_subject_bands(data_dir, subject):
    idx = data.channel_indices(config.SSVEP_CHANNELS)
    raw = pp.demean(data.load_subject(data_dir, subject)[:, idx, :])          # (240, 9, 1500)
    bands = fb.filter_bank(raw, config.FS, config.BENCHMARK["n_subbands"])     # (240, 5, 9, 1500)
    return raw, bands


def fit_trca(bands, targets, blocks, test_block, window_s):
    w = fb.band_weights(config.BENCHMARK["n_subbands"])
    xw = fb.window(bands, config.FS, window_s)
    tr = blocks != test_block
    return calibrated.EnsembleTRCA(w).fit(xw[tr], targets[tr]), xw


def style_axis(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TXT)


# ============================================================================= speller
def plan_speller(word, bands, targets, blocks, layout, window_s, start_block=0, max_retries=2):
    """Build the sequence of real selections (with real errors and backspaces)."""
    c2t = char_to_target(layout)
    models, scores_cache = {}, {}
    for b in range(config.N_BLOCKS):
        models[b], xw = fit_trca(bands, targets, blocks, b, window_s)
        te = np.where(blocks == b)[0]
        scores_cache[b] = dict(zip(te, models[b].decision_function(xw[te])))
    next_block = {t: start_block for t in range(config.N_TARGETS)}

    def select(target):
        b = next_block[target] % config.N_BLOCKS
        next_block[target] += 1
        trial = int(np.where((targets == target) & (blocks == b))[0][0])
        sc = scores_cache[b][trial]
        return dict(target=target, block=b, trial=trial, scores=sc, pred=int(np.argmax(sc)))

    events, typed = [], []
    for ch in word:
        if ch not in c2t:
            raise ValueError(f"Character '{ch}' is not on the keyboard (use A-Z, 0-9, _ , .)")
        for _ in range(max_retries + 1):
            ev = select(c2t[ch]); events.append(ev)
            typed.append(ev["pred"])
            if ev["pred"] == ev["target"]:
                break
            # wrong character: the user selects backspace (a real trial of '<')
            bs = select(c2t["<"]); events.append(bs)
            if bs["pred"] == bs["target"]:
                typed.pop()
            else:
                typed.append(bs["pred"])
                break
    return events


def render_speller(args, raw, bands, targets, blocks, layout, stim_freqs, events, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as mpla
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    t2rc = {t: (r, c) for t, (r, c, _) in layout.items()}
    t2ch = {t: ch for t, (_, _, ch) in layout.items()}
    disp = lambda ch: "␣" if ch == "_" else ("⌫" if ch == "<" else ch)

    # ---- timeline (real time): cue 0.5 s, stimulation (latency + window), decision 0.7 s
    t_cue, t_stim, t_dec = 0.5, config.LATENCY_S + args.window, 0.7
    n_cue, n_stim, n_dec = (int(round(x * FPS)) for x in (t_cue, t_stim, t_dec))
    frames = []
    for i in range(len(events)):
        frames += [(i, "cue", k / n_cue) for k in range(n_cue)]
        frames += [(i, "stim", k / n_stim) for k in range(n_stim)]
        frames += [(i, "dec", k / n_dec) for k in range(n_dec)]
    frames += [(len(events) - 1, "end", k / FPS) for k in range(int(1.5 * FPS))]

    # ---- display EEG: sub-band 1 (6-90 Hz), 4 occipital channels, -0.5 .. +0.8 s around onset
    show_ch = ["POz", "O1", "Oz", "O2"]
    ch_idx = [config.SSVEP_CHANNELS.index(c) for c in show_ch]
    t0 = -config.STIM_ONSET_S
    t_axis = np.arange(int((config.STIM_ONSET_S + 0.8) * config.FS)) / config.FS + t0
    win = (config.LATENCY_S, config.LATENCY_S + args.window)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax_k = fig.add_axes([0.03, 0.10, 0.50, 0.66]); ax_k.set_facecolor(BG); ax_k.axis("off")
    ax_k.set_xlim(-0.1, 8.1); ax_k.set_ylim(-0.1, 5.1); ax_k.invert_yaxis()
    ax_e = fig.add_axes([0.58, 0.40, 0.40, 0.42]); style_axis(ax_e)
    ax_s = fig.add_axes([0.58, 0.10, 0.40, 0.20]); style_axis(ax_s)

    fig.text(0.03, 0.95, "SSVEP speller — replay of real EEG", color=TXT, fontsize=18, weight="bold")
    fig.text(0.03, 0.915, f"Subject S{args.subject:02d} · ensemble TRCA trained on the other 5 blocks · "
             f"{args.window:g} s of EEG per character", color=MUTED, fontsize=10.5)
    fig.text(0.03, 0.03, "Flicker slowed 8× for display (real stimuli: 8–15.8 Hz). "
             "Data: Tsinghua SSVEP Benchmark (Wang et al., 2017).", color=MUTED, fontsize=8.5)

    # typed text bar
    bar = FancyBboxPatch((0.03, 0.80), 0.50, 0.075, transform=fig.transFigure,
                         boxstyle="round,pad=0.005,rounding_size=0.01", fc=PANEL, ec=GRID)
    fig.patches.append(bar)
    max_chars, dx = 26, 0.0172
    typed_chars = [fig.text(0.045 + k * dx, 0.837, "", color=TXT, fontsize=24, family="monospace",
                            va="center") for k in range(max_chars + 1)]
    status_txt = fig.text(0.52, 0.837, "", color=MUTED, fontsize=11, ha="right", va="center")

    # keys
    key_patch, key_label = {}, {}
    for t, (r, c) in t2rc.items():
        p = FancyBboxPatch((c + 0.08, r + 0.08), 0.84, 0.84, boxstyle="round,pad=0,rounding_size=0.08",
                           fc=PANEL, ec=GRID, lw=1.5)
        ax_k.add_patch(p); key_patch[t] = p
        key_label[t] = ax_k.text(c + 0.5, r + 0.45, disp(t2ch[t]), ha="center", va="center",
                                 color=TXT, fontsize=17, weight="bold")
        ax_k.text(c + 0.5, r + 0.78, f"{stim_freqs[t]:.1f} Hz", ha="center", va="center",
                  color=MUTED, fontsize=6.5)

    # EEG panel
    ax_e.set_xlim(t_axis[0], t_axis[-1]); ax_e.set_ylim(-0.8, len(show_ch) - 0.2)
    ax_e.set_yticks(range(len(show_ch)), show_ch[::-1]); ax_e.set_xlabel("time from stimulus onset (s)")
    ax_e.set_title("EEG, occipital channels (6–90 Hz)", fontsize=11, loc="left", color=TXT)
    ax_e.axvspan(*win, color=ACCENT, alpha=0.12, lw=0)
    ax_e.text(np.mean(win), len(show_ch) - 0.35, "analysis window", color=ACCENT, fontsize=8, ha="center")
    ax_e.axvline(0, color=GRID, lw=1, ls="--")
    eeg_lines = [ax_e.plot([], [], color=TXT, lw=0.9)[0] for _ in show_ch]
    cursor = ax_e.axvline(t0, color=ACCENT, lw=1.2)

    # score panel
    order = np.argsort(stim_freqs)
    ax_s.set_xlim(-0.7, config.N_TARGETS - 0.3); ax_s.set_xticks([])
    ax_s.set_title("TRCA score for each of the 40 targets", fontsize=10, loc="left", color=TXT)
    score_bars = ax_s.bar(range(config.N_TARGETS), np.zeros(config.N_TARGETS), color=MUTED, width=0.75)
    ax_s.set_yticks([])
    sc_all = np.array([ev["scores"] for ev in events])
    ax_s.set_ylim(min(0, sc_all.min()), sc_all.max() * 1.1)

    def trace(ev):
        x = bands[ev["trial"], 0][ch_idx, :len(t_axis)]
        x = x / (3 * x.std(axis=1, keepdims=True) + 1e-12)
        return x

    traces = [trace(ev) for ev in events]
    state = {"done": -1, "typed": []}

    def apply_decision(i):
        ev = events[i]
        if t2ch[ev["target"]] == "<" and ev["pred"] == ev["target"]:
            if state["typed"]:
                state["typed"].pop()
        else:
            state["typed"].append((disp(t2ch[ev["pred"]]), ev["pred"] == ev["target"]))

    def draw_typed(blink_on):
        shown = state["typed"][-max_chars:]
        for k, txt in enumerate(typed_chars):
            if k < len(shown):
                txt.set_text(shown[k][0]); txt.set_color(TXT if shown[k][1] else BAD)
            elif k == len(shown):
                txt.set_text("▌" if blink_on else ""); txt.set_color(ACCENT)
            else:
                txt.set_text("")

    def update(frame):
        i, phase, u = frame
        ev = events[i]
        while state["done"] < i - 1:               # decisions of earlier events
            state["done"] += 1; apply_decision(state["done"])
        # keys
        for t in key_patch:
            key_patch[t].set_facecolor(PANEL); key_patch[t].set_edgecolor(GRID); key_patch[t].set_linewidth(1.5)
        if phase == "cue":
            key_patch[ev["target"]].set_edgecolor(CUE); key_patch[ev["target"]].set_linewidth(4)
        elif phase == "stim":
            tt = u * t_stim
            for t in key_patch:
                lum = 0.5 + 0.5 * np.sin(2 * np.pi * (stim_freqs[t] / 8) * tt)
                key_patch[t].set_facecolor(matplotlib.colors.to_hex(
                    np.array(matplotlib.colors.to_rgb(PANEL)) * (1 - 0.35 * lum)
                    + np.array(matplotlib.colors.to_rgb("#3a4a74")) * 0.35 * lum))
        elif phase in ("dec", "end"):
            if phase == "dec" and state["done"] < i:
                state["done"] = i; apply_decision(i)
            ok = ev["pred"] == ev["target"]
            key_patch[ev["pred"]].set_facecolor(OK if ok else BAD)
            key_patch[ev["pred"]].set_edgecolor(TXT)
        # EEG
        if phase == "cue":
            t_now = t0 + u * t_cue
        elif phase == "stim":
            t_now = u * t_stim
        else:
            t_now = t_stim
        n = int(np.searchsorted(t_axis, t_now))
        for k, ln in enumerate(eeg_lines):
            ln.set_data(t_axis[:n], traces[i][k, :n] + (len(show_ch) - 1 - k))
        cursor.set_xdata([t_now, t_now])
        # scores
        show = phase in ("dec", "end")
        for j, t in enumerate(order):
            score_bars[j].set_visible(show)
            score_bars[j].set_height(ev["scores"][t] if show else 0)
            score_bars[j].set_color(OK if (show and t == ev["target"] and ev["pred"] == t)
                                    else BAD if (show and t == ev["pred"]) else
                                    ACCENT if (show and t == ev["target"]) else MUTED)
        # text
        draw_typed((frame_count[0] // 15) % 2 == 0)
        target_ch = disp(t2ch[ev["target"]])
        status_txt.set_text({"cue": f"look at  {target_ch}", "stim": "decoding…",
                             "dec": ("✓ correct" if ev["pred"] == ev["target"] else "✗ error"),
                             "end": f"{n_ok}/{len(events)} selections correct"}[phase])
        status_txt.set_color(OK if (phase == "dec" and ev["pred"] == ev["target"])
                             else BAD if phase == "dec" else MUTED)
        frame_count[0] += 1
        return []

    n_ok = sum(ev["pred"] == ev["target"] for ev in events)
    frame_count = [0]
    anim = mpla.FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)
    save_animation(anim, fig, out_dir / "speller.mp4", out_dir / "speller.gif")
    # poster: final frame
    state.update(done=-1, typed=[]); frame_count[0] = 0
    for f in frames:
        update(f)
    fig.savefig(out_dir / "speller_poster.png", facecolor=BG, dpi=100)
    print(f"[png]   {out_dir / 'speller_poster.png'}")
    plt.close(fig)


# ============================================================================= anatomy
def render_anatomy(args, raw, bands, targets, blocks, layout, stim_freqs, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as mpla
    import matplotlib.pyplot as plt

    c2t = char_to_target(layout)
    t2ch = {t: ch for t, (_, _, ch) in layout.items()}
    target = c2t[args.char]
    tb = args.block - 1
    trial = int(np.where((targets == target) & (blocks == tb))[0][0])
    fs, lat = config.FS, config.LATENCY_S

    # data lengths shown, and the decision after each
    lengths = np.round(np.arange(0.10, args.max_window + 1e-9, 1 / fs * 2), 4)
    print(f"[anatomy] fitting TRCA for {len(lengths)} window lengths ...")
    scores = []
    for L in lengths:
        model, xw = fit_trca(bands, targets, blocks, tb, L)
        scores.append(model.decision_function(xw[trial:trial + 1])[0])
    scores = np.array(scores)                                     # (n_lengths, 40)
    preds = scores.argmax(1)
    lock = next((k for k in range(len(lengths)) if np.all(preds[k:] == target)), None)
    lock_s = lengths[lock] if lock is not None else None
    print(f"[anatomy] target {args.char}, block {args.block}: correct from "
          f"{lock_s:.2f} s onward" if lock_s else "[anatomy] never locks in on the target")

    # display model for panel (c): filter for the true class, fitted on the longest window
    model_full, xw_full = fit_trca(bands, targets, blocks, tb, args.max_window)
    w_k = model_full.W_[0][:, list(model_full.classes_).index(target)]
    tr = (blocks != tb) & (targets == target)
    t_rel = np.arange(-int(0.2 * fs), int((lat + args.max_window + 0.1) * fs)) / fs   # rel. to onset
    s0 = int(config.STIM_ONSET_S * fs)
    seg = slice(s0 + int(t_rel[0] * fs), s0 + int(t_rel[0] * fs) + len(t_rel))
    proj_trial = w_k @ bands[trial, 0][:, seg]
    proj_tmpl = w_k @ bands[tr, 0].mean(axis=0)[:, seg]
    if np.corrcoef(proj_trial, proj_tmpl)[0, 1] < 0:   # sign of an eigenvector is arbitrary
        w_k = -w_k; proj_trial, proj_tmpl = -proj_trial, -proj_tmpl
    sc = max(np.abs(proj_trial).max(), np.abs(proj_tmpl).max())
    proj_trial, proj_tmpl = proj_trial / sc, proj_tmpl / sc

    raw_seg = raw[trial][:, seg]; raw_seg = raw_seg / (4 * raw_seg.std(axis=1, keepdims=True))
    oz = config.SSVEP_CHANNELS.index("Oz")
    fb_seg = bands[trial, :, oz, seg]; fb_seg = fb_seg / (4 * fb_seg.std(axis=1, keepdims=True))

    # ---- frames: 1 s intro, data grows slowed x args.slow, 2.5 s hold
    n_intro, n_hold = FPS, int(2.5 * FPS)
    n_grow = int(round((lengths[-1] - lengths[0]) * args.slow * FPS))
    frame_L = np.concatenate([np.full(n_intro, np.nan), np.linspace(lengths[0], lengths[-1], n_grow),
                              np.full(n_hold, lengths[-1])])

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    fig.text(0.03, 0.95, "Anatomy of one decision", color=TXT, fontsize=18, weight="bold")
    fig.text(0.03, 0.915, f"Subject S{args.subject:02d}, target '{args.char}' ({stim_freqs[target]:.1f} Hz), "
             f"block {args.block} held out · ensemble TRCA", color=MUTED, fontsize=10.5)
    gs = fig.add_gridspec(3, 2, left=0.06, right=0.97, top=0.86, bottom=0.08, wspace=0.18, hspace=0.45,
                          width_ratios=[1.15, 1])
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[i, 0]) for i in range(3))
    ax_d = fig.add_subplot(gs[:, 1])
    for ax in (ax_a, ax_b, ax_c, ax_d):
        style_axis(ax)

    shades, cursors = [], []
    for ax, title in ((ax_a, "① raw EEG, 9 parieto-occipital channels"),
                      (ax_b, "② filter bank at Oz: 5 harmonic sub-bands"),
                      (ax_c, "③ TRCA spatial filter: this trial vs the user's template")):
        ax.set_xlim(t_rel[0], t_rel[-1]); ax.set_title(title, fontsize=10.5, loc="left", color=TXT)
        ax.axvline(0, color=GRID, lw=1, ls="--")
        shades.append(ax.axvspan(lat, lat, color=ACCENT, alpha=0.15, lw=0))
        cursors.append(ax.axvline(lat, color=ACCENT, lw=1.2, alpha=0))
    ax_c.set_xlabel("time from stimulus onset (s)")
    for i in range(raw_seg.shape[0]):
        ax_a.plot(t_rel, raw_seg[i] + (8 - i), color=TXT, lw=0.6, alpha=0.85)
    ax_a.set_yticks(range(9), config.SSVEP_CHANNELS[::-1]); ax_a.tick_params(axis="y", labelsize=6.5)
    ax_a.set_ylim(-1, 9)
    band_cols = ["#4cc9f0", "#4895ef", "#4361ee", "#7209b7", "#b5179e"]
    for i in range(fb_seg.shape[0]):
        ax_b.plot(t_rel, fb_seg[i] + (4 - i), color=band_cols[i], lw=0.7)
    ax_b.set_yticks(range(5), [f"SB{5 - i}" for i in range(5)]); ax_b.set_ylim(-1, 5)
    ax_c.plot(t_rel, proj_tmpl, color=MUTED, lw=1.4, label="template (calibration average)")
    trial_line, = ax_c.plot([], [], color=ACCENT, lw=1.4, label="this trial")
    ax_c.set_ylim(-1.15, 1.15); ax_c.set_yticks([])
    ax_c.legend(loc="lower left", fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, ncol=2)

    order = np.argsort(stim_freqs)
    labels = [t2ch[t].replace("_", "␣").replace("<", "⌫") for t in order]
    ax_d.set_title("④ score for each of the 40 targets", fontsize=10.5, loc="left", color=TXT)
    bars = ax_d.barh(range(config.N_TARGETS), np.zeros(config.N_TARGETS), color=MUTED, height=0.72)
    ax_d.set_yticks(range(config.N_TARGETS), [f"{l}  {stim_freqs[t]:.1f}" for l, t in zip(labels, order)],
                    fontsize=6.3)
    ax_d.invert_yaxis()
    ax_d.set_xlim(min(0, scores.min()) * 1.05, scores.max() * 1.15)
    ax_d.set_xlabel("ensemble TRCA score")
    big = fig.text(0.585, 0.885, "", color=TXT, fontsize=14, weight="bold")
    sub = fig.text(0.585, 0.862, "", color=MUTED, fontsize=9.5)
    row_of = {t: j for j, t in enumerate(order)}

    def update(k):
        L = frame_L[k]
        if np.isnan(L):
            big.set_text("waiting for data…"); sub.set_text("")
            return []
        idx = int(np.argmin(np.abs(lengths - L)))
        end = lat + L
        for sh in shades:
            if hasattr(sh, "set_width"):          # matplotlib >= 3.9: axvspan returns a Rectangle
                sh.set_x(lat); sh.set_width(end - lat)
            else:                                 # older matplotlib: a Polygon
                sh.set_xy([[lat, 0], [lat, 1], [end, 1], [end, 0], [lat, 0]])
        for cu in cursors:
            cu.set_xdata([end, end]); cu.set_alpha(1)
        m = (t_rel >= lat) & (t_rel <= end)
        trial_line.set_data(t_rel[m], proj_trial[m])
        sc_now, pred = scores[idx], preds[idx]
        for t in range(config.N_TARGETS):
            j = row_of[t]
            bars[j].set_width(sc_now[t])
            bars[j].set_color(OK if (t == pred == target) else BAD if t == pred else
                              ACCENT if t == target else MUTED)
        ok = pred == target
        big.set_text(f"{L:.2f} s of EEG → decision: {labels[row_of[pred]]}  {'✓' if ok else '✗'}")
        big.set_color(OK if ok else BAD)
        sub.set_text(f"locked on the correct target from {lock_s:.2f} s" if (lock_s and L >= lock_s)
                     else "true target highlighted in blue")
        return []

    anim = mpla.FuncAnimation(fig, update, frames=len(frame_L), interval=1000 / FPS, blit=False)
    save_animation(anim, fig, out_dir / "anatomy.mp4", out_dir / "anatomy.gif")
    update(len(frame_L) - 1)
    fig.savefig(out_dir / "anatomy_poster.png", facecolor=BG, dpi=100)
    print(f"[png]   {out_dir / 'anatomy_poster.png'}")
    plt.close(fig)


# ============================================================================= main
def main():
    p = argparse.ArgumentParser(description="Website animations from real SSVEP data")
    p.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR)
    p.add_argument("--which", choices=["speller", "anatomy", "both"], default="both")
    p.add_argument("--subject", type=int, default=3)
    p.add_argument("--word", default="HELLO_DTU", help="A-Z, 0-9, _ (space), comma, period")
    p.add_argument("--window", type=float, default=0.5, help="speller: seconds of EEG per character")
    p.add_argument("--start-block", type=int, default=1, help="speller: first test block (1-6)")
    p.add_argument("--char", default="T", help="anatomy: target character")
    p.add_argument("--block", type=int, default=1, help="anatomy: held-out block (1-6)")
    p.add_argument("--max-window", type=float, default=1.0, help="anatomy: longest data length (s)")
    p.add_argument("--slow", type=float, default=5.0, help="anatomy: slow-motion factor")
    p.add_argument("--plan-only", action="store_true", help="print the speller sequence and exit")
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR / "media")
    args = p.parse_args()
    args.word = args.word.upper().replace(" ", "_")
    args.char = args.char.upper()

    data.check_data_dir(args.data_dir, [args.subject])
    stim_freqs, _ = data.load_freq_phase(args.data_dir)
    layout = key_layout(stim_freqs)
    targets, blocks = data.trial_labels()
    print(f"[data] loading subject {args.subject} ...")
    raw, bands = load_subject_bands(args.data_dir, args.subject)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.which in ("speller", "both") or args.plan_only:
        events = plan_speller(args.word, bands, targets, blocks, layout, args.window, args.start_block - 1)
        t2ch = {t: ch for t, (_, _, ch) in layout.items()}
        print("\nSpeller plan (real held-out predictions):")
        for ev in events:
            mark = "ok " if ev["pred"] == ev["target"] else "ERR"
            print(f"  block {ev['block'] + 1}  target '{t2ch[ev['target']]}'  ->  predicted "
                  f"'{t2ch[ev['pred']]}'  {mark}")
        n_ok = sum(ev["pred"] == ev["target"] for ev in events)
        print(f"  {n_ok}/{len(events)} correct\n")
        if args.plan_only:
            return
        render_speller(args, raw, bands, targets, blocks, layout, stim_freqs, events, args.out)
    if args.which in ("anatomy", "both"):
        render_anatomy(args, raw, bands, targets, blocks, layout, stim_freqs, args.out)


if __name__ == "__main__":
    main()
