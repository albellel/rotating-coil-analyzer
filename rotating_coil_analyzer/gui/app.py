from __future__ import annotations

from pathlib import Path

import ipywidgets as w
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from rotating_coil_analyzer.ingest.discovery import MeasurementDiscovery
from rotating_coil_analyzer.ingest.readers_sm18 import Sm18CorrSigsReader


def _browse_for_folder() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select measurement folder")
        root.destroy()
        return folder or None
    except Exception:
        return None


def build_catalog_gui() -> w.VBox:
    discovery = MeasurementDiscovery()
    reader = Sm18CorrSigsReader()

    folder = w.Text(
        description="Folder",
        placeholder="Paste measurement folder path here (or use Browse...)",
        layout=w.Layout(width="75%"),
    )
    browse_btn = w.Button(description="Browse…")
    load_btn = w.Button(description="Load Catalog", button_style="primary")

    run_dd = w.Dropdown(description="Run", options=[])
    seg_dd = w.Dropdown(description="Segment", options=[])
    mode_dd = w.Dropdown(description="Mode", options=[("abs", "abs"), ("cmp", "cmp")], value="abs")

    preview_btn = w.Button(description="Preview")
    diag_btn = w.Button(description="Diagnostics", button_style="warning")

    out = w.Output(layout=w.Layout(border="1px solid #ccc", padding="8px"))

    state = {"catalog": None, "last_seg_frame": None, "last_mode": None}

    def _refresh_dropdowns():
        cat = state["catalog"]
        if cat is None:
            run_dd.options = []
            seg_dd.options = []
            return
        run_dd.options = [(r.run_id, r.run_id) for r in cat.runs]
        seg_dd.options = [(s.segment, s.segment) for s in cat.segments]

    def _on_browse(_btn):
        picked = _browse_for_folder()
        with out:
            if picked is None:
                print("Browse failed (likely headless environment). Please paste the folder path manually.")
                return
            folder.value = picked
            print(f"Selected folder: {picked}")

    def _on_load(_btn):
        with out:
            out.clear_output()
            try:
                p = Path(folder.value).expanduser()
                print("=== INPUT DEBUG ===")
                print("selected:", p)
                print("exists:", p.exists(), "is_dir:", p.is_dir())

                cat = discovery.build_catalog(p)
                state["catalog"] = cat

                print("\n=== CATALOG DEBUG ===")
                print("root_dir:", cat.root_dir)
                print("Parameters.txt:", cat.parameters_path)
                print("samples_per_turn:", cat.samples_per_turn)
                print("shaft_speed_rpm:", cat.shaft_speed_rpm)
                print("enabled_apertures:", list(cat.enabled_apertures))
                print("segments:", [(s.segment, s.fdi_abs, s.fdi_cmp, s.length_m) for s in cat.segments])
                print("runs_found:", len(cat.runs))
                print("segment_files_found:", len(cat.segment_files))

                if cat.warnings:
                    print("\nWARNINGS:")
                    for wmsg in cat.warnings:
                        print(" -", wmsg)

                _refresh_dropdowns()

            except Exception as e:
                print("ERROR:", repr(e))

    def _load_selected_segment_frame() -> tuple[Path, str, str]:
        cat = state["catalog"]
        if cat is None:
            raise RuntimeError("Load a catalog first.")
        run_id = run_dd.value
        seg = seg_dd.value
        if run_id is None or seg is None:
            raise RuntimeError("Select Run and Segment.")
        key = (run_id, seg)
        if key not in cat.segment_files:
            raise RuntimeError(f"No segment file for run={run_id}, segment={seg}.")
        return cat.segment_files[key], run_id, seg

    def _on_preview(_btn):
        with out:
            out.clear_output()
            try:
                cat = state["catalog"]
                if cat is None:
                    print("Load a catalog first.")
                    return

                fpath, run_id, seg = _load_selected_segment_frame()
                mode = mode_dd.value

                print("=== PREVIEW DEBUG ===")
                print("file:", fpath)

                seg_frame = reader.read(
                    fpath,
                    run_id=run_id,
                    segment=seg,
                    samples_per_turn=cat.samples_per_turn,
                    shaft_speed_rpm=cat.shaft_speed_rpm,
                )
                state["last_seg_frame"] = seg_frame
                state["last_mode"] = mode

                for msg in seg_frame.warnings:
                    print("CHECK:", msg)

                df = seg_frame.df
                if mode == "abs":
                    ch = df[["t", "df_abs", "I"]].rename(columns={"df_abs": "df"})
                else:
                    ch = df[["t", "df_cmp", "I"]].rename(columns={"df_cmp": "df"})

                print(f"rows: {len(ch)}  n_turns: {seg_frame.n_turns}")
                print("head:")
                display(ch.head(10))

                Ns = cat.samples_per_turn
                n_show = min(len(ch), 3 * Ns)

                plt.figure()
                plt.plot(ch["t"].iloc[:n_show], ch["df"].iloc[:n_show])
                plt.title(f"{seg} {mode} — first 3 turns")
                plt.xlabel("t [s]")
                plt.ylabel(r"$\Delta\Phi$ (selected channel)")
                plt.show()

                plt.figure()
                plt.plot(ch["t"].iloc[:n_show], ch["I"].iloc[:n_show])
                plt.title(f"{seg} current — first 3 turns")
                plt.xlabel("t [s]")
                plt.ylabel("I [A]")
                plt.show()

            except Exception as e:
                print("ERROR:", repr(e))

    def _on_diag(_btn):
        with out:
            out.clear_output()
            seg_frame = state.get("last_seg_frame")
            mode = state.get("last_mode")
            cat = state.get("catalog")

            if seg_frame is None or mode is None or cat is None:
                print("Run Preview first, then Diagnostics.")
                return

            Ns = cat.samples_per_turn
            df = seg_frame.df
            t = df["t"].to_numpy(dtype=np.float64, copy=False)

            n_turns = len(t) // Ns
            tR = t[: n_turns * Ns].reshape((n_turns, Ns))
            dR = np.diff(tR, axis=1)

            per_turn_mono_frac = float(np.mean(np.all(dR > 0, axis=1)))
            dt_global = np.diff(t[: n_turns * Ns])
            global_mono_frac = float(np.mean(dt_global > 0))

            time_mode = "global_monotone" if global_mono_frac > 0.999 else ("per_turn_sawtooth" if per_turn_mono_frac > 0.95 else "invalid")

            print("=== TIME AXIS DIAGNOSTICS (Phase-1 quick tool) ===")
            print(f"segment: {seg_frame.segment} mode: {mode} run: {seg_frame.run_id}")
            print(f"samples_per_turn: {Ns}")
            print(f"time_mode={time_mode}, global_mono_frac={global_mono_frac:.6f}, per_turn_mono_frac={per_turn_mono_frac:.3f}")

            if time_mode == "global_monotone":
                dt = dt_global
            elif time_mode == "per_turn_sawtooth":
                dt = dR.ravel()
            else:
                print("Time axis invalid; reader should normally have rejected this.")
                return

            dt_med = float(np.median(dt))
            alpha = 10.0
            thr = alpha * dt_med
            jumps = np.where(dt > thr)[0]

            print(f"dt median={dt_med:.6g} s, threshold={alpha:.0f}*median={thr:.6g} s")
            print(f"n dt-jumps: {len(jumps)} (showing first 30)")

            for k, i in enumerate(jumps[:30]):
                print(f"[{k}] index={i}  dt={dt[i]:.6g} s")

            Tturn = tR[:, -1] - tR[:, 0]
            Tmed = float(np.median(Tturn))
            tol = 0.01
            bad = np.where(np.abs(Tturn / Tmed - 1.0) > tol)[0]

            print("\n=== TURN DURATION QC (Phase-1 quick tool) ===")
            print(f"median(T)={Tmed:.6g} s, tolerance=±{tol*100:.0f}%")
            print(f"n bad turns: {len(bad)}")
            if len(bad) > 0:
                print("bad turns (1-based, first 80):", (bad[:80] + 1).tolist())

    browse_btn.on_click(_on_browse)
    load_btn.on_click(_on_load)
    preview_btn.on_click(_on_preview)
    diag_btn.on_click(_on_diag)

    folder_row = w.HBox([folder, browse_btn, load_btn])
    selector_row = w.HBox([run_dd, seg_dd, mode_dd, preview_btn, diag_btn])
    return w.VBox([folder_row, selector_row, out])
