#!/usr/bin/env python3
"""
GUI.PY
Simple Tkinter GUI for the land cover project.
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox

# Make sure local modules are importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config


MAIN_PIPELINE = "main_pipeline.py"
PREDICT_NEW_IMAGE = "predict_new_image.py"
NDVI_EXPLORER = "NDVI_THRESHOLD_EXPLORER.py"


def run_script(script):
    if not os.path.exists(os.path.join(CURRENT_DIR, script)):
        messagebox.showerror("Error", f"Script not found:\n{script}")
        return
    cmd = [sys.executable, os.path.join(CURRENT_DIR, script)]
    subprocess.Popen(cmd)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Land Cover Classification GUI")
        self.geometry("520x320")
        self._build_bbox_section()
        self._build_buttons()

    def _build_bbox_section(self):
        frame = tk.LabelFrame(self, text="Area of Interest (BBOX)", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=10)

        # Current BBOX from config
        bbox = getattr(config, "BBOX", [-4.5, 36.7, -4.3, 36.8])
        min_lon, min_lat, max_lon, max_lat = bbox

        tk.Label(frame, text="Min Lon:").grid(row=0, column=0, sticky="e")
        tk.Label(frame, text="Min Lat:").grid(row=0, column=2, sticky="e")
        tk.Label(frame, text="Max Lon:").grid(row=1, column=0, sticky="e")
        tk.Label(frame, text="Max Lat:").grid(row=1, column=2, sticky="e")

        self.entry_min_lon = tk.Entry(frame, width=10)
        self.entry_min_lat = tk.Entry(frame, width=10)
        self.entry_max_lon = tk.Entry(frame, width=10)
        self.entry_max_lat = tk.Entry(frame, width=10)

        self.entry_min_lon.grid(row=0, column=1, padx=5, pady=2)
        self.entry_min_lat.grid(row=0, column=3, padx=5, pady=2)
        self.entry_max_lon.grid(row=1, column=1, padx=5, pady=2)
        self.entry_max_lat.grid(row=1, column=3, padx=5, pady=2)

        self.entry_min_lon.insert(0, str(min_lon))
        self.entry_min_lat.insert(0, str(min_lat))
        self.entry_max_lon.insert(0, str(max_lon))
        self.entry_max_lat.insert(0, str(max_lat))

        btn = tk.Button(frame, text="Apply BBOX to config.py", command=self.update_bbox)
        btn.grid(row=2, column=0, columnspan=4, pady=6)

    def _build_buttons(self):
        frame = tk.LabelFrame(self, text="Actions", padx=10, pady=10)
        frame.pack(fill="both", expand=True, padx=10, pady=5)

        btn_full = tk.Button(frame, text="Run FULL Pipeline", width=25,
                             command=lambda: run_script(MAIN_PIPELINE))
        btn_ml = tk.Button(frame, text="Classify NEW Image (Best ML)", width=25,
                           command=lambda: run_script(PREDICT_NEW_IMAGE))
        btn_ndvi = tk.Button(frame, text="Open NDVI Threshold Explorer", width=25,
                             command=lambda: run_script(NDVI_EXPLORER))

        btn_full.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        btn_ml.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        btn_ndvi.grid(row=2, column=0, padx=5, pady=5, sticky="w")

    def update_bbox(self):
        """Write BBOX values back into config.py."""
        try:
            min_lon = float(self.entry_min_lon.get())
            min_lat = float(self.entry_min_lat.get())
            max_lon = float(self.entry_max_lon.get())
            max_lat = float(self.entry_max_lat.get())
        except ValueError:
            messagebox.showerror("Error", "BBOX values must be numbers.")
            return

        if not (min_lon < max_lon and min_lat < max_lat):
            messagebox.showerror("Error", "Require: min_lon < max_lon and min_lat < max_lat.")
            return

        bbox_str = f"BBOX = [{min_lon}, {min_lat}, {max_lon}, {max_lat}]\n"

        config_path = os.path.join(CURRENT_DIR, "config.py")
        if not os.path.exists(config_path):
            messagebox.showerror("Error", f"config.py not found:\n{config_path}")
            return

        # Replace existing BBOX line or append if missing
        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith("BBOX"):
                lines[i] = bbox_str
                found = True
                break
        if not found:
            lines.append("\n" + bbox_str)

        with open(config_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Update in-memory config too
        config.BBOX = [min_lon, min_lat, max_lon, max_lat]

        messagebox.showinfo("BBOX updated",
                            f"New BBOX saved to config.py:\n{bbox_str.strip()}")

if __name__ == "__main__":
    app = App()
    app.mainloop()

