#!/usr/bin/env python3
"""
PROJECT_UI.PY
Simple text-based user interface for the land cover project.
"""

import os
import sys
import subprocess

import config

# Paths to scripts (adjust names/paths if different)
MAIN_PIPELINE = "main_pipeline.py"
PREDICT_NEW_IMAGE = "predict_new_image.py"
NDVI_EXPLORER = "NDVI_THRESHOLD_EXPLORER.py"


def run_python(script):
    if not os.path.exists(script):
        print(f"[ERROR] Script not found: {script}")
        return
    cmd = [sys.executable, script]
    print(f"\n>>> Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def menu():
    while True:
        print("\n" + "="*70)
        print(" LAND COVER CLASSIFICATION - USER INTERFACE")
        print("="*70)
        print("Current BBOX:", getattr(config, "BBOX", "not set"))
        print("\nChoose an option:")
        print("  1) Run FULL pipeline (all stages)")
        print("  2) Run ONLY data acquisition (download image)")
        print("  3) Run ONLY traditional ML (Stages 3–4)")
        print("  4) Run ONLY deep learning (Stage 5)")
        print("  5) Classify NEW image with best ML model")
        print("  6) Launch NDVI threshold explorer")
        print("  0) Exit")
        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            run_python(MAIN_PIPELINE)

        elif choice == "2":
            from data_acquisition import LandsatDataAcquisition
            acq = LandsatDataAcquisition()
            imagery, metadata, stats = acq.run()
            if imagery is not None:
                print("✓ Acquisition complete.")
            else:
                print("✗ Acquisition failed.")

        elif choice == "3":
            # Minimal partial run: labels + traditional ML
            from main_pipeline import LandCoverClassificationPipeline
            p = LandCoverClassificationPipeline()
            p.run_stage_1_acquisition()
            p.run_stage_2_feature_extraction()
            p.run_stage_3_labels()
            p.run_stage_4_traditional_ml()
            print("✓ Traditional ML training complete.")

        elif choice == "4":
            from main_pipeline import LandCoverClassificationPipeline
            p = LandCoverClassificationPipeline()
            p.run_stage_1_acquisition()
            p.run_stage_2_feature_extraction()
            p.run_stage_3_labels()
            p.run_stage_5_deep_learning()
            print("✓ Deep learning training complete.")

        elif choice == "5":
            run_python(PREDICT_NEW_IMAGE)

        elif choice == "6":
            run_python(NDVI_EXPLORER)

        elif choice == "0":
            print("Goodbye.")
            break

        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    menu()
