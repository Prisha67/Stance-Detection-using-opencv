# Stance-Detection-using-opencv

This repository contains a small demo and a MediaPipe-based pose detection notebook.

Files of interest
- [demo.py]— simple Hello World script.
- [.vir_proj/poseDetection.ipynb] — MediaPipe Holistic notebook that captures webcam frames, draws landmarks, exports coords and trains a model.
- [.vscode/settings.json]— workspace settings.

Prerequisites
- Python 3.8+ and pip
- Optional: virtual environment (venv)

Quick start
1. Create and activate a venv:
   - Windows (PowerShell): `python -m venv .venv && .\.venv\Scripts\Activate.ps1`
   - macOS / Linux: `python -m venv .venv && source .venv/bin/activate`
2. Install dependencies (example):

pip install opencv-python mediapipe numpy pandas scikit-learn

3. Run the demo:
- Script: `python demo.py` — see [demo.py](demo.py)
- Notebook: open [.vir_proj/poseDetection.ipynb](.vir_proj/poseDetection.ipynb) in Jupyter / VS Code

Notes
- Exclude virtual environments from the repo. See `.gitignore` below.

# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Jupyter
.ipynb_checkpoints

# VS Code
.vscode/settings.json
.vscode/.*
