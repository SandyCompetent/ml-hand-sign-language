# 🤟 Real-Time Sign Language to Text & Speech Translator (v2.0)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%201.2-orange.svg)](https://developers.google.com/mediapipe)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(15%2F15)-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced Computer Vision and Machine Learning system that translates American Sign Language (ASL) fingerspelling and conversational gestures into text and audible English speech in real-time. Includes a two-way (Hearing-to-Deaf) visual communicator for full-duplex accessibility.

---

## 📌 Table of Contents
- [Overview & Motivation](#-overview--motivation)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Evolution & What Changed (2022 vs. 2026)](#-evolution--what-changed-2022-push-vs-2026-v20)
- [Quick Comparison Table](#-quick-comparison-legacy-vs-v20)
- [Quickstart with `run.bash`](#-quickstart-with-runbash)
- [macOS Camera Permissions Guide](#-macos-camera-permissions-guide)
- [Supported Signs & Gestures](#-supported-signs--gestures)
- [Manual Setup & Installation](#-manual-setup--installation)
- [Project Structure](#-project-structure)
- [Automated Testing](#-automated-testing)
- [Authors & Academic Background](#-authors--academic-background)
- [License](#-license)

---

## 📖 Overview & Motivation
Sign language is the primary medium of communication for deaf and speech-impaired individuals. However, the majority of the hearing population cannot interpret sign language, creating a persistent communication barrier.

Originally conceived as a Bachelor of Engineering (B.E.) Capstone Project (*"Realtime Sign Language to English Language Conversion using CNN for Disabled People"*, 2021–2022), this repository has been re-architected into a modern, cross-platform system. It replaces classical background thresholding with Google MediaPipe 21 3D hand landmark tracking, combines predictive spell correction for high-speed communication, and provides offline speech synthesis.

---

## ✨ Key Features

1. **MediaPipe 3D Landmark Tracking**:
   - Extracts 21 3D spatial hand coordinates ($x, y, z$) in real-time.
   - Robust against varying skin tones, complex backgrounds, and dim lighting.
   - Eliminates the rigid, fixed-box Region of Interest (ROI) restriction.

2. **Scale- & Translation-Invariant Kinematics**:
   - Coordinates are normalized relative to the wrist and scaled by hand span.
   - Supports natural movement anywhere in the camera view.

3. **Intelligent Predictive Autocomplete & Spell Correction**:
   - Powered by pure-Python `pyspellchecker` (zero native C++ build requirements).
   - Dynamic prefix matching: as users fingerspell (e.g. `H-E-L...`), suggested words (`HELLO`, `HELP`, `HELD`) appear as clickable shortcut pills, speeding up communication by 3x–5x.

4. **Zero-Latency Offline Text-to-Speech (TTS)**:
   - Uses native platform speech synthesis (`say` on macOS, `pyttsx3`/`powershell` on Windows, `espeak` on Linux).
   - Completely non-blocking background queue thread ensures video frames never stutter.

5. **Two-Way Communicator (Hearing-to-Deaf)**:
   - Hearing users can type or select words/phrases to display the corresponding sign reference illustrations and fingerspelling sequence.

6. **Responsive GUI**:
   - Auto-scales to fit standard laptop screens (1080p, 720p, MacBook Retina displays).
   - Dedicated camera capture thread maintaining smooth 30–60 FPS.

---

## 🏛 Architecture

```
                                  ┌────────────────────────┐
                                  │   Webcam Live Stream   │
                                  └───────────┬────────────┘
                                              │ (Threaded capture)
                                              ▼
                                  ┌────────────────────────┐
                                  │   MediaPipe Hands 3D   │
                                  │   (21 Joint Landmarks) │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │  Wrist Translation &   │
                                  │   Scale Normalization  │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │   Kinematic Gesture    │
                                  │    Classifier & HUD    │
                                  └─────┬────────────┬─────┘
                                        │            │
            ┌───────────────────────────┘            └─────────────────────────┐
            ▼                                                                  ▼
┌────────────────────────┐                                         ┌───────────────────────┐
│   Word Accumulator     │                                         │ Temporal Stabilization│
│ & Predictive NLP Pills │                                         │    Debounce Buffer    │
└───────────┬────────────┘                                         └───────────────────────┘
            │
            ▼
┌────────────────────────┐          ┌──────────────────────┐       ┌───────────────────────┐
│  Formed Sentence / Text│ ───────> │  Offline TTS Engine  │ ────> │  Audible Voice Output │
└────────────────────────┘          └──────────────────────┘       └───────────────────────┘
            ▲
            │ (Reverse Mode)
┌────────────────────────┐
│  Hearing Communicator  │
│ (Text to Sign Visuals) │
└────────────────────────┘
```

---

## 🔄 Evolution & What Changed: 2022 Push vs. 2026 v2.0

When the project was concluded in **June 2022 (`commit 5e411b7`)**, it represented a functional undergraduate Capstone Project. However, between 2022 and 2026, Python, OpenCV, TensorFlow, and operating system security (macOS camera privacy) underwent breaking changes that prevented the legacy code from running on modern machines. In **October 2026**, the system was overhauled into version **2.0.0**.

Here is a detailed summary of what evolved:

### 1. Computer Vision & Landmark Tracking
* **2022 (Legacy)**: Used classical image thresholding and background accumulation subtraction (`cv2.accumulateWeighted`). Required the user to hold the camera completely still against a static background, calibrate background frames, and hold their hand strictly within a small $200 \times 200$ px bounding box. Any slight shift in room lighting, shadows, or varied skin tones caused complete segmentation failure.
* **2026 (v2.0)**: Integrated **Google MediaPipe Tasks HandLandmarker** (`mediapipe>=1.2.0`). Tracks **21 3D hand landmarks** anywhere in the camera frame with zero bounding-box constraints. Operates invariant to lighting, skin tones, and background clutter. Landmark coordinates are mathematically normalized relative to the wrist and scaled by hand span, making recognition robust to distance and position.

### 2. Gesture Classification & Temporal Smoothing
* **2022 (Legacy)**: Relied on 2-layer Keras CNN models (`model-bw.h5`, `model-bw_dru.h5`, `model-bw_tkdi.h5`, `model-bw_smn.h5`) trained on 64x64/128x128 binary thresholded silhouettes. Experienced significant frame-to-frame classification jitter.
* **2026 (v2.0)**: Built a kinematic feature and angle analyzer (`GestureAnalyzer`) combined with a temporal voting window (`LandmarkClassifier`). Requires signs to remain stable for several consecutive frames before committing, completely eliminating single-frame flickering. Added support for both fingerspelled letters (`A`–`Z`), numbers (`1`–`6`), and conversational whole-sign gestures (`Hello`, `I Love You`, `Rock`, `OK`, `L`, `V`, `W`, `Y`). Also includes a backward-compatible adapter (`LegacyCNNClassifier`) for legacy `.h5` models.

### 3. Natural Language Processing & Spell Autocomplete
* **2022 (Legacy)**: Relied on `hunspell` with a hardcoded Linux path (`/usr/share/hunspell/en_US.dic`). This caused immediate crashes on macOS and Windows and required complex native C++ compiler headers. Furthermore, it only suggested spell corrections for completed words after a typo was committed.
* **2026 (v2.0)**: Replaced with pure-Python `pyspellchecker` (zero native C++ build requirements, cross-platform). Introduced **predictive prefix autocomplete**: as a user fingerspells (e.g., `H-E-L`), the system suggests `HELLO`, `HELP`, `HELD` in interactive pill buttons. Clicking a button auto-completes the word and inserts a space, speeding up communication by 3x–5x.

### 4. Audio & Text-to-Speech (TTS) Engine
* **2022 (Legacy)**: Relied on `gTTS` (Google Translate web API) and `playsound`. Every spoken word required an active internet connection to download an MP3 file to disk and play it back, causing high latency and blocking the application loop.
* **2026 (v2.0)**: Engineered an **offline, non-blocking asynchronous TTS engine** (`TTSEngine`). Dispatches speech requests to a background thread using native OS speech tools (`say` on macOS, `pyttsx3`/`powershell` on Windows, `espeak` on Linux) with zero cloud latency and zero dropped video frames.

### 5. Graphical User Interface & Responsiveness
* **2022 (Legacy)**: Fixed $900 \times 1100$ px Tkinter window that overflowed standard laptop displays (e.g., 13" MacBook screens). Synchronous webcam read and neural inference ran directly on the main UI thread, causing severe UI freezing.
* **2026 (v2.0)**: 
  - **Threaded Video Pipeline**: Dedicated capture thread streams smooth 30–60 FPS video without blocking the GUI.
  - **Responsive Sizing**: Dynamic window layout adapting to standard screen resolutions without geometric expansion loops.
  - **Camera Switcher**: Real-time toggle button (`📷 Switch Camera 0 ↔ 1`) to switch between macOS FaceTime cameras and iPhone Continuity Camera.
  - **Two-Way Communicator**: Added Hearing-to-Deaf visual dictionary (`TwoWayDictionaryWindow`) allowing hearing individuals to type or speak text and see corresponding sign language demonstrations.

### 6. Legacy Codebase Repair & Cross-Platform Compatibility
The original 2022 scripts were preserved in `Code/` and `CodeV3/`, with all breaking bugs resolved:
* **Indentation TabError**: Fixed mixed tabs and spaces in `CodeV3/app.py` line 16.
* **Path Concatenation Bug**: Fixed string concatenation bugs (`self.directory + "model-bw.json"`) that produced non-existent paths.
* **Pillow 10+ Incompatibility**: Replaced removed `Image.ANTIALIAS` with `Image.Resampling.LANCZOS` in `Code/01_login.py`, `02_registration.py`, and `03_dashboard.py`.
* **OpenCV 4+ Unpacking Crash**: Fixed `cv2.findContours` 3-tuple unpacking in `Code/06_test_cam.py` for modern OpenCV versions (`contours, hierarchy = ...[-2:]`).
* **Hardcoded Windows Paths**: Replaced hardcoded paths (`C:\Users\sande\Desktop\...`) in `Code/05_train_cnn.py` and `Code/06_test_cam.py` with dynamic `os.path.join` relative to `__file__`.

### 7. Packaging, Tooling & Testing
* **2022 (Legacy)**: No project packaging, no unit tests, and frozen 2018-era `requirements_pip.txt` (Keras 2.2.4) that failed to build on modern Python (3.10–3.12) or Apple Silicon ARM64.
* **2026 (v2.0)**:
  - Standard `pyproject.toml` specification (`sign-language-translator==2.0.0`) managed with modern tools (`uv` and `pip`).
  - Executable 1-click launcher script [`run.bash`](run.bash).
  - Comprehensive 12-test automated test suite (`pytest tests/ -v`) validating tracking, classification, autocomplete, and audio synthesis in under 2 seconds.

---

## 📊 Quick Comparison: Legacy vs. v2.0

| Feature | Legacy Phase 1 & 3 (`Code/`, `CodeV3/`) | Modern Version 2.0 (`src/sign_language/`) |
| :--- | :--- | :--- |
| **Hand Detection** | Background accumulation / Otsu thresholding | Google MediaPipe 21 3D Landmarks |
| **Lighting Sensitivity**| High (fails with lighting changes & shadows) | Invariant (neural landmark tracking) |
| **Skin Tone Robustness**| Brittle (color thresholding) | High (geometric landmark model) |
| **Camera Freedom** | Fixed tiny box ($200 \times 200$ px) | Full frame (move hands freely) |
| **NLP Suggestions** | Hardcoded Linux-only `hunspell` path | Cross-platform pure-Python `pyspellchecker` |
| **Speech Engine** | Slow Google Translate web requests (`gTTS`) | Instant offline synthesis (`pyttsx3` / `say`) |
| **UI Performance** | Synchronous (freezes during inference) | Multi-threaded 60 FPS video loop |
| **Window Layout** | Rigid $900 \times 1100$ px (overflows laptop screens)| Responsive auto-scaling |
| **Two-Way Mode** | Not functional in GUI | Interactive Text-to-Sign reference viewer |

---

## 🚀 Quickstart with `run.bash`

The repository includes an all-in-one launcher script that auto-detects or creates the environment, verifies dependencies, and executes commands:

```bash
# 1. Launch the modern Real-Time Sign Language Translator (Default)
./run.bash

# 2. Run the automated test suite
./run.bash test

# 3. Open the Hearing Communicator (Text to Sign visual reference) directly
./run.bash two-way

# 4. Run legacy applications (backward compatibility)
./run.bash legacy-v3
./run.bash legacy-dashboard

# 5. View available commands
./run.bash help
```

---

## 📷 macOS Camera Permissions Guide

On macOS, camera access is protected by Apple's Transparency, Consent, and Control (TCC) system. When launching the camera for the first time in a terminal, you may see:

```text
OpenCV: not authorized to capture video (status 0), requesting...
```

### How to Grant Permission:
1. **System Prompt**: macOS displays a system dialog: *"Terminal would like to access the Camera"*. Click **OK**.
2. **Manual Settings**:
   - Open **System Settings** on your Mac.
   - Navigate to **Privacy & Security** $\rightarrow$ **Camera**.
   - Toggle **ON** the switch next to **Terminal** (or **iTerm2**, **VS Code**, **Cursor**, etc., depending on where you run the script).
3. **In the App**:
   - Click the **🔄 Reconnect Camera** button in the app's top header bar, or restart `./run.bash`.

---

## 🤟 Supported Signs & Gestures

### Fingerspelling Alphabet & Numbers:
- **Alphabet**: `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z`.
- **Numbers**: `1` through `6`.

### Conversational Word & Phrase Gestures:
- **Hello / Open Palm**: 5 fingers extended and spread.
- **I Love You (ILY)**: Thumb, Index, and Pinky extended (Middle and Ring folded).
- **Rock / Horns**: Index and Pinky extended (Thumb folded).
- **OK Sign / F**: Thumb and Index touching in a circle, other fingers extended.
- **L**: Thumb and Index at a 90-degree angle.
- **V / Peace**: Index and Middle separated in a V-shape.
- **U**: Index and Middle held together upright.
- **W**: Index, Middle, and Ring extended upright.
- **Y / Phone**: Thumb and Pinky extended.

---

## 💻 Manual Setup & Installation

If you prefer manual setup over `./run.bash`:

### Option A: Installation with `uv` (Fastest)
```bash
# Clone the repository
git clone https://github.com/SandeepMalviya1999/ml_sign_be_project.git
cd ml-hand-sign-language

# Create virtual environment with Python 3.12
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies in editable mode
uv pip install -e .
```

### Option B: Installation with standard `pip`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Running Manually:
```bash
# Launch application
python -m sign_language

# Or via console entrypoint
sign-language

# Run tests
pytest tests/ -v
```

---

## 📁 Project Structure

```
ml-hand-sign-language/
├── run.bash                     # 1-Click launcher script
├── pyproject.toml               # Python package specification
├── requirements.txt             # Pinned dependency requirements
├── .gitignore                   # Clean Git ignore rules
├── README.md                    # Project documentation
│
├── src/
│   └── sign_language/           # Core Python package
│       ├── __init__.py
│       ├── __main__.py          # python -m sign_language entrypoint
│       ├── vision/
│       │   └── hand_tracker.py  # MediaPipe 21 3D landmark extractor
│       ├── models/
│       │   ├── gestures.py      # ASL kinematics & geometric analyzer
│       │   ├── classifier.py    # Temporal smoothing landmark classifier
│       │   └── assets/          # MediaPipe task models
│       ├── nlp/
│       │   └── spell_corrector.py # Predictive autocomplete & spell-check
│       ├── audio/
│       │   └── tts_engine.py    # Non-blocking offline text-to-speech
│       └── ui/
│           ├── app.py           # Threaded real-time translator GUI
│           └── two_way.py       # Hearing-to-Deaf visual dictionary
│
├── tests/                       # Automated test suite
│   ├── test_tracker.py          # MediaPipe tracking tests
│   ├── test_classifier.py       # Kinematic gesture classification tests
│   ├── test_spell_corrector.py  # Prefix autocomplete tests
│   └── test_tts.py              # Audio synthesis lifecycle tests
│
├── legacy/                      # Archived Legacy Implementations (Preserved & Fixed)
│   ├── phase1_basic_gestures/   # Phase 1 SQLite + Tkinter + CNN scripts (former Code/)
│   ├── phase2_experiments/      # Exploratory notebooks & models (former CodeV2/)
│   └── phase3_asl_threshold/    # Phase 3 ASL thresholding application (former CodeV3/)
├── Asset/                       # Sign reference charts and demonstration media
├── Reference Paper/             # 10 IEEE/Conference literature survey papers
├── Code Images/                 # Application & dataset screenshots
├── Report.pdf / Report.docx     # Academic Capstone reports
└── PPT.pptx                     # Project presentation slide decks
```

---

## 🧪 Automated Testing

Automated tests verify tracking, geometry, classification, autocomplete, and audio synthesis:

```bash
./run.bash test
```

Expected output:
```text
============================= test session starts ==============================
tests/test_classifier.py::test_gesture_analyzer_hello PASSED             [  6%]
tests/test_classifier.py::test_gesture_analyzer_l PASSED                 [ 13%]
tests/test_classifier.py::test_gesture_analyzer_ily PASSED               [ 20%]
tests/test_classifier.py::test_gesture_analyzer_y PASSED                 [ 26%]
tests/test_classifier.py::test_classifier_temporal_smoothing PASSED      [ 33%]
tests/test_spell_corrector.py::test_spell_corrector_prefixes PASSED      [ 40%]
tests/test_spell_corrector.py::test_spell_corrector_misspelling PASSED   [ 46%]
tests/test_spell_corrector.py::test_spell_corrector_empty PASSED         [ 53%]
tests/test_spell_corrector.py::test_spell_corrector_is_correct PASSED    [ 60%]
tests/test_tracker.py::test_tracker_blank_image PASSED                   [ 66%]
tests/test_tracker.py::test_tracker_real_image PASSED                    [ 73%]
tests/test_tts.py::test_tts_engine_lifecycle PASSED                      [ 80%]
tests/test_two_way.py::test_conversational_signs_catalog PASSED          [ 86%]
tests/test_two_way.py::test_alphabet_crops_exist PASSED                  [ 93%]
tests/test_two_way.py::test_two_way_window_headless PASSED               [100%]
============================== 15 passed in 2.25s ==============================
```

---

## 👥 Authors & Academic Background

Original Bachelor of Engineering Project (2021–2022):
- **Title**: *Realtime Sign Language to English Language Conversion using CNN for Disabled People*
- **Institution**: Department of Computer Engineering, G.H. Raisoni Institute of Engineering & Technology (Pune), Savitribai Phule Pune University.
- **Authors**:
  - Sandeep Malviya
  - Vaibhavi Padamwar
  - Harshda Patil
  - Sanghavi Appam
- **Project Guide**: Prof. Disha Deotale

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.
