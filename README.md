# Advanced Hand Gesture Mouse Control

A real-time computer vision system that transforms hand gestures into operating system input events without physical peripherals.
Control your computer cursor, clicks, scroll, and zoom using hand gestures through a webcam.

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![Status](https://img.shields.io/badge/status-active-green)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## Overview

This project implements a real-time vision-based virtual mouse system that converts geometric hand landmark patterns into operating system input events using rule-based gesture classification.

### Key Features

- Cursor movement using index finger tracking
- Left click and right click using pinch gestures
- Vertical scrolling using two-finger directional gestures
- Zoom in / zoom out using dynamic pinch distance
- Exponential smoothing for stable, low-jitter cursor motion
- Cooldown-based gesture debouncing to reduce repeated triggers
- High reliability under controlled lighting
- Real-time performance (30+ FPS on mid-range hardware)

---

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the application

```bash
python GestureCursor.py
```

### 3) Use gestures

1. Face the webcam from a comfortable distance.
2. Wait for hand landmarks to appear.
3. Use configured gestures to control the cursor and actions.
4. Press `q` to quit.

---

## Requirements

- Python 3.7+
- Webcam (built-in or external)
- Windows, macOS, or Linux

---

## Gesture Guide

| Gesture                      | Action             |
| ---------------------------- | ------------------ |
| Index finger pointing        | Move cursor        |
| Thumb + index pinch          | Left click         |
| Index + middle pinch         | Right click        |
| Two-finger upward movement   | Scroll up          |
| Two-finger downward movement | Scroll down        |
| Pinch expand / contract      | Zoom in / Zoom out |

---

## How Gesture Recognition Works

1. **Frame capture:** Webcam frames are captured continuously using **OpenCV (computer vision library)**.
2. **Hand landmark detection:** Each frame is processed with **MediaPipe (Google hand tracking framework)** to estimate **21 hand landmarks**.
3. **Distance normalization:** Finger distances are normalized relative to hand size so gesture thresholds remain more stable across users and camera positions.
4. **Rule-based gesture logic:** Landmark geometry is evaluated using deterministic rules for cursor movement, click, scroll, and zoom actions.
5. **Motion stabilization:** Exponential smoothing is applied to cursor coordinates to reduce jitter and improve control stability.
6. **OS interaction:** Recognized gestures are mapped to system input events through **PyAutoGUI (Python automation library)**.

---

## System Flow

Webcam → Frame Capture → Hand Landmark Detection → 
Geometric Gesture Logic → Smoothing → OS Event Trigger


## Tech Stack

- Python
- OpenCV (computer vision library)
- MediaPipe (Google hand tracking framework)
- PyAutoGUI (Python automation library)
- NumPy (numerical computing library)

---

## Project Structure

```text
.
├── GestureCursor.py
├── requirements.txt
└── README.md
```

---

## Customization

Tune parameters directly in `GestureCursor.py`, such as:

- Smoothing factor for cursor movement
- Pinch thresholds for click/zoom gestures
- Cooldown intervals for debouncing repeated actions

---



## Troubleshooting

### Camera not opening

- Confirm webcam permissions for terminal/Python.
- Check if another app is already using the camera.

### Hand landmarks unstable

- Use front lighting and reduce strong backlight.
- Keep your hand within clear camera view.

### Cursor feels jumpy

- Increase smoothing configuration in `GestureCursor.py`.
- Move hand more steadily and remain in frame center.

---

## Author

Siva Surya P
**Computer Vision & AI Enthusiast**

---

## Limitations

- Performance depends on lighting conditions.
- Rule-based gesture detection may produce false positives in edge cases.
- Single-hand tracking only (current version).
- Not optimized for low-power embedded devices.

Future versions can incorporate ML-based gesture classification for improved robustness.
