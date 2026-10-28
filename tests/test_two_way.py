"""Tests for Bidirectional Communicator (Text to Sign)."""

import os
import sys

# Ensure Tcl/Tk libraries are found in virtual environment
for base in [
    os.path.expanduser("~/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/lib"),
    sys.base_prefix + "/lib"
]:
    tcl_dir = os.path.join(base, "tcl9.0")
    tk_dir = os.path.join(base, "tk9.0")
    if os.path.exists(tcl_dir):
        os.environ.setdefault("TCL_LIBRARY", tcl_dir)
    if os.path.exists(tk_dir):
        os.environ.setdefault("TK_LIBRARY", tk_dir)

import tkinter as tk
import pytest
from sign_language.ui.two_way import TwoWayDictionaryWindow, CONVERSATIONAL_SIGNS, ensure_alphabet_crops


def test_conversational_signs_catalog():
    assert "YES" in CONVERSATIONAL_SIGNS
    assert "NO" in CONVERSATIONAL_SIGNS
    assert "HELLO" in CONVERSATIONAL_SIGNS
    assert "THANK YOU" in CONVERSATIONAL_SIGNS
    assert "I LOVE YOU" in CONVERSATIONAL_SIGNS
    assert "WATER" in CONVERSATIONAL_SIGNS

    for sign, data in CONVERSATIONAL_SIGNS.items():
        assert "motion" in data
        assert "title" in data
        assert len(data["motion"]) > 10


def test_alphabet_crops_exist():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(repo_root, "Asset")
    alphabet_dir = ensure_alphabet_crops(assets_dir)

    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789":
        img_path = os.path.join(alphabet_dir, f"{ch}.png")
        assert os.path.exists(img_path), f"Missing crop for {ch}"


def test_two_way_window_headless():
    root = tk.Tk()
    root.withdraw()

    try:
        win = TwoWayDictionaryWindow(root)

        # Test single word translation
        win.text_var.set("YES")
        win.on_translate()
        assert len(win._cached_images) == 3

        # Test multi-word translation
        win.text_var.set("HI ALL")
        win.on_translate()
        assert len(win._cached_images) == 5

        # Test animation toggle
        win.toggle_animation()
        assert win._anim_running is True
        win.stop_animation()
        assert win._anim_running is False

        # Test chart display
        win.show_full_chart()
        assert len(win._cached_images) == 1

        win.on_close()
    finally:
        root.destroy()
