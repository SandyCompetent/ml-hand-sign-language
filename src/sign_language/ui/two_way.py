"""Two-Way Sign Language Translator: Speech / Text to Sign Reference Visualizer."""

from __future__ import annotations
import os
import tkinter as tk
from tkinter import ttk
from typing import Optional
from PIL import Image, ImageTk


class TwoWayDictionaryWindow:
    """Window allowing hearing users to type or speak text and see the corresponding sign gestures."""

    def __init__(self, parent: Optional[tk.Tk] = None):
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("Bidirectional Communicator: Text / Speech to Sign Language")
        self.root.geometry("850x650")
        self.root.minsize(700, 500)
        self.root.configure(bg="#1e1e2e")

        # Discover repository root & asset images
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(curr_dir)))
        self.assets_dir = os.path.join(repo_root, "Asset")
        self.codev3_dir = os.path.join(repo_root, "CodeV3")

        self._setup_ui()

    def _setup_ui(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dark.TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Helvetica", 12))
        style.configure("Title.TLabel", background="#1e1e2e", foreground="#89b4fa", font=("Helvetica", 18, "bold"))
        style.configure("Action.TButton", font=("Helvetica", 11, "bold"), background="#89b4fa", foreground="#11111b")

        header_frame = tk.Frame(self.root, bg="#181825", padx=16, pady=12)
        header_frame.pack(fill=tk.X)

        title_lbl = ttk.Label(
            header_frame,
            text="Hearing-to-Deaf Communicator (Text to Sign)",
            style="Title.TLabel",
        )
        title_lbl.pack(side=tk.LEFT)

        # Input Frame
        input_frame = tk.Frame(self.root, bg="#1e1e2e", padx=20, pady=15)
        input_frame.pack(fill=tk.X)

        prompt_lbl = ttk.Label(
            input_frame,
            text="Enter word or phrase to translate into Sign Language:",
            style="Dark.TLabel",
        )
        prompt_lbl.pack(anchor=tk.W, pady=(0, 6))

        entry_subframe = tk.Frame(input_frame, bg="#1e1e2e")
        entry_subframe.pack(fill=tk.X)

        self.text_var = tk.StringVar(value="HELLO")
        self.entry = tk.Entry(
            entry_subframe,
            textvariable=self.text_var,
            font=("Helvetica", 14),
            bg="#313244",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            relief=tk.FLAT,
            bd=6,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self.on_translate())

        translate_btn = tk.Button(
            entry_subframe,
            text="Translate to Sign",
            command=self.on_translate,
            bg="#89b4fa",
            fg="#11111b",
            font=("Helvetica", 12, "bold"),
            relief=tk.FLAT,
            padx=14,
            pady=4,
            cursor="hand2",
        )
        translate_btn.pack(side=tk.LEFT)

        # Quick phrases bar
        quick_frame = tk.Frame(self.root, bg="#1e1e2e", padx=20)
        quick_frame.pack(fill=tk.X, pady=(0, 10))

        quick_lbl = ttk.Label(quick_frame, text="Quick Signs:", style="Dark.TLabel")
        quick_lbl.pack(side=tk.LEFT, padx=(0, 8))

        for phrase in ["HELLO", "THANK YOU", "YES", "NO", "I LOVE YOU", "WATER"]:
            btn = tk.Button(
                quick_frame,
                text=phrase,
                command=lambda p=phrase: self._quick_fill(p),
                bg="#45475a",
                fg="#cdd6f4",
                font=("Helvetica", 9),
                relief=tk.FLAT,
                cursor="hand2",
                padx=6,
                pady=2,
            )
            btn.pack(side=tk.LEFT, padx=3)

        # Display Frame for Sign Images / Visuals
        self.display_frame = tk.Frame(self.root, bg="#181825", bd=1, relief=tk.SOLID)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        self.image_label = tk.Label(self.display_frame, bg="#181825")
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Status label
        self.status_lbl = ttk.Label(self.root, text="Ready", style="Dark.TLabel")
        self.status_lbl.pack(side=tk.BOTTOM, anchor=tk.W, padx=20, pady=(0, 8))

        # Initial render
        self.on_translate()

    def _quick_fill(self, phrase: str) -> None:
        self.text_var.set(phrase)
        self.on_translate()

    def on_translate(self) -> None:
        phrase = self.text_var.get().strip().upper()
        if not phrase:
            return

        self.status_lbl.config(text=f"Showing visual reference for: {phrase}")

        # Try to load reference sign image (signs.png or sign1..sign4)
        target_img = None
        asset_signs = os.path.join(self.assets_dir, "signs.png")
        sign1_img = os.path.join(self.assets_dir, "sign1.png")
        legacy_signs = os.path.join(self.codev3_dir, "signs.png")

        if os.path.exists(asset_signs):
            target_img = asset_signs
        elif os.path.exists(legacy_signs):
            target_img = legacy_signs
        elif os.path.exists(sign1_img):
            target_img = sign1_img

        if target_img:
            try:
                pil_img = Image.open(target_img)
                # Resize keeping aspect ratio to fit inside display frame
                w_max, h_max = 800, 420
                pil_img.thumbnail((w_max, h_max), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(pil_img)
                self.image_label.config(image=tk_img, text="")
                self.image_label.image = tk_img
            except Exception as e:
                self.image_label.config(text=f"Could not load image: {e}", fg="white")
        else:
            self.image_label.config(
                text=f"Fingerspelling sequence: {' '.join(list(phrase))}",
                fg="#89b4fa",
                font=("Courier", 24, "bold"),
            )
