"""Two-Way Sign Language Translator: Speech / Text to Sign Reference Visualizer."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from typing import Optional, List, Dict
from PIL import Image, ImageTk, ImageDraw, ImageFont


CONVERSATIONAL_SIGNS: Dict[str, Dict[str, str]] = {
    "YES": {
        "title": "YES (Agreement)",
        "motion": "Make a closed fist (S-hand shape) at shoulder/chest height and nod it forward and down twice, mimicking a nodding head.",
        "asl_type": "ASL Conversational Gesture",
        "tip": "Move from the wrist in a smooth, firm double nod.",
    },
    "NO": {
        "title": "NO (Negative)",
        "motion": "Hold dominant hand up; snap your index and middle fingers down together onto your thumb twice (resembling talking fingers closing).",
        "asl_type": "ASL Conversational Gesture",
        "tip": "Firm, crisp double tap. Often paired with a subtle head shake.",
    },
    "HELLO": {
        "title": "HELLO (Greeting)",
        "motion": "Touch the tips of your flat dominant hand to your temple, then move your hand outward and away in an open salute-wave.",
        "asl_type": "ASL Conversational Gesture",
        "tip": "Start near the eye/forehead with fingers together and move forward with a warm smile.",
    },
    "THANK YOU": {
        "title": "THANK YOU (Gratitude)",
        "motion": "Touch the fingertips of your flat open dominant hand to your chin/lips, then extend your hand forward toward the person.",
        "asl_type": "ASL Conversational Gesture",
        "tip": "Ensure your palm faces inward near the chin, then outward as it moves forward.",
    },
    "THANKS": {
        "title": "THANKS (Gratitude)",
        "motion": "Touch the fingertips of your flat hand to your lips, then move forward toward the other person.",
        "asl_type": "ASL Conversational Gesture",
        "tip": "Casual form of 'THANK YOU'.",
    },
    "I LOVE YOU": {
        "title": "I LOVE YOU (ILY)",
        "motion": "Simultaneously extend your thumb, index finger, and pinky finger while curling your middle and ring fingers down, palm facing forward.",
        "asl_type": "Iconic ASL Emblem",
        "tip": "Combines the fingerspelled letters I, L, and Y into a single iconic gesture.",
    },
    "WATER": {
        "title": "WATER",
        "motion": "Form the letter 'W' with your three middle fingers upright (thumb holding pinky) and tap your index finger to your chin twice.",
        "asl_type": "ASL Vocabulary Sign",
        "tip": "Lightly tap the chin near the bottom lip twice.",
    },
    "PLEASE": {
        "title": "PLEASE",
        "motion": "Place the flat palm of your dominant hand on the center of your chest and rub in a gentle clockwise circle.",
        "asl_type": "ASL Conversational Gesture",
        "tip": "Keep fingers flat and relaxed against the chest.",
    },
    "SORRY": {
        "title": "SORRY (Apology)",
        "motion": "Make a closed 'A' fist with your dominant hand and rub it in a circle over your heart with a remorseful facial expression.",
        "asl_type": "ASL Conversational Gesture",
        "tip": "Rub in a small circular motion 2 to 3 times.",
    },
    "GOOD": {
        "title": "GOOD",
        "motion": "Touch fingertips of flat right hand to lips, then bring it down to rest palm-up onto the flat palm of your non-dominant hand.",
        "asl_type": "ASL Conversational Gesture",
        "tip": "A clean downward motion landing softly on the supporting palm.",
    },
    "HELP": {
        "title": "HELP",
        "motion": "Place a closed 'A' fist with thumb upright onto the flat open palm of your other hand, then lift both hands upward together.",
        "asl_type": "ASL Conversational Gesture",
        "tip": "Lifting upward signals offering or requesting assistance.",
    },
}


def ensure_alphabet_crops(assets_dir: str) -> str:
    """Ensure that individual letter/number sign crops exist in Asset/alphabet."""
    alphabet_dir = os.path.join(assets_dir, "alphabet")
    os.makedirs(alphabet_dir, exist_ok=True)

    expected = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [str(i) for i in range(1, 10)]
    missing = [c for c in expected if not os.path.exists(os.path.join(alphabet_dir, f"{c}.png"))]
    if not missing:
        return alphabet_dir

    signs_path = os.path.join(assets_dir, "signs.png")
    if not os.path.exists(signs_path):
        return alphabet_dir

    try:
        import cv2
        import numpy as np

        img = cv2.imread(signs_path)
        if img is None:
            return alphabet_dir

        exact_crops = {
            # Row 1 (A-I)
            "A": (10, 240, 60, 160), "B": (10, 240, 160, 252), "C": (10, 240, 255, 415),
            "D": (10, 240, 415, 508), "E": (10, 240, 510, 605), "F": (10, 240, 606, 717),
            "G": (30, 240, 717, 902), "H": (30, 240, 905, 1108), "I": (20, 240, 1110, 1225),
            # Row 2 (J-Q)
            "J": (240, 475, 50, 215), "K": (240, 475, 215, 310), "L": (240, 475, 312, 470),
            "M": (240, 475, 472, 592), "N": (240, 475, 594, 725), "O": (240, 475, 727, 855),
            "P": (240, 475, 860, 1088), "Q": (240, 475, 1090, 1245),
            # Row 3 (R-Z)
            "R": (475, 715, 60, 165), "S": (475, 715, 170, 275), "T": (475, 715, 285, 385),
            "U": (475, 715, 395, 490), "V": (475, 715, 500, 600), "W": (475, 715, 608, 718),
            "X": (475, 715, 725, 852), "Y": (510, 715, 855, 1044), "Z": (475, 715, 1044, 1200),
            # Row 4 (1-9)
            "1": (715, 955, 65, 171), "2": (715, 955, 171, 272), "3": (715, 955, 275, 420),
            "4": (715, 955, 420, 550), "5": (715, 955, 555, 718), "6": (715, 955, 718, 840),
            "7": (715, 955, 842, 962), "8": (715, 955, 965, 1095), "9": (715, 955, 1098, 1238),
        }

        for label, (y1, y2, x1, x2) in exact_crops.items():
            crop = img[y1:y2, x1:x2].copy()
            if label == "Y":
                crop[:25, 150:] = 255
            if label == "Z":
                crop[100:, :25] = 255

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            mask = gray < 240
            if np.any(mask):
                ys, xs = np.where(mask)
                pad = 6
                ymin = max(0, ys.min() - pad)
                ymax = min(crop.shape[0], ys.max() + pad)
                xmin = max(0, xs.min() - pad)
                xmax = min(crop.shape[1], xs.max() + pad)
                crop = crop[ymin:ymax, xmin:xmax]

            cv2.imwrite(os.path.join(alphabet_dir, f"{label}.png"), crop)
    except Exception:
        pass

    return alphabet_dir


class TwoWayDictionaryWindow:
    """Window allowing hearing users to type or speak text and see the corresponding sign gestures."""

    def __init__(self, parent: Optional[tk.Tk] = None):
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("Bidirectional Communicator: Text / Speech to Sign Language")
        self.root.geometry("920x720")
        self.root.minsize(760, 560)
        self.root.configure(bg="#1e1e2e")

        # Discover repository root & asset paths
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(curr_dir)))
        self.assets_dir = os.path.join(repo_root, "Asset")
        self.legacy_v3_dir = os.path.join(repo_root, "legacy", "phase3_asl_threshold")
        self.codev3_dir = os.path.join(repo_root, "CodeV3")

        # Ensure letter-level crop images exist
        self.alphabet_dir = ensure_alphabet_crops(self.assets_dir)

        # Animation playback tracking
        self._anim_running = False
        self._anim_index = 0
        self._anim_chars: List[str] = []
        self._anim_timer_id: Optional[str] = None

        self._cached_images: List[ImageTk.PhotoImage] = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dark.TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Helvetica", 11))
        style.configure("Title.TLabel", background="#1e1e2e", foreground="#89b4fa", font=("Helvetica", 17, "bold"))
        style.configure("Action.TButton", font=("Helvetica", 11, "bold"), background="#89b4fa", foreground="#11111b")

        # 1. Header Frame
        header_frame = tk.Frame(self.root, bg="#181825", padx=18, pady=12)
        header_frame.pack(fill=tk.X)

        title_lbl = ttk.Label(
            header_frame,
            text="⇄ Hearing-to-Deaf Communicator (Text to Sign)",
            style="Title.TLabel",
        )
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = tk.Label(
            header_frame,
            text="Real-time Visual Sign Translation & Fingerspelling",
            bg="#181825",
            fg="#a6adc8",
            font=("Helvetica", 10, "italic"),
        )
        subtitle_lbl.pack(side=tk.RIGHT)

        # 2. Input Frame
        input_frame = tk.Frame(self.root, bg="#1e1e2e", padx=20, pady=12)
        input_frame.pack(fill=tk.X)

        prompt_lbl = ttk.Label(
            input_frame,
            text="Enter word, phrase, or sentence to translate into Sign Language:",
            style="Dark.TLabel",
        )
        prompt_lbl.pack(anchor=tk.W, pady=(0, 6))

        entry_subframe = tk.Frame(input_frame, bg="#1e1e2e")
        entry_subframe.pack(fill=tk.X)

        self.text_var = tk.StringVar(value="YES")
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
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT,
            padx=14,
            pady=4,
            cursor="hand2",
        )
        translate_btn.pack(side=tk.LEFT, padx=3)

        self.play_btn = tk.Button(
            entry_subframe,
            text="▶ Play Sequence",
            command=self.toggle_animation,
            bg="#a6e3a1",
            fg="#11111b",
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
        )
        self.play_btn.pack(side=tk.LEFT, padx=3)

        chart_btn = tk.Button(
            entry_subframe,
            text="Full Chart",
            command=self.show_full_chart,
            bg="#585b70",
            fg="#cdd6f4",
            font=("Helvetica", 11),
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
        )
        chart_btn.pack(side=tk.LEFT, padx=3)

        # 3. Quick phrases bar
        quick_frame = tk.Frame(self.root, bg="#1e1e2e", padx=20)
        quick_frame.pack(fill=tk.X, pady=(0, 8))

        quick_lbl = tk.Label(quick_frame, text="Quick Signs:", bg="#1e1e2e", fg="#a6adc8", font=("Helvetica", 10, "bold"))
        quick_lbl.pack(side=tk.LEFT, padx=(0, 8))

        for phrase in ["YES", "NO", "HELLO", "THANK YOU", "I LOVE YOU", "WATER", "PLEASE", "GOOD"]:
            btn = tk.Button(
                quick_frame,
                text=phrase,
                command=lambda p=phrase: self._quick_fill(p),
                bg="#313244",
                fg="#cdd6f4",
                activebackground="#45475a",
                activeforeground="#ffffff",
                font=("Helvetica", 9),
                relief=tk.FLAT,
                cursor="hand2",
                padx=7,
                pady=2,
            )
            btn.pack(side=tk.LEFT, padx=2)

        # 4. Conversational Gesture Info Banner (Visible when matching a known phrase)
        self.gesture_card = tk.Frame(self.root, bg="#24273a", padx=16, pady=10, bd=1, relief=tk.SOLID)
        self.gesture_card.pack(fill=tk.X, padx=20, pady=(0, 10))

        self.gesture_badge = tk.Label(
            self.gesture_card,
            text="★ ASL Conversational Gesture",
            bg="#f9e2af",
            fg="#11111b",
            font=("Helvetica", 10, "bold"),
            padx=6,
            pady=2,
        )
        self.gesture_badge.pack(anchor=tk.W, pady=(0, 4))

        self.gesture_title_lbl = tk.Label(
            self.gesture_card,
            text="",
            bg="#24273a",
            fg="#89b4fa",
            font=("Helvetica", 13, "bold"),
        )
        self.gesture_title_lbl.pack(anchor=tk.W)

        self.gesture_desc_lbl = tk.Label(
            self.gesture_card,
            text="",
            bg="#24273a",
            fg="#cdd6f4",
            font=("Helvetica", 11),
            wraplength=860,
            justify=tk.LEFT,
        )
        self.gesture_desc_lbl.pack(anchor=tk.W, pady=(2, 2))

        self.gesture_tip_lbl = tk.Label(
            self.gesture_card,
            text="",
            bg="#24273a",
            fg="#a6adc8",
            font=("Helvetica", 9, "italic"),
            wraplength=860,
            justify=tk.LEFT,
        )
        self.gesture_tip_lbl.pack(anchor=tk.W)

        # 5. Main Display Frame with Scrollable Canvas for Sign Cards
        self.display_frame = tk.Frame(self.root, bg="#181825", bd=1, relief=tk.SOLID)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        self.canvas = tk.Canvas(self.display_frame, bg="#181825", highlightthickness=0)
        self.scrollbar_x = tk.Scrollbar(self.display_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set)

        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind horizontal scrolling with mousewheel / touchpad
        self.canvas.bind_all("<Shift-MouseWheel>", lambda e: self.canvas.xview_scroll(int(-1 * (e.delta / 120)), "units"))

        # 6. Status label
        status_bar = tk.Frame(self.root, bg="#1e1e2e", padx=20, pady=4)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_lbl = tk.Label(status_bar, text="Ready", bg="#1e1e2e", fg="#a6adc8", font=("Helvetica", 10))
        self.status_lbl.pack(side=tk.LEFT)

        self.counter_lbl = tk.Label(status_bar, text="", bg="#1e1e2e", fg="#89b4fa", font=("Helvetica", 10, "bold"))
        self.counter_lbl.pack(side=tk.RIGHT)

        # Clean teardown
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initial render
        self.on_translate()

    def _quick_fill(self, phrase: str) -> None:
        self.text_var.set(phrase)
        self.on_translate()

    def on_translate(self) -> None:
        """Translate the entered text into conversational ASL guidance and letter cards."""
        self.stop_animation()
        phrase = self.text_var.get().strip().upper()
        if not phrase:
            self.status_lbl.config(text="Please enter text to translate.")
            return

        self.status_lbl.config(text=f"Translating: \"{phrase}\"")

        # 1. Check for whole-phrase conversational sign match
        sign_info = CONVERSATIONAL_SIGNS.get(phrase)
        if sign_info:
            self.gesture_card.pack(fill=tk.X, padx=20, pady=(0, 10), before=self.display_frame)
            self.gesture_badge.config(text=f"★ {sign_info['asl_type']}")
            self.gesture_title_lbl.config(text=sign_info["title"])
            self.gesture_desc_lbl.config(text=f"Movement: {sign_info['motion']}")
            self.gesture_tip_lbl.config(text=f"Execution Tip: {sign_info['tip']}")
        else:
            self.gesture_card.pack_forget()

        # 2. Decompose into fingerspelling characters
        valid_chars = [ch for ch in phrase if ch.isalnum() or ch.isspace()]
        self._anim_chars = [ch for ch in valid_chars if ch.isalnum()]
        self.counter_lbl.config(text=f"{len(self._anim_chars)} Sign{'s' if len(self._anim_chars) != 1 else ''}")

        # 3. Render cards onto canvas
        self._render_sign_cards(phrase)

    def _render_sign_cards(self, phrase: str, highlight_idx: Optional[int] = None) -> None:
        """Render letter sign cards onto the scrollable canvas."""
        self.canvas.delete("all")
        self._cached_images.clear()

        # Card styling constants
        card_w, card_h = 160, 240
        card_margin = 16
        start_x = 20
        start_y = 20

        current_x = start_x
        letter_count = 0

        # Split into words to display clean spacing between words
        words = phrase.split()
        if not words:
            return

        for w_idx, word in enumerate(words):
            for ch in word:
                if not ch.isalnum():
                    continue

                is_highlighted = (highlight_idx is not None and letter_count == highlight_idx)
                letter_count += 1

                # Load letter sign image
                img_path = os.path.join(self.alphabet_dir, f"{ch}.png")
                pil_crop = None
                if os.path.exists(img_path):
                    try:
                        pil_crop = Image.open(img_path).convert("RGB")
                    except Exception:
                        pil_crop = None

                # Build styled card
                card_img = Image.new("RGB", (card_w, card_h), (255, 255, 255))
                draw = ImageDraw.Draw(card_img)

                # Card border and header
                border_color = (137, 180, 250) if is_highlighted else (205, 214, 244)
                border_width = 4 if is_highlighted else 2
                header_fill = (180, 210, 255) if is_highlighted else (235, 240, 255)
                text_fill = (20, 20, 60)

                draw.rectangle([0, 0, card_w - 1, card_h - 1], outline=border_color, width=border_width)
                draw.rectangle([border_width, border_width, card_w - 1 - border_width, 36], fill=header_fill)
                draw.text((card_w // 2 - 6, 10), ch, fill=text_fill)

                # Paste hand sign image inside card body
                if pil_crop:
                    max_w = card_w - 24
                    max_h = card_h - 52
                    pil_crop.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                    ox = (card_w - pil_crop.width) // 2
                    oy = 42 + (max_h - pil_crop.height) // 2
                    card_img.paste(pil_crop, (ox, oy))
                else:
                    draw.text((card_w // 2 - 30, card_h // 2), "No Sign", fill=(120, 120, 120))

                tk_card = ImageTk.PhotoImage(card_img, master=self.root)
                self._cached_images.append(tk_card)

                self.canvas.create_image(current_x, start_y, anchor=tk.NW, image=tk_card)
                current_x += card_w + card_margin

            # Space between words
            if w_idx < len(words) - 1:
                # Draw space marker
                space_w = 40
                self.canvas.create_rectangle(
                    current_x, start_y + card_h // 2 - 2, current_x + space_w, start_y + card_h // 2 + 2,
                    fill="#45475a", outline=""
                )
                self.canvas.create_text(
                    current_x + space_w // 2, start_y + card_h // 2 + 16,
                    text="SPACE", fill="#6c7086", font=("Helvetica", 8, "bold")
                )
                current_x += space_w + card_margin

        # Configure scrollregion
        total_w = current_x + 20
        total_h = start_y + card_h + 30
        self.canvas.config(scrollregion=(0, 0, total_w, total_h))

    def show_full_chart(self) -> None:
        """Display the complete ASL alphabet and numbers reference chart."""
        self.stop_animation()
        self.gesture_card.pack_forget()

        chart_path = os.path.join(self.assets_dir, "signs.png")
        if not os.path.exists(chart_path):
            chart_path = os.path.join(self.legacy_v3_dir, "signs.png")

        if not os.path.exists(chart_path):
            self.status_lbl.config(text="Alphabet chart image not found.")
            return

        try:
            pil_img = Image.open(chart_path)
            # Resize keeping aspect ratio to fit inside canvas nicely
            w_max, h_max = 860, 520
            pil_img.thumbnail((w_max, h_max), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img, master=self.root)
            self._cached_images = [tk_img]

            self.canvas.delete("all")
            self.canvas.create_image(20, 20, anchor=tk.NW, image=tk_img)
            self.canvas.config(scrollregion=(0, 0, tk_img.width() + 40, tk_img.height() + 40))

            self.status_lbl.config(text="Displaying Full ASL Alphabet & Number Reference Chart")
            self.counter_lbl.config(text="35 Signs (A-Z, 1-9)")
        except Exception as e:
            self.status_lbl.config(text=f"Failed to display chart: {e}")

    def toggle_animation(self) -> None:
        """Start or stop automated sequential fingerspelling animation."""
        if self._anim_running:
            self.stop_animation()
        else:
            self.start_animation()

    def start_animation(self) -> None:
        """Play through the fingerspelled letters sequentially."""
        if not self._anim_chars:
            return

        self._anim_running = True
        self._anim_index = 0
        self.play_btn.config(text="⏹ Stop", bg="#f38ba8")
        self._step_animation()

    def _step_animation(self) -> None:
        if not self._anim_running or not self._anim_chars:
            return

        phrase = self.text_var.get().strip().upper()
        self._render_sign_cards(phrase, highlight_idx=self._anim_index)

        curr_char = self._anim_chars[self._anim_index]
        self.status_lbl.config(
            text=f"Playing Sequence: Step {self._anim_index + 1}/{len(self._anim_chars)} - Sign '{curr_char}'"
        )

        # Auto-scroll canvas to keep current card in view
        card_total_w = 176
        scroll_fraction = (self._anim_index * card_total_w) / max(1, len(self._anim_chars) * card_total_w)
        self.canvas.xview_moveto(max(0.0, scroll_fraction - 0.1))

        self._anim_index = (self._anim_index + 1) % len(self._anim_chars)
        self._anim_timer_id = self.root.after(850, self._step_animation)

    def stop_animation(self) -> None:
        """Halt animation playback."""
        self._anim_running = False
        if self._anim_timer_id:
            try:
                self.root.after_cancel(self._anim_timer_id)
            except Exception:
                pass
            self._anim_timer_id = None
        self.play_btn.config(text="▶ Play Sequence", bg="#a6e3a1")

    def on_close(self) -> None:
        """Teardown on window close."""
        self.stop_animation()
        try:
            self.root.destroy()
        except Exception:
            pass


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    app = TwoWayDictionaryWindow(None)
    root.mainloop()


if __name__ == "__main__":
    main()
