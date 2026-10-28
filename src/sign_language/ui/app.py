"""Main Real-Time Sign Language Translator Application GUI."""

from __future__ import annotations
import os
import sys
import time
import threading
import queue
from typing import Optional, List, Tuple
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

from sign_language.vision import HandTracker
from sign_language.models import LandmarkClassifier, LegacyCNNClassifier
from sign_language.nlp import SpellCorrector
from sign_language.audio import TTSEngine
from sign_language.ui.two_way import TwoWayDictionaryWindow


class VideoCaptureThread:
    """Dedicated background thread for webcam video frame capture with auto-retry and multi-index scanning."""

    def __init__(self, src: int = 0):
        self.src = src
        self.cap: Optional[cv2.VideoCapture] = None
        self.ret = False
        self.frame: Optional[np.ndarray] = None
        self.running = True
        self.is_connected = False
        self.status_message = "Initializing camera..."
        self.lock = threading.Lock()
        self._init_camera()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _init_camera(self) -> bool:
        """Attempt to open camera, trying candidate device indices."""
        candidates = [self.src]
        for alt in [0, 1, 2]:
            if alt not in candidates:
                candidates.append(alt)

        for candidate_idx in candidates:
            try:
                cap = cv2.VideoCapture(candidate_idx)
                if cap.isOpened():
                    # Probe a test read
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.src = candidate_idx
                        self.cap = cap
                        self.ret = True
                        self.frame = test_frame
                        self.is_connected = True
                        self.status_message = f"Camera #{candidate_idx} Active"
                        return True
                    cap.release()
            except Exception as e:
                pass

        # If all candidates fail, initialize fallback
        self.cap = cv2.VideoCapture(self.src)
        self.is_connected = False
        self.status_message = "Camera access denied or device busy (Check macOS Privacy & Security > Camera)"
        return False

    def reconnect(self) -> bool:
        """Re-scan and reconnect to camera."""
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            success = self._init_camera()
            return success

    def set_source(self, new_src: int) -> bool:
        """Switch to a specific camera device index."""
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.src = new_src
            self.ret = False
            self.frame = None
            try:
                cap = cv2.VideoCapture(new_src)
                if cap.isOpened():
                    self.cap = cap
                    self.is_connected = True
                    self.status_message = f"Camera #{new_src} Active"
                    return True
                else:
                    self.cap = cap
                    self.is_connected = False
                    self.status_message = f"Camera #{new_src} unavailable"
                    return False
            except Exception as e:
                self.is_connected = False
                self.status_message = f"Error opening Camera #{new_src}: {e}"
                return False

    def _capture_loop(self) -> None:
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
                        self.is_connected = True
                else:
                    with self.lock:
                        self.ret = False
                    time.sleep(0.05)
            else:
                time.sleep(0.1)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self) -> None:
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        with self.lock:
            if hasattr(self, "cap") and self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass


class SignLanguageApp:
    """Modern Real-Time Sign Language Translator Application."""

    def __init__(self, root: Optional[tk.Tk] = None):
        self.root = root or tk.Tk()
        self.root.title("Sign Language to Text & Speech Translator (v2.0)")
        self.root.geometry("1100x750")
        self.root.minsize(960, 640)
        self.root.configure(bg="#11111b")

        # Core subsystems
        self.tracker = HandTracker(max_num_hands=1)
        self.classifier = LandmarkClassifier()
        self.corrector = SpellCorrector()
        self.tts = TTSEngine()

        # State tracking
        self.current_symbol = "None"
        self.confidence = 0.0
        self.current_word = ""
        self.current_sentence = ""
        self.auto_speak = tk.BooleanVar(value=False)
        self.mode = "MediaPipe 3D"

        # Frame rate & stabilization
        self._last_frame_time = time.time()
        self._fps = 0.0
        self._hold_counter = 0
        self._last_committed_char = ""
        self._hold_threshold = 14  # Approx ~0.5 sec at 30 fps
        self._blank_counter = 0

        # Video thread
        self.video_thread = VideoCaptureThread(src=0)

        # Build interface
        self._setup_styles()
        self._build_layout()

        # Handle graceful shutdown
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Kickoff main video render loop
        self.root.after(10, self._render_loop)

    def _setup_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dark.TFrame", background="#11111b")
        style.configure("Card.TFrame", background="#181825", relief=tk.FLAT)
        style.configure("Header.TLabel", background="#11111b", foreground="#cdd6f4", font=("Helvetica", 18, "bold"))
        style.configure("Label.TLabel", background="#181825", foreground="#a6adc8", font=("Helvetica", 11))
        style.configure("Value.TLabel", background="#181825", foreground="#89b4fa", font=("Courier", 16, "bold"))

    def _build_layout(self) -> None:
        # Top Header Bar
        header = tk.Frame(self.root, bg="#181825", padx=20, pady=12)
        header.pack(fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="🤟 Real-Time Sign Language Translator",
            font=("Helvetica", 16, "bold"),
            bg="#181825",
            fg="#cdd6f4",
        )
        title_lbl.pack(side=tk.LEFT)

        mode_badge = tk.Label(
            header,
            text="● MediaPipe 3D AI",
            font=("Helvetica", 10, "bold"),
            bg="#313244",
            fg="#a6e3a1",
            padx=10,
            pady=4,
        )
        mode_badge.pack(side=tk.LEFT, padx=15)

        reconnect_btn = tk.Button(
            header,
            text="🔄 Reconnect Camera",
            command=self.reconnect_camera,
            bg="#313244",
            fg="#89b4fa",
            font=("Helvetica", 10, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
        )
        reconnect_btn.pack(side=tk.LEFT, padx=5)

        self.switch_cam_btn = tk.Button(
            header,
            text="📷 Switch Camera (0 ↔ 1)",
            command=self.switch_camera,
            bg="#313244",
            fg="#fab387",
            font=("Helvetica", 10, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
        )
        self.switch_cam_btn.pack(side=tk.LEFT, padx=5)

        two_way_btn = tk.Button(
            header,
            text="⇄ Hearing Communicator",
            command=self.open_two_way,
            bg="#cba6f7",
            fg="#11111b",
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
        )
        two_way_btn.pack(side=tk.RIGHT, padx=5)

        # Main Body (Video Left, HUD Right)
        body = tk.Frame(self.root, bg="#11111b", padx=16, pady=12)
        body.pack(fill=tk.BOTH, expand=True)

        # Left Column: Video Feed (pack_propagate=False prevents expanding feedback loop)
        self.video_card = tk.Frame(body, bg="#181825", bd=1, relief=tk.SOLID, width=640, height=480)
        self.video_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.video_card.pack_propagate(False)

        self.video_label = tk.Label(self.video_card, bg="#181825")
        self.video_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Right Column: Detection, Text, Suggestions & Controls
        sidebar = tk.Frame(body, bg="#11111b", width=380)
        sidebar.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        sidebar.pack_propagate(False)

        # 1. Active Gesture Card
        gesture_card = tk.Frame(sidebar, bg="#181825", padx=14, pady=10)
        gesture_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(gesture_card, text="CURRENT SIGN", font=("Helvetica", 10, "bold"), bg="#181825", fg="#6c7086").pack(anchor=tk.W)
        self.gesture_val = tk.Label(gesture_card, text="Waiting...", font=("Courier", 32, "bold"), bg="#181825", fg="#a6e3a1")
        self.gesture_val.pack(anchor=tk.W)

        self.conf_bar = ttk.Progressbar(gesture_card, orient=tk.HORIZONTAL, length=100, mode="determinate")
        self.conf_bar.pack(fill=tk.X, pady=(4, 0))

        # 2. Accumulated Word & Sentence Card
        text_card = tk.Frame(sidebar, bg="#181825", padx=14, pady=10)
        text_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(text_card, text="WORD BEING SPELLED", font=("Helvetica", 10, "bold"), bg="#181825", fg="#6c7086").pack(anchor=tk.W)
        self.word_val = tk.Label(text_card, text="", font=("Courier", 22, "bold"), bg="#181825", fg="#89b4fa")
        self.word_val.pack(anchor=tk.W, pady=(0, 6))

        tk.Label(text_card, text="FULL SENTENCE", font=("Helvetica", 10, "bold"), bg="#181825", fg="#6c7086").pack(anchor=tk.W)
        self.sentence_val = tk.Label(
            text_card,
            text="",
            font=("Helvetica", 13),
            bg="#181825",
            fg="#cdd6f4",
            wraplength=340,
            justify=tk.LEFT,
        )
        self.sentence_val.pack(anchor=tk.W, fill=tk.X, pady=(0, 4))

        # 3. Suggestion Buttons Card
        sugg_card = tk.Frame(sidebar, bg="#181825", padx=14, pady=10)
        sugg_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(sugg_card, text="SUGGESTIONS (CLICK TO COMMIT)", font=("Helvetica", 10, "bold"), bg="#181825", fg="#fab387").pack(anchor=tk.W, pady=(0, 6))

        self.sugg_buttons = []
        for i in range(4):
            btn = tk.Button(
                sugg_card,
                text="",
                command=lambda idx=i: self._apply_suggestion(idx),
                bg="#313244",
                fg="#cdd6f4",
                font=("Helvetica", 11, "bold"),
                relief=tk.FLAT,
                cursor="hand2",
                padx=8,
                pady=4,
            )
            btn.pack(fill=tk.X, pady=2)
            self.sugg_buttons.append(btn)

        # 4. Action Buttons Card
        action_card = tk.Frame(sidebar, bg="#181825", padx=14, pady=10)
        action_card.pack(fill=tk.BOTH, expand=True)

        speak_btn = tk.Button(
            action_card,
            text="🔊 Speak Sentence",
            command=self.speak_sentence,
            bg="#89b4fa",
            fg="#11111b",
            font=("Helvetica", 12, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=6,
            cursor="hand2",
        )
        speak_btn.pack(fill=tk.X, pady=(0, 6))

        btn_row1 = tk.Frame(action_card, bg="#181825")
        btn_row1.pack(fill=tk.X, pady=2)

        space_btn = tk.Button(
            btn_row1,
            text="␣ Space",
            command=self.commit_word,
            bg="#45475a",
            fg="#cdd6f4",
            font=("Helvetica", 10, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            padx=6,
            pady=4,
        )
        space_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        back_btn = tk.Button(
            btn_row1,
            text="⌫ Backspace",
            command=self.backspace,
            bg="#45475a",
            fg="#cdd6f4",
            font=("Helvetica", 10, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            padx=6,
            pady=4,
        )
        back_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        clear_btn = tk.Button(
            action_card,
            text="🗑 Clear All",
            command=self.clear_all,
            bg="#f38ba8",
            fg="#11111b",
            font=("Helvetica", 10, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            pady=4,
        )
        clear_btn.pack(fill=tk.X, pady=(6, 0))

        # Bottom Status Bar
        self.status_bar = tk.Label(
            self.root,
            text="Camera: Active | FPS: 0.0 | MediaPipe 3D Hand Landmarks Active",
            bg="#181825",
            fg="#a6adc8",
            font=("Helvetica", 9),
            anchor=tk.W,
            padx=16,
            pady=4,
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _render_loop(self) -> None:
        """Process video frames and update UI."""
        try:
            if not hasattr(self, "root") or not self.root.winfo_exists():
                return
        except Exception:
            return

        ret, frame = self.video_thread.read()
        if ret and frame is not None:
            # Mirror horizontally for natural webcam interaction
            frame = cv2.flip(frame, 1)

            # Detect hand & 3D landmarks
            annotated_frame, hands = self.tracker.find_hands(frame, draw=True)

            detected_label = "blank"
            confidence = 0.0

            if hands:
                self._blank_counter = 0
                hand_info = hands[0]
                raw_lms = hand_info["raw_landmarks"]
                norm_feats = hand_info["features"]
                bbox = hand_info["bbox"]

                # Crop hand patch for legacy model if needed
                x1, y1, x2, y2 = bbox
                hand_patch = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

                detected_label, confidence, _ = self.classifier.predict(
                    raw_landmarks=raw_lms,
                    normalized_features=norm_feats,
                    hand_image_bgr=hand_patch,
                )
            else:
                self._blank_counter += 1
                # If hand is away for 20 frames, reset hold counter
                if self._blank_counter > 20:
                    self._hold_counter = 0
                    self._last_committed_char = ""

            self._process_detection(detected_label, confidence)

            # Update FPS calculation
            now = time.time()
            dt = now - self._last_frame_time
            self._last_frame_time = now
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

            # Render frame in video_label
            self._display_frame(annotated_frame)
        else:
            # Display helpful placeholder when camera frame is unavailable
            self._display_placeholder()

        try:
            if hasattr(self, "root") and self.root.winfo_exists():
                self._after_id = self.root.after(25, self._render_loop)
        except Exception:
            pass

    def reconnect_camera(self) -> None:
        """Trigger re-scan and reconnection to video camera."""
        success = self.video_thread.reconnect()
        if success:
            self.status_bar.config(text=f"Camera reconnected successfully (Device #{self.video_thread.src})")
        else:
            self.status_bar.config(text="Camera connection failed. Verify macOS System Settings > Privacy & Security > Camera")

    def switch_camera(self) -> None:
        """Toggle between Camera 0 and Camera 1 (e.g. Built-in Mac camera vs iPhone Continuity)."""
        new_idx = 1 if self.video_thread.src == 0 else 0
        success = self.video_thread.set_source(new_idx)
        if success:
            self.switch_cam_btn.config(text=f"📷 Camera {new_idx} (Switch)")
            self.status_bar.config(text=f"Switched to Camera #{new_idx}")
        else:
            # Fall back to 0
            self.video_thread.set_source(0)
            self.switch_cam_btn.config(text="📷 Camera 0 (Active)")
            self.status_bar.config(text=f"Camera #{new_idx} unavailable. Defaulting to Camera #0")

    def _display_placeholder(self) -> None:
        """Render a helpful diagnostic screen when camera permission or device is pending."""
        card_w = self.video_card.winfo_width()
        card_h = self.video_card.winfo_height()
        if card_w < 50 or card_h < 50:
            card_w, card_h = 640, 480

        lbl_w = max(300, card_w - 16)
        lbl_h = max(200, card_h - 16)
        canvas = np.zeros((lbl_h, lbl_w, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 37)  # Dark purple/gray background (#181825)

        # Center diagnostic text
        cv2.putText(
            canvas,
            "Camera Feed Connecting / Select Camera",
            (max(20, lbl_w // 2 - 210), lbl_h // 2 - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (205, 214, 244),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "To switch between iPhone and Mac camera: click [Switch Camera] above",
            (max(20, lbl_w // 2 - 250), lbl_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (250, 179, 135),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Or tap 'Disconnect' on your iPhone screen",
            (max(20, lbl_w // 2 - 160), lbl_h // 2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (166, 227, 161),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "macOS: System Settings -> Privacy & Security -> Camera",
            (max(20, lbl_w // 2 - 210), lbl_h // 2 + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (137, 180, 250),
            1,
            cv2.LINE_AA,
        )

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        self.video_label.config(image=tk_img)
        self.video_label.image = tk_img

    def _process_detection(self, label: str, conf: float) -> None:
        """Handle gesture stabilization, letter accumulation, and autocomplete."""
        self.current_symbol = label
        self.confidence = conf

        # Update HUD label
        if label != "blank":
            self.gesture_val.config(text=label, fg="#a6e3a1")
            self.conf_bar["value"] = int(conf * 100)
        else:
            self.gesture_val.config(text="--", fg="#6c7086")
            self.conf_bar["value"] = 0

        # Letter accumulation with temporal debounce
        if label != "blank" and conf > 0.82:
            if label == self._last_committed_char:
                self._hold_counter += 1
            else:
                self._last_committed_char = label
                self._hold_counter = 1

            if self._hold_counter == self._hold_threshold:
                # Commit sign
                if len(label) == 1 and label.isalnum():
                    # Finger-spelled character
                    self.current_word += label
                    self.word_val.config(text=self.current_word)
                    self._update_suggestions()
                elif label in ["Hello", "Thank You", "Yes", "No", "ILY", "OK", "Rock"]:
                    # Whole-word / phrase gesture
                    if self.current_sentence and not self.current_sentence.endswith(" "):
                        self.current_sentence += " "
                    self.current_sentence += label
                    self.sentence_val.config(text=self.current_sentence)
                    if self.auto_speak.get():
                        self.tts.speak(label)
        else:
            self._hold_counter = 0

        # Update status bar
        self.status_bar.config(
            text=f"Camera: Active | FPS: {self._fps:.1f} | Detection: {self.current_symbol} ({self.confidence:.0%})"
        )

    def _update_suggestions(self) -> None:
        """Query NLP SpellCorrector and refresh suggestion pill buttons."""
        suggestions = self.corrector.suggest(self.current_word, max_suggestions=4)
        for i, btn in enumerate(self.sugg_buttons):
            if i < len(suggestions):
                btn.config(text=suggestions[i], state=tk.NORMAL)
            else:
                btn.config(text="", state=tk.DISABLED)

    def _apply_suggestion(self, index: int) -> None:
        """Commit selected suggestion button word into the sentence."""
        text = self.sugg_buttons[index].cget("text")
        if text:
            if self.current_sentence and not self.current_sentence.endswith(" "):
                self.current_sentence += " "
            self.current_sentence += text
            self.sentence_val.config(text=self.current_sentence)
            self.current_word = ""
            self.word_val.config(text="")
            self._update_suggestions()

    def commit_word(self) -> None:
        """Move current word into the sentence."""
        if self.current_word:
            if self.current_sentence and not self.current_sentence.endswith(" "):
                self.current_sentence += " "
            self.current_sentence += self.current_word
            self.sentence_val.config(text=self.current_sentence)
            self.current_word = ""
            self.word_val.config(text="")
            self._update_suggestions()

    def backspace(self) -> None:
        """Delete last character of word or sentence."""
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self.word_val.config(text=self.current_word)
            self._update_suggestions()
        elif self.current_sentence:
            self.current_sentence = self.current_sentence[:-1].rstrip()
            self.sentence_val.config(text=self.current_sentence)

    def clear_all(self) -> None:
        """Reset word and sentence."""
        self.current_word = ""
        self.current_sentence = ""
        self.word_val.config(text="")
        self.sentence_val.config(text="")
        self._update_suggestions()

    def speak_sentence(self) -> None:
        """Trigger TTS audio synthesis for accumulated sentence."""
        text = self.current_sentence.strip() or self.current_word.strip()
        if text:
            self.tts.speak(text)

    def open_two_way(self) -> None:
        """Open Two-Way Hearing Communicator visual reference window."""
        TwoWayDictionaryWindow(self.root)

    def _display_frame(self, frame_bgr: np.ndarray) -> None:
        """Fit frame into video label bounded by video_card without expanding."""
        card_w = self.video_card.winfo_width()
        card_h = self.video_card.winfo_height()
        if card_w < 50 or card_h < 50:
            card_w, card_h = 640, 480

        # Leave 16px inner padding
        max_w = max(50, card_w - 16)
        max_h = max(50, card_h - 16)

        # Resize preserving aspect ratio
        h, w, _ = frame_bgr.shape
        scale = min(max_w / w, max_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(image=pil_img)

        self.video_label.config(image=tk_img)
        self.video_label.image = tk_img

    def on_close(self) -> None:
        """Clean resource shutdown."""
        if hasattr(self, "_after_id") and self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
        self.video_thread.stop()
        self.tracker.close()
        self.tts.stop()
        try:
            self.root.destroy()
        except Exception:
            pass


def main() -> int:
    """Application CLI entrypoint."""
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
