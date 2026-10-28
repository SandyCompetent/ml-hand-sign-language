"""Spell correction and predictive word auto-completion for sign language fingerspelling."""

from __future__ import annotations
from typing import List, Optional
import os

try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None

# High-frequency conversational English vocabulary to prioritize in autocomplete
COMMON_CONVERSATIONAL_WORDS = [
    "hello", "how", "are", "you", "good", "morning", "night", "thanks", "thank",
    "please", "welcome", "fine", "help", "need", "what", "where", "when", "why",
    "who", "name", "nice", "meet", "friend", "happy", "sorry", "excuse", "yes",
    "no", "maybe", "today", "tomorrow", "yesterday", "water", "food", "time",
    "doctor", "hospital", "family", "mother", "father", "brother", "sister",
    "love", "like", "want", "work", "school", "home", "call", "understand",
]


class SpellCorrector:
    """Predictive spell-corrector and prefix auto-completer for finger-spelled text."""

    def __init__(self, language: str = "en"):
        self.language = language
        self._sc = None
        if SpellChecker is not None:
            self._sc = SpellChecker(language=language)
            self._sc.word_frequency.load_words(COMMON_CONVERSATIONAL_WORDS)

    def suggest(self, query: str, max_suggestions: int = 5) -> List[str]:
        """Return top word suggestions for partial typing or misspelled words.

        Prioritizes:
        1. Prefix completions (e.g. 'hel' -> 'hello', 'help') ranked by frequency.
        2. Levenshtein edit-distance candidates if query is a completed misspelled word.
        """
        if not query:
            return []

        clean_query = query.strip().lower()
        if not clean_query:
            return []

        if self._sc is None:
            # Basic fallback without spellchecker
            matches = [w for w in COMMON_CONVERSATIONAL_WORDS if w.startswith(clean_query)]
            return [m.upper() for m in matches[:max_suggestions]]

        suggestions: List[str] = []

        # 1. Prefix completions (user is currently fingerspelling a word)
        if len(clean_query) >= 2:
            prefix_matches = []
            for word in self._sc.word_frequency:
                if word.startswith(clean_query) and word != clean_query and len(word) > len(clean_query):
                    freq = self._sc.word_frequency[word]
                    prefix_matches.append((word, freq))
            # Sort by frequency descending
            prefix_matches.sort(key=lambda x: x[1], reverse=True)
            for w, _ in prefix_matches:
                if w not in suggestions:
                    suggestions.append(w)
                if len(suggestions) >= max_suggestions:
                    break

        # 2. Spelling correction candidates (if query might be misspelled)
        if len(suggestions) < max_suggestions:
            cands = self._sc.candidates(clean_query)
            if cands:
                # Rank candidates by frequency
                cands_ranked = [(w, self._sc.word_frequency[w]) for w in cands if w != clean_query]
                cands_ranked.sort(key=lambda x: x[1], reverse=True)
                for w, _ in cands_ranked:
                    if w not in suggestions:
                        suggestions.append(w)
                    if len(suggestions) >= max_suggestions:
                        break

        # Capitalize for display consistency with ASL capital letters
        return [s.upper() for s in suggestions[:max_suggestions]]

    def is_correct(self, word: str) -> bool:
        """Check whether the word is in the dictionary."""
        if not word or self._sc is None:
            return False
        clean = word.strip().lower()
        return clean in self._sc
