"""Unit tests for SpellCorrector."""

from sign_language.nlp import SpellCorrector


def test_spell_corrector_prefixes():
    sc = SpellCorrector()
    suggestions = sc.suggest("hel", max_suggestions=4)
    assert len(suggestions) > 0
    # Suggestions should be capitalized
    for s in suggestions:
        assert s.startswith("HEL")


def test_spell_corrector_misspelling():
    sc = SpellCorrector()
    suggestions = sc.suggest("hellp", max_suggestions=4)
    assert len(suggestions) > 0
    assert any("HELP" in s or "HELLO" in s for s in suggestions)


def test_spell_corrector_empty():
    sc = SpellCorrector()
    assert sc.suggest("") == []
    assert sc.suggest("   ") == []


def test_spell_corrector_is_correct():
    sc = SpellCorrector()
    assert sc.is_correct("hello") is True
    assert sc.is_correct("xyzqwertyyy123") is False
