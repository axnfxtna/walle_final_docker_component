"""
Thai Reader Pipeline (MCP)
===========================
Converts raw Thai LLM output into syllable-spaced text that the
TTS engine (VachanaTTS / MMS-TTS-THAI) can pronounce correctly.

Example
-------
Input:  "ปี 4 ของคุณจะเน้นการปฏิบัติงานจริง"
Output: "ปี 4 ของ คุณ จะ เน้น การ ปะ ติ บัด งาน จริง"

The syllabifier uses pythainlp.tokenize.syllable_tokenize which is
already present in the project requirements.  Non-Thai tokens
(ASCII letters, digits, punctuation) are passed through unchanged.
"""

import re
from typing import List

# pythainlp must be installed (it is already in requirements.txt)
try:
    from pythainlp.tokenize import syllable_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Match a "word" as either a consecutive run of Thai characters or any
# non-whitespace run of non-Thai characters (English words, numbers, etc.)
_THAI_RANGE = "\u0e00-\u0e7f"
_TOKEN_RE = re.compile(
    rf"[{_THAI_RANGE}]+|[^{_THAI_RANGE}\s]+"
)


def _syllabify_thai_token(token: str) -> List[str]:
    """
    Split a single Thai word into its spoken syllables.
    Falls back to returning the token as-is if pythainlp is unavailable.
    """
    if not PYTHAINLP_AVAILABLE:
        return [token]
    try:
        syllables = syllable_tokenize(token)
        return [s for s in syllables if s.strip()]
    except Exception:
        return [token]


def _is_thai(text: str) -> bool:
    return bool(re.search(rf"[{_THAI_RANGE}]", text))


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class ThaiReaderPipeline:
    """
    MCP that transforms a Thai text string into space-separated syllables
    suitable for TTS.

    Usage::

        reader = ThaiReaderPipeline()
        tts_text = reader.process("ปี 4 ของคุณจะเน้นการปฏิบัติงานจริง")
        # → "ปี 4 ของ คุณ จะ เน้น การ ปะ ติ บัด งาน จริง"
    """

    def __init__(self):
        if not PYTHAINLP_AVAILABLE:
            print(
                "⚠️  pythainlp not found — Thai syllabification will be skipped.\n"
                "    Install with: pip install pythainlp"
            )

    # ------------------------------------------------------------------
    def process(self, text: str) -> str:
        """
        Syllabify all Thai words in *text* and return the result as a
        single space-joined string.

        Non-Thai sequences (numbers, English words, punctuation) are kept
        exactly as they appear between the Thai words.

        Args:
            text: Raw Thai string from the LLM response.

        Returns:
            Syllable-spaced string ready for TTS.
        """
        if not text or not text.strip():
            return text

        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()

        output_parts: List[str] = []

        # Walk through the original text character by character, preserving
        # existing whitespace structure while replacing Thai words with their
        # syllabified form.
        pos = 0
        for m in _TOKEN_RE.finditer(text):
            start, end = m.span()
            # Preserve any whitespace / punctuation between previous match and this one
            if pos < start:
                output_parts.append(text[pos:start])

            token = m.group()
            if _is_thai(token):
                syllables = _syllabify_thai_token(token)
                output_parts.append(" ".join(syllables))
            else:
                output_parts.append(token)

            pos = end

        # Append any trailing non-matched text
        if pos < len(text):
            output_parts.append(text[pos:])

        result = "".join(output_parts)
        # Collapse multiple spaces that may have accumulated
        result = re.sub(r" {2,}", " ", result).strip()
        return result

    # ------------------------------------------------------------------
    def process_lines(self, text: str) -> str:
        """
        Process a multi-paragraph text line-by-line and return the result.
        Blank lines are preserved.
        """
        lines = text.splitlines()
        return "\n".join(
            self.process(line) if line.strip() else ""
            for line in lines
        )
