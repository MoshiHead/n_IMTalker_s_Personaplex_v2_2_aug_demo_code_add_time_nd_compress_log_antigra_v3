"""Regression tests for the pipeline bugs found in conversation_log_1.

Pure-Python: no GPU, no model load, no network. Run with:
    python -m pytest IMTalker/test_search_helpers.py -v
or directly:
    python IMTalker/test_search_helpers.py

These exercise the actual reported failures with the actual reported inputs
(transcripts, search passages, compressor answers) so a future regression
against the real conversation_log_1 examples fails loudly here first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import search_helpers as sh


# ---------------------------------------------------------------------------
# Compressor faithfulness (Task 5 / Example 3: "Aravilev does not provide...")
# ---------------------------------------------------------------------------

def test_compressor_overlap_rejects_the_aravilev_case():
    """conversation_log_1 turn 6, verbatim: the compressor invented a claim
    about an entity ("Aravilev") that appears nowhere in the passages -- only
    in the user's own question. The content-word overlap gate must reject it
    (this was the exact case the old raw-word overlap check let through)."""
    question = "So, Aravilev, can you help me to invest in Bitcoin?"
    chunks = [
        {"text": (
            "way to store and save bitcoin. The third way is one that some people. "
            "consider that want exposure to Bitcoin without actually directly buying "
            "Bitcoin, which is buying a crypto focused ETF. So there ar"
        )},
        {"text": (
            "Bitcoin can function either as an investment or a medium of exchange. So "
            "you can either spend it, trade it or hold it. If you're spending Bitcoin, "
            "there are a handful of retailers and digital services"
        )},
    ]
    bad_answer = "No, Aravilev does not provide information on investing in Bitcoin."
    overlap = sh.compressor_faithfulness_overlap(bad_answer, chunks)
    assert overlap < sh.COMPRESSOR_OVERLAP_MIN, (
        f"the Aravilev non-answer scored {overlap:.3f} overlap, expected it to fall "
        f"below the {sh.COMPRESSOR_OVERLAP_MIN} rejection threshold"
    )


def test_compressor_overlap_accepts_real_grounded_answers():
    """The same gate must not reject genuinely grounded answers. All four
    cases below are real (question, passages, answer) triples from
    conversation_log_1 where the compressor's answer WAS correct and must
    keep reaching the model."""
    cases = [
        (
            "What is today's bull market price?",
            [
                {"text": (
                    "As of Aug 20, 2026, $BULL has a total market cap of $--, "
                    "representing a 0.00% change over the past 24 hours. The current "
                    "price of $BULL is $0.00007147, while the 24-hour trading volume "
                    "stands at $4.63"
                )},
                {"text": (
                    "NasdaqCM - Delayed Quote * USD. Webull Corporation (BULL). Is BULL "
                    "a long-term buy? 8.78 -0.07 (-0.79%)."
                )},
            ],
            "The current price of $BULL is $0.00007147.",
        ),
        (
            "What is Google is top market price today?",
            [
                {"text": (
                    "Market cap. 4.22T. Market cap4.22T. Price-Earnings ratio. 17.32. "
                    "Dividend yield. 0.25%. Average volume. 20.55M. High today. $346.20."
                )},
                {"text": "Rank: Name, Market Cap, Price, Today, Price (30 days), Country."},
            ],
            "The current high price of Google stock is $346.20.",
        ),
        (
            "What is that today's exchange rate of euro to dollar?",
            [{"text": (
                "EUR to USD frequently asked questions. What is the EUR to USD "
                "exchange rate today? As of 06:15 UTC, the mid-market EUR to USD rate "
                "is 1 euro = $1.1679."
            )}],
            "The current exchange rate of Euro to Dollar is 1 euro = $1.1679.",
        ),
        (
            "What is the today's Bitcoin exact rate in dollar?",
            [{"text": (
                "All-time high. $126,080.00. Buy Bitcoin. Bitcoin price today. Bitcoin "
                "price today is $63,076.00. In the last 24 hours Bitcoin's price moved "
                "-0.40%. The current BTC to USD conversion rate is $63,076.00 per BTC."
            )}],
            "The current Bitcoin exact rate in dollars is $63,076.00.",
        ),
    ]
    for question, chunks, answer in cases:
        overlap = sh.compressor_faithfulness_overlap(answer, chunks)
        assert overlap >= sh.COMPRESSOR_OVERLAP_MIN, (
            f"legitimate grounded answer {answer!r} scored only {overlap:.3f} overlap "
            f"and would be wrongly rejected"
        )


# ---------------------------------------------------------------------------
# Tag / control-token stripping (Task 7: corrupted prefixes)
# ---------------------------------------------------------------------------

def test_strip_injected_tags_removes_logged_corrupted_prefixes():
    """Every corrupted-prefix example actually observed in conversation_log_1
    must come out clean (or at least without the leaked markup fragment)."""
    # Expected values only strip the CONFIRMED markup artifact (the stray
    # '>', the byte-fallback token, the literal BOS spelling) -- a short
    # leading word like "an"/"not what I" that survives is left alone on
    # purpose: there is no reliable way to tell a genuine truncated fragment
    # apart from a real short word/phrase without guessing, and guessing is
    # exactly the kind of regex-over-symptoms fix this gate is not meant to
    # be (the actual generation-time fix for the tag echo itself is
    # decode_piece's BOS/EOS/byte-fallback filtering plus the lookup-phrase
    # rotation, both in liveTry.py / search_helpers.py).
    cases = {
        ",.> Today's Bitcoin rate in dollars is $63,076.":
            "Today's Bitcoin rate in dollars is $63,076.",
        "an.> The current high price of Google stock is $346.20.":
            "an The current high price of Google stock is $346.20.",
        "not what I.> Aravilev is robot labs and they help build the robotic systems":
            "not what I Aravilev is robot labs and they help build the robotic systems",
        "<0x0A> Never say that you think this was a mistake.":
            "Never say that you think this was a mistake.",
        "<s><s> Oh, it's so funny. That's perfect.":
            "Oh, it's so funny. That's perfect.",
    }
    for raw, expected in cases.items():
        cleaned = sh.strip_injected_tags(raw)
        assert ">" not in cleaned, f"{raw!r} -> {cleaned!r} still has a stray '>'"
        assert "<0x" not in cleaned.lower(), f"{raw!r} -> {cleaned!r} still has a byte-fallback token"
        assert "<s>" not in cleaned, f"{raw!r} -> {cleaned!r} still has a literal <s>"
        assert cleaned == expected, f"{raw!r} -> {cleaned!r}, expected {expected!r}"


def test_strip_injected_tags_leaves_normal_text_alone():
    normal = "Bitcoin is a digital currency you can use online or hold as an investment."
    assert sh.strip_injected_tags(normal) == normal


# ---------------------------------------------------------------------------
# Lookup-phrase rotation (Task 6 contributor: identical filler repeated)
# ---------------------------------------------------------------------------

def test_lookup_tags_are_not_one_fixed_sentence():
    """conversation_log_1 injected the exact literal string
    "<lookup> Please wait a minute." six times in four minutes, and the model
    was later observed reproducing "please wait a minute" as its own genuine
    speech right before a 11-31s freeze. The phrase pool must vary and must
    not be that exact sentence for consecutive turns."""
    seen = {sh.wrap_with_lookup_tags(i) for i in range(8)}
    assert len(seen) > 1, "lookup phrasing never varies"
    assert sh.wrap_with_lookup_tags(0) != sh.wrap_with_lookup_tags(1)
    for phrase in seen:
        assert phrase.startswith("<lookup>")


# ---------------------------------------------------------------------------
# Router regression (make sure the faithfulness/tag fixes didn't touch this)
# ---------------------------------------------------------------------------

def test_router_rules_match_logged_decisions():
    """A handful of the exact routing decisions from conversation_log_1,
    pinned so a future change can't silently break routing while fixing
    something else."""
    cases = [
        ("What is your name?", False),
        ("What is Bitcoin?", None),  # undecided by rules -> the model router
        ("So is it safe to invest in Bitcoin?", False),
        ("How I can invest in Bitcoin.", False),
        ("What is today's bull market price?", True),
        ("What is cryptocurrency", None),
        ("Okay, thank you. Have a nice day.", False),
    ]
    for transcript, expected in cases:
        verdict, reason = sh.rule_route_explain(transcript)
        assert verdict == expected, (
            f"{transcript!r}: rule_route_explain returned {verdict!r} ({reason}), "
            f"expected {expected!r}"
        )


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print()
    if failed:
        print(f"{failed}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"ALL {len(tests)} TESTS PASSED")
