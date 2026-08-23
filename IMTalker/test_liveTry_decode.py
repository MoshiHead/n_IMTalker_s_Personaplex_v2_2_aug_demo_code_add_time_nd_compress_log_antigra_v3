"""Regression tests for liveTry.py's decode-time control-token filtering
(the root-cause fix for the "<s><s>"/"<0x0A>" leaks in conversation_log_1).

Pure-Python: tests the module-level helpers directly without instantiating
MoshiOnlyEngine (which needs a loaded model). Run with:
    python IMTalker/test_liveTry_decode.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# liveTry.py's top-level imports include cv2/torchaudio/fastapi, none of
# which this test needs (it only exercises the pure control-token-filtering
# helpers, defined before any of those are touched) and none of which may be
# installed in a CPU-only / non-avatar test environment. Stub them out rather
# than skip the test entirely.
for _name in ("cv2", "torchaudio"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
if "fastapi" not in sys.modules:
    _fastapi = types.ModuleType("fastapi")
    for _cls in ("FastAPI", "WebSocket", "WebSocketDisconnect"):
        setattr(_fastapi, _cls, type(_cls, (), {}))
    sys.modules["fastapi"] = _fastapi
    _fastapi_responses = types.ModuleType("fastapi.responses")
    for _cls in ("FileResponse", "HTMLResponse", "JSONResponse"):
        setattr(_fastapi_responses, _cls, type(_cls, (), {}))
    sys.modules["fastapi.responses"] = _fastapi_responses

import liveTry as lt


def test_special_ids_cover_epad_bos_eos_pad():
    # 0=EPAD, 1=BOS, 2=EOS, 3=PAD -- see moshi/models/lm.py's
    # "0..3 = EPAD/BOS/EOS/PAD" comment. The pre-fix code only special-cased
    # 0 and 3; BOS(1)/EOS(2) fell through to id_to_piece and got spoken.
    assert lt._SPECIAL_TEXT_TOKEN_IDS == frozenset((0, 1, 2, 3))


def test_control_piece_detection():
    # Real leaks observed verbatim in conversation_log_1.
    must_be_control = ["<s>", "</s>", "<0x0A>", "<0x09>", "<unk>", "<pad>",
                        "<ref>", "</ref>", "<lookup>", "<system>"]
    for piece in must_be_control:
        assert lt._is_control_piece(piece), f"{piece!r} should be treated as control markup"

    must_not_be_control = [
        "hello", " world", "Bitcoin", "▁the", "$63,076.00", "an", ".", ",", "",
    ]
    for piece in must_not_be_control:
        assert not lt._is_control_piece(piece), f"{piece!r} should NOT be filtered as control markup"


def test_control_piece_is_case_insensitive_and_tolerant_of_spacing():
    for piece in ["<S>", "< s >", "<0X0A>", "<REF>"]:
        assert lt._is_control_piece(piece), f"{piece!r} should be treated as control markup"


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
