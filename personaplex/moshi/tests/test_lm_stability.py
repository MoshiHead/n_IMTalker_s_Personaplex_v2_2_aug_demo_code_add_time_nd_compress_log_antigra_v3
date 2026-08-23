"""CPU regression tests for the long-conversation stability mechanisms in
moshi/models/lm.py: the prompt-snapshot / context-refresh position-cliff fix,
the text repetition penalty, the silence breaker, and BOS/EOS masking.

These run real forward passes against a tiny synthetic LMModel (no GPU, no
downloaded checkpoint needed) rather than mocking anything -- the point is to
catch a real regression in the tensor-shape/offset bookkeeping, which a mock
would hide. Run with:

    python tests/test_lm_stability.py

from moshi/, or with pytest from anywhere once moshi is importable.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from moshi.models.lm import LMModel, LMGen

# AUDIO_TOKENS_PER_STREAM=8 is hardcoded in lm.py (8 assistant + 8 user audio
# codebooks + 1 text = 17), and dep_q must equal n_q for step()'s gathered
# output to actually be 17-wide (the depformer produces all 16 audio
# positions; the 8 "user" ones are overridden by the true input_tokens via
# the provided_ mask) -- see refresh_context_async's docstring and the
# _history comment in LMGen.__init__. A synthetic test model must match this
# shape exactly or prepare_step_input's assertions fail immediately.
N_Q = 16
DEP_Q = 16
DIM = 32
CARD = 2048  # SILENCE_TOKENS/SINE_TOKENS use hardcoded ids up to 2008
TEXT_CARD = 48
CONTEXT = 40
NUM_CODEBOOKS = N_Q + 1
NEEDED_INPUT_TOKENS = NUM_CODEBOOKS - 8 - 1


def _build_model(context: int = CONTEXT) -> LMModel:
    torch.manual_seed(0)
    model = LMModel(
        delays=[0] * NUM_CODEBOOKS, n_q=N_Q, dep_q=DEP_Q, card=CARD, text_card=TEXT_CARD,
        dim=DIM, num_heads=2, hidden_scale=2, depformer_dim=16, context=context,
        device="cpu", dtype=torch.float32, num_layers=2,
        depformer_num_heads=2, depformer_num_layers=2,
        depformer_weights_per_step=True, depformer_gating="silu", causal=True,
    )
    model.eval()
    return model


def _rand_audio_tokens() -> torch.Tensor:
    return torch.randint(0, CARD, (1, NEEDED_INPUT_TOKENS, 1))


def test_refresh_rewinds_offset_and_model_keeps_working():
    """The actual fix for the RoPE absolute-position cliff: a refresh must
    REWIND the offset (not just replay history at the same growing offset),
    and the model must keep producing valid output afterward."""
    lm_model = _build_model()
    lm_gen = LMGen(
        lm_model, device="cpu", audio_silence_frame_cnt=2,
        text_prompt_tokens=list(range(4, 20)), top_k=16, top_k_text=16,
    )
    with lm_gen.streaming(1), lm_model.streaming(1):
        lm_gen.step_system_prompts(mimi=None)
        prompt_end = lm_gen._streaming_state.offset
        assert prompt_end > 0

        lm_gen.save_prompt_snapshot()
        assert lm_gen.has_prompt_snapshot
        lm_gen.start_history_recording(30)

        for _ in range(60):
            out = lm_gen.step(input_tokens=_rand_audio_tokens())
            if out is not None:
                assert out.shape == (1, NUM_CODEBOOKS, 1)
                assert not torch.isnan(out.float()).any()
        offset_before = lm_gen._streaming_state.offset
        assert offset_before == prompt_end + 60

        stats = asyncio.run(lm_gen.refresh_context_async(
            is_alive=None, batch_size=4, history_steps=30))
        assert stats["completed"]
        assert stats["restored_snapshot"] is True
        offset_after = lm_gen._streaming_state.offset
        assert offset_after < offset_before, "refresh did not rewind the offset"
        assert offset_after == prompt_end + 30

        produced = 0
        for _ in range(20):
            out = lm_gen.step(input_tokens=_rand_audio_tokens())
            if out is not None:
                produced += 1
                assert not torch.isnan(out.float()).any()
        assert produced > 0, "model stopped producing output after a refresh"


def test_bos_eos_never_sampled_mid_stream():
    """conversation_log_1 turn 0/22: BOS sampled mid-stream decodes to the
    literal string "<s>" and gets spoken. mask_bos_eos_mid_stream must make
    that impossible regardless of what the (here: random-weight) model would
    otherwise have preferred."""
    lm_model = _build_model()
    lm_gen = LMGen(
        lm_model, device="cpu", audio_silence_frame_cnt=2, top_k=16, top_k_text=16,
        mask_bos_eos_mid_stream=True,
    )
    with lm_gen.streaming(1), lm_model.streaming(1):
        text_tokens = []
        for _ in range(150):
            out = lm_gen.step(input_tokens=_rand_audio_tokens())
            if out is not None:
                text_tokens.append(int(out[0, 0, 0].item()))
        bad = [t for t in text_tokens if t in (1, 2)]
        assert not bad, f"BOS/EOS leaked into the text channel: {bad}"


def test_silence_breaker_forces_a_real_word():
    """conversation_log_1 turns 14/18/19/21: the model produced zero spoken
    text for 11-31s repeatedly. The breaker must force a non-PAD/EPAD token
    once the configured run length is reached, and reset afterward."""
    lm_model = _build_model()
    lm_gen = LMGen(
        lm_model, device="cpu", audio_silence_frame_cnt=2, top_k=16, top_k_text=16,
        text_silence_break_steps=3,
    )
    with lm_gen.streaming(1), lm_model.streaming(1):
        # Warm up past the depformer's output delay (step() returns None
        # until then regardless of the breaker) before arming it.
        for _ in range(10):
            lm_gen.step(input_tokens=_rand_audio_tokens())
        lm_gen._silence_run.fill_(lm_gen.text_silence_break_steps)
        out = lm_gen.step(input_tokens=_rand_audio_tokens())
        assert out is not None
        forced = int(out[0, 0, 0].item())
        assert forced not in (0, 3), f"breaker failed to exclude EPAD/PAD, sampled {forced}"
        assert int(lm_gen._silence_run.item()) == 0, "run counter did not reset after breaking"


def test_clear_silence_run_and_clear_text_repetition_reset_state():
    lm_model = _build_model()
    lm_gen = LMGen(lm_model, device="cpu", text_silence_break_steps=5, text_rep_penalty=1.2)
    lm_gen._silence_run.fill_(9)
    lm_gen.clear_silence_run()
    assert int(lm_gen._silence_run.item()) == 0

    lm_gen._rep_counts[5] = 3
    lm_gen.clear_text_repetition()
    assert bool((lm_gen._rep_counts == 0).all())
    assert bool((lm_gen._rep_buffer == -1).all())


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print()
    if failed:
        print(f"{failed}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"ALL {len(tests)} TESTS PASSED")
