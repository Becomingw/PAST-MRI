from __future__ import annotations

import pytest
import torch


def test_restore_rng_state_accepts_cpu_state() -> None:
    from past_mri.train_stage2 import restore_rng_state

    state = torch.get_rng_state()
    restore_rng_state({"torch": state})
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for map_location regression")
def test_restore_rng_state_moves_cuda_mapped_cpu_state_back_to_cpu() -> None:
    from past_mri.train_stage2 import restore_rng_state

    state = torch.get_rng_state()
    restore_rng_state({"torch": state.cuda()})
    assert torch.equal(torch.get_rng_state(), state)
