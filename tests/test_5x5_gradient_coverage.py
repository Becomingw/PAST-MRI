from __future__ import annotations

import torch


def test_all_five_output_channels_receive_supervised_gradients():
    from past_mri.losses import Task3DistillLoss
    from past_mri.models import CondNAFUNet5x5

    torch.manual_seed(7)
    model = CondNAFUNet5x5(
        in_channels=5,
        output_channels=5,
        width=8,
        enc_blocks=[1],
        mid_blocks=1,
        dec_blocks=[1],
        cond_dim=32,
        use_calibration_head=False,
        use_hyper_residual_head=False,
    )
    source = torch.rand(2, 5, 24, 24)
    target = torch.stack(
        [torch.full((2, 24, 24), 0.1 + 0.15 * channel) for channel in range(5)],
        dim=1,
    )
    ids = (
        torch.tensor([0, 1]),
        torch.tensor([4, 3]),
        torch.tensor([0, 2]),
    )
    output = model(source, *ids, base_center=source)
    criterion = Task3DistillLoss(
        charbonnier=1.0,
        msssim=0.2,
        lpips=0.0,
        gradient=0.0,
        residual_reg=0.0,
    )

    loss, _ = criterion(output, target, source=source)
    loss.backward()

    head_grad = model.residual_head.weight.grad
    assert head_grad is not None
    per_channel = head_grad.abs().flatten(1).sum(dim=1)
    assert per_channel.shape == (5,)
    assert torch.all(per_channel > 0), per_channel

