# PAST-MRI Model Card

## Intended Use

PAST-MRI performs condition-controlled translation between MRI field strengths
for T1-weighted, T2-weighted, and T2-FLAIR images. It is research software and
is not a medical device.

## Training

- Stage 1: 2.5D masked-denoising pretraining on unpaired retrospective MRI.
- Stage 2: end-to-end supervised translation on paired multi-field subjects.
- Input and output: five adjacent axial slices.
- Backbone: width-128 conditional NAF U-Net with 145,546,501 parameters.
- Conditioning: source field, target field, and contrast.
- Selection: terminal EMA at step 40,000.

The published Stage 1 adaptation is inspired by masked-diffusion
autoencoding. It uses a shared 2D block mask over a five-slice stack and
reconstructs the clean center slice; it is not an exact reproduction of a 3D
masked-diffusion implementation.

## Observed External Performance

On the held-out 15-subject, 180-volume evaluation used in the accompanying
study:

| SSIM | nRMSE | LPIPS |
|---:|---:|---:|
| 0.915138 | 0.222335 | 0.097913 |

These aggregate values do not imply uniform superiority for every contrast or
field direction.

## Limitations

- Stage 2 uses only three paired multi-field subjects.
- Cross-site, cross-vendor, and pathological generalization are not
  established.
- Generated images must not replace acquired diagnostic MRI.
- Stage 2 interruption recovery restores model, EMA, optimizer, scaler,
  global RNG, and dataset RNG, but multi-worker sample order is not claimed
  bit-identical.

