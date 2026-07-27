# Release Manifest

Release date: 2026-07-27

## Scope

This source-only release contains the final PAST-MRI workflow. Datasets,
checkpoints, generated predictions, historical experiment branches, private
paths, and service credentials are excluded.

## Compatibility Contract

The release was checked against the terminal research artifacts:

| Artifact | Expected result |
|---|---:|
| Stage 1 model parameters | 143,114,964 |
| Stage 1 state keys | 474 |
| Stage 1 terminal step | 40,000 |
| Stage 2 model parameters | 145,546,501 |
| Stage 2 state keys | 492 |
| Transferred Stage 1 keys | 264 |
| Transfer shape mismatches | 0 |
| Stage 2 terminal step | 40,000 |

Both terminal model state dictionaries load with `strict=True`. Stage 1 and
Stage 2 full checkpoints retain optimizer and random-state data for
continuation; inference uses the terminal Stage 2 EMA checkpoint.

## Source Hashes

```text
98717b1861b3fb6d87b817e7fb9fcb64384fcd224020008c8664e641636e08fa  configs/stage1_pretrain.yaml
08a7d2f9757b6e5d3a8bfb178e9d7ae7f77d783cc0227d2cc02de784d699f229  configs/stage2_translation.yaml
290b48767f4abecb68cce210a87c4ba7a4fef3120303a3b2795cb6eba3d5b032  past_mri/train_stage1.py
2d488b54b0e3239b295c23de93a833b03b495d1c4e5e947e50bacfe1efe52b41  past_mri/train_stage2.py
c06ebab61c28cdfbeada6d4c13d78b29282115ff0c2d9c49a655ced849b7ef87  past_mri/infer.py
a2d797d318adbc7c4165245c43673b62092fd37f9f553561c188047e01171a4a  scripts/run_pipeline.sh
3620bc03338a92b8d8fe657cc8b5cec5818dc1b42eecaefc828ac4f93b1a569b  scripts/verify_checkpoint.py
3c0fd1c6052ef10179ef909bb956ade8e3773d664e9105d9ca196ff761f302c2  scripts/verify_submission.py
```

## Verification

The release passes 26 focused tests covering data preparation, masking and
noise construction, five-slice supervision and gradient coverage, checkpoint
resume state, frozen recipe values, and strict 180-file package validation.
The shell entry point passes `bash -n`, every documented Python CLI imports,
and a private-path and credential scan returns no findings.
