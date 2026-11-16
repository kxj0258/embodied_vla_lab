# Phase 5A: small world-model data contract

Date: 2026-08-28  
Status: `COMPLETE_DATA_GO`

## Objective

Determine whether the existing turn-on-stove demonstrations provide aligned temporal data for a small latent world
model, without downloading data or training a model. The intended comparison is copy-current-frame versus learned
action-free and action-conditioned dynamics.

## Frozen protocol

- Reuse the established 40-train/10-validation episode split; never split frames from one episode across sets.
- Primary view is `agentview_rgb`; `eye_in_hand_rgb` is a secondary diagnostic, not an automatic second training run.
- A one-step item is `(image_t, action_t, image_t+1)`; an H-step item contains actions `t…t+H−1` and target `t+H`.
- Horizons are `[1,2,4,8]`; windows never cross episode or consume the terminal source action.
- Compare copy-frame, action-free latent dynamics and action-conditioned latent dynamics on identical samples.
- Report pixel MAE/MSE/PSNR at every horizon. Candidate-action ranking remains diagnostic until prediction quality
  passes a separate offline gate.

Static copy-frame error measures temporal difficulty only. It is not learned-model performance, reward prediction,
environment success or evidence that candidate actions can be ranked.

## Audit result

All 12 contract checks PASSed. The unchanged 40/10 split contains 3,571/889 frames. Train transition counts at
H=`[1,2,4,8]` are `[3531,3491,3411,3251]`; validation counts are `[879,869,849,809]`. Episode lengths range
77–119 in train and 75–103 in validation. All 4,460 actions are finite and every dimension is nonconstant; state,
reward and terminal arrays align with image/action length. All 50 demonstrations end in recorded success.

The validation copy-frame baseline is much stronger on the static third-person view than the moving wrist camera.
Agentview MAE for H=1/2/4/8 is 0.769/1.181/1.739/2.402 pixels, while wrist MAE is
6.278/9.942/15.927/24.594. Corresponding agentview PSNR is 33.28/30.61/28.35/26.55 dB versus wrist
22.39/19.62/17.04/14.85 dB. This is baseline difficulty evidence, not learned performance.

Decision: `DATA_CONTRACT_PASS_BASELINE_IMPLEMENTATION_ALLOWED`. Phase 5B may implement CPU/static model and resource
contracts, but no GPU training is authorized. Canonical result:
`results/phase5a_libero_world_model_data_contract_v1.yaml`.
