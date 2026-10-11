# Artifact inventory and experiment provenance

Original host: `/workspace/GR00T_DualArm_Project`, RunPod persistent network FUSE. Full project is approximately190GB; it is not uploaded to GitHub.

| Artifact | Original relative path | Git contents |
|---|---|---|
| Official base | checkpoints/GR00T-N1.7-3B | version reference only |
| B | checkpoints/lift_pot_rgb_B/checkpoint-2000 | config/provenance references; no weights |
| C | checkpoints/lift_pot_C_v1/checkpoint-2000 | config/provenance references; no weights |
| C prepared train/validation/test | datasets/lerobot/lift_pot_C_{train,validation,test}_v1 | manifest and QC summaries |
| Paired raw rollouts | evaluation/BC_blind_v2 | aggregate CSV + failure classification; raw files external |
| Success/failure video | videos/BC_blind_v2_{B,C}_seed*_raw.mp4 | path references; MP4 external |
| C stable success candidate | videos/BC_blind_v2_C_seed410000_raw.mp4 | path reference only |
| Historical B successes | videos/B_contact_feedback_{qpos_full_h5_seed19,drive_h5_seed107}_raw.mp4 | diagnostic provenance only; excluded from formal0/20 |
| Local daily save check | results/end_of_day_2026-10-10/save_manifest.json | kept local because includes large detailed inventory/host metadata |

B and C each have3 safetensors shards and1030 tensors with config, processor_config, statistics, embodiment ID and experiment config. C previously passed persistent-copy SHA256 and real-model forward checks. CPU patches never load/write these weights. Training was model-only: optimizer state is unavailable for exact continuation.

Preserve full datasets and MP4 on the volume or an authenticated separate artifact store. Git records are evidence summaries, not a backup of large artifacts. A separate GPU cannot train solely from these CSV summaries.

Before making a publication or external benchmark claim, report: base0-only selection, expert_check=False, fixed drive_h5,60ms,4 diffusion steps,200 actions before first success, same policy for subsequent2s, paused simulation during inference. After CPU analysis these20 seeds are no longer unused tuning data.
