# AgentTrainer implementation and validation ledger

The acceptance contract is [OBJECTIVE.md](OBJECTIVE.md). The project is not complete until the normal Record → Library → AI Model → Pre-train → Train → Run workflow works on representative macOS applications and the final audit has passed.

## Milestones

- [ ] Native signed application, repository, build/run entry point, reference-informed navigation.
- [ ] Versioned recording journal, raw transitions, static-scene observations, recovery, causal alignment tests.
- [ ] ScreenCaptureKit targets, recording UI, global input capture, permissions, start/stop shortcuts.
- [ ] Folder organization, preview/action inspection, nondestructive trimming and metadata.
- [ ] Model architecture configuration, capability grammar, separate data assignments, compatibility fingerprints.
- [ ] MLX vision/temporal policy, shared observation/action semantics, self-supervised pre-training.
- [ ] Local imitation training, validation, live metrics, pause/resume/cancel, atomic complete checkpoints.
- [ ] Local closed-loop inference, target bounds, action grammar, human override and emergency release.
- [ ] Architecture comparison with equivalent data and budgets; measured performance review.
- [ ] End-to-end real-app tests, every major visible control, full source audit, final signed build.

## Engineering constraints

No individual test, training run, benchmark, stability run, architecture comparison or validation task may exceed 20–30 minutes without an explicit request for a longer run. Prefer short representative runs; split validation into bounded cases.

All application execution, recording, learning and inference are local. No telemetry or runtime network dependency. Build-time package downloads are pinned. Never ad hoc sign app builds; fail if the configured Apple Development identity is unavailable.

Record event transitions on a monotonic clock independently of capture cadence. Observations carry both source display time and availability time. Never train an earlier action against pixels from its future. Keep original data immutable; use editable metadata overlays. Durable manifests must be atomically replaced after their referenced data is flushed. Interrupted artifacts remain inspectable and are never silently called complete.

No mock training progress, invented GPU utilization, or untrained checkpoint presented as runnable. Visible unavailable features must explain their actual prerequisite. Distinguish implemented behavior, automated evidence and user-workflow validation.

## Current evidence

2026-10-07: Empty project initialized on `main`. Four UI references inspected. Xcode 27, Swift 6.4, arm64, 36 GiB memory. A non-revoked Apple Development signing identity is available. Implementation has begun; no end-to-end validation yet.
