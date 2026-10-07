# AgentTrainer implementation and validation ledger

The acceptance contract is [OBJECTIVE.md](OBJECTIVE.md). The project is not complete until the normal Record → Library → AI Model → Pre-train → Train → Run workflow works on representative macOS applications and the final audit has passed.

## Milestones

- [x] Native signed application, repository, build/run entry point, reference-informed navigation.
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

### Foundation implementation (2026-10-07)

- Private repository: https://github.com/UnendlessGit/AgentTrainer, primary branch `main`; initial and implementation commits pushed.
- Signed Xcode app builds with the existing non-revoked Apple Development certificate. No ad hoc fallback; MLX package resource bundles also receive the development team/identity. `script/build_and_run.sh` and the Codex Run action are configured.
- Native SwiftUI navigation, Record settings/preview, Library folders/table/inspector, model architecture/data editor, Settings permissions/storage/appearance. Train and Run explicitly remain unavailable pending orchestration; they do not simulate progress.
- Recoverable JSONL observation/input journals and atomic manifests. Input capture preserves sub-frame transitions, raw pointer locations, repeats and modifiers. Static pixels are reused. Capture has independent bounded input and visual queues. Swift 6 timer isolation crash fixed after reproducing it through the UI; even capture dimensions fixed window stream startup.
- Display recording through the public UI completed with 294 observations/72 input events, including 205 static observations. TextEdit window capture completed with 728 observations/12 input events. Captured image inspected visually. Raw key press durations down to roughly 76 microseconds remained in the journal. These were short smoke validations, not sufficient final accuracy evidence.
- Library auto-inclusion and preview/action inspector verified through the UI. The attempted AppleScript metadata edit focused the wrong field; it saved a test recording name as `1`. Do not count that as a passing trim workflow. The test recording is disposable; original pixels/events remain preserved. Improve robust UI interaction/verification before the final audit.
- A synthetic modifier test revealed that querying global key state missed injected modifier transitions. Capture now reads the event's device-specific left/right flags, with a unit regression. Needs a fresh real recording verification after rebuild.
- 25 unit/runtime tests passed in under one second on the last warm run (cold policy runtime checks approximately four seconds). Coverage includes causal alignment, action grammar, journal recovery, failed checkpoint writes, checksum corruption, configuration mismatch, streamed event-level targets, MLX gradients, causal-attention future exclusion, recurrent carry equivalence, optimizer resume equivalence, pixel orientation and letterbox transforms.
- MLX native policy implemented: patch transformer vision, joint spatial pointer distribution plus patch-conditioned offsets, explicit recurrent carry or bounded causal attention, state/action/instruction conditioning, action-conditioned future spatial RGB pretraining head. Self-supervised loss gradient test verifies vision gradients and no imitation-head gradients. It is NOT trained or evaluated in closed-loop behavior yet.
- `ObservationPreprocessor` and `PolicyActionCodec` share letterbox geometry. Model arrays and optimizer are designed for one worker. `ResumableAdamW` exposes all moments for exact resume; `CheckpointStore` writes immutable payload directories before latest/best pointers.
- XCTest UI test target is prepared with an isolated temporary workspace. Its first run failed before test execution: macOS Automation Mode requires user authentication and initialization timed out after 60 seconds. Do not describe it as a passed UI test. Existing Accessibility-based UI inspection remains available; no system security setting was weakened.

### Immediate remaining implementation

1. Finish bounded disk-backed dataset/sequence index and tensor batching, data eligibility/capability validation and recording-level hold-out splits. Wire imitation and self-supervised losses into a native MLX training worker with real metrics, pause/resume/cancel and checkpoint restoration. Connect Train controls; no fake progress.
2. Implement live capture observation source and policy run worker using the exact same preprocessor/action grammar/temporal semantics. Global configurable shortcuts, emergency release, human override, permission intersection and target bounds checks must precede enabling input execution.
3. Validate dynamic window/crop geometry introduced after the last capture smoke test. Recheck modifiers, clicks, scroll, dragging, relative input, regions, full desktop, secure-input interruption, stop-during-start and graceful application quit. Avoid claiming full recording correctness from keyboard-only checks.
4. Model deletion, recording deletion/recovery approval, persistent Record form state, storage migration during mutation, instruction quality and architecture alternatives still need product work/review. Cache management and resource measurements are incomplete.
5. Complete every final-audit gate in OBJECTIVE.md, including actual real-app Record → Library → Model → Pre-train → Train → Run, controlled equivalent-budget architecture comparisons, measured learned behavior and repeated end-to-end checks after fixes. Individual runs must stay below 20–30 minutes.

### Native training workflow (2026-10-07)

- Disk-backed example/offset index with streaming hashes; deterministic recording-level held-out split; persistent recurrent lanes and truncated backpropagation; no sequence crosses recording boundaries. Unsupported targets fail with the recording/action named. Actions outside the capture target are counted and excluded explicitly.
- Real MLX imitation losses for grammar-masked action tokens, conditional timing, joint pointer patches/offsets and selected continuous arguments. Separate action-conditioned future-pixel pre-training objective, weighted toward changing patches. Future frames that precede the action are excluded unless the capture explicitly reports static reuse.
- Serial training worker, finite loss/gradient checks, clipping, memory preflight estimate, real CPU/process/MLX memory metrics, loss chart, deadline pause, batch-boundary pause/cancel, checksummed atomic weights/optimizer/recurrent state checkpoints, latest/best pointers. Resume restores saved settings and verifies unchanged dataset/configuration. Pre-training and imitation fingerprints are tracked independently.
- Instruction conditioning now includes positions and attention so word/byte order is not erased. Architecture implementation version 2 invalidates earlier development checkpoints; the earlier UI smoke checkpoint is intentionally incompatible after this change.
- 29 tests pass. New integration coverage exercises pre-training → imitation, complete held-out evaluation, disk-backed sequence boundaries, explicit capability rejection, and pause/resume equivalence against uninterrupted training (all parameter errors < 1e-5). Instruction-order sensitivity is checked; this is not evidence of language understanding.
- A normal UI workflow created/configured a model, selected a TextEdit recording, encountered and explained unsupported-key errors, then completed one imitation epoch: 375 decisions, 24 steps, approximately 5.84 seconds, final training loss 0.4493, 249 MB process memory at completion, checkpoint saved. No validation score was shown because only one recording was assigned. This is workflow evidence only, not policy-quality or closed-loop evidence. Screenshot is local at `.validation/train-completed.png` and is not committed.
- TextEdit recording `136783F2-19F9-407D-8298-256488FBF0E1` completed with 371 observations and six injected key transitions; observation bounds followed a window move from (185,83) to (200,190). System Events injected modifier flags without separate modifier transitions, so this does NOT verify raw modifier transition capture.
- Fixed a tensor-axis crash found by the integrated imitation test. Added explicit keyboard-selection buttons after the native disclosure triangle was not operable through Accessibility. Expanded the key catalog and general-desktop defaults, including keypad, navigation and function keys.
- Recording form now persists across tabs/launches. Carbon global shortcut registration/configuration, clean quit handling and storage-migration activity guards are implemented and compile; user-facing shortcut/quit validation is still pending. Run shortcut currently opens Run because inference is not yet connected.

Next priorities: validate global recording shortcuts/clean quit and normal pre-training UI; implement shared live inference with emergency release/human override/focus/bounds guards; complete remaining Library CRUD/recovery review, capture edge cases, controlled architecture comparisons and full real-app end-to-end audit. Do not mistake the passing training tests or low smoke-test loss for completion.
