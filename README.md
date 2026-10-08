# AgentTrainer

A native macOS workspace for learning computer interaction from demonstrations. Recording, pretraining, imitation learning and execution run locally with SwiftUI, ScreenCaptureKit and MLX on Apple Silicon.

Development is ongoing. The [implementation ledger](docs/DEVELOPMENT.md) records observed behavior and remaining limits; [OBJECTIVE.md](docs/OBJECTIVE.md) contains the acceptance scope.

## Build and run

Requires Apple Silicon, Xcode, and the configured Apple Development certificate. Every build uses that identity; the scripts never fall back to ad hoc signing.

```sh
./script/build_and_run.sh          # Debug build and launch
./script/build_and_run.sh --release
open build/Build/Products/Release/AgentTrainer.app
```

The release build is optimized for arm64. Run the unit/runtime checks with `./script/build_and_run.sh --test`. The test host uses an isolated temporary workspace.

## Workflow

1. In **Settings**, grant Screen Recording, Input Monitoring and Accessibility permissions.
2. In **Library**, create demonstration folders and optional, separate pretraining folders.
3. In **Record**, choose a capture target, folder and input capture settings. Record varied examples, including waiting after completion. The default recording shortcut is **Control–Option–Command R**.
4. In **AI Models**, choose vision, memory and action capabilities, then assign data. Save the configuration. Unsaved edits survive tab changes while the app stays open; Discard restores the saved configuration.
5. In **Train**, optionally run **Pre-train**, then **Train**. By default, a new imitation run uses compatible pretrained weights when available; otherwise it starts from scratch. Select **Latest trained checkpoint** to fine-tune after adding demonstrations or changing training settings, or **New random weights** for a fresh start. These new runs reset the optimizer and memory. **Resume** continues an interrupted run with its saved settings, optimizer, memory and data split.
6. In **Run**, select a trained model, target and allowed actions. Compare the latest checkpoint with the one selected by validation loss. The default run shortcut is **Control–Option–Command P**; **Control–Option–Command Escape** stops and releases agent-held input.

Architecture or capability changes require retraining. Existing linear pretraining checkpoints retain compatibility; new models offer a spatial predictor that can learn localized action effects. Pretraining is optional and does not guarantee better task performance.

Models with task conditioning accept instructions up to 96 UTF-8 bytes. The visible counter accounts for multibyte characters. Library preserves longer original instructions for editing; training and Run explain the limit instead of silently truncating them. Models without instruction conditioning ignore this text.

Held-out scores use recorded histories. Check actual task completion, including whether the model waits after finishing. The validation workspace has demonstrated text clearing, Calculator pointer control and short visual-cue recall; these do not establish general competence on unfamiliar tasks.

Recordings, models and checkpoints live outside this repository, by default under `~/Library/Application Support/AgentTrainer`. Storage changes copy and verify data before switching paths, preserving the source. Do not commit recordings, credentials or model weights.
