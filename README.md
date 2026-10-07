# AgentTrainer

A native, local macOS workspace for learning general computer interaction from demonstrations. Built with SwiftUI, ScreenCaptureKit and MLX on Apple Silicon.

**Development in progress.** See [the implementation ledger](docs/DEVELOPMENT.md) for implemented and unverified work, and [the acceptance objective](docs/OBJECTIVE.md) for the full scope.

## Build

Requires Apple Silicon, Xcode, and the configured Apple Development certificate. The build script never falls back to ad hoc signing.

```sh
./script/build_and_run.sh
```

Recordings and model data stay outside the source repository. No recordings, credentials, or model weights should be committed.
