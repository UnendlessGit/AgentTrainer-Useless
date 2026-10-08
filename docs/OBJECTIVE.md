Build a new native macOS application called **AgentTrainer** using Swift and SwiftUI.

The application should let me:

1. Record demonstrations of myself using games, applications, browsers, and macOS.
2. Organize those recordings into reusable training libraries.
3. Create and configure AI models.
4. Train those models locally from my demonstrations.
5. Run the trained models locally so they can control either the whole computer or a selected display, window, or region.

The application must be designed as a **general computer-interaction learning system**. Do not architect the observation, action, model, memory, training, or inference systems specifically around games, browsers, or any single type of application.

Everything must operate locally.

## Main tabs

The application should have these primary tabs:

- Record
- Library
- AI Models
- Train
- Run
- Settings

Keep the workflow understandable and cohesive. Avoid spreading settings across unrelated tabs or requiring hidden prerequisites.

## Record

The Record tab should let me record as much training data as I want.

I should be able to select:

- Full desktop
- A specific display
- A specific window
- A region of a display or window

I should be able to configure useful recording settings such as:

- Capture resolution
- Capture rate
- Capture quality
- Cursor inclusion
- Keyboard capture
- Mouse buttons
- Pointer movement
- Scrolling
- Relative mouse movement when appropriate
- Recording name
- Target Library folder, including normal training folders or pre-training folders
- Any other settings that materially affect training quality

Record **raw timestamped input transitions** accurately.

Do not represent keyboard and mouse behavior only as periodic held-state snapshots. Short key presses, mouse clicks, button transitions, dragging, scrolling, chords and other transient actions must not disappear between visual observations.

Preserve precise timing between observations and actions.

Training targets must be causally aligned with the observation that existed before the action occurred.

Recording must remain reliable when the visual scene is static or changes slowly.

Interrupted recordings should be recoverable whenever possible.

Every recording must automatically appear inside the selected folder in Library.

## Library

Library is where recordings are organized.

Library should support folders.

Folders should be easy to:

- Create
- Rename
- Delete
- Reorganize
- Browse

Each folder contains recordings.

A recording should expose useful information such as:

- Name
- Duration
- Recording date
- Instruction/task if applicable
- Number of observations
- Number of input events
- Training eligibility/status

I should be able to preview a recording and inspect the actions that were captured.

Edits such as trimming or metadata changes should preferably be nondestructive so the original recording remains recoverable.

The storage format should be versioned, reliable and designed for large amounts of training data.

### Pre-training data

Library should also have a separate section of folders specifically for **pre-training data**.

These folders should be kept separate from normal imitation-learning folders so I can independently choose which data is used for pre-training and which data teaches the model to imitate actions.

Pre-training recordings can contain visual observations, actions and temporal changes that help the model learn how computer environments behave before imitation learning begins.

## AI Models

The AI Models tab should let me create, duplicate, rename, configure, and delete AI models.

When creating or configuring a model, expose meaningful architecture and capability settings.

Examples include:

### Vision

- Visual encoder
- Input resolution
- Detail/high-resolution crops if useful
- Pretrained versus scratch vision where appropriate
- Fine-tuning options

### Memory / temporal understanding

- Recurrent memory or another appropriate temporal architecture
- Memory size/depth
- Sequence length
- Other meaningful temporal settings

### Action capabilities

I should be able to control which actions the model is capable of producing, including:

- Specific keyboard keys
- Mouse movement
- Mouse buttons
- Scrolling
- Dragging
- Relative pointer movement
- Chords / combinations
- Waiting/timing where required

The model should never learn or emit impossible combinations simply because unrelated action parameters are predicted independently.

Use a coherent structured action representation.

The same action semantics must be used during:

- Training
- Evaluation
- Inference

Avoid train/run representation mismatches.

### Training data

Each AI model should let me select which normal Library folders and/or individual recordings are assigned as its imitation-learning training data.

It should also separately let me select which **Pre-training Library folders** are assigned to that model for pre-training.

The pre-training and imitation-learning datasets should be clearly separated in the UI.

Changing model configuration should clearly indicate which changes require a new model or retraining.

Do not silently make incompatible checkpoints appear usable.

## Model architecture

Treat model architecture as one of the most important parts of the project.

Design it from first principles for **general visual computer interaction**.

Pay particular attention to things like:

- Visual understanding
- Fine spatial understanding for pointer control
- Temporal understanding
- Persistent memory
- Delayed dependencies
- Previous actions
- Current keyboard/mouse state
- Cursor state
- Elapsed time
- Action sequencing
- Generalization
- Training stability
- Closed-loop behavior

The architecture should be capable of learning tasks, playing games, requiring both immediate reactions and memory of earlier observations.

Do not choose an architecture just because it is simpler to implement.

At the same time, do not add architectural complexity that has no measurable benefit.

Where there are meaningful architectural alternatives, run short controlled comparisons using equivalent data and budgets.

For example, temporal memory alternatives may include recurrent state, bounded causal attention, or another justified approach.

### Observation representation

The policy should receive all information that materially improves general computer interaction.

Consider things like:

- Visual observations
- Spatial geometry
- Cursor position
- Keyboard and mouse state
- Previous executed action
- Timing
- Instruction/task conditioning

## Pre-training

Support an optional **pre-training stage before imitation learning**.

Pre-training should help the model learn useful representations of how computer environments/games behave without directly teaching it to imitate the recorded actions.

It should focus on things such as:

- Understanding visual changes over time
- Learning temporal relationships
- Learning how actions affect what happens next
- Learning useful visual and spatial representations
- Learning persistent state and memory where appropriate

Use appropriate self-supervised or action-conditioned learning objectives rather than simply performing imitation learning under a different name.

## Train

The Train tab should let me select an AI model and train it locally.

There should be separate actions for:

- **Pre-train**
- **Train**

**Pre-train** should use the pre-training folders assigned to the selected model.

**Train** should perform the normal imitation-learning training using the model's selected normal Library data.

If a model has already been pretrained, normal training should continue from the pretrained model rather than starting over.

Training must happen locally.

Use **MLX** and Apple Silicon acceleration where appropriate.

The Train tab should clearly show useful live information such as:

- Current step
- Epoch
- Training loss
- Validation loss
- Learning rate
- Gradient information where useful
- Training progress
- Estimated dataset progress
- GPU/Metal utilization where measurable
- CPU utilization
- Memory usage
- Active/cache MLX memory
- Training speed
- Checkpoint status
- Current training state

Training should support:

- Pause
- Resume
- Cancel safely
- Periodic checkpoints
- Best checkpoint tracking
- Latest resumable checkpoint
- Recovery after interruption where practical

Checkpoints must be atomic and versioned.

A failed checkpoint write must never destroy the previous valid checkpoint.

Training and inference must use compatible preprocessing, action representations, temporal state behavior and model semantics.

## Run

The Run tab should let me run a trained model locally.

I should be able to configure things like:

- AI model
- Full desktop / display / window / region
- Task instruction if the architecture uses one
- Allowed keyboard keys
- Mouse permissions
- Scroll permissions
- Relative pointer behavior
- Deterministic versus sampled action selection if applicable
- if human input stops run
- Other useful inference settings

## Settings

Settings should include application-wide configuration such as:

- Recording start/stop keybind
- Run start/stop keybind
- Emergency stop keybind
- Storage location for recordings
- Storage location for models
- Storage location for checkpoints/cache
- Appearance
- Hardware/memory limits
- Cache management
- Permissions
- Other useful application preferences

Changing storage locations must not lose data.

## Apple Silicon performance

Design training and inference specifically for Apple Silicon.

Use:

- MLX
- Metal
- Unified memory
- Efficient CPU/GPU coordination
- Native Apple capture APIs
- Hardware video encoding where appropriate

Avoid unnecessary:

- CPU/GPU copies
- Tensor materialization
- Synchronization
- Readbacks
- Duplicate buffers
- Unbounded queues
- Main-thread work
- Memory spikes

Use bounded pipelines and asynchronous processing where appropriate.

Quality is the priority.

Do not reduce final model or training quality merely to produce attractive throughput numbers.

Optimize performance aggressively where the optimization does not materially harm learning quality or correctness.

## UI and product quality

Build this as a real polished macOS application, not an ML demo wrapped in SwiftUI.

Use native SwiftUI/macOS behavior where appropriate.

The UI should be:

- Clean
- Modern
- Fast
- Consistent
- Easy to understand
- Responsive during recording and training
- Accessible
- Polished

I have put UI references in the **"UI References"** folder in the project directory. Use them as the visual design reference for the application.

## Validation

Validation quality is extremely important.

Do not consider the project finished merely because:

- It compiles
- Unit tests pass
- Training loss decreases
- A synthetic benchmark passes
- A special-purpose test application works

Those are useful checks but are not sufficient.

No single test, training run, benchmark, stability run, architecture comparison, or validation task should exceed 20–30 minutes unless I explicitly request a longer run. Prefer shorter representative runs over exhaustive long-running ones.

Validate the application through the **same normal user-facing workflows** I will use.

Test the actual flow:

Record → Library → AI Model → Pre-train → Train → Run

Use representative real macOS applications and environments in addition to controlled test fixtures.

## Final audit

Before considering the application complete:

1. Use the app as a normal user from beginning to end.
2. Test every major visible control.
3. Find and fix silent failures.
4. Find and fix confusing workflows.
5. Verify default settings actually make sense.
6. Verify recording accuracy.
7. Verify training data alignment.
8. Verify pre-training.
9. Verify model training.
10. Verify inference.
11. Verify input safety.
12. Verify real closed-loop learned behavior.
13. Review CPU, GPU and memory behavior.
14. Review the full source for high-severity defects.
15. Fix the issues you find.
16. Repeat the important end-to-end tests after the fixes.

## Signing and build

Throughout development and for all local builds, use my existing **Apple Development signing identity**.

Do not use or fall back to ad hoc signing.

Once the implementation, validation, bug fixing and polish are complete, produce the final working macOS build.

## GitHub repository

Create a new GitHub repository named **AgentTrainer** for this project.

Use this repository throughout development and keep the project source committed to it as development progresses.

Use `main` as the primary branch and keep the repository up to date with the completed implementation.

## Updated implementation priority

Keep testing targeted and proportionate. Prioritize further implementation over repeated or exhaustive testing; testing remains important.
