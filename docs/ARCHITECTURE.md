# General computer interaction architecture

This document records implementation contracts and the design to be validated. It is not evidence of completed learning behavior. The implementation ledger tracks what exists.

## Observation and action contract

An observation consists of RGB pixels, global capture geometry, cursor location, physical keyboard/button state, the preceding executed action, elapsed time, and an optional task instruction. Coordinates must retain the transform from global desktop points to encoded pixels, including multiple displays with negative origins and window crops. Every dataset and run uses the same preprocessing function and version.

Record raw input independently from image cadence. A journal stores monotonic host-clock nanoseconds, transition order, repeat flags and modifiers. Observations have a source display timestamp and a later availability timestamp. Dataset alignment uses the latest observation strictly before an event, never the following image. Events at identical observation timestamps cannot prove causality and attach to the preceding observation. Events before the first observation are context only. Trailing transitions are retained.

Static scenes still produce observation ticks, referencing immutable pixels already on disk. No new frame is invented. Capture suspension stops the session rather than reusing stale pixels indefinitely. The capture queue holds only the newest frame for each source; the input journal has a separate bounded queue and stops with a visible failure on overflow.

Actions are a tagged union of key down/up/repeat, pointer movement, relative movement, button down/up, scroll and wait. Raw journals preserve keyboard repeat flags; dataset normalization converts them to explicit repeat targets. A posted hold does not automatically reproduce macOS typing repeats, so the policy learns each repeat's timing. Repeats require an already-held allowed key and never restart the hold watchdog. The optional repeat capability extends the vocabulary and fingerprint; older down/up-only configurations remain compatible until it is enabled.

The grammar masks invalid transitions using input state and configured capabilities. A chord is an ordered sequence of downs followed by releases. A drag is pointer movement while a button is held. The model must not emit unrelated independent booleans that permit contradictory actions. The same grammar runs in dataset validation, evaluation, sampling and execution. Emergency cleanup releases only the keys/buttons owned by the agent. Policy event sources do not suppress local hardware input; human override remains observable. Run displays owned input and retains its final released state. Pointer occlusion checks ignore the system cursor window layer, which cannot receive clicks, while retaining checks for covering panels and other windows.

## Policy and comparisons

Preserve spatial information through a patch-level vision encoder and position embeddings. A global view captures layout; a cursor-centered crop preserves local target detail. A spatial pointer head should score locations using visual tokens, then refine coordinates, rather than regress positions from a globally averaged feature vector.

Fuse vision with projected input state, previous action, geometry and time. Compare recurrent gated memory with bounded causal attention under equal data, parameter budgets and step budgets. Both candidates must support the identical public observation/action contract. Recurrent carry must be explicitly reset at episode boundaries, with burn-in and truncation handled consistently. Attention must use causal masking and the same bounded history in training and inference. No architecture choice is justified solely by implementation convenience or training loss.

Use event-level targets so several transitions between images remain learnable. Teacher forcing supplies preceding actions during training; inference conditions on actions actually executed. Mask action family, key/button choice and action-dependent arguments coherently. Timing is part of the target, including wait intervals. Track event likelihood, transition validity, pointer error, timing error and actual closed-loop completion.

## Pre-training

Pre-training uses only separately assigned pre-training data. Predict future visual representations or spatial changes conditioned on preceding actions; include a representation-preserving reconstruction/variance objective to prevent collapse. The action decoder is not trained to copy recorded behavior in this stage. Imitation initializes the shared encoder and temporal weights from the compatible pretrained checkpoint. Hold-out sets split by recording, not adjacent frames, to avoid temporal leakage.

## Checkpoints and resource ownership

A resumable checkpoint contains model weights, optimizer state, recurrent carry, stage, step/epoch/cursor, deterministic shuffle seed, data selection/split identity and preprocessing/action/configuration fingerprints. The current optimizer loop has no stochastic augmentation or dropout; adding either requires preserving its RNG state as well. Write a complete new checkpoint directory, synchronize it, then atomically replace the latest/best pointer. Never overwrite the previous valid checkpoint in place. A configuration mismatch makes the checkpoint unusable until the user restores the compatible configuration or retrains.

Library previews build cancellable disk time/offset indexes and decode only the selected frame and a bounded input interval. Temporary indexes do not replace source journals. Reviewed interrupted or failed recordings may use a valid complete prefix; malformed complete rows and missing images still fail validation. Checkpoint cleanup preserves every best/latest pointer, every model reference and three recent copies per stage, and fails closed for a model with malformed metadata.

MLX arrays and model/optimizer state belong to one dedicated execution context. SwiftUI receives small immutable metric snapshots. Batch decoding currently uses two image-cache slots per recording lane; it does not cache the corpus or asynchronously prefetch. Imitation omits future-frame decoding and pixel targets used only by pre-training. Profile allocation, active/cache MLX memory, physical app footprint, CPU and throughput before introducing further pipeline complexity. The app footprint uses TASK_VM_INFO.phys_footprint; resident bytes alone understate unified-memory cost. Report GPU utilization only if actually measured; Metal availability does not equal utilization.

## Validation gates

Core tests establish event survival, alignment, grammar and recovery. Short controlled experiments compare architectures and pre-training. Real workflows must then prove Record → Library → Model → Pre-train → Train → Run through the public UI in representative macOS applications. Report success rates over repeated held-out conditions, including delayed dependencies, cursor targeting, interruption, human override and emergency release. Every individual experiment/test/validation task is bounded to at most 20–30 minutes, with shorter representative runs preferred.
