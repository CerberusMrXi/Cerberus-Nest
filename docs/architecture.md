# Architecture

## Baseline Modules

The v0.1 prototype currently combines:

1. `CerberusNest2D` — simulated environment and internal drives.
2. `VisionEncoder` — CNN representation of raw pixel observations.
3. `QNetwork` — action-value decision model.
4. `RNDCuriosity` — intrinsic novelty signal.
5. `EpisodicMemory` — prioritized experience storage.
6. `SemanticMemory` — online feature clustering.
7. `ProceduralMemory` — successful action sequences.
8. `AttentionWorkspace` — competing-signal focus selection.
9. `SelfModel` — prediction of internal state and movement.
10. `WorldModel` — prediction of next visual representation.
11. `LanguageAssociator` — early sound/object association experiment.
12. `DevelopmentalTimeline` — staged capability availability.

## Key Research Constraint

These modules are computational mechanisms. Their existence does not establish consciousness.

Future work should replace heuristic measurements with controlled experiments and measurable behavioral tasks.
