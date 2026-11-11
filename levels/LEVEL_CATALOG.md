# Puzzler V2: 25-Level Benchmark

The level IDs are deliberately grouped into five categories. Each category has
five levels that increase in object count, geometric constraint, persistence,
or causal depth. The JSON remains compatible with `BrainItOnGeneralEnv`: no new
level, object, goal, joint, or draw-rule fields were introduced.

## 1–5: Trajectory and directional control

| ID | Level | Primary reasoning challenge |
|---:|---|---|
| 1 | Redirect the falling ball right | Convert vertical fall into rightward motion with one ramp. |
| 2 | Redirect the falling ball left | Mirrored directional generalisation. |
| 3 | Guide the ball around the blocker | Compose two deflections around fixed geometry. |
| 4 | Bank the moving ball back to the target | Reverse an existing horizontal trajectory. |
| 5 | Route two balls to separate targets | Coordinate two trajectories without confusing object–goal identity. |

## 6–10: Target placement and containment

| ID | Level | Primary reasoning challenge |
|---:|---|---|
| 6 | Place the ball in the wide bay | Enter a broad bounded pocket. |
| 7 | Place the cube in the narrow bay | Account for cube extent, rotation, and a narrow opening. |
| 8 | Land the ball in the elevated tray | Reach a target above the ground rather than merely falling. |
| 9 | Thread the ball through the slot | Control overshoot through a 64-pixel opening. |
| 10 | Sort both balls into centre bays | Place two identified objects into distinct bounded regions. |

## 11–15: Stability and settling

| ID | Level | Primary reasoning challenge |
|---:|---|---|
| 11 | Catch and settle the falling ball | Maintain containment for 120 frames at low velocity. |
| 12 | Stabilise the sliding cube | Construct a shelf and arrest lateral motion for 150 frames. |
| 13 | Dampen the pendulum | Use a three-surface trap to dissipate constrained motion. |
| 14 | Arrest the fast ball | Catch a high-speed object and keep it settled in the end zone. |
| 15 | Settle both falling balls | Maintain two independent stable configurations simultaneously. |

## 16–20: Support and structural construction

| ID | Level | Primary reasoning challenge |
|---:|---|---|
| 16 | Bridge the single gap | Create a load-bearing span between two banks. |
| 17 | Build a supported catching shelf | Connect two columns and catch a falling cube. |
| 18 | Support two loads over the pit | Support multiple loads on one constructed span. |
| 19 | Build a containment barrier | Construct a barrier that absorbs horizontal momentum. |
| 20 | Complete the two-span bridge | Connect two gaps via a central pier while cargo is moving. |

## 21–25: Release and causal manipulation

These levels use dynamic strokes. Their draw regions prevent direct target-side
construction and encourage impact or indirect causal mechanisms.

| ID | Level | Primary reasoning challenge |
|---:|---|---|
| 21 | Knock the ball off its pedestal | Use asymmetric impact to remove effective support. |
| 22 | Tip the cube into the bin | Cause rotation and release from a precarious shelf. |
| 23 | Strike the lever | Transfer force through a pivoted lever to release a payload. |
| 24 | Trigger the domino chain | Initiate multi-object momentum transfer. |
| 25 | Trigger the pendulum striker | Activate an intermediate constrained object to hit the payload. |

## Validation

Run the deterministic regression check after editing the level set:

```bash
python script/validate_level_set.py --render-check
```

It verifies the unchanged JSON contract, exact category counts, all object/goal
references, environment reset and rendering, a no-draw baseline, and at least
one legal reference solution per level.
