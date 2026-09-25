# Synthetic sensory media

`CharacterMedium` maps 7-bit ASCII characters to their standard integer IDs
(0..127), then maps those IDs through a fixed,
seeded embedding table to a bounded `float32` pA pattern over an explicit
EXTERNAL `InterfacePort`. Text is an early synthetic sensory medium. It does
not claim to model biological receptors or language comprehension.

The medium does not schedule frames or equate a character with a simulation
tick. Its output is a current vector suitable for `network.step(current)`.
The experiment chooses when to change that current, how many integration
steps to use, and whether to insert silence or overlap characters. Token
position is separate from elapsed physical simulation time.

No tokenizer library, learned text embedding, Unicode handling, output
decoder, or task objective is implied. Non-ASCII input fails explicitly.
The seeded lookup values can later be replaced or fitted by an experiment
without changing neuron kernels or the global pA input contract. Audio and
vision can use their own media later without sharing this text encoding.

Run `python -m examples.character_input` for a small explicit presentation
rule, or `python debug.py --only character_medium` for a smoke check.
