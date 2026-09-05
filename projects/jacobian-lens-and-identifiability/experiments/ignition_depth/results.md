# Results: ignition-depth alignment

Design frozen in [`PREREG.md`](PREREG.md) before any data.

**Verdict: NO VERDICT. The test did not run on enough models to issue one, and that is our
design's fault, not the models'.**

## What happened

| model | ignited | ignition reldepth | lens late boundary | gap | null median | beats null? |
|---|---|---|---|---|---|---|
| gemma-2-9b | 9/10 | 0.805 | 0.463 | 0.341 | 0.146 | no |
| qwen3.5-0.8b | 8/10 | 0.978 | 0.652 | 0.326 | 0.326 | no |
| qwen3-1.7b | 5/10 | 0.815 | 0.185 | 0.630 | 0.111 | **excluded, head gate failed** |
| gemma-3-270m | 2/10 | | | | | excluded, too few ignited |
| qwen3-4b | 1/10 | | | | | excluded, too few ignited |
| gpt2-small | 0/10 | | | | | excluded, too few ignited |

Four of six models were excluded, leaving two, below the frozen minimum of three. The frozen
rule therefore issues **no verdict**, and we are not going to quietly pool two models and call it
a null.

**Correction (2026-09-05).** The "lens late boundary" column above came from the own-vocabulary
atlas map (`atlas_out/<slug>.npz`); `PREREG.md` specifies the shared-vocabulary map, which lives in
`atlas_out/shared_maps/`. `analyze.py` now fits the boundary on the shared map (own values kept in
the output; `--own-vocab-boundaries` reproduces the table above). Corrected rows: gemma-2-9b late
boundary 0.415 (17/41), gap 0.390, null median 0.098, p = 1.000, does not beat its null;
qwen3.5-0.8b unchanged at 0.652, gap 0.326, null median 0.326, p = 0.680. Verdict unchanged:
**NO VERDICT** (2 usable models). Superseded output: `results_ownvocab_boundaries.json`.

## Why it failed, plainly

**The prompt set was too hard for half the frozen model set.** We chose ten factual completions
with single-token answers to keep the test cheap, and never checked that small models can
actually answer them. `gpt2-small` gets **zero** of ten right at any depth; `qwen3-4b` gets one.
A question about *where* the answer becomes readable is meaningless in a model that never reads
it out at all.

That is a pre-registration failure of a specific and avoidable kind: we gated on "at least 5 of
10 ignite" without first checking that the models could clear it. A five-minute CPU check on
`gpt2-small` before freezing would have caught it.

**One model was excluded by the head gate doing its job.** `qwen3-1.7b` ignited on 5 prompts but
our output-head readout reproduced its actual final-layer prediction on fewer than 8 of 10, which
the pre-registration says means the readout path is wrong for that model. A first pass of the
analyzer applied only the ignition bar and wrongly counted it; the analyzer now applies both
conditions, as the frozen design requires. Its inclusion would not have changed the direction,
but it would have been a rule violation.

## What the two surviving models suggest, labelled as suggestion

Both put ignition **much deeper** than the fitted late boundary: gemma-2-9b ignites at 0.805
relative depth against a boundary at 0.463, and qwen3.5-0.8b at 0.978 against 0.652. Neither
beats its own random-segmentation null. So the direction of the evidence is **away from**
alignment, consistent with our pre-registered prediction of NO ALIGNMENT and with every other
result in this campaign.

We are not calling that a result. Two models, ten prompts, no verdict under the frozen rule.

## What a working version would need

1. A prompt set validated on the **smallest** model in the set before freezing, so the usability
   gate is reachable.
2. More prompts. Ten was chosen for cost and leaves the median ignition depth very noisy.
3. Models large enough to answer factual questions at all, which means the cheap end of our zoo
   is simply not available for behavioural work.

Whether it is worth building is a separate question. Given that five earlier designs and Test B
all point the same way, we are recording this as attempted and not pursuing it further.

## Cost

Ran on the shared pod alongside the Test B re-run. Marginal cost **under $1**. Receipts carry the
full per-layer rank trajectory and top-10 for every prompt, so a different definition of ignition
can be evaluated without renting a GPU again.
