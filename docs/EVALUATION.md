# Evaluation protocol

Evaluation uses fixed model weights and reference motion durations. Each test repeat changes the sampling seed.

## HumanML3D

Use official test data and the standard frozen Guo motion/text matching evaluator with its own preprocessing and normalization. The protocol uses 4,640 samples per repeat, candidate groups of 32 in the evaluator loader order, AR guidance 3.5, Flow guidance 3.0, 32 Euler steps, temperature 1, top-p 0.95, and test seeds 7–26. Each repeat produces one FID, R@1–3, and matching-distance value. Report the mean and `1.96 × sample_std / sqrt(20)` interval half-width across the 20 repeats.

The generator returns normalized Guo263 features. Restore the correct feature units and follow the official evaluator's length and batch rules.

## Ω-MotionVerse

Use all 97,642 v5 test pairs with reference durations. Generate UMR499 at 50 Hz, recover the canonical SOMA motion with the same initial-pose convention, resample to 30 Hz, and use the independent V2 evaluator trained only on the Ω training split. The evaluator uses DistilBERT text features and a 30-joint SOMA motion representation; it is distinct from the frozen tokenizer TMR teacher and the generation text conditioner.

The main setting uses sampling seed 7, AR guidance 3.0, Flow guidance 1.5, and 32 Euler steps. FID and matching distance use all samples. Retrieval uses one fixed seed-7 permutation and disjoint groups of 32, so 97,632 samples contribute to retrieval and the final ten remain included in FID and matching distance. Optional 256-way retrieval uses 97,536 samples. Ω retrieval ranks evaluator inner products; HumanML3D ranks Euclidean distance. These feature spaces and scores are not interchangeable.

## Scoring

Extract motion and text embeddings using the dataset evaluator, then score the aligned arrays. For Ω-MotionVerse, use `semoco.evaluation.soma77_to_tmr186` and `semoco.evaluation.OmegaV2Evaluator.load("checkpoints/omega/evaluator_v2.pt", "checkpoints/omega/norm_stats.json")` on recovered 50 Hz SOMA77 joints and test captions.

One NPZ per repeat contains `generated`, `reference`, and `text`, all `[N,D]`, with consistent sample order. For HumanML3D, preserve the evaluator's groups of 32. For Ω, pass the complete arrays; the scorer performs the specified permutation for retrieval only.

```bash
semoco-score --dataset humanml3d --embeddings runs/test/seed7.npz runs/test/seed8.npz
semoco-score --dataset omega --embeddings runs/omega/test.npz --seed 7
```

Report HumanML3D results over 20 repeats. The scorer computes FID over all
motion embeddings, matching distance over all paired text/motion embeddings,
and retrieval over complete candidate groups. Repeat statistics use equal weights.
