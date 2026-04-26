# RT-FSAS Evaluation Results

This file summarizes ablation-style metrics for the semester submission.

## Run Command

Use the command below from project root (adapt model dimensions to your checkpoint):

```bash
python -m src.evaluation.ablation \
  --graphs_path graphs/la_liga_2015_16_full_shards \
  --checkpoint_path checkpoints/gnn_weighted.pt \
  --index_dir index \
  --output_path report/ablation_metrics.json \
  --sample_size 50 \
  --k 5 \
  --hidden_dim 96 \
  --embed_dim 128 \
  --num_classes 11
```

## Metrics Source

Generated JSON: `report/ablation_metrics.json`

## Results Snapshot

Fill from generated JSON:

- Samples evaluated: `50`
- Retrieval Top-1 next-event match rate: `0.78`
- Retrieval Top-k contains-event rate: `1.0`
- Average Top-1 similarity: `0.9999999797344208`
- Full pipeline average advice words: `59.36`
- Dumb baseline average advice words: `59`
- Full pipeline average specificity score: `6.1`
- Dumb baseline average specificity score: `6`

## Interpretation

- The retrieval rates indicate how often nearest-neighbor tactical contexts agree with observed next events.
- Advice length and specificity compare grounded (retrieval + q_delta) guidance vs baseline unguided guidance.
- This provides a measurable proxy for whether RAG-style context improves practical coaching output quality.
