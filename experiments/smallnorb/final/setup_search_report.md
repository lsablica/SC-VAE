# smallNORB validation-only setup search

No official test image or test metric was accessed during setup selection.

| Candidate | Numeric health | Decision | Gap NLL | Observed NLL | KL | Median rho | Pose gap error | Distance Spearman | Cross-gap Spearman | Interpolation MSE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_b050_w20_s020 | True | rejected: gap +15.0% | 195.5134 | 177.2823 | 55.436 | 0.839818 | 150.66 | 0.004 | -0.191 | 0.009412 |
| deep_cnn_b025_w30_s020 | True | rejected: gap +56.2% | 265.6980 | 230.1449 | 66.674 | 0.889403 | 152.49 | -0.007 | -0.236 | 0.009654 |
| lower_beta_slow_lr_b025_w30_s020 | True | accepted | 170.0825 | 154.3874 | 73.953 | 0.909832 | 153.37 | 0.004 | -0.170 | 0.009836 |
| slower_lr_b050_w30_s020 | True | rejected: gap +2.3% | 173.9949 | 156.6071 | 58.558 | 0.858127 | 153.22 | 0.006 | -0.172 | 0.009759 |
| stronger_beta_b100_w20_s020 | True | rejected: gap +31.9% | 224.3180 | 198.9883 | 40.451 | 0.756481 | 151.79 | -0.016 | -0.220 | 0.009341 |

Numeric health requires finite completion, KL above 2 nats, median rho below 0.9995, non-catastrophic gap reconstruction, improvement from the start of KL warmup, and no more than five percent regression from the best post-warmup gap NLL in the final five-epoch median.

The setup is chosen lexicographically from numerically healthy runs using gap reconstruction, observed reconstruction among candidates within two percent of the best gap result, pose geometry, then schedule simplicity.

No candidate made absolute held-out-gap pose linearly recoverable. This limitation is retained explicitly; the Stage 2 comparative geometry and interpolation gate determines whether a full sweep is justified.
