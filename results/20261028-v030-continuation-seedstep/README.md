# 30 s continuation with a different seed per window (`H3_TP2_WINDOW_SEED_STEP=1`), 960x576, single-take prompt

Upstream re-creates the noise generators from the request seed for every window, so equal-shaped windows start from
the same noise tensor. Test: seed + k for window k (1101, 1102, 1103); otherwise the single-take prompt request.

| | same seed every window | seed + k |
|---|---|---|
| Wall clock | 3663.6 s (offset-0 run, same hour) | 3598.2 s |
| Scene cuts | none | none |
| Seams 1 / 2 (mean before) | 4.51 (4.75) / 1.42 (1.61) | 4.03 (4.75) / 1.55 (1.38) |
| Apparent motion, window 1 -> 3 | 6.1 -> 2.4 -> 1.x | 6.1 -> 2.5 -> 1.x |
| Audio mean per 10 s | -21.2 / -28.3 / -37.3 dB | -21.2 / -29.1 / -33.2 dB |

Rejected: the noise reuse is not what drives the decay either. Together with the local-time test this leaves the
latent-tail conditioning itself (a quiet, slow tail begets a quieter, slower window) as the mechanism, and the two
measured remedies are a driving soundtrack (removes both decays) and per-window prompts (removes the motion decay).
