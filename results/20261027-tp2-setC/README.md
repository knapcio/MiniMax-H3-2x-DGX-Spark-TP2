# Set C: final configuration (uint8 output path, VAE split over both GPUs, cache freed before decode), boot B

Head spark-06, worker spark-05, seed 1101, 20 steps, 1344x768, bench prompt. Memory minima are MemAvailable sampled
every 2 s during the request.

| Case | Elapsed | Peak (X-Peak-Memory-MB) | Min free head / worker | Phases (encode / denoise / decode) | framemd5 | frames sha256 |
|---|---|---|---|---|---|---|
| 10 s T2VA | 1690.5 s | 84.2 GB | 26.1 / 29.6 GB | 0.2 / 1602.6 / 84.2 s | 6fe8c1b1 | f48347ae… |
| 15 s FL2VA from `source-4k.jpg` (first frame) | 3166.3 s | 84.4 GB | 24.1 / 26.5 GB | 0.9 / 3030.9 / 125.9 s | b7237007 | 79da6364… |
