# XTraj

Reference implementation for **XTraj**, a heat-aware multimodal generative
framework for urban GPS trajectories. The code is split into three coupled
stages:

1. **Road encoder** (`models/gat_multimodal_encoder.py`) — a multimodal
   GATv2 encoder that fuses road geometry, POI features and (optionally)
   node-level heat features into a per-segment embedding.
2. **Trajectory autoregressive transformer**
   (`models/traj_ar_transformer.py`) — generates a road-segment sequence
   conditioned on origin/destination embeddings under graph-reachability
   constraints.
3. **Road-progress decoder** (`models/road_progress_decoder.py`) — turns a
   road-segment sequence into a sequence of (segment, progress, time) tuples
   that are interpolated to GPS coordinates.

## Repository layout

```
XTraj/
├── experiments/                # Training and generation entry points
│   ├── train_encoder_linkpred.py             # Stage 1: encoder pretraining
│   ├── export_road_embeddings.py             # Stage 1: dump embeddings to disk
│   ├── train_traj_ar_transformer.py          # Stage 2: AR transformer training
│   ├── train_road_progress_decoder_ddp.py    # Stage 3: road-progress decoder (DDP)
│   ├── sample_and_generate_trajectories.py   # End-to-end generation pipeline
│   ├── batch_generate_trajectories.py        # Stage 2 batched generation
│   ├── batch_generate_gps_from_road_progress.py       # Stage 3 batched generation
│   ├── batch_generate_gps_from_road_progress_full.py  # variant for full-time models
│   ├── build_road_progress_dataset.py        # Dataset builder for stage 3
│   ├── reconstruct_gps_from_road_progress.py # GPS reconstruction utility
│   └── build_reachability_table.py           # Precompute graph-reachability table
├── models/                     # Network definitions
├── utils/                      # Datasets, graph utilities, geometry helpers
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# torch-geometric needs matching torch/CUDA wheels — see
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

## Expected data layout

All entry points read from a city-keyed dataset directory and a road-network
CSV. The expected schema:

* `data-root/<city>/*.npy` — per-day trajectory dumps. Each entry is a
  4-tuple `[meta, road_ids_unique, s_seq, t_seq]` where `s_seq` is the GPS
  trajectory and `t_seq` is the per-step duration in seconds.
* `<city>_road_network/edges.csv` — road segments with at least the columns
  `fid`, `u`, `v`, `wkt`, plus optional POI / category fields.

Replace the placeholder `/path/to/...` paths in the example commands below
with your local locations.

## End-to-end pipeline

### 1. Pretrain the road encoder

```bash
python experiments/train_encoder_linkpred.py \
    --city chengdu \
    --data-root /path/to/datasets \
    --edges /path/to/chengdu_road_network/edges.csv \
    --save ./checkpoints/encoder
```

### 2. Export node embeddings

```bash
python experiments/export_road_embeddings.py \
    --city chengdu \
    --edges /path/to/chengdu_road_network/edges.csv \
    --ckpt ./checkpoints/encoder/<your-ckpt>.pt \
    --out ./checkpoints/embedding/chengdu_embedding.pt
```

### 3. Build the reachability table (once per city)

```bash
python experiments/build_reachability_table.py \
    --edges /path/to/chengdu_road_network/edges.csv \
    --out ./checkpoints/chengdu_reachability.npz
```

### 4. Train the AR transformer (segment-level generator)

```bash
python experiments/train_traj_ar_transformer.py \
    --city chengdu \
    --data-root /path/to/datasets \
    --edges /path/to/chengdu_road_network/edges.csv \
    --emb ./checkpoints/embedding/chengdu_embedding.pt \
    --save ./checkpoints/segment_level_generator
```

### 5. Train the road-progress decoder (point-level generator, DDP)

```bash
torchrun --nproc_per_node=4 \
    experiments/train_road_progress_decoder_ddp.py \
    --city chengdu \
    --data-root /path/to/datasets \
    --edges /path/to/chengdu_road_network/edges.csv \
    --emb ./checkpoints/embedding/chengdu_embedding.pt \
    --save ./checkpoints/point_level_generator
```

### 6. End-to-end generation

```bash
python experiments/sample_and_generate_trajectories.py \
    --input-dir /path/to/datasets/chengdu_7days \
    --output-dir ./sampled_generation/chengdu \
    --num-samples 1000 \
    --segment-ckpt ./checkpoints/segment_level_generator/<your-ckpt>.pt \
    --point-ckpt   ./checkpoints/point_level_generator/<your-ckpt>.pt \
    --emb          ./checkpoints/embedding/chengdu_embedding.pt \
    --city chengdu \
    --data-root /path/to/datasets \
    --edges /path/to/chengdu_road_network/edges.csv \
    --gpu 0
```

The pipeline writes both the resampled ground-truth trajectories
(`sampled_original.npy`) and the generated trajectories
(`generated_trajectories.npy`) to `--output-dir`.

## Notes

* All scripts expose `--help` for the full list of options.
* GPU is required for training and recommended for generation.
* The code is released for review purposes; author and affiliation
  identifiers have been stripped.
