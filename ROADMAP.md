# Roadmap

This roadmap builds the smallest reproducible system that can answer the project question. Each milestone has a concrete acceptance check; a later phase does not start until the previous one works on a small dataset.

## Phase 0 — Project foundation

**Goal:** establish a repeatable local environment and a stable experiment contract.

- Create the Python package, `uv` environment, and dependency lockfile.
- Add configuration files for paths, device, seed, model ID, layer, pooling, resolution, and dataset split.
- Define a standard run directory containing configuration, logs, metrics, predictions, and figures.
- Add `.gitignore` rules for ImageNet, Hugging Face caches, embeddings, and generated outputs.

**Done when:** a smoke command writes its resolved configuration and runs on CPU or GPU without requiring ImageNet.

## Phase 1 — Data and backbone access

**Goal:** reliably load the official pretrained DINOv3 ViT-S/16 model and ImageNet-compatible samples.

- Load `facebook/dinov3-vits16-pretrain-lvd1689m` using a pinned revision.
- Implement the official evaluation image transform for the selected checkpoint family.
- Validate ImageNet train/validation folder structure and class-index mapping.
- Add a deterministic small-subset mode for local iteration.

**Done when:** ten images produce global embeddings and final-layer patch tokens with recorded shapes and no gradients.

**Status:** acceptance check passed with the official gated Hugging Face checkpoint on
an Imagenette small subset: ten images produced global embeddings and final patch tokens
without gradients. The earlier locally supplied checkpoint remains development-only.

## Phase 2 — Feature extraction and cache

**Goal:** create reusable frozen-backbone embeddings without storing unnecessary dense tensors.

- Extract `[CLS]` and mean-patch global embeddings in batches using inference mode.
- L2-normalize the representation required by k-NN and retrieval.
- Write cache metadata keyed by model revision, transform, layer, pooling, resolution, split, and dtype.
- Support resumable extraction and validate cache completeness.

**Done when:** train and validation embeddings for the small subset can be extracted twice with identical metadata and reused without executing the backbone.

**Status:** acceptance check passed with the official gated Hugging Face checkpoint on
a 20-image-per-split Imagenette subset. The second run reused every complete cache
without loading the backbone. The earlier locally supplied checkpoint remains
development-only.

## Phase 3 — ImageNet frozen-feature benchmark

**Goal:** establish a trustworthy ViT-S/16 reference result.

- Implement k-NN evaluation, including chunked similarity search suitable for a 24 GB GPU.
- Implement multinomial logistic regression and a PyTorch linear probe.
- Report Top-1, Top-5, macro per-class accuracy, confusion matrix, and runtime.
- Save predictions and class names so individual errors can be inspected.
- Scale from the subset to ImageNet-1k after subset checks pass.

**Done when:** all three methods run from cached global features, and one report compares `[CLS]` with mean patch pooling on the full validation set.

## Phase 4 — Dense patch-token visualization

**Goal:** turn local features into interpretable evidence.

- Extract a selected layer's patch-token grid for one image.
- Fit a documented three-component PCA on a defined sample of patch tokens.
- Render PCA components as a normalized RGB map aligned with the input image.
- Render cosine-similarity heatmaps from a selected query patch.
- Save side-by-side input, PCA, similarity, and metadata figures.

**Done when:** the same command produces deterministic PCA and similarity figures for a supplied image and chosen layer.

## Phase 5 — Controlled representation analyses

**Goal:** answer the central question with matched experiments rather than isolated demos.

| Experiment | Controlled variable | Output |
| --- | --- | --- |
| Depth | early / middle / final layer | probe scores, PCA maps, similarity maps |
| Pooling | `[CLS]` / mean patches | probe-score table |
| Resolution | 224 / 448 px | accuracy, runtime, VRAM, qualitative maps |
| Robustness | crop / rotation / colour / occlusion | cosine similarity and retrieval-consistency curves |
| Retrieval | backbone and layer | nearest-neighbour grids and failure taxonomy |

**Done when:** one report states whether evidence supports a transition from local to semantic representations across depth, with quantitative results and qualitative counterexamples.

## Phase 6 — Baselines and final report

**Goal:** make every DINOv3 claim comparative and reproducible.

- Run the identical protocol for DINOv2 and ImageNet-supervised ResNet-50.
- Keep input transforms, splits, evaluation code, and random seeds explicit.
- Summarize results in tables and figures; distinguish observed facts from interpretation.
- Publish a concise report covering hardware, software versions, model revisions, protocol, results, limitations, and compute cost.

**Done when:** a reader can reproduce the headline table and regenerate every published figure from committed configurations plus local data access.

## Compute policy

- Use ViT-S/16 at 224 px as the default for development and the primary benchmark.
- Run every new feature first on the deterministic subset.
- Use ViT-B/16 and 448 px only for selected, justified comparisons.
- Generate patch tokens on demand or on a named subset; never cache dense tokens for the full ImageNet training set by default.
- Do not add pre-training or full backbone fine-tuning to this roadmap.
