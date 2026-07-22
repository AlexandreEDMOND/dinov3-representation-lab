# DINOv3 Representation Lab

An evaluation and analysis lab for pretrained DINOv3 visual representations. The project benchmarks frozen backbones on ImageNet-1k, visualizes dense patch features, and studies how semantic structure emerges across depth, pooling choices, resolution, and image perturbations.

> This is **not** a DINOv3 pre-training reproduction. It studies publicly released pretrained backbones on a single consumer GPU.

## Project goals

The central question is:

> At which Transformer depth do DINOv3 representations transition from local texture cues to semantic object structure?

The lab answers it through three complementary tracks:

| Track | Objective | Main outputs |
| --- | --- | --- |
| Global evaluation | Measure frozen image embeddings on ImageNet-1k | k-NN, logistic regression, linear probe, Top-1/Top-5, per-class metrics, confusion matrix |
| Dense visualization | Make patch-level representations interpretable | PCA RGB maps, cosine-similarity maps, nearest-neighbour retrieval |
| Comparative analysis | Identify what makes a representation useful and stable | DINOv3 vs DINOv2 vs supervised ResNet-50; layer, pooling, resolution, and robustness studies |

## Scope and constraints

- **Backbones are frozen.** Only lightweight evaluation heads are trained.
- **Primary model:** `facebook/dinov3-vits16-pretrain-lvd1689m` (ViT-S/16).
- **Secondary models:** DINOv3 ViT-B/16, DINOv2 ViT-S/14 or ViT-B/14, and ImageNet-supervised ResNet-50.
- **Hardware target:** one NVIDIA RTX 3090 (24 GB). ViT-S/16 at 224 px is the reference configuration; larger models and 448 px experiments use smaller batches.
- **Dataset:** ImageNet-1k, obtained separately under its terms of use. A small subset validates each pipeline before full-scale runs.

The project excludes DINOv3 pre-training and full backbone fine-tuning: they are outside the compute budget and unnecessary for representation evaluation.

## Experimental protocol

### 1. Frozen-backbone benchmark

For every model, layer, pooling strategy, and input resolution:

1. Extract normalized global image embeddings for ImageNet-1k train and validation splits.
2. Train/evaluate k-NN, multinomial logistic regression, and a linear probe.
3. Report Top-1, Top-5, macro per-class accuracy, and a confusion matrix.
4. Save predictions, metrics, configuration, model revision, and random seed.

The primary comparison is between the `[CLS]` token and mean-pooled patch tokens. Results use a fixed ImageNet transform and the same split for all models.

### 2. Patch-token analysis

For an input image, patch embeddings form a matrix \(X \in \mathbb{R}^{N_{patches} \times D}\). PCA is fitted on a defined sample and the first three components are mapped to RGB:

\[
X_{PCA} = PCA_3(X)
\]

The grid is resized to image resolution to inspect foreground/background separation, object-part coherence, and repeated structures. A selected patch can query cosine similarity against every patch to produce a dense similarity map.

### 3. Original analyses

- **Depth:** compare early, middle, and final Transformer layers.
- **Pooling:** compare `[CLS]` and mean patch pooling.
- **Resolution:** compare 224 px and 448 px using matched evaluation settings.
- **Robustness:** quantify embedding stability under crop, rotation, colour perturbation, and occlusion.
- **Retrieval:** inspect nearest ImageNet neighbours and failure cases.
- **Baselines:** compare DINOv3 with DINOv2 and supervised ResNet-50 under the same protocol.

## Success criteria

The first complete milestone is reached when:

- [ ] ViT-S/16 embeddings are reproducibly extracted for an ImageNet-1k subset and then the full validation set.
- [ ] k-NN, logistic regression, and linear-probe results run from saved embeddings without re-running the backbone.
- [ ] Every result records model ID, revision, transform, resolution, layer, pooling, seed, and dataset split.
- [ ] PCA maps and patch cosine-similarity maps can be generated for a chosen image.
- [ ] At least one controlled comparison answers the depth question with metrics and qualitative examples.
- [ ] DINOv3, DINOv2, and ResNet-50 are compared under an identical evaluation protocol.

## Repository layout (target)

```text
src/dinov3_lab/       feature extraction, evaluation, visualization, utilities
configs/              versioned experiment configurations
scripts/              small command-line entry points
notebooks/            exploratory, non-source-of-truth analysis
tests/                unit and smoke tests
data/                 local, git-ignored dataset metadata and caches
outputs/              git-ignored embeddings, figures, predictions, metrics
docs/                 experiment reports and figures selected for publication
```

## Reproducibility principles

- Configurations are committed and every run writes a machine-readable result file.
- Feature caches are keyed by backbone, model revision, transform, layer, pooling, resolution, split, and embedding dtype.
- Global embeddings may be cached; full ImageNet patch-token tensors are not. Dense features are computed on demand or for a selected subset.
- Qualitative figures identify the exact checkpoint and layer that produced them.
- A smoke-test dataset validates the pipeline before expensive ImageNet runs.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for ordered milestones, acceptance criteria, and the planned experiment matrix.

## Conceptual note

DINOv3 is a self-supervised joint-embedding representation learner. It is related to the line of work that led to JEPA, but it is not itself a predictive JEPA with the defining objective \(P(E(X)) \approx E(Y)\). This repository studies representation quality; it does not claim to implement a JEPA.

## GitHub metadata

**Description**

`Benchmarking and visualizing pretrained DINOv3 image representations: ImageNet probes, patch-token PCA, retrieval, and robustness analysis.`

**Topics**

`dinov3`, `computer-vision`, `self-supervised-learning`, `representation-learning`, `vision-transformer`, `pytorch`, `huggingface`, `imagenet`, `feature-visualization`, `linear-probe`, `knn`, `pca`
