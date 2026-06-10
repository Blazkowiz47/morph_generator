# Morph Generator

This repository contains the morph-generation code used for face morphing
experiments. It includes utilities for FFHQ-style face alignment, face
recognition feature extraction/pair selection, gender annotation, and multiple
morph generators.

The code is organised so that a reviewer can start from aligned face images and
a CSV file of image pairs, then generate morphs with one of the supported
methods.

## Supported Methods

| Method name | Code path | Main environment | Notes |
| --- | --- | --- | --- |
| `lma` | `morphs/lma/` | `morph_env.yaml` | Landmark-based alpha blending. Requires the dlib 68-point landmark model. |
| `ubo` | `morphs/lmaubo/` | `morph_env.yaml` | Wrapper around the UBO landmark morphing executable in `morphs/lmaubo/`. |
| `mipgan1` | `morphs/mipgan1/` | `morph_env.yaml` | MIPGAN-I based morphing. Requires TensorFlow 1.x-compatible environment and model weights. |
| `mipgan2` | `morphs/mipgan2/` | `morph_env.yaml` | MIPGAN-II based morphing. Requires TensorFlow 1.x-compatible environment and model weights. |
| `pipe` | `morphs/pipe/` | `pipe_env.yaml` | Morph-PIPE style diffusion autoencoder morphing. |
| `mordiff` | `morphs/mordiff/` | `pipe_env.yaml` | Diffusion autoencoder morphing. The current wrapper limits runs to the first 50 pairs for testing. |
| `greedy` | `morphs/greedy/` | `pipe_env.yaml` | Greedy-DiM style morphing with identity loss. |
| `ladimo` | `morphs/ladimo/` | `pipe_env.yaml` | Latent diffusion morphing conditioned on MagFace features. |

## Repository Layout

```text
.
├── ffhq_align_images.py        # FFHQ-style alignment helper
├── ffhq_dataset/               # Landmark detection/alignment utilities
├── frs/                        # FRS wrappers used for feature extraction and pair selection
├── frs_feature_extractor.py    # Feature extraction helper
├── gender_mapper.py            # Interactive gender mapping helper
├── get_topk_frs_match.py       # Select top-k FRS-similar morphing pairs
├── morph_images.py             # Shared morphing driver utilities
├── morph_partial.py            # Direct runner for MIPGAN-I/II chunks
├── morphs/                     # Individual morph-generation methods
├── morph_env.yaml              # Environment for landmark/MIPGAN methods
└── pipe_env.yaml               # Environment for diffusion-based methods
```

## Installation

Create one or both conda environments depending on the morphing methods you want
to run.

```bash
conda env create -f morph_env.yaml
conda activate gan
```

For diffusion-based methods:

```bash
conda env create -f pipe_env.yaml
conda activate pipe
```

The environments were exported from the original experimental setup. GPU-based
methods were run on Linux with CUDA-enabled NVIDIA GPUs. If you recreate the
environment on a different CUDA/PyTorch/TensorFlow stack, keep the model-specific
versions consistent with the files in the environment YAMLs.

## Model Files

Large model files are expected under `models/`. They are ignored by git, so they
must be downloaded or copied separately before running the corresponding method.
All model files required by the morphing methods are provided as a single
archive:

```text
Google Drive: https://drive.google.com/drive/folders/175ZdnHvW5gPcTUYpKSj3f0FsyMHJq6e1?usp=sharing
```

The folder is access-restricted. Please use Google Drive's **Request access**
option on the link above; access will be granted for reproducibility review and
research use.

Download both files from the folder:

```text
models.zip
models.zip.sha256
```

Then extract the archive in the repository root:

```bash
unzip models.zip
```

This should create the top-level `models/` directory:

```text
morph_generator/
├── models/
├── morphs/
├── morph_images.py
└── ...
```

Optionally verify the downloaded archive before extracting:

```bash
shasum -a 256 -c models.zip.sha256
```

The code expects the paths under `models/` exactly as listed below unless the
corresponding method wrapper is modified.

Expected files include:

```text
models/temp/shape_predictor_68_face_landmarks.dat
models/frs/configs/config_ms1m_100.yaml
models/frs/config_ms1m_100_1006k/best-m-1006000.*
models/StyleGAN_finetuned_ICAO.pkl
models/stylegan2_finetuned_ICAO.pkl
models/finetuned_resnet.h5
models/resnet_18_20191231.h5
models/checkpoints/ffhq256_autoenc/last.ckpt
models/ffhq/last.ckpt
models/glint360k_cosface_r100_fp16_0.1/backbone.pth
models/magface/magface_epoch_00025.pth
models/logs/2023-11-15T10-04-11_ffhq-ldm-vq-f8/checkpoints/epoch=000096.ckpt
```

The dlib landmark predictor can be obtained from:

```text
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

After downloading it, extract the file and place it at:

```text
models/temp/shape_predictor_68_face_landmarks.dat
```

For reproducibility, document the source URL, checksum, and license/access terms
for every external checkpoint used in your experiment.

## Input Format

The morphing drivers expect:

1. A directory containing aligned face images.
2. A CSV file listing image pairs.
3. An output directory.

The pair CSV has no header. Each line contains two image filenames relative to
the source image directory:

```csv
subject001_01.png,subject002_01.png
subject003_01.png,subject004_01.png
```

For most methods, generated morphs are saved as:

```text
<output_dir>/<image1-stem>-vs-<image2-stem>.png
```

Example:

```text
outputs/lma/subject001_01-vs-subject002_01.png
```

## Quick Start

From the repository root, create a small pair CSV:

```bash
mkdir -p data/example
printf "subject001_01.png,subject002_01.png\n" > data/example/pairs.csv
```

Place the corresponding aligned images in:

```text
data/example/aligned/subject001_01.png
data/example/aligned/subject002_01.png
```

Run a morphing method through the shared driver:

```bash
conda activate gan
python - <<'PY'
from morph_images import perform_morphing

perform_morphing(
    morph="lma",
    src_dir="data/example/aligned",
    morph_list_csv="data/example/pairs.csv",
    output_dir="outputs/example/lma",
)
PY
```

For diffusion-based methods, activate the `pipe` environment and change the
method name:

```bash
conda activate pipe
python - <<'PY'
from morph_images import perform_morphing

perform_morphing(
    morph="greedy",
    src_dir="data/example/aligned",
    morph_list_csv="data/example/pairs.csv",
    output_dir="outputs/example/greedy",
)
PY
```

Supported method names are:

```text
lma, ubo, mipgan1, mipgan2, pipe, mordiff, greedy, ladimo
```

## Running MIPGAN-I/II in Chunks

MIPGAN-I and MIPGAN-II can also be run directly with `morph_partial.py`.

```bash
conda activate gan
python morph_partial.py 0 data/example/aligned data/example/pairs.csv outputs/example/mipgan2 mipgan2
```

Arguments:

```text
python morph_partial.py <process_num> <src_dir> <pair_csv> <output_dir> <morph>
```

where `<morph>` is either `mipgan1` or `mipgan2`.

## FFHQ Alignment

`ffhq_align_images.py` aligns raw face images using dlib landmarks and the FFHQ
alignment routine. The helper currently expects a dataset structure like:

```text
<dataset_root>/
└── digital/
    └── bonafide/
        └── raw/
            └── <subject_id>/
                └── image.png
```

Aligned outputs are written under:

```text
<dataset_root>/digital/aligned/<subject_id>/image.png
```

Example:

```bash
conda activate gan
python - <<'PY'
from ffhq_align_images import driver

driver(
    CLEAN_DIR="data/my_dataset",
    printers=["digital"],
    num_process=2,
)
PY
```

Make sure the dlib landmark predictor exists at:

```text
models/temp/shape_predictor_68_face_landmarks.dat
```

## Selecting FRS-Similar Pairs

`get_topk_frs_match.py` can generate a new pair CSV by selecting the top-k most
similar identities according to an FRS backbone.

```bash
conda activate gan
python get_topk_frs_match.py \
  --rdir data/example/aligned \
  --input_csv data/example/seed_pairs.csv \
  --output_csv data/example/top_5_pairs.csv \
  --backbone arcface \
  --top_k 5
```

The available FRS backbones are provided through `frs/get_frs_initializers` and
include ArcFace, AdaFace, and MagFace wrappers.

## Gender Mapping

`gender_mapper.py` is an interactive helper for assigning a gender label to each
subject. It displays sample images and writes a JSON mapping. This is useful
when experiments require same-gender pair generation or reporting.

Example usage from Python:

```bash
python - <<'PY'
from gender_mapper import splitted_datasets

splitted_datasets(
    rdir="data/my_dataset/digital/aligned/test",
    oname="data/my_dataset/test_gender.json",
)
PY
```

The tool expects a terminal capable of displaying images with `wezterm imgcat`.
If that is unavailable, either adapt the display command or prepare the JSON
mapping manually.

## Method-Specific Notes

- `lma` is the easiest method to run and is useful as a smoke test because it
  only needs aligned images and the dlib landmark predictor.
- `ubo` depends on the executable files inside `morphs/lmaubo/`; run it on a
  system where those binaries are supported.
- `mipgan1` and `mipgan2` require TensorFlow 1.x APIs and the corresponding
  StyleGAN/FRS/resnet checkpoints listed above.
- `pipe`, `mordiff`, and `greedy` require CUDA and diffusion autoencoder
  checkpoints.
- `ladimo` uses a latent diffusion checkpoint plus MagFace features. The current
  code sets `CUDA_VISIBLE_DEVICES=0` in `morphs/ladimo/morph_images.py`.
- `mordiff` currently slices the pair list to the first 50 pairs in
  `morphs/mordiff/morph_loop.py`. Remove that line for full-dataset generation.

## Reproducibility Checklist

For an RRPR-style release, provide the following alongside this code:

- Official links and access instructions for the original face datasets.
- Exact train/test/protocol split files used to create the pair CSVs.
- Pair CSVs used for each morphing method.
- Download links, checksums, and licenses for all external checkpoints.
- The conda environment file used for each method.
- The command used to run each method.
- Hardware details, including GPU model, CUDA version, RAM, and approximate
  runtime per dataset/method.
- A small public/sample dataset or a tiny synthetic fixture for smoke testing
  the pipeline without requiring full dataset access.

## Acknowledgements and Third-Party References

This repository combines local experiment code with third-party algorithms,
pretrained models, and utilities. Please verify licenses for each artifact before
redistribution or commercial use. The table below records the provenance that is
known from the repository and public project pages; entries marked `TODO` should
be completed before a final archival release.

### Morph Generation Methods

| Component | Used for | Local files/checkpoints | Reference / source to acknowledge | License / note |
| --- | --- | --- | --- | --- |
| Landmark morphing (`lma`) | Landmark-based alpha blending with Delaunay triangulation | `morphs/lma/` | TODO: add original source if this implementation was adapted from an external repository/tutorial. | Uses dlib landmarks and OpenCV. |
| UBO morphing executable (`ubo`) | External landmark morphing executable wrapper | `morphs/lmaubo/MorphedImageGenerator.exe`, `morphs/lmaubo/FaceMorphingTool.exe` | TODO: add original author/tool URL and citation. | TODO: confirm redistribution permission for binaries. |
| MIPGAN-I | GAN-based identity-prior morph generation | `morphs/mipgan1/`, `models/StyleGAN_finetuned_ICAO.pkl`, `models/finetuned_resnet.h5`, `models/frs/config_ms1m_100_1006k/` | Haoyu Zhang, Sushma Venkatesh, Raghavendra Ramachandra, Kiran Raja, Naser Damer, Christoph Busch, "MIPGAN - Generating Strong and High Quality Morphing Attacks Using Identity Prior Driven GAN," IEEE TBIOM 2021. Public portal: https://github.com/ZHYYYYYYYYYYYY/MIPGAN-face-morphing-algorithm | Original MIPGAN code/model access may require a signed license. Confirm terms before redistribution. |
| MIPGAN-II | StyleGAN2-based MIPGAN variant | `morphs/mipgan2/`, `models/stylegan2_finetuned_ICAO.pkl`, `models/resnet_18_20191231.h5`, `models/frs/config_ms1m_100_1006k/` | TODO: confirm the exact MIPGAN-II citation/source. Also acknowledge MIPGAN and StyleGAN2. | TODO: confirm checkpoint provenance and redistribution terms. |
| Morph-PIPE (`pipe`) | Diffusion-based morphing with identity prior | `morphs/pipe/`, `models/checkpoints/ffhq256_autoenc/last.ckpt` | Haoyu Zhang, Raghavendra Ramachandra, Kiran Raja, Christoph Busch, "Morph-PIPE: Plugging in Identity Prior to Enhance Face Morphing Attack Based on Diffusion Model," NISK 2023. Source portal: https://github.com/ZHYYYYYYYYYYYY/Morph-PIPE | Confirm model checkpoint source and access terms. |
| MorDIFF (`mordiff`) | Diffusion autoencoder face morphing | `morphs/mordiff/`, `models/ffhq/last.ckpt` | Naser Damer, Meiling Fang, Patrick Siebke, Jan Niklas Kolf, Marco Huber, Fadi Boutros, "MorDIFF: Recognition Vulnerability and Attack Detectability of Face Morphing Attacks Created by Diffusion Autoencoders," IWBF 2023 / arXiv:2302.01843. Source: https://github.com/naserdamer/MorDIFF | Confirm model checkpoint source and access terms. |
| Greedy-DiM (`greedy`) | Greedy identity-guided diffusion morphing | `morphs/greedy/`, `models/checkpoints/ffhq256_autoenc/last.ckpt`, `models/glint360k_cosface_r100_fp16_0.1/backbone.pth` | Zander W. Blasingame, Chen Liu, "Greedy-DiM: Greedy Algorithms for Unreasonably Effective Face Morphs," IJCB 2024 / arXiv:2404.06025. Source: https://github.com/zblasingame/Greedy-DiM | `morphs/greedy/LICENSE` is included locally; confirm checkpoint redistribution terms. |
| LADIMO (`ladimo`) | Latent diffusion morphing from biometric templates | `morphs/ladimo/`, `models/logs/2023-11-15T10-04-11_ffhq-ldm-vq-f8/checkpoints/epoch=000096.ckpt`, `models/magface/magface_epoch_00025.pth` | Marcel Grimmer, Christoph Busch, "LADIMO: Face Morph Generation through Biometric Template Inversion with Latent Diffusion," IJCB 2024 / arXiv:2410.07988. Source: https://github.com/dasec/LADIMO | Official LADIMO code/models may require contacting the authors. Confirm terms before redistribution. |

### Generative Backbones and Utilities

| Component | Used for | Local files/checkpoints | Reference / source to acknowledge | License / note |
| --- | --- | --- | --- | --- |
| FFHQ dataset/alignment procedure | Face alignment and StyleGAN-family preprocessing | `ffhq_align_images.py`, `ffhq_dataset/`, copied FFHQ alignment routine | NVIDIA FFHQ dataset: https://github.com/NVlabs/ffhq-dataset. StyleGAN paper: Tero Karras, Samuli Laine, Timo Aila, "A Style-Based Generator Architecture for Generative Adversarial Networks," CVPR 2019 / arXiv:1812.04948 | FFHQ images and metadata have their own licensing constraints; this repo uses the alignment procedure. |
| dlib 68-point landmark predictor | Landmark detection for FFHQ alignment and LMA morphing | `models/temp/shape_predictor_68_face_landmarks.dat` | Vahid Kazemi, Josephine Sullivan, "One Millisecond Face Alignment with an Ensemble of Regression Trees," CVPR 2014. dlib model page: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 | dlib notes that the predictor was trained on iBUG 300-W; iBUG 300-W excludes commercial use. |
| StyleGAN | MIPGAN-I generator backbone and FFHQ checkpoint | `models/karras2019stylegan-ffhq-1024x1024.pkl`, `models/StyleGAN_finetuned_ICAO.pkl` | Tero Karras, Samuli Laine, Timo Aila, "A Style-Based Generator Architecture for Generative Adversarial Networks," CVPR 2019 / arXiv:1812.04948. Source: https://github.com/NVlabs/stylegan | NVIDIA StyleGAN code/checkpoints have their own license. |
| StyleGAN2 | MIPGAN-II generator backbone | `models/stylegan2-ffhq-config-f.pkl`, `models/stylegan2_finetuned_ICAO.pkl`, `models/stylegan2-finetuned_ICAO_less_epochs.pkl` | Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila, "Analyzing and Improving the Image Quality of StyleGAN," CVPR 2020 / arXiv:1912.04958. Source: https://github.com/NVlabs/stylegan2 | NVIDIA StyleGAN2 code/checkpoints have their own license. |
| Diffusion Autoencoders (DiffAE) | Diffusion autoencoder backbone for MorDIFF, Morph-PIPE, and Greedy-DiM style methods | `morphs/mordiff/diffae/`, `morphs/pipe/`, `morphs/greedy/`, `models/checkpoints/*`, `models/ffhq/last.ckpt` | Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, Supasorn Suwajanakorn, "Diffusion Autoencoders: Toward a Meaningful and Decodable Representation," CVPR 2022 / arXiv:2111.15640. Source: https://github.com/konpatp/diffae | Confirm checkpoint source and license. |
| Latent Diffusion Models | LADIMO model architecture and sampling code | `morphs/ladimo/ldm/`, `models/logs/.../epoch=000096.ckpt` | Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022 / arXiv:2112.10752. Source: https://github.com/CompVis/latent-diffusion | Confirm checkpoint source and license. |
| Taming Transformers / VQGAN | LADIMO first-stage / VQGAN dependencies | `morphs/ladimo/taming-transformers/`, `morphs/ladimo/src/taming-transformers/`, `models/first_stage_models/vq-f4/` | Patrick Esser, Robin Rombach, Björn Ommer, "Taming Transformers for High-Resolution Image Synthesis," CVPR 2021 / arXiv:2012.09841. Source: https://github.com/CompVis/taming-transformers | Local upstream license files are included under the LADIMO folders. |
| CLIP | Dependency bundled with LADIMO/latent-diffusion code | `morphs/ladimo/src/clip/` | Alec Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021 / arXiv:2103.00020. Source: https://github.com/openai/CLIP | Local CLIP license is included under `morphs/ladimo/src/clip/LICENSE`. |
| LPIPS / perceptual similarity | Perceptual losses in GAN/diffusion morphing components | `models/vgg16_zhang_perceptual.pkl`, LPIPS imports/code | Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang, "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," CVPR 2018 / arXiv:1801.03924. Source: https://github.com/richzhang/PerceptualSimilarity | Check local dependency license if redistributed. |

### Face Recognition Models

| Component | Used for | Local files/checkpoints | Reference / source to acknowledge | License / note |
| --- | --- | --- | --- | --- |
| ArcFace | FRS feature extraction, pair selection, identity loss/backbone | `frs/arcface/`, `models/frs_models/arcface/`, `models/glint360k_cosface_r100_fp16_0.1/backbone.pth`, `models/frs/config_ms1m_100_1006k/` | Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," CVPR 2019 / arXiv:1801.07698. InsightFace source: https://github.com/deepinsight/insightface | For Glint360K-trained InsightFace models, InsightFace notes non-commercial research use. |
| CosFace / Glint360K InsightFace backbone | Identity loss in Greedy-DiM configuration | `models/glint360k_cosface_r100_fp16_0.1/backbone.pth` | Hao Wang et al., "CosFace: Large Margin Cosine Loss for Deep Face Recognition," CVPR 2018 / arXiv:1801.09414. InsightFace model zoo/source: https://github.com/deepinsight/insightface | Confirm exact model-card/license for the downloaded checkpoint. |
| AdaFace | FRS feature extraction | `frs/adaface/`, `models/frs_models/adaface/adaface_ir101_ms1mv2.ckpt`, `models/frs_models/adaface/adaface_ir101_webface12m.ckpt` | Minchul Kim, Anil K. Jain, Xiaoming Liu, "AdaFace: Quality Adaptive Margin for Face Recognition," CVPR 2022 / arXiv:2204.00964. Source: https://github.com/mk-minchul/AdaFace | Confirm checkpoint license from the AdaFace model release. |
| MagFace | FRS feature extraction and LADIMO conditioning | `frs/magface/`, `models/frs_models/magface/magface_epoch_00025.pth`, `models/magface/magface_epoch_00025.pth` | Qiang Meng, Shichao Zhao, Zhida Huang, Feng Zhou, "MagFace: A Universal Representation for Face Recognition and Quality Assessment," CVPR 2021 / arXiv:2103.06627. Source: https://github.com/IrvingMeng/MagFace | Local MagFace license is included under `frs/magface/LICENSE`. |
| TensorFlow FRS checkpoint | Identity loss / embeddings in MIPGAN-I and MIPGAN-II | `models/frs/config_ms1m_100_1006k/best-m-1006000.*`, `models/frs/configs/config_ms1m_100.yaml` | TODO: confirm exact upstream model/training source. Likely InsightFace-style FRS used by the original MIPGAN code. | TODO: confirm license and redistribution terms. |

## Citation

If you use this repository, please cite the associated paper:

```bibtex
@inproceedings{patwardhan2026trusted,
  title={Trusted but Tainted: Enrolment Perturbations that Undermine Morphing Attack Detection and Face Recognition},
  author={Patwardhan, Sushrut Dattatraya and others},
  booktitle={International Conference on Pattern Recognition},
  year={2026}
}
```
