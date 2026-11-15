# Training and evaluation

Run these commands from the repository folder after installing the dependencies and
extracting the release data.

## Data setup

Download the `DisasterM3_Bench/` and `DisasterM3_Instruct/` folders from
[DisasterM3](https://huggingface.co/datasets/Kingdrone-Junjue/DisasterM3) into one folder.
Inside `DisasterM3_Instruct/`, unzip `train_images.zip` and `masks.zip`. Then set the dataset
root:

```sh
export DM3_ROOT=/path/to/DisasterM3
```

`assets/metadata/split.csv` lists each scene's images and road masks under `${DM3_ROOT}`. SIREN's train, validation and
test splits use scenes from both folders. `siren.inference.export` takes the root as `--data-root`; the other commands read
`DM3_ROOT` from the environment, so keep it exported in every shell.

To train entity extraction, also extract `siren-training-initialization.tar.gz`.
`data/README.md` describes each data file.

Check the images and road masks before a long training run:

```sh
python -m siren.inference.export --assets configs/assets.json \
  --data-root "$DM3_ROOT" --split train --preflight
python -m siren.inference.export --assets configs/assets.json \
  --data-root "$DM3_ROOT" --split val --preflight
```

## Evaluate SIREN

The paper's quantitative evaluations use fixed query programs.

```sh
python -m siren.inference.export --assets configs/assets.json \
  --data-root "$DM3_ROOT" --split test --output outputs/entities
python -m siren.evaluation.counts --entities-dir outputs/entities \
  --references data/counts/test/damaged_buildings.csv --question damaged_buildings \
  --output outputs/damaged_metrics.json
```

Change the question and reference filename to evaluate another query. An interrupted export
resumes in the same folder with the same inputs and settings. Exporting a full split takes days on a
CPU, so use a GPU. The export's `--scene-ids` selects particular scenes; to score such a
partial export, pass `counts` a file of those IDs, one per line, with `--scene-ids`.
The two road queries are scored on the scenes in their reference files;
`data/road_query_exclusions.json` lists the test scenes left out because their road
annotations are incomplete.

| Question name | What it counts |
| --- | --- |
| `damaged_buildings` | Buildings with minor damage, major damage or destruction |
| `destroyed_buildings` | Destroyed buildings |
| `blocked_roads` | Blocked road segments |
| `damaged_buildings_without_clear_road` | Damaged buildings with no clear road within 50 pixels |
| `destroyed_buildings_without_intact_neighbor` | Destroyed buildings with no intact building within 50 pixels |
| `large_destroyed_buildings` | Destroyed buildings with footprint area of at least 2,000 square pixels |
| `destroyed_residential_buildings` | Destroyed residential buildings |
| `destroyed_buildings_surrounded_by_vegetation` | Destroyed buildings surrounded by vegetation (trees or shrubs) |

## Train models

Before CUDA training, run `export CUBLAS_WORKSPACE_CONFIG=:4096:8`. `siren.concepts.train` uses
deterministic operations and needs it; the other training commands can differ slightly between
GPU runs.

### SIREN training

Stage 1 trains the encoder, building detector and road segmentation head. Stage 2 trains the blocked-road classifier on the frozen final
Stage 1 weights and writes `base.pth`, in the same format as `assets/checkpoints/base.pth`.

```sh
python -m siren.perception.train --stage 1 --output outputs/entity_extraction \
  --split-csv assets/metadata/split.csv --building-entities-dir data/building_entities \
  --detector-checkpoint assets/checkpoints/detector.pth \
  --unreadable-scenes assets/metadata/unreadable_scenes.json
python -m siren.perception.train --stage 2 --output outputs/road_classifier \
  --split-csv assets/metadata/split.csv --labels-dir data/labels --roads-csv data/roads.csv \
  --stage1-checkpoint outputs/entity_extraction/stage1_last.pth \
  --unreadable-scenes assets/metadata/unreadable_scenes.json
```

When retraining the full model, use `outputs/road_classifier/base.pth` in place of
`assets/checkpoints/base.pth` in every classifier command below.

### Entity extraction

`siren.perception.eval` reports building-detection F1 and road-mask F1 for a model file.
`--road-sweep` chooses the road segmentation threshold on validation data; pass the result to
`--road-thresh` on another split.

```sh
python -m siren.perception.eval --checkpoint assets/checkpoints/base.pth \
  --split-csv assets/metadata/split.csv --building-entities-dir data/building_entities \
  --unreadable-scenes assets/metadata/unreadable_scenes.json --split val --road-sweep \
  --output outputs/extraction_val
```

### Damage and destruction classifiers

The classifiers train on frozen features from both images; their thresholds minimize
validation Count-MAE. Each random seed has its own folder, such as `seed_0/`, containing
`damaged.pth`, `destroyed.pth` and validation candidates.

```sh
python -m siren.concepts.train_damage --base-checkpoint assets/checkpoints/base.pth \
  --split-csv assets/metadata/split.csv \
  --unreadable-scenes assets/metadata/unreadable_scenes.json \
  --labels-dir data/labels --buildings-csv data/buildings.csv --output outputs/damage
python -m siren.concepts.calibrate_damage --candidates outputs/damage/seed_0/candidates_val \
  --counts-dir data/counts --unreadable-scenes assets/metadata/unreadable_scenes.json \
  --output outputs/damage/seed_0/thresholds.json
```

### Residential and vegetation classifiers

The residential classifier trains on frozen `base.pth` features. It writes `classifier.pth` and `classifier.json` (settings and validation
results). Vegetation training labels will be added later.

```sh
python -m siren.concepts.train --concept residential \
  --base-checkpoint assets/checkpoints/base.pth \
  --labels data/labels/residential.json --output outputs/residential
```

### Use newly trained models

Copy `configs/assets.json` and set each new model's path (relative to the config file) and
SHA-256 hash: Stage 2's `base.pth` as `base`, `damaged.pth` and `destroyed.pth` as
`damaged` and `destroyed`, and the classifier's `classifier.pth` as
`residential` or `vegetation`.

Calibration does not edit `src/siren/settings.py`. To evaluate new models, copy their
validation thresholds into it; the learning-curve `score` command reads each run's
`thresholds.json` instead.

### Road thresholds

The road-access threshold maximizes macro-F1 on annotated validation road segments. Stage 2
saves these validation scores; `data/validation_road_scores.json` holds them for the released
model. The blocked-road counting threshold minimizes validation Count-MAE, using exports from
the model being calibrated.

```sh
python -m siren.concepts.calibrate_roads \
  --road-scores outputs/road_classifier/validation_road_scores.json \
  --output outputs/road_access_threshold.json
python -m siren.inference.export --assets configs/assets.json \
  --data-root "$DM3_ROOT" --split val --output outputs/entities_val
python -m siren.concepts.calibrate_roads --entities-dir outputs/entities_val \
  --references data/counts/val/blocked_roads.csv --split-csv assets/metadata/split.csv \
  --unreadable-scenes assets/metadata/unreadable_scenes.json \
  --output outputs/road_count_threshold.json
```

### Count baselines

```sh
python -m siren.baselines.train --encoder satlas --question damaged_buildings \
  --split-csv assets/metadata/split.csv --counts-dir data/counts \
  --unreadable-scenes assets/metadata/unreadable_scenes.json --output outputs/satlas_damaged
```

`--counts-dir` holds `train`, `val` and `test` folders with one CSV per question. The two road
questions use only scenes with road masks in `split.csv`; their test score covers the scenes
in the test reference files.

| `--encoder` | Input | Learning rate | Batch x accumulation | Pretrained weights |
| --- | ---: | ---: | ---: | --- |
| `satlas` | 1024 | 1e-4 | 2 x 16 | [`Aerial_SwinB_SI`](https://github.com/allenai/satlaspretrain_models#available-pretrained-models), downloaded automatically |
| `dinov3` | 1024 | 1e-5 | 1 x 32 | [DINOv3 SAT-493M](https://github.com/facebookresearch/dinov3) (`vit_large_patch16_dinov3.sat493m`), loaded by timm or from `--ckpt` |
| `remoteclip` | 1022 | 1e-5 | 2 x 16 | [`RemoteCLIP-ViT-L-14.pt`](https://huggingface.co/chendelong/RemoteCLIP), passed with `--ckpt` |
| `skyclip` | 1022 | 1e-5 | 2 x 16 | [SkyCLIP ViT-L/14, top 30%, CLIP-laion-RS filtered](https://github.com/wangzhecheng/SkyScript#download-model-checkpoints), extracted and passed with `--ckpt` |

Training writes `best.pth` and `config.json` in the output folder; keep them together. It also
writes the test Count-MAE to `results.json` and per-scene predictions to `predictions_val.csv` and
`predictions_test.csv`. SkyCLIP training also downloads OpenAI CLIP ViT-L/14 through open_clip.

### Learning curves

`--train-subset 400 --subset-seed 0` trains a baseline on 400 scenes and writes `train_chips.txt`;
pass that file with `--train-chips` to the other baselines and to `train_damage` (for example with
`--output outputs/damage_n400`). Alternatively, use the supplied lists in `data/subsets/`
with `--train-chips` for each method.
Save each SIREN run's calibration output as `<run>/seed_0/thresholds.json`, then cache the test
features, apply the classifiers and score. The full-data point is the main SIREN evaluation
above.

```sh
python -m siren.evaluation.learning cache --assets configs/assets.json \
  --entities-dir outputs/entities --output outputs/learning_cache
python -m siren.evaluation.learning apply --cache outputs/learning_cache \
  --heads outputs/damage_n400/seed_0 --output outputs/damage_n400/seed_0/test_scores.npz
python -m siren.evaluation.learning score --run-dir outputs/damage_n400/seed_0 \
  --entities-dir outputs/entities --counts-dir data/counts --output outputs/damage_n400/scores
```

### External events

| Events | Data sources |
| --- | --- |
| Eaton and Palisades | [Maxar Open Data](https://registry.opendata.aws/maxar-open-data/) imagery; CAL FIRE DINS [Eaton](https://data.lacounty.gov/datasets/CALFIRE-Forestry::dins-2025-eaton-public-view/about) and [Palisades](https://data.lacounty.gov/datasets/CALFIRE-Forestry::dins-2025-palisades-public-view/about) records |
| Hurricane Ian | Pre-event [NAIP imagery](https://naip-usdaonline.hub.arcgis.com/), post-event [NOAA imagery](https://storms.ngs.noaa.gov/storms/ian/index.html) and [DesignSafe PRJ-5700](https://doi.org/10.17603/ds2-c1g8-qy41) damage records |

The external-event images and their preparation are not included. `data/counts/external/`
holds the reference counts, and `data/predictions/external.csv` holds the predictions behind
the reported metrics (see the last section). With your own aligned pairs, the commands are:

```sh
python -m siren.evaluation.external --pairs /path/to/ian_pairs.csv \
  --siren-assets configs/assets.json --question destroyed_buildings --output outputs/ian_siren
python -m siren.evaluation.external --pairs /path/to/ian_pairs.csv \
  --baseline-run outputs/satlas_destroyed --question destroyed_buildings --output outputs/ian_satlas
python -m siren.evaluation.counts --predictions outputs/ian_siren/predictions.csv \
  --references data/counts/external/ian.csv --question destroyed_buildings --system SIREN \
  --output outputs/ian_metrics.json
```

`--pairs` lists aligned 1024-pixel pairs (`chip_id,pre_image_path,post_image_path`); SIREN also
saves each scene's entities. The Ian road example (blocked roads within 40 m of a residential
building) is `blocked_roads_near_homes` in `src/siren/programs/recipes.py`.

### Vision-language model baselines

These models answer from the image pair directly. Install the `language` extra first.

```sh
python -m siren.evaluation.vlm --pairs /path/to/damaged_test_pairs.csv \
  --prompts configs/vlm_prompts.json --provider anthropic --question damaged_buildings \
  --output outputs/claude_damaged --uploads outputs/claude_uploads.json
```

Without `--execute`, the command only checks the inputs. `--execute` sends the requests, which
incur API charges, using `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` with `--provider openai`.
The defaults are the paper's models and reasoning settings. For `--pairs`, keep the rows of
`assets/metadata/split.csv` whose `chip_id` appears in `data/counts/test/<question>.csv`.
`--variant`
selects the prompt: `direct` (the default, used for the test results), `listing` or `examples`.
The last two are for the prompt comparison and accept only validation pairs; `examples` also
needs `--examples`, a CSV of three validation pairs.
`data/subsets/vlm_timing_400.json` lists, for each question, the test scenes used to time the
GPT and Claude runs.

### Computation

```sh
python -m siren.evaluation.efficiency --pairs /path/to/timing_pairs.csv \
  --siren-assets configs/assets.json --output outputs/efficiency.json
```

For a baseline, replace `--siren-assets` with `--baseline-runs` followed by its eight run
folders. Timing uses the test pairs in `data/subsets/timing_300.txt`; build `--pairs` from
their rows in `assets/metadata/split.csv`. Three warm-up pairs, preprocessing and device
transfer are not timed. `--profile-scenes N` counts FLOPs and parameters on the first N pairs only.

## Recompute reported metrics from saved predictions

The data bundle also includes the paper's benchmark and external-event predictions:

```sh
python -m siren.evaluation.counts --predictions data/predictions/benchmark.csv \
  --output outputs/benchmark_metrics.json
python -m siren.evaluation.counts --predictions data/predictions/external.csv \
  --output outputs/external_metrics.json
```

These commands recompute the per-question Count-MAE, by event for the external predictions, and
for the benchmark the reduction in count error relative to predicting zero
(`count_error_reduction_vs_zero`). They do not rerun the models.
