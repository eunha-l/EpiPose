# EpiPose: Antibody-Specific Epitope Prediction by Weighting Rigid Docking Poses

Given an unbound antigen and an antibody, EpiPose predicts which antigen residues
the antibody binds. A rigid FFT docking run produces a pool of candidate poses, and
the model learns how much to trust each one.

Pose tokens attend to each other, so a pose is judged against the other candidates
for the same antigen, and attend to antigen residue tokens. The model emits one
scalar per pose, and a softmax over those scalars weights the pose-residue contact
matrix into an epitope map.

An auxiliary loss uses the bound/unbound structural change of the antigen surface.
For training antigens both forms are available, so the change can be measured per
residue; ranking the pose scores by the mean change of the residues each pose covers
teaches the model to prefer poses that cover sites that open on binding. The change
needs the bound structure and is therefore a training target only, never used at
inference.

## Results

Test set: 98 antigen-antibody units, 3 seeds.

| | AUROC | AUPRC | MCC | F1 | F1@30 |
|---|---|---|---|---|---|
| without the auxiliary loss | 0.8439 ± 0.005 | 0.2480 ± 0.021 | 0.2580 ± 0.039 | 0.2909 ± 0.043 | 0.2958 ± 0.016 |
| **EpiPose** | **0.8486 ± 0.003** | **0.2571 ± 0.009** | **0.2801 ± 0.017** | **0.3137 ± 0.015** | **0.3030 ± 0.008** |

The decision threshold is chosen on the validation split and applied to test.
F1@30 takes the 30 highest scoring residues of each unit and needs no threshold.

## Environment

Developed with the versions pinned in `environment.yml`:

- Python 3.13, CUDA 13.0, torch 2.11.0+cu130
- numpy 1.26, scipy 1.17, scikit-learn 1.8, pandas 3.0

These are not strict requirements; any recent torch, numpy and scikit-learn should
work. Inference on the distributed features runs on CPU in a few minutes. Training wants
a GPU.

```bash
conda env create --file environment.yml
conda activate epipose
```

If the torch build does not match your driver, install it separately following
<https://pytorch.org>.

## Data

The repository holds the code, the configs, the split table and the three
checkpoints. The bulk data is distributed as release archives, in two tiers: the
**quick path** reproduces the reported numbers from the shipped intermediate
features, and the **full path** adds the structures needed to re-run docking from
scratch. Every archive stores repository-relative paths, so unzip it at the
repository root.

| tier | archive | size | unpacks to | needed for |
|---|---|---|---|---|
| quick | `epipose_features_core.zip` | 57.9 MB | `data/features/{pose_rep,residue}.pkl` | `eval`, `predict` |
| quick | `epipose_features_train.zip` | 139.0 MB | `data/features/{pose_multi_apo,delta_target}.pkl` | `train` |
| full | `epipose_structures_cif.zip` | 58.2 MB | `data/structures/complex_cif/` | the auxiliary target |
| full | `epipose_structures_dock.zip` | 307.2 MB | `data/structures/dock_fft/` | docking, pose features |
| full | `epipose_structures_contacts.zip` | 18.2 MB | `data/structures/{consasa,apoc}/` | pose contact matrices |

Checksums are in `SHA256SUMS` next to the archives.

### Quick path: reproduce the reported numbers

Two archives, 324 MB unpacked. Evaluation runs on CPU in a few minutes.

```bash
unzip epipose_features_core.zip -d .        # eval and predict
unzip epipose_features_train.zip -d .       # add this to train

python main.py eval --config configs/predict.yaml \
                    --ckpt checkpoints/epipose_seed0.pt
```

`epipose_features_core.zip` alone is enough for `eval` and `predict`;
`pose_multi_apo.pkl` and `delta_target.pkl` are loaded only if present, so
training without them falls back to the representative apo structure and to the
plain pose loss, which is not the reported configuration.

### Full path: re-run the docking pipeline

All five archives, 860 MB unpacked. This regenerates the feature pickles the
quick path downloads, and needs a GPU. See
[Re-running the docking pipeline](#re-running-the-docking-pipeline).

```bash
for z in epipose_features_core.zip epipose_features_train.zip \
         epipose_structures_cif.zip epipose_structures_dock.zip \
         epipose_structures_contacts.zip; do unzip -q "$z" -d .; done
```

## Layout

```
epipose/
├── main.py                 train / eval / predict entry point
├── epipose/
│   ├── config.py           yaml config with command line overrides
│   ├── data.py             feature loading and batching
│   ├── model.py            network and pose scores to map
│   ├── losses.py           pose loss and auxiliary ranking loss
│   ├── metrics.py          AUROC, AUPRC, MCC, F1, F1@30
│   └── runner.py           training, evaluation, prediction
├── docking/
│   ├── run_pipeline.py     docking and feature extraction entry point
│   ├── residue_feature.py  antigen residue features
│   ├── delta_target.py     auxiliary training target
│   └── _vendor/            original research code, paths externalised
├── configs/
│   ├── train.yaml          training
│   ├── train_no_aux.yaml   training without the auxiliary loss
│   └── predict.yaml        evaluation and prediction
├── data/
│   ├── assets/             tables the pipeline reads (in the repository)
│   ├── features/           pose and residue features (quick-path archives)
│   └── structures/         docking inputs, results and structures (full-path archives)
└── checkpoints/            epipose_seed{0,1,2}.pt (in the repository)
```

## Evaluate a checkpoint

```bash
python main.py eval --config configs/predict.yaml \
                    --ckpt checkpoints/epipose_seed0.pt
```

## Predict maps

```bash
python main.py predict --config configs/predict.yaml \
                       --ckpt checkpoints/epipose_seed0.pt \
                       --split test --out pred.npz
```

`pred.npz` holds one epitope map per unit (`map_<gid>`) and the pose scores
(`pose_logit_<gid>`).

## Train

```bash
python main.py train --config configs/train.yaml --gpu 0

# without the auxiliary loss
python main.py train --config configs/train_no_aux.yaml --gpu 0
```

Any yaml key can be overridden on the command line. Each run writes
`runs/<tag>/{config.yaml, best.pt, result.json}`.

| option | default | meaning |
|---|---|---|
| `use_delta_rank` | `true` | auxiliary ranking loss on or off |
| `lambda_delta` | `0.125` | weight of the auxiliary loss |
| `rank_margin` | `0.1` | minimum target gap for a pair to enter the ranking loss |
| `multi_apo` | `true` | resample the apo structure of a unit each step |
| `top_poses` | `500` | poses kept per unit, by docking score |
| `map_tau` | `1.0` | softmax temperature over pose scores |

## Re-running the docking pipeline

Unpack the three full-path archives first; `data/structures/` then holds the
inputs, so the steps below run on the distributed set.
Paths come from `docking/_vendor/paths.py` and can be overridden with
`EPIPOSE_WORK`, `EPIPOSE_DOCK_OUT`, `EPIPOSE_CONSASA`, `EPIPOSE_APOC`,
`EPIPOSE_RAW`.

```
data/structures/dock_fft/     antigen and antibody atom tensors, docking results
data/structures/consasa/      per-pose contact matrices
data/structures/apoc/         contact matrices of the further apo structures
data/structures/complex_cif/  bound structures, for the auxiliary target
```

```bash
export EPIPOSE_WORK=/path/to/scratch

# 1. complex cif to atom tensors (only for antigens outside the bundled set)
EPIPOSE_RAW=/path/to/complex/cif python docking/run_pipeline.py tensor --nproc 16

# 2. FFT docking
python docking/run_pipeline.py dock --units units.txt --gpu 0 --keep 500

# 3. pose features
python docking/run_pipeline.py pose --gpu 0 --out data/features/pose_rep.pkl

# 4. antigen residue features
python docking/run_pipeline.py residue --out data/features/residue.pkl
```

The auxiliary target is needed for training only:

```bash
python docking/run_pipeline.py delta --out data/features/delta_target.pkl
```

The files under `docking/_vendor/` are the original research scripts. Hardcoded
paths were moved into `_vendor/paths.py` and code for models this repository does
not use was removed; the pose reconstruction and scoring path is unchanged.
`residue_feature.py` and `delta_target.py` were rewritten and reproduce the shipped
features exactly.

## Data format

`data/features/` holds four pickles, each `{"name": [...], "feat": {...}}`:

| file | keyed by | contents |
|---|---|---|
| `pose_rep.pkl` | `gid` | `F` (P, 33) pose features, `C` contact matrix as packed bits, `y` epitope labels, `q` pose quality |
| `pose_multi_apo.pkl` | `(gid, k)` | the same for further apo structures of one antigen |
| `residue.pkl` | `gid` | (nres, 9) prop, kl, id, rsa, kd, protrusion, dens6, dens10, q_abs |
| `delta_target.pkl` | `gid` | (nres, 4) auxiliary target, training units only |

`q = |C and y| / |C|` is the fraction of a pose's contacts that lie on the true
epitope, and it is the target of the main loss.

`data/assets/epipose_split.csv` lists the units:

| column | meaning |
|---|---|
| `gid` | antigen-antibody unit id |
| `split` | train, val or test |
| `use` | whether the unit is usable |
| `scored` | whether a test unit counts toward the reported metrics |
| `apo_code` | PDB entry of the apo structure used for this antigen |
| `cluster` | sequence cluster; the splits were cut along these |
| `seq_len` | canonical antigen sequence length |
| `epitope` | epitope labels as a bit string, for reference |
