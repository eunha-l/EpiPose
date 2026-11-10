# EpiPose: Antibody-Specific Epitope Prediction from Unbound Structures by Learning to Reweight Docking Poses

Given an experimentally determined unbound antigen and an antibody modelled from
its sequence, EpiPose predicts which antigen residues the antibody binds. FFT
rigid-body docking produces a pool of candidate poses, and an attention-based
reranking network scores each one.

Pose tokens attend to each other, so a pose is judged against the other candidates
for the same antigen, and cross-attention brings in the antigen residue
descriptors. The network emits one reranking score per pose. Those scores are
standardised across the pose pool, passed through a softmax at temperature `tau`
to give the pose weights, and the weights average the contact matrix into a
per-residue epitope score.

The auxiliary ranking objective uses the binding-induced change in antigen surface
properties as supervision. Training units have both the unbound structure and the
bound complexes, so the change can be measured per residue; ranking pairs of poses
by the mean change over the residues each pose contacts teaches the model to
prefer poses covering sites that rearrange on binding. The target needs the bound
structure and is therefore used during training only, never at inference.

## Environment

Developed with the versions pinned in `environment.yml`:

- Python 3.13, CUDA 13.0, torch 2.11.0+cu130
- numpy 1.26, scipy 1.17, scikit-learn 1.8, pandas 3.0

These are not strict requirements; any recent torch, numpy and scikit-learn should
work. Inference on the distributed descriptors runs on CPU in a few minutes.
Training wants a GPU.

```bash
conda env create --file environment.yml
conda activate epipose
```

If the torch build does not match your driver, install it separately following
<https://pytorch.org>.

## Data

The repository holds the code, the configs, the split table and the three
checkpoints. The bulk data is distributed as release archives, in two tiers. The
**quick path** ships the intermediate products of docking, that is the pose
descriptors, the residue descriptors and the auxiliary ranking target, so the
reranking model can be trained and evaluated directly without running docking. The
**full path** adds the structures needed to rebuild those descriptors from FFT
docking onward. Every archive stores repository-relative paths, so unzip it at the
repository root.

| tier | archive | size | unpacks to | needed for |
|---|---|---|---|---|
| quick | `epipose_features_core.zip` | 57.9 MB | `data/features/{pose_rep,residue}.pkl` | `eval`, `predict` |
| quick | `epipose_features_train.zip` | 139.0 MB | `data/features/{pose_multi_apo,delta_target}.pkl` | `train` |
| full | `epipose_structures_cif.zip` | 58.2 MB | `data/structures/complex_cif/` | the auxiliary target |
| full | `epipose_structures_dock.zip` | 307.2 MB | `data/structures/dock_fft/` | docking, pose descriptors |
| full | `epipose_structures_contacts.zip` | 18.2 MB | `data/structures/{consasa,apoc}/` | pose contact matrices |

Checksums are in `SHA256SUMS` next to the archives.

### Quick path: reproduce the numbers in the paper

Two archives, 324 MB unpacked. FFT docking has already been run, and the pose
descriptors, residue descriptors and auxiliary ranking target extracted from its
output are in the archives. The reranking model
can therefore be trained directly, with no docking and no other preprocessing, and
it reproduces the numbers reported in the paper. Evaluation alone needs no GPU and
finishes on CPU in a few minutes.

```bash
unzip epipose_features_core.zip -d .        # eval and predict
unzip epipose_features_train.zip -d .       # add this to train

# 1. reproduce directly from the shipped checkpoints
python main.py eval --config configs/predict.yaml \
                    --ckpt checkpoints/epipose_seed0.pt

# 2. or train the reranking model yourself on the same descriptors
python main.py train --config configs/train.yaml --gpu 0
```

`epipose_features_core.zip` alone is enough for `eval` and `predict`; training also
needs `epipose_features_train.zip`. `pose_multi_apo.pkl` and `delta_target.pkl` are
loaded only if present, so training without them falls back to the representative
unbound structure and to the pose loss alone, which is not the configuration
reported in the paper.

### Full path: re-run the docking pipeline

All five archives, 860 MB unpacked. This regenerates the descriptor pickles the
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
│   ├── data.py             descriptor loading and batching
│   ├── model.py            reranking network and pose scores to epitope map
│   ├── losses.py           pose loss and auxiliary ranking loss
│   ├── metrics.py          AUROC, AUCPR, MCC, BAcc, F1, F1@30, macro AUROC
│   └── runner.py           training, evaluation, prediction
├── docking/
│   ├── run_pipeline.py     docking and descriptor extraction entry point
│   ├── residue_feature.py  antigen residue descriptors
│   ├── delta_target.py     auxiliary ranking target
│   └── _vendor/            original research code, paths externalised
├── configs/
│   ├── train.yaml          training
│   ├── train_no_aux.yaml   training without the auxiliary ranking objective
│   └── predict.yaml        evaluation and prediction
├── data/
│   ├── assets/             tables the pipeline reads (in the repository)
│   ├── features/           pose and residue descriptors (quick-path archives)
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

`pred.npz` holds one epitope map per unit (`map_<gid>`) and the reranking scores
(`pose_logit_<gid>`).

## Train

```bash
python main.py train --config configs/train.yaml --gpu 0

# without the auxiliary ranking objective
python main.py train --config configs/train_no_aux.yaml --gpu 0
```

Any yaml key can be overridden on the command line. Each run writes
`runs/<tag>/{config.yaml, best.pt, result.json}`.

## Re-running the docking pipeline

Unpack the three full-path archives first; `data/structures/` then holds the
inputs, so the steps below run on the distributed set.
Paths come from `docking/_vendor/paths.py` and can be overridden with
`EPIPOSE_WORK`, `EPIPOSE_DOCK_OUT`, `EPIPOSE_CONSASA`, `EPIPOSE_APOC`,
`EPIPOSE_RAW`.

```
data/structures/dock_fft/     antigen and antibody atom tensors, docking results
data/structures/consasa/      per-pose contact matrices
data/structures/apoc/         contact matrices of the further unbound structures
data/structures/complex_cif/  bound complexes, for the auxiliary target
```

```bash
export EPIPOSE_WORK=/path/to/scratch

# 1. complex cif to atom tensors (only for antigens outside the bundled set)
EPIPOSE_RAW=/path/to/complex/cif python docking/run_pipeline.py tensor --nproc 16

# 2. FFT rigid-body docking
python docking/run_pipeline.py dock --units units.txt --gpu 0 --keep 500

# 3. pose descriptors
python docking/run_pipeline.py pose --gpu 0 --out data/features/pose_rep.pkl

# 4. antigen residue descriptors
python docking/run_pipeline.py residue --out data/features/residue.pkl
```

The auxiliary ranking target is needed for training only:

```bash
python docking/run_pipeline.py delta --out data/features/delta_target.pkl
```

## Data format

`data/features/` holds four pickles, each `{"name": [...], "feat": {...}}`:

| file | keyed by | contents |
|---|---|---|
| `pose_rep.pkl` | `gid` | `F` (P, 33) pose descriptors, `C` contact matrix as packed bits, `y` epitope labels, `q` contact precision |
| `pose_multi_apo.pkl` | `(gid, k)` | the same for further unbound structures of one antigen |
| `residue.pkl` | `gid` | (nres, 9) prop, kl, id, rsa, kd, protrusion, dens6, dens10, q_abs |
| `delta_target.pkl` | `gid` | (nres, 4) auxiliary ranking target, training units only |

`q = |C and y| / |C|` is the fraction of a pose's contacts that lie on the true
epitope, its contact precision, and it is the target of the pose loss.

`data/assets/epipose_split.csv` lists the units:

| column | meaning |
|---|---|
| `gid` | antibody-antigen unit id |
| `split` | train, val or test |
| `use` | whether the unit is usable |
| `scored` | whether a test unit counts toward the reported metrics |
| `apo_code` | PDB entry of the unbound structure used for this antigen |
| `cluster` | sequence cluster; the splits were cut along these |
| `seq_len` | canonical antigen sequence length |
| `epitope` | epitope labels as a bit string, for reference |
