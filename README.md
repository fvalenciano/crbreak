# crbreak

Search for a spectral break in the diffuse gamma-ray emission of the Galactic plane measured by
*Fermi*-LAT. The plane is cut into windows of Galactic longitude; in each window the
foreground-subtracted spectrum is fitted with a power law and with a smooth broken power law,
per window and jointly (one break energy $E_{\rm br}$ and one $\Delta\gamma$ for all windows),
with and without nuisance parameters for the effective-area systematic. The significance of the
break is computed with Wilks' theorem and with Monte Carlo pseudo-experiments.

**Start with [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb)**: it explains the code and
reproduces the main results of the paper.

## Install

Python 3.10.

```bash
git clone https://github.com/fvalenciano/crbreak.git
cd crbreak
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e .                    # or: poetry install
bash scripts/download_data.sh       # 6.3 GB of input data into data/ (see data/README.md)
```

The paper checkpoints in `results/paper1/` and the first part of the tutorial do not need the data.

## Run

Every option of the analysis is in a YAML config: `configs/baseline.yaml` holds all of them;
any other config starts with `extends:` and lists only what it changes.

```bash
python -m crbreak.run configs/examples/pass8_nomc.yaml          # baseline without Monte Carlo, ~10 min on 20 cores
python -m crbreak.run configs/baseline.yaml                      # the paper baseline (2M global toys: days)
sbatch scripts/slurm/run_analysis.sbatch configs/baseline.yaml   # same, on SLURM (edit partition/account)
```

A run writes its figures, log, merged `config.yaml` and `checkpoint.pkl.gz` to
`results/runs/<ddMonyy>/b<lo>-<hi>/<name>/`. Figures can be redrawn from a checkpoint without refitting:

```bash
python -m crbreak.run_plots_from_checkpoint <run folder>/checkpoint.pkl.gz
```

## Paper results

| Result | Config | Checkpoint |
|---|---|---|
| Baseline: 9 windows in $\|l\| < 85^\circ$, $\|b\| < 4^\circ$ | `configs/baseline.yaml` | `results/paper1/baseline` |
| $\|b\| < 2^\circ$ | `configs/paper1/b0-2.yaml` | `results/paper1/b0-2` |
| 4 and 18 windows | `configs/paper1/windows4.yaml`, `windows18.yaml` | `results/paper1/windows4`, `windows18` |
| No foreground subtraction | `configs/paper1/unsubtracted.yaml` | `results/paper1/unsubtracted` |
| Interstellar emission model | `configs/paper1/iem.yaml` | `results/paper1/iem` |
| Piecewise-linear nuisance basis | `configs/paper1/piecewise.yaml` | `results/paper1/piecewise` |
| Gaussian nuisance prior | `configs/paper1/gaussian_prior.yaml` | `results/paper1/gaussian_prior` |
| Point sources doubled | no config key in this release | `results/paper1/ps_x2` |

Figures from the checkpoints:

```bash
python scripts/regen_single_checkpoint.py results/paper1/baseline   # E_br, Δγ vs |l| + contours; SED + nuisances
python scripts/regen_sed_v1.py results/paper1/baseline              # SED grid
python scripts/regen_8panel_grid.py                                 # robustness figure -> results/paper1/figures/
```

## Layout

```
configs/            baseline.yaml (every option), paper1/ variants, examples/
data/               input data: bubbles/ in the repo, the rest from scripts/download_data.sh
notebooks/          tutorial.ipynb
results/paper1/     checkpoints of the paper runs and the figures made from them
scripts/            paper figures, data download, SLURM launcher
src/crbreak/
  config/           YAML loading and validation (analysis.py)
  data_extract/     sky windows, data reading, foreground subtraction (main.py, data_reduction.py)
  fitting/          models, χ² fits per window and global, nuisances, scans, Monte Carlo;
                    pipeline.py runs the whole analysis
  plotting/         figures
  run.py            command line: one analysis from a config
tests/              pytest tests/
```

## Citation

If you use this code, please cite the paper (reference to be added).
