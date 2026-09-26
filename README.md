# TABPFN-MTO

TabPFN-based evolutionary multitask optimization on the nine CEC2017
single-objective, two-task benchmark problems. This repository contains the
fixed main method and exports final best-so-far objective values for both tasks.

## Associated paper

**Representative-Context-Guided TabPFN for Online Transfer in Evolutionary Multitask Optimization**

Hao Sun, Jingwen Zhang, Ziyu Hu, and Bingxuan Yu.

## Repository contents

| File | Purpose |
| --- | --- |
| `main.py` | Main optimization algorithm and public Python function |
| `utils.py` | Population state and uncertainty-filtered pairing |
| `CEC2017MTSO.py` | Benchmark objectives and MAT-file loader |
| `run.py` | Batch runner and final-result CSV output |
| `requirements.txt` | Pinned main dependencies |
| `Tasks/` | Nine MAT files and benchmark source notes |
| `LICENSE` | MIT license for project code |

## Installation

Use Python **3.10.11 (64-bit)** as reported in Supplementary Table S4.
Create a separate environment. From this repository directory:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell with `.venv\Scripts\Activate.ps1`, or on
Linux/macOS with `source .venv/bin/activate`.

For a compatible NVIDIA GPU on Linux or Windows, install the CUDA 12.1 PyTorch
build recorded in the supplied environment, followed by the other dependencies:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
```

For a CPU-only Linux/Windows environment, replace the first command with:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

See the official [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/)
for other platforms. `requirements.txt` pins the main dependencies;
the CUDA command above selects the `2.5.1+cu121` build used in the reported environment.

### TabPFN model weights

The algorithm uses `TabPFNRegressor` from `tabpfn==2.2.1` with
`fit_mode="low_memory"`, `device="auto"` and `random_state=42`.
The target-dimension feature is categorical. The reported model settings include
`n_estimators=8` and `model_path="auto"`; fine-tuning is not used.

The regression checkpoint in the experiment-machine cache is
`tabpfn-v2-regressor.ckpt`. Classifier checkpoints are not used by this algorithm.
Weights are not bundled with this repository. On first use, allow TabPFN to
retrieve its required regression weights, or make the checkpoint available in
its model cache for offline execution. The code retains the package's automatic
checkpoint selection. See [TabPFN](https://github.com/PriorLabs/TabPFN)
for model access and caching instructions.

## Run

One problem and one seed (the full evaluation budget still applies):

```bash
python run.py --problems CIHS --seeds 11
```

All nine problems with seeds 11 through 40 (270 runs):

```bash
python run.py
```

Selected problems/seeds and a separate output file:

```bash
python run.py --problems CIHS PILS --seeds 11 12 --output results/example.csv
```

The default output is `final_results.csv` beside `run.py`. An explicitly supplied
relative output path is resolved from the current working directory. The CSV has
four columns:

| Column | Meaning |
| --- | --- |
| `Problem` | Benchmark identifier |
| `Seed` | Optimizer initialization seed |
| `BestA` | Final best-so-far objective value of task 1 |
| `BestB` | Final best-so-far objective value of task 2 |

Both tasks are minimized. One row is saved after each completed run. Existing
output files are not overwritten; select a new filename for another invocation.
There is no automatic resume. Convergence curves, checkpoints, ablation options,
sensitivity options and aggregate statistics are not exported.

Python interface:

```python
from main import run_online_single_problem
result = run_online_single_problem("CIHS", init_seed=11)
print(result["bestA"], result["bestB"])
```

## Fixed algorithm settings

| Setting | Value |
| --- | --- |
| Population per task | 50 |
| Combined function-evaluation budget | 100000, including initialization |
| Generation cap | 1500; evaluation budget terminates the run earlier |
| Task dimensions | 50/50; PILS uses 50/25 |
| DE parameters | F = 0.5, CR = 0.9 |
| Migration interval | 5 generations |
| Initial migration ratio | 0.2 per direction |
| Migration learning rate | 0.05 |
| Migration feedback | Rank-weighted survival |
| Context size | 20 |
| Pairing | Mutual nearest neighbors; uncertainty threshold 0.8 |
| Minimum pairs | 3 |
| Current / near / far context allocation | 0.7 / 0.2 / 0.1 |
| Elite / middle / diverse source allocation | 0.5 / 0.3 / 0.2 |
| Selection | Diversity-aware |

If a mapping fit or prediction fails, the algorithm may skip the corresponding
migration and issue a console warning. Check these warnings when running experiments.

## Data and attribution

All nine required MAT files are included under `Tasks/`; MATLAB is not required.
Paths are resolved relative to the source files. See [Tasks/README.md](Tasks/README.md)
for the benchmark references, download source and file mapping.
Project code is provided under the MIT license. Benchmark files and model weights
retain their respective upstream terms and attribution.

## Experimental environment

The environment reported in Supplementary Table S4 is:

| Item | Reported value |
| --- | --- |
| Python | 3.10.11, 64-bit |
| OS identification string | `Windows-10-10.0.26200-SP0` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA runtime | 12.1 |
| cuDNN version code | 90100 |
| TabPFN | 2.2.1 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| scikit-learn | 1.6.1 |

The OS entry is the recorded platform-identification string. CUDA refers to the
runtime used by PyTorch.
