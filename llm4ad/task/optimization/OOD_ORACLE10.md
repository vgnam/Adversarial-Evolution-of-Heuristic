# ID training and OOD Oracle@10 evaluation

The existing `train_dataset.pkl` and `train_datasets/` files are the
in-distribution (ID) training split. OOD files are held out under:

```text
<task>/ood_test_datasets/mixture/size_<N>.pkl
```

Supported constructive tasks are TSP, CVRP, OVRP, VRPTW, 1D/2D bin packing,
knapsack, JSSP, QAP, CFLP, and set cover. Each mixture contains four or six
task-valid distribution families, recorded in `metadata.json`.

## Generate OOD data

Generate all tasks with 64 instances per benchmark size:

```powershell
py -3.11 llm4ad/task/optimization/generate_ood_datasets.py
```

Generate one task or a smaller debugging set:

```powershell
py -3.11 llm4ad/task/optimization/generate_ood_datasets.py `
  --task tsp_construct --sizes 20 50 --n-instances 8
```

This command never modifies the ID training files.

## Train on ID

Keep `load_from_file: true` with `dataset_split: train` (the existing default
fixed-dataset workflow). Every compared method must retain at least 10 unique
heuristics in its final population. AdvEoH, EoH, and MCTS-AHD are configured
with final heuristic population size 10.

## Evaluate every method/run

Pass every completed run using `METHOD=LOG_DIR`:

```powershell
py -3.11 llm4ad/task/optimization/oracle_ood_eval.py `
  --task tsp_construct `
  --run EoH=logs/<eoh-run> `
  --run AdvEoH=logs/<adveoh-run> `
  --run MCTS_AHD=logs/<mcts-run> `
  --output-root logs/tsp_ood_oracle10
```

For each OOD instance, all 10 members of the final population are evaluated
and the highest task score is selected. The mean is computed only after this
per-instance selection. A run with fewer than 10 unique final members is
rejected instead of being mislabeled Oracle@10.

The GUI training path invokes the same OOD Oracle@10 evaluator automatically
after a run completes. Results are written under
`<run>/eval/ood_oracle_at_10/`.
