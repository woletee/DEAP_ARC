# DEAP_ARC

Genetic Programming-based program synthesis for ARC-AGI tasks using the DEAP evolutionary computation framework.

## Repository Structure

```
DEAP_ARC/
│
├── training/                   # ARC-AGI task JSON files
│
├── trees/                      # Saved GP tree outputs per task
│
├── fitness_plots/              # Fitness history plots generated during evolution
│
├── benchmarks/                 # Benchmark evaluation scripts and results
│
├── tools/                      # Custom DEAP operators and utilities
│   ├── __init__.py
│   ├── mutation.py             # Standard GP mutation operators
│   └── semantic_mutation.py   # LLM-augmented semantic mutation operator
│
├── main.py                     # Entry point — runs GP over all training tasks
├── arc_gp.py                   # GP setup: pset, toolbox, fitness, evolution loop
├── hodel_dsl.py                # DSL primitives for ARC grid manipulation
├── ice_cuber_dsl.py            # Alternative DSL primitives
├── mistral.py                  # Mistral LLM API interface
├── plot_utils.py               # Fitness history and tree visualisation utilities
├── base.py                     # DEAP base fitness and individual definitions
├── creator.py                  # DEAP creator configuration
├── algorithms.py               # Evolutionary algorithm implementations
├── gp.py                       # GP tree operations and primitive set utilities
├── cma.py                      # CMA-ES strategy integration
└── __init__.py
```

## Requirements

```
pip install deap numpy matplotlib requests
```

## Usage

Run GP over all tasks in the `training/` folder:

```bash
python main.py
```

Available arguments:

| Argument | Short | Default | Description |
|---|---|---|---|
| `--max_height` | `-H` | 70 | Maximum GP tree height |
| `--cx_rate` | `-C` | 0.5 | Initial crossover probability |
| `--mut_rate` | `-M` | 0.4 | Initial mutation probability |
| `--generations` | `-G` | 100 | Number of generations |
| `--g_plateau` | `-P` | 5 | Stagnant generations before LLM trigger |
| `--n_cand` | `-K` | 3 | LLM candidate programs per elite |
| `--n_elites` | `-E` | 3 | Number of elites shown to LLM |

Example:

```bash
python main.py --generations 200 --max_height 50 --cx_rate 0.4
```

## Environment Variables

```bash
export MISTRAL_API_KEY=your_key_here   # Linux/Mac
$env:MISTRAL_API_KEY="your_key_here"   # Windows PowerShell
```

## Output

After running, the following are generated:

- `tasks_eval_results_summary.json` — per-task results showing best program and whether a solution was found
- `fitness_plots/` — fitness history plots per task
- `trees/` — saved GP tree visualisations and outputs per task
