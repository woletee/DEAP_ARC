#!/usr/bin/env python3
import argparse
from typing import Callable
import numpy as np
import sys, os, json, glob
import deap.creator as creator
import deap.gp      as gp
import deap.base   as base
import deap.algorithms as algorithms
import tools
import deap.cma    as cma
import hodel_dsl as dsl   
import multiprocessing
import operator
import plot_utils
sys.setrecursionlimit(10000)

# === PARSE COMMAND-LINE ARGS ===
parser = argparse.ArgumentParser(
    description="Run genetic programming over all tasks in ./training/"
)
parser.add_argument(
    "--max_height", "-H",
    type=int,
    default=70,
    help="maximum tree height"
)
parser.add_argument(
    "--cx_rate", "-C",
    type=float,
    default=0.5,
    help="initial crossover probability"
)
parser.add_argument(
    "--mut_rate", "-M",
    type=float,
    default=0.4,
    help="initial mutation probability"
)
parser.add_argument(
    "--generations", "-G",
    type=int,
    default=100,
    help="number of generations"
)
args = parser.parse_args()

# Override constants from args
MAX_HEIGHT  = args.max_height
CX0         = args.cx_rate
MUT0        = args.mut_rate
GENERATIONS = args.generations

# === 1) GP SETUP ===

# Clean up any existing DEAP definitions
for name in ("FitnessMax", "Individual"):
    if name in creator.__dict__:
        delattr(creator, name)

# 1a) Fitness: maximize a single scalar
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 1b) Primitive set typing
pset = gp.PrimitiveSetTyped("MAIN", [tuple], tuple)

def identity(x):
    return x

# 2) Add primitives & terminals from your DSL
pset.addPrimitive(dsl.hline,      [frozenset],                bool,      name="hline_pred")
pset.addTerminal(identity, object, name="IDENTITY")
pset.addPrimitive(dsl.objects,  [tuple, bool, bool, bool], frozenset, name="objects")
pset.addTerminal(frozenset(), frozenset, name="EmptySet")
pset.addPrimitive(dsl.replace,    [tuple, int, int],        tuple,     name="replace")
pset.addPrimitive(dsl.leastcolor, [tuple],                  int,       name="leastcolor")
pset.addPrimitive(dsl.fill,       [tuple, int, frozenset],  tuple,     name="fill")
pset.addPrimitive(dsl.vmirror,    [tuple],                  tuple,     name="vmirror")
pset.addPrimitive(dsl.lefthalf,   [tuple],                  tuple,     name="lefthalf")
pset.addPrimitive(dsl.righthalf,  [tuple],                  tuple,     name="righthalf")
pset.addPrimitive(dsl.cellwise,   [tuple, tuple, int],      tuple,     name="cellwise")

# Add integer and boolean constants
for i in range(10):
    pset.addTerminal(i, int, name=str(i))
pset.addTerminal(True,  bool, name="T")
pset.addTerminal(False, bool, name="F")

# Ephemeral random int generator
def rand_int():
    return np.random.randint(0, 10)
pset.addEphemeralConstant("randInt", rand_int, int)





"""
DEAP Toolbox Configuration
──────────────────────────
This block sets up DEAP's central registry of genetic programming operations.
Instead of hard-coding function calls throughout script, we register each
operator under a string key. Later, we can invoke them uniformly via toolbox.NAME().

- compile:   Turns a GP tree into a runnable function by wiring up  DSL primitives.
- select:    Chooses parents via tournament selection (tournsize=5).
- mate:      Performs one-point subtree crossover on matching node types.
- expr_mut:  Generates a “full” tree up to depth 3 for use in mutation.
- mutate:    Replaces a random subtree with a new one from expr_mut, preserving types.
- decorate:  Wraps mate and mutate to enforce a maximum tree height (Bloat Control).

* just call toolbox.population(), toolbox.compile(), toolbox.select(),
toolbox.mate(), and toolbox.mutate() without worrying about their inner workings.
"""
toolbox = base.Toolbox()
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate",   gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate",   gp.staticLimit(operator.attrgetter("height"), MAX_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), MAX_HEIGHT))



# Population init parameters
POP_SIZE = 100
INIT_MIN, INIT_MAX = 2, 8
"""
Population Initialization
─────────────────────────
Registers how to build the initial population of candidate programs.
- initRepeat: repeatedly calls the provided generator function to fill a list.
- The generator: creates a new Individual by growing a half and half GP tree
  (mixing “full” and “grow” methods-refer to the deap gp.py) with depth between INIT_MIN and INIT_MAX.
- n=POP_SIZE: produces exactly POP_SIZE individuals for the starting population.
"""
toolbox.register("population", tools.initRepeat, list, 
                 lambda: creator.Individual(gp.genHalfAndHalf(pset, min_=INIT_MIN, max_=INIT_MAX)), n=POP_SIZE)


"""
    DEAP_custom “evaluate_task” function:
    ─────────────────────
    1. Compiles a GP tree into an executable Python function.
    2. Runs it on each training example.
    3. Counts the total number of correctly predicted cells.
    4. Returns that count as a 1.tuple, matching the single-objective FitnessMax.

    This differs from the `creator.FitnessMax` class itself—`FitnessMax` only *stores*
    fitness values and knows how to compare them (higher is better), whereas
    `evaluate_task` actually *computes* the raw score that gets stuffed into the `.fitness`
    attribute by DEAP's engine.

    They benefit each other because:
     - `evaluate_task` produces a numeric performance measure.
     - 'FitnessMax' wraps that measure and lets DEAP sort, compare, and select individuals
       based on it.
    """
def evaluate_task(individual):
    func = toolbox.compile(expr=individual)
    total = 0
    for ex in current_task["train"]:
        inp = tuple(map(tuple, ex["input"]))
        tgt = np.array(ex["output"])
        try:
            out = np.array(func(inp))
            if out.shape == tgt.shape:
                total += np.sum(out == tgt)
        except Exception:
            pass
    return (total,)

"""
Register the fitness evaluation function that scores each individual via evaluate_task
"""
toolbox.register("evaluate", evaluate_task)




"""""
used during parallel evaluation
"""""
def _init_task(task):
    global current_task
    current_task = task

# === 5) MAIN LOOP ===
def main():
    training_folder = "./training/"
    task_files = glob.glob(os.path.join(training_folder, "*.json"))
    results = []

    for task_file in task_files:
        task_name = os.path.basename(task_file)
        print(f"Processing {task_name}")
        with open(task_file) as f:
            task = json.load(f)

        with multiprocessing.Pool(initializer=_init_task, initargs=(task,)) as pool:
            toolbox.register("map", pool.map)
            pop = toolbox.population()
            hof = tools.HallOfFame(5)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("max", np.max)

            # initial evaluation
            fitnesses = toolbox.map(toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            history = []
            for gen in range(1, GENERATIONS + 1):
                # dynamic cx/mut scheduling
                cxpb = CX0 * (1 - gen/GENERATIONS)
                mutpb = MUT0 + (1 - MUT0)*(gen/GENERATIONS)
                if cxpb + mutpb > 1.0:
                    mutpb = 1.0 - cxpb

                offspring = algorithms.varOr(pop, toolbox,
                                             lambda_=(POP_SIZE - len(hof)),
                                             cxpb=cxpb, mutpb=mutpb)

                fitnesses = toolbox.map(toolbox.evaluate, offspring)
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit

                hof.update(offspring)
                pop = toolbox.select(offspring + list(hof), k=POP_SIZE)

                record = stats.compile(pop)
                history.append(record)
                print(f"Gen {gen} | Max {record['max']} | Avg {record['avg']:.1f}")

            plot_utils.plot_history(history, task_name)

            # test best individual
            best = hof[0]
            func = toolbox.compile(expr=best)
            correct = True
            for tc in task["test"]:
                inp = tuple(map(tuple, tc["input"]))
                tgt = np.array(tc["output"])
                try:
                    out = np.array(func(inp))
                    if not np.array_equal(out, tgt):
                        correct = False
                        break
                except Exception:
                    correct = False
                    break

            results.append({
                "task_name": task_name,
                "best_program": str(best),
                "solution_found": correct
            })
            print(f"{task_name} -> {'✔' if correct else '✘'}")

    # save summary
    with open("tasks_eval_results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done. Results saved.")

if __name__ == "__main__":
    main()
