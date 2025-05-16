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

"""
1) PARSE COMMAND-LINE ARGUMENTS
────────────────────────────────
Uses Python's argparse to allow users to override default GP hyperparameters:
  • --max_height / -H : Maximum allowed tree height (default 70)
  • --cx_rate    / -C : Initial crossover probability (default 0.5)
  • --mut_rate   / -M : Initial mutation probability (default 0.4)
  • --generations / -G : Number of generations to evolve (default 100)
Parsed values override the hard-coded constants in the script.
"""
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

""" constants from args
"""
MAX_HEIGHT  = args.max_height
CX0         = args.cx_rate
MUT0        = args.mut_rate
GENERATIONS = args.generations

# === 1) GP SETUP ===

# Clean up any existing DEAP definitions
for name in ("FitnessMax", "Individual"):
    if name in creator.__dict__:
        delattr(creator, name)



"""
 DEFINE FITNESS & INDIVIDUAL
─────────────────────────────────
• creator.create("FitnessMax", base.Fitness, weights=(1.0,)) (refer to creator.py ):
  Defines a new fitness class where higher scalar values are better (single-objective maximization).
• creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)(refer to gp.py):
  Defines an Individual type as a GP tree whose .fitness attribute uses the FitnessMax class.
This hooks DEAP's selection and comparison mechanisms to the problem of maximizing correct outputs.
"""
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)






def identity(x):
    return x
"""
2) Add primitives & terminals from hodel_dsl.py or ice_cuber.py
─────────────────────────────────────────
• Each call to `addPrimitive(python_function, input_types, output_type, string_name)` registers a DSL function
  as a GP node, specifying exactly what input types it accepts and what type it returns.
• `addTerminal(value, value_type, string_name)` registers constant values or zero-ary functions.
• This enforces **strict typing** at tree-construction time: you can only connect subtrees
  whose output type matches the primitive's expected input type.
By assembling these DSL operations in the primitive set, GP can build complex,
type-safe programs that manipulate grids, colors, and objects per the task logic.

Make sure you know the name of the DSL , the input type, output type when we add the primitves into the this set
Then we can as many DSLs as we have 
"""
pset = gp.PrimitiveSetTyped("MAIN", [tuple], tuple)
pset.addPrimitive(dsl.hline,[frozenset],bool,name="hline_pred")
pset.addTerminal(identity, object, name="IDENTITY")
pset.addPrimitive(dsl.objects,  [tuple, bool, bool, bool], frozenset, name="objects")
pset.addTerminal(frozenset(), frozenset, name="EmptySet")
pset.addPrimitive(dsl.replace, [tuple, int, int],tuple,name="replace")
pset.addPrimitive(dsl.leastcolor,[tuple],int, name="leastcolor")
pset.addPrimitive(dsl.fill, [tuple, int, frozenset],  tuple, name="fill")
pset.addPrimitive(dsl.vmirror,[tuple],tuple, name="vmirror")
pset.addPrimitive(dsl.lefthalf, [tuple],tuple, name="lefthalf")
pset.addPrimitive(dsl.righthalf,[tuple],tuple, name="righthalf")
pset.addPrimitive(dsl.cellwise,[tuple, tuple, int],tuple,name="cellwise")



"""
 Add integer and boolean constants, plus an ephemeral random-int generator
────────────────────────────────────────────────────────────────────────
• addTerminal(value, type, name) registers literal constants-in the case of ARC -could be color (0-9) that GP trees can use.
   Here integers 0-9 are added as terminals of type int, named “0” through “9”.
   also the Boolean constants `True` and False of type `bool`, named “T” and “F”.
• addEphemeralConstant(name, generator, type) registers a zero-argument “constant”
  whose value is re-drawn each time a new tree is created(refer to gp.py).
   randInt produces a random integer in [0,10) for radmoness- during the mutation and cross-over process.
"""
for i in range(10):
    pset.addTerminal(i, int, name=str(i))
pset.addTerminal(True,  bool, name="T")
pset.addTerminal(False, bool, name="F")
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


def main():

    """
    1) TASK LOADING & WORKER INITIALIZATION
    ────────────────────────────────────────────
    • Discover all JSON task files in the training folder.
    • Prepare a results list to collect outcomes per task.
    • Initialize a multiprocessing pool per task, setting `current_task`.
    """
      
    training_folder = "./training/"
    task_files = glob.glob(os.path.join(training_folder, "*.json"))
    results = []

    for task_file in task_files:
        task_name = os.path.basename(task_file)
        print(f"Processing {task_name}")
        with open(task_file) as f:
            task = json.load(f)

        with multiprocessing.Pool(initializer=_init_task, initargs=(task,)) as pool:
            """
            2) EVOLUTIONARY OPTIMIZATION PER TASK
            ────────────────────────────────────────
            • Create initial population, Hall-of-Fame, and statistics trackers.
            • Perform initial fitness evaluation.
            • For each generation:
                Adapt crossover and mutation rates over time.
                Generate offspring via varOr (crossover + mutation).
                Evaluate offspring fitness in parallel.
                Update the Hall-of-Fame (elitism).
                Select the next generation (tournament).
                Record stats and print progress.
            • Plot fitness history.
            """
            toolbox.register("map", pool.map)
            pop = toolbox.population()
            hof = tools.HallOfFame(5)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("max", np.max)
            fitnesses = toolbox.map(toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            history = []
            for gen in range(1, GENERATIONS + 1):
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

            """
            3) POST-EVOLUTION TESTING & RESULTS SAVING
            ──────────────────────────────────────────────
            • Compile the best individual into a function.
            • Test it on held-out examples.
            • Record success and append to the results list.
            """
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
