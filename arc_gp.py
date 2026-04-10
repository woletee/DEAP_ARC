#!/usr/bin/env python3
import argparse
import numpy as np
import sys, os, json, glob
import deap.creator as creator
import deap.gp      as gp
import deap.base    as base
import deap.algorithms as algorithms
import tools
import deap.cma     as cma
import hodel_dsl as dsl
import multiprocessing
import operator
import plot_utils
from plot_utils import plot_history, save_tree_and_outputs_dot
from tools.semantic_mutation import SemanticMutationOperator

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser(description="Run genetic programming over all tasks in ./training/")
parser.add_argument("--max_height",  "-H", type=int,   default=70)
parser.add_argument("--cx_rate",     "-C", type=float, default=0.5)
parser.add_argument("--mut_rate",    "-M", type=float, default=0.4)
parser.add_argument("--generations", "-G", type=int,   default=100)
parser.add_argument("--g_plateau",   "-P", type=int,   default=5)
parser.add_argument("--n_cand",      "-K", type=int,   default=3)
parser.add_argument("--n_elites",    "-E", type=int,   default=3)
args = parser.parse_args()

MAX_HEIGHT  = args.max_height
CX0         = args.cx_rate
MUT0        = args.mut_rate
GENERATIONS = args.generations

for name in ("FitnessMax", "Individual"):
    if name in creator.__dict__:
        delattr(creator, name)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

def identity(x):
    return x

pset = gp.PrimitiveSetTyped("MAIN", [tuple], tuple)
pset.addPrimitive(dsl.hline,     [frozenset],              bool,      name="hline_pred")
pset.addTerminal(identity,        object,                             name="IDENTITY")
pset.addPrimitive(dsl.objects,   [tuple, bool, bool, bool], frozenset, name="objects")
pset.addTerminal(frozenset(),     frozenset,                          name="EmptySet")
pset.addPrimitive(dsl.replace,   [tuple, int, int],         tuple,    name="replace")
pset.addPrimitive(dsl.leastcolor,[tuple],                   int,      name="leastcolor")
pset.addPrimitive(dsl.fill,      [tuple, int, frozenset],   tuple,    name="fill")
pset.addPrimitive(dsl.vmirror,   [tuple],                   tuple,    name="vmirror")
pset.addPrimitive(dsl.lefthalf,  [tuple],                   tuple,    name="lefthalf")
pset.addPrimitive(dsl.righthalf, [tuple],                   tuple,    name="righthalf")
pset.addPrimitive(dsl.cellwise,  [tuple, tuple, int],       tuple,    name="cellwise")

for i in range(10):
    pset.addTerminal(i, int, name=str(i))
pset.addTerminal(True,  bool, name="T")
pset.addTerminal(False, bool, name="F")

def rand_int():
    return np.random.randint(0, 10)
pset.addEphemeralConstant("randInt", rand_int, int)

toolbox = base.Toolbox()
toolbox.register("compile",  gp.compile, pset=pset)
toolbox.register("select",   tools.selTournament, tournsize=5)
toolbox.register("mate",     gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
toolbox.register("mutate",   gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate",     gp.staticLimit(operator.attrgetter("height"), MAX_HEIGHT))
toolbox.decorate("mutate",   gp.staticLimit(operator.attrgetter("height"), MAX_HEIGHT))

POP_SIZE       = 100
INIT_MIN, INIT_MAX = 2, 8

toolbox.register("population", tools.initRepeat, list,
                 lambda: creator.Individual(gp.genHalfAndHalf(pset, min_=INIT_MIN, max_=INIT_MAX)),
                 n=POP_SIZE)

def evaluate_task(individual):
    func  = toolbox.compile(expr=individual)
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

toolbox.register("evaluate", evaluate_task)

def _init_task(task):
    global current_task
    current_task = task


def main():
    training_folder = "./training/"
    task_files      = glob.glob(os.path.join(training_folder, "*.json"))
    results         = []

    for task_file in task_files:
        task_name = os.path.basename(task_file)
        print(f"Processing {task_name}")
        with open(task_file) as f:
            task = json.load(f)

        with multiprocessing.Pool(initializer=_init_task, initargs=(task,)) as pool:
            toolbox.register("map", pool.map)

            pop  = toolbox.population()
            hof  = tools.HallOfFame(5)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("max", np.max)

            fitnesses = toolbox.map(toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            sem_mut = SemanticMutationOperator(
                pset=pset,
                toolbox=toolbox,
                task=task,
                g_plateau=args.g_plateau,
                n_cand=args.n_cand,
                n_elites=args.n_elites,
                verbose=True,
            )

            history = []
            for gen in range(1, GENERATIONS + 1):
                cxpb  = CX0 * (1 - gen / GENERATIONS)
                mutpb = MUT0 + (1 - MUT0) * (gen / GENERATIONS)
                if cxpb + mutpb > 1.0:
                    mutpb = 1.0 - cxpb

                offspring = algorithms.varOr(pop, toolbox,
                                             lambda_=(POP_SIZE - len(hof)),
                                             cxpb=cxpb, mutpb=mutpb)

                offspring, llm_triggered = sem_mut.step(pop, gen, offspring)

                fitnesses = toolbox.map(toolbox.evaluate, offspring)
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit

                hof.update(offspring)
                pop = toolbox.select(offspring + list(hof), k=POP_SIZE)

                record = stats.compile(pop)
                history.append(record)
                print(f"Gen {gen} | Max {record['max']} | Avg {record['avg']:.1f}"
                      + (" | LLM" if llm_triggered else ""))

            plot_utils.plot_history(history, task_name)
            sem_mut.summary()

            best  = hof[0]
            func  = toolbox.compile(expr=best)
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

            save_tree_and_outputs_dot(task_name, best, func, task["test"])
            results.append({
                "task_name":      task_name,
                "best_program":   str(best),
                "solution_found": correct
            })
            print(f"{task_name} -> {'Correct-Solution Found' if correct else 'Failed-No Solution Found'}")

    with open("tasks_eval_results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done. Results saved.")


if __name__ == "__main__":
    main()
