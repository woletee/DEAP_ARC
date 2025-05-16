import numpy as np
import sys, os, json, glob
from deap import base, creator, gp, tools, algorithms
import hodel_dsl as dsl
import sys
sys.setrecursionlimit(10000)
from typing import (
    List,
    Union,
    Tuple,
    Any,
    Container,
    FrozenSet,
    Iterable
)
from collections.abc import Callable

# Optional wrapper types
class GridList(list):
    """Used by some DSL primitives that return lists of grids."""
    pass

class Grid:
    """A lightweight wrapper around a tuple-of-tuples."""
    def __init__(self, grid, position=(0, 0)):
        arr = np.array(grid)
        self.grid = tuple(tuple(row) for row in arr)
        self.position = position

    @property
    def size(self):
        return (len(self.grid), len(self.grid[0])) if self.grid else (0, 0)

    def newgrid(self, grid, position=None):
        if position is None:
            position = self.position
        return Grid(grid, position)

    def count(self):
        return sum(v != dsl.ZERO for row in self.grid for v in row)

Colour = int


# 2) Build your PSET over Python-native tuple type
pset = gp.PrimitiveSetTyped("MAIN", [tuple], tuple)

# --- Terminals returning callables ---
pset.addTerminal(dsl.contained, Callable, name="contained")
pset.addTerminal(dsl.palette,   Callable, name="palette")
pset.addTerminal(dsl.ulcorner,  Callable, name="ulcorner")
pset.addTerminal(dsl.sfilter,   Callable, name="sfilter")
pset.addTerminal(dsl.center,    Callable, name="center")
pset.addTerminal(dsl.shift,     Callable, name="shift")
pset.addTerminal(dsl.identity,  Callable, name="identity")

# --- Grid primitives ---
pset.addPrimitive(dsl.vconcat,    [tuple, tuple],     tuple, name="vconcat")
pset.addPrimitive(dsl.vmirror,    [tuple],            tuple, name="vmirror")
pset.addPrimitive(dsl.hmirror,    [tuple],            tuple, name="hmirror")
pset.addPrimitive(dsl.bottomhalf, [tuple],            tuple, name="bottomhalf")

# --- Object‐set and tuple primitives ---
pset.addPrimitive(dsl.objects,  [tuple, bool, bool, bool], frozenset, name="objects")
pset.addPrimitive(dsl.astuple,  [int, int],                   tuple,     name="astuple")
pset.addPrimitive(dsl.initset,  [object],                     frozenset, name="initset")
pset.addPrimitive(dsl.insert,   [object, frozenset],          frozenset, name="insert")

# --- Binder and composition primitives ---
pset.addPrimitive(dsl.lbind,    [Callable, object], Callable, name="lbind")
pset.addPrimitive(dsl.compose,  [Callable, Callable], Callable, name="compose")
pset.addPrimitive(dsl.rbind,    [Callable, object], Callable, name="rbind")
# fork: combines two unary functions via an outer binary

from typing import Callable

pset.addPrimitive(dsl.add,        [int, int],               int,       name="add_int")
pset.addPrimitive(dsl.add,        [tuple, tuple],           tuple,     name="add_tuple")
pset.addPrimitive(dsl.argmax,     [object, Callable],       object,    name="argmax")
pset.addPrimitive(dsl.argmin,     [frozenset, Callable],    object,    name="argmin")
pset.addPrimitive(dsl.canvas,     [int, tuple],             tuple,     name="canvas")
from collections.abc import Callable

pset.addPrimitive(dsl.leastcolor,
                  [tuple],
                  int,
                  name="leastcolor")
pset.addPrimitive(dsl.chain,
                  [Callable, Callable, Callable],
                  Callable,
                  name="chain")

pset.addPrimitive(dsl.combine,    [frozenset, frozenset],   frozenset, name="combine")
pset.addPrimitive(dsl.colorcount, [tuple, int],             int,       name="colorcount")
pset.addPrimitive(dsl.crop,       [tuple, tuple, tuple],    tuple,     name="crop")
pset.addPrimitive(dsl.delta,      [frozenset],              frozenset, name="delta")
pset.addPrimitive(dsl.divide,     [int, int],               int,       name="divide_int")
pset.addPrimitive(dsl.divide,     [tuple, tuple],           tuple,     name="divide_tuple")
pset.addPrimitive(dsl.downscale,  [tuple, int],             tuple,     name="downscale")
pset.addPrimitive(dsl.fill,       [tuple, int, frozenset],  tuple,     name="fill")
pset.addPrimitive(dsl.first,      [object],                 object,    name="first")
pset.addPrimitive(dsl.fork,       [Callable, Callable, Callable], Callable, name="fork")
pset.addPrimitive(dsl.hsplit,     [tuple, int],             tuple,     name="hsplit")
pset.addPrimitive(dsl.identity,   [object],                 object,    name="identity")
pset.addPrimitive(dsl.mapply,     [Callable, frozenset],    frozenset, name="mapply")
pset.addPrimitive(dsl.move,       [tuple, frozenset, tuple], tuple,    name="move")
pset.addPrimitive(dsl.mostcolor,  [tuple],                  int,       name="mostcolor_grid")
pset.addPrimitive(dsl.mostcolor,  [frozenset],              int,       name="mostcolor_objs")
pset.addPrimitive(dsl.multiply,   [int, int],               int,       name="multiply_int")
pset.addPrimitive(dsl.multiply,   [tuple, tuple],           tuple,     name="multiply_tuple")
pset.addPrimitive(dsl.ofcolor,    [tuple, int],             frozenset, name="ofcolor")
pset.addPrimitive(dsl.paint,      [tuple, frozenset],       tuple,     name="paint")
pset.addPrimitive(dsl.remove,     [object, object],         object,    name="remove")
pset.addPrimitive(dsl.remove,     [object, frozenset],      frozenset, name="remove")
pset.addPrimitive(dsl.replace,    [tuple, int, int],        tuple,     name="replace")
pset.addPrimitive(dsl.subtract,   [int, int],               int,       name="subtract_int")
pset.addPrimitive(dsl.subtract,   [tuple, tuple],           tuple,     name="subtract_tuple")
pset.addPrimitive(dsl.tojvec,     [int],                    tuple,     name="tojvec")
pset.addPrimitive(dsl.sfilter,    [frozenset, Callable],    frozenset, name="sfilter")

pset.addPrimitive(dsl.branch,         [bool, object, object],    object,    name="branch")
pset.addPrimitive(dsl.colorfilter,    [frozenset, int],           frozenset, name="colorfilter")
pset.addPrimitive(dsl.compress,       [tuple],                    tuple,     name="compress")
pset.addPrimitive(dsl.crement,        [int],                      int,       name="crement_int")
pset.addPrimitive(dsl.crement,        [tuple],                    tuple,     name="crement_tuple")
pset.addPrimitive(dsl.decrement,      [int],                      int,       name="decrement_int")
pset.addPrimitive(dsl.decrement,      [tuple],                    tuple,     name="decrement_tuple")
pset.addPrimitive(dsl.either,         [bool, bool],               bool,      name="either")
pset.addPrimitive(dsl.extract,        [frozenset, Callable],      object,    name="extract")
pset.addPrimitive(dsl.first,          [frozenset],                object,    name="first")
pset.addPrimitive(dsl.frontiers,      [tuple],                    frozenset, name="frontiers")
pset.addPrimitive(dsl.hperiod,        [frozenset],                int,       name="hperiod")
pset.addPrimitive(dsl.hupscale,       [tuple, int],               tuple,     name="hupscale")
pset.addPrimitive(dsl.increment,      [int],                      int,       name="increment_int")
pset.addPrimitive(dsl.increment,      [tuple],                    tuple,     name="increment_tuple")
pset.addPrimitive(dsl.last,           [frozenset],                object,    name="last")
pset.addPrimitive(dsl.mfilter,        [frozenset, Callable],      frozenset, name="mfilter")
pset.addPrimitive(dsl.pair,           [tuple, tuple],             tuple,     name="pair")
pset.addPrimitive(dsl.portrait,       [tuple],                    bool,      name="portrait")
pset.addPrimitive(dsl.positive,       [int],                      bool,      name="positive")
pset.addPrimitive(dsl.shape,          [tuple],                    tuple,     name="shape")
pset.addPrimitive(dsl.sign,           [int],                      int,       name="sign_int")
pset.addPrimitive(dsl.sign,           [tuple],                    tuple,     name="sign_tuple")
pset.addPrimitive(dsl.sizefilter,     [frozenset, int],           frozenset, name="sizefilter")
pset.addPrimitive(dsl.sfilter,        [frozenset, Callable],      frozenset, name="sfilter")
pset.addPrimitive(dsl.totuple,        [frozenset],                tuple,     name="totuple")
pset.addPrimitive(dsl.underfill,      [tuple, int, frozenset],    tuple,     name="underfill")
pset.addPrimitive(dsl.underpaint,     [tuple, frozenset],         tuple,     name="underpaint")
pset.addPrimitive(dsl.vperiod,        [frozenset],                int,       name="vperiod")
pset.addPrimitive(dsl.vupscale,       [tuple, int],               tuple,     name="vupscale")
pset.addPrimitive(dsl.width,          [tuple],                    int,       name="width")
pset.addPrimitive(dsl.width,          [frozenset],                int,       name="width_obj")

# === Indices Transformations ===
pset.addPrimitive(dsl.asindices,    [tuple],               frozenset, name="asindices")
pset.addPrimitive(dsl.occurrences,  [tuple, frozenset],    frozenset, name="occurrences")
pset.addPrimitive(dsl.trim,         [tuple],               tuple,     name="trim")

# === Grid Editing ===
pset.addPrimitive(dsl.cover,        [tuple, frozenset],    tuple,     name="cover")

pset.addPrimitive(dsl.switch,       [tuple, int, int],     tuple,     name="switch")

# === Patch Geometry ===
pset.addPrimitive(dsl.center,       [frozenset],           tuple,     name="center")
pset.addPrimitive(dsl.position,     [frozenset, frozenset],tuple,    name="position")
pset.addPrimitive(dsl.inbox,        [frozenset],           frozenset, name="inbox")
pset.addPrimitive(dsl.outbox,       [frozenset],           frozenset, name="outbox")
pset.addPrimitive(dsl.box,          [frozenset],           frozenset, name="box")
pset.addPrimitive(dsl.gravitate,    [frozenset, frozenset],tuple,    name="gravitate")

# === Container Merging ===
pset.addPrimitive(dsl.merge,        [frozenset],           frozenset, name="merge")

# === Numeric Aggregation ===
pset.addPrimitive(dsl.maximum,      [frozenset],           int,       name="maximum")
pset.addPrimitive(dsl.minimum,      [frozenset],           int,       name="minimum")
pset.addPrimitive(dsl.valmax,       [frozenset, Callable], int,       name="valmax")
pset.addPrimitive(dsl.valmin,       [frozenset, Callable], int,       name="valmin")

# === Ray Casting ===
pset.addPrimitive(dsl.shoot,        [tuple, tuple],        frozenset, name="shoot")

Int = int
Bool = bool
Num = (int, tuple)
AnyObj = object
Container = (tuple, frozenset)
Func = Callable
class Numerical:
    pass

# Begin registration of missing primitives
pset.addPrimitive(dsl.intersection, [frozenset, frozenset], frozenset, name="intersection")
pset.addPrimitive(dsl.difference, [frozenset, frozenset], frozenset, name="difference")
pset.addPrimitive(dsl.dedupe, [tuple], tuple, name="dedupe")

pset.addPrimitive(dsl.repeat, [AnyObj, Int], tuple, name="repeat")
pset.addPrimitive(dsl.greater, [Int, Int], Bool, name="greater")
pset.addPrimitive(dsl.even, [Int], Bool, name="even")

pset.addPrimitive(dsl.flip, [Bool], Bool, name="flip")
pset.addPrimitive(dsl.equality, [AnyObj, AnyObj], Bool, name="equality")
pset.addPrimitive(dsl.both, [Bool, Bool], Bool, name="both")
pset.addPrimitive(dsl.interval, [Int, Int, Int], tuple, name="interval")
pset.addPrimitive(dsl.matcher, [Func, AnyObj], Func, name="matcher")
pset.addPrimitive(dsl.power, [Func, Int], Func, name="power")

pset.addPrimitive(dsl.normalize, [frozenset], frozenset, name="normalize")
pset.addPrimitive(dsl.dneighbors, [tuple], frozenset, name="dneighbors")
pset.addPrimitive(dsl.ineighbors, [tuple], frozenset, name="ineighbors")
pset.addPrimitive(dsl.neighbors, [tuple], frozenset, name="neighbors")
pset.addPrimitive(dsl.partition, [tuple], frozenset, name="partition")
pset.addPrimitive(dsl.fgpartition, [tuple], frozenset, name="fgpartition")
pset.addPrimitive(dsl.uppermost, [frozenset], Int, name="uppermost")
pset.addPrimitive(dsl.lowermost, [frozenset], Int, name="lowermost")

pset.addPrimitive(dsl.leftmost, [frozenset], Int, name="leftmost")
pset.addPrimitive(dsl.rightmost, [frozenset], Int, name="rightmost")
pset.addPrimitive(dsl.square, [tuple], Bool, name="square")
pset.addPrimitive(dsl.vline, [frozenset], Bool, name="vline")
pset.addPrimitive(dsl.hline, [frozenset], Bool, name="hline")
pset.addPrimitive(dsl.hmatching, [frozenset, frozenset], Bool, name="hmatching")
pset.addPrimitive(dsl.vmatching, [frozenset, frozenset], Bool, name="vmatching")
pset.addPrimitive(dsl.manhattan, [frozenset, frozenset], Int, name="manhattan")
pset.addPrimitive(dsl.adjacent, [frozenset, frozenset], Bool, name="adjacent")
pset.addPrimitive(dsl.bordering, [frozenset, tuple], Bool, name="bordering")
pset.addPrimitive(dsl.centerofmass, [frozenset], tuple, name="centerofmass")
pset.addPrimitive(dsl.numcolors, [tuple], Int, name="numcolors")
pset.addPrimitive(dsl.color, [frozenset], Int, name="color")
pset.addPrimitive(dsl.toobject, [frozenset, tuple], frozenset, name="toobject")
pset.addPrimitive(dsl.asobject, [tuple], frozenset, name="asobject")
pset.addPrimitive(dsl.rot90, [tuple], tuple, name="rot90")
pset.addPrimitive(dsl.rot180, [tuple], tuple, name="rot180")
pset.addPrimitive(dsl.rot270, [tuple], tuple, name="rot270")
pset.addPrimitive(dsl.dmirror, [tuple], tuple, name="dmirror")
pset.addPrimitive(dsl.cmirror, [tuple], tuple, name="cmirror")
pset.addPrimitive(dsl.subgrid, [frozenset, tuple], tuple, name="subgrid")
pset.addPrimitive(dsl.vsplit, [tuple, Int], tuple, name="vsplit")
pset.addPrimitive(dsl.hconcat, [tuple, tuple], tuple, name="hconcat")
pset.addPrimitive(dsl.cellwise, [tuple, tuple, Int], tuple, name="cellwise")
pset.addPrimitive(dsl.tophalf, [tuple], tuple, name="tophalf")
pset.addPrimitive(dsl.lefthalf, [tuple], tuple, name="lefthalf")
pset.addPrimitive(dsl.righthalf, [tuple], tuple, name="righthalf")
pset.addPrimitive(dsl.vfrontier, [tuple], frozenset, name="vfrontier")
pset.addPrimitive(dsl.hfrontier, [tuple], frozenset, name="hfrontier")
pset.addPrimitive(dsl.backdrop, [frozenset], frozenset, name="backdrop")
pset.addPrimitive(dsl.toindices, [frozenset], frozenset, name="toindices")
pset.addPrimitive(dsl.recolor, [Int, frozenset], frozenset, name="recolor")
pset.addPrimitive(dsl.index, [tuple, tuple], Int, name="index")
pset.addPrimitive(
    dsl.upscale,
    [tuple, int],
    tuple,
    name="upscale_grid"
)
# the four corner‐cells of a patch
pset.addPrimitive(
    dsl.corners,              # corners(patch: Patch) -> Indices
    [frozenset],
    frozenset,
    name="corners"
)




# --- Numeric inversion ---
pset.addPrimitive(dsl.invert,   [int],    int,   name="invert_int")
pset.addPrimitive(dsl.invert,   [tuple],  tuple, name="invert_tuple")

# --- Boolean primitive ---
pset.addPrimitive(dsl.contained, [object, object], bool, name="contained_bool")

# --- Empty set terminal ---
pset.addTerminal(frozenset(), frozenset, name="EmptySet")

# --- Integer terminals ---
for name, val in [
    ("ZERO", 0), ("ONE", 1), ("TWO", 2), ("THREE", 3),
    ("NEG_ONE", -1), ("NEG_TWO", -2),
]:
    pset.addTerminal(val, int, name=name)

# --- Boolean terminals ---
pset.addTerminal(False, bool, name="F")
pset.addTerminal(True,  bool, name="T")

# --- Direction‐tuple terminals ---
for name, val in [
    ("DOWN",  (1,0)),
    ("RIGHT", (0,1)),
    ("UP",    (-1,0)),
    ("LEFT",  (0,-1)),
]:
    pset.addTerminal(val, tuple, name=name)

# Ephemeral random-int generator (pickleable)
def rand_int():
    return np.random.randint(0, 10)
pset.addEphemeralConstant("randInt", rand_int, int)




# === TOOLBOX & GP BOILERPLATE ===
toolbox = base.Toolbox()
toolbox.register("compile", gp.compile, pset=pset)
import multiprocessing
import operator


from deap import base, creator, gp

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
def evaluate_task(individual):
    """Evaluate an individual on the global current_task."""
    func = toolbox.compile(expr=individual)
    total = 0
    for ex in current_task["train"]:
        inp = tuple(map(tuple, ex["input"]))
        tgt_arr = np.array(ex["output"])
        try:
            out_arr = np.array(func(inp))
            if out_arr.shape == tgt_arr.shape:
                total += np.sum(out_arr == tgt_arr)
        except Exception:
            pass
    return (total,)

# Register evaluation for multiprocessing
toolbox.register("evaluate", evaluate_task)
# === VARIATION OPERATORS ===
POP_SIZE = 100
INIT_MIN, INIT_MAX = 2, 6
toolbox.register(
    "population",
    tools.initRepeat,
    list,
    lambda: creator.Individual(
        gp.genHalfAndHalf(pset, min_=INIT_MIN, max_=INIT_MAX)
    ),
    n=POP_SIZE
)
def smart_crossover(ind1, ind2):
    matches = []
    for i, n1 in enumerate(ind1):
        for j, n2 in enumerate(ind2):
            if n1.ret == n2.ret:
                if abs(len(ind1) - len(ind2)) < max(len(ind1), len(ind2)) // 2:
                    matches.append((i, j))
    if not matches:
        return ind1, ind2
    i, j = matches[np.random.randint(len(matches))]
    s1, s2 = ind1.searchSubtree(i), ind2.searchSubtree(j)
    ind1[s1], ind2[s2] = ind2[s2], ind1[s1]
    return ind1, ind2

toolbox.register("mate", smart_crossover)

toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

MAX_H0 = 5
def decorate_limits(gen):
    limit = MAX_H0 + gen // 50
    toolbox.decorate("mate", gp.staticLimit(operator.attrgetter("height"), limit))
    toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), limit))

# === SELECTION ===
toolbox.register("select", tools.selTournament, tournsize=5)

# ===  INITIALIZER ===
def _init_task(task):
    """Initializer for worker processes to set the current task."""
    global current_task
    current_task = task

# === MAIN EXECUTION ===
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
            # Initialize population and hall-of-fame
            pop = toolbox.population()
            hof = tools.HallOfFame(5)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("max", np.max)
            # Initial evaluation
            fitnesses = toolbox.map(toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            CX0, MUT0 = 0.5, 0.6
            GENERATIONS = 100
            # compute total cells across all train outputs
            sample_out = np.array(task["train"][0]["output"])
            target = len(task["train"]) * np.prod(sample_out.shape)
            history=[]
            for gen in range(1, GENERATIONS + 1):
                decorate_limits(gen)
                cxpb = CX0 * (1 - gen / GENERATIONS)
                mutpb = MUT0 + (1 - MUT0) * (gen / GENERATIONS)
                if cxpb + mutpb > 1.0:
                    mutpb = 1.0 - cxpb
                # variation
                offspring = algorithms.varOr(
                    pop, toolbox,
                    lambda_=(POP_SIZE - len(hof)),
                    cxpb=cxpb,
                    mutpb=mutpb
                )

                # evaluate offspring
                fitnesses = toolbox.map(toolbox.evaluate, offspring)
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit
                # update hall-of-fame and select next gen
                hof.update(offspring)
                combined = offspring + list(hof)
                pop = tools.selNSGA2(combined, POP_SIZE)
                record = stats.compile(pop)
                history.append(record)
                print(f"Gen {gen} | Max {record['max']} | Avg {record['avg']:.1f}")
                if record['max'] >= target:
                    print(f"Converged at generation {gen}")
                    break
            import matplotlib.pyplot as plt
            generations = list(range(len(history)))
            max_fitness = [record['max'] for record in history]
            avg_fitness = [record['avg'] for record in history]
            plt.figure(figsize=(10, 6))
            plt.plot(generations, max_fitness, label="Best Fitness", marker="")
            plt.plot(generations, avg_fitness, label="Average Fitness", marker="")
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.title(f"Evolution in Task {task_name}")
            plt.legend()
            plt.grid(True)
            plt.show()
            # Test best
            best = hof[0]
            func = toolbox.compile(expr=best)
            correct = True
            for tc in task["test"]:
                inp = tuple(map(tuple, tc["input"]))
                tgt_arr = np.array(tc["output"])
                try:
                    out_arr = np.array(func(inp))
                    if not np.array_equal(out_arr, tgt_arr):
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

    with open("tasks_eval_results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("All tasks processed. Results saved.")
if __name__ == "__main__":
    main()
