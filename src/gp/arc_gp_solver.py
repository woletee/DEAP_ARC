import argparse
import numpy as np
import sys, os, json, glob
from deap import base, creator, gp, tools, algorithms
import hodel_dsl as dsl
import sys
import multiprocessing
import operator
from deap import base, creator, gp
import plot_utils
from plot_utils import plot_history, save_tree_and_outputs_dot

sys.setrecursionlimit(10000)
from collections.abc import Callable
parser = argparse.ArgumentParser(
    description="Run genetic programming over ARC"
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
pset=gp.PrimitiveSetTyped("MAIN",[tuple],tuple)
pset.addTerminal(dsl.contained,Callable,name="contained")
pset.addTerminal(dsl.palette,Callable,name="palette")
pset.addTerminal(dsl.ulcorner,Callable,name="ulcorner")
pset.addTerminal(dsl.sfilter,Callable,name="sfilter")
pset.addTerminal(dsl.center,Callable,name="center")
pset.addTerminal(dsl.shift,Callable,name="shift")
pset.addTerminal(dsl.identity,Callable,name="identity")
pset.addPrimitive(dsl.vconcat,[tuple,tuple],tuple,name="vconcat")
pset.addPrimitive(dsl.vmirror,[tuple],tuple,name="vmirror")
pset.addPrimitive(dsl.hmirror,[tuple],tuple,name="hmirror")
pset.addPrimitive(dsl.bottomhalf,[tuple],tuple,name="bottomhalf")
pset.addPrimitive(dsl.objects,[tuple,bool,bool,bool],frozenset,name="objects")
pset.addPrimitive(dsl.astuple,[int,int],tuple,name="astuple")
pset.addPrimitive(dsl.initset,[object],frozenset,name="initset")
pset.addPrimitive(dsl.insert,[object,frozenset],frozenset,name="insert")
pset.addPrimitive(dsl.lbind,[Callable,object],Callable,name="lbind")
pset.addPrimitive(dsl.compose,[Callable,Callable],Callable,name="compose")
pset.addPrimitive(dsl.rbind,[Callable,object],Callable,name="rbind")
pset.addPrimitive(dsl.add,[int,int],int,name="add_int")
pset.addPrimitive(dsl.add,[tuple,tuple],tuple,name="add_tuple")
pset.addPrimitive(dsl.argmax,[object,Callable],object,name="argmax")
pset.addPrimitive(dsl.argmin,[frozenset,Callable],object,name="argmin")
pset.addPrimitive(dsl.canvas,[int,tuple],tuple,name="canvas")
pset.addPrimitive(dsl.leastcolor,
                  [tuple],
                  int,
                  name="leastcolor")
pset.addPrimitive(dsl.chain,
                  [Callable, Callable, Callable],
                  Callable,
                  name="chain")
pset.addPrimitive(dsl.combine,[frozenset,frozenset],frozenset,name="combine")
pset.addPrimitive(dsl.colorcount,[tuple,int],int,name="colorcount")
pset.addPrimitive(dsl.crop,[tuple,tuple,tuple],tuple,name="crop")
pset.addPrimitive(dsl.delta,[frozenset],frozenset,name="delta")
pset.addPrimitive(dsl.divide,[int,int],int,name="divide_int")
pset.addPrimitive(dsl.divide,[tuple,tuple],tuple,name="divide_tuple")
pset.addPrimitive(dsl.downscale,[tuple,int],tuple,name="downscale")
pset.addPrimitive(dsl.fill,[tuple,int,frozenset],tuple,name="fill")
pset.addPrimitive(dsl.first,[object],object,name="first")
pset.addPrimitive(dsl.fork,[Callable,Callable,Callable],Callable,name="fork")
pset.addPrimitive(dsl.hsplit,[tuple,int],tuple,name="hsplit")
pset.addPrimitive(dsl.identity,[object],object,name="identity")
pset.addPrimitive(dsl.mapply,[Callable,frozenset],frozenset,name="mapply")
pset.addPrimitive(dsl.move,[tuple,frozenset,tuple],tuple,name="move")
pset.addPrimitive(dsl.mostcolor,[tuple],int,name="mostcolor_grid")
pset.addPrimitive(dsl.mostcolor,[frozenset],int,name="mostcolor_objs")
pset.addPrimitive(dsl.multiply,[int,int],int,name="multiply_int")
pset.addPrimitive(dsl.multiply,[tuple,tuple],tuple,name="multiply_tuple")
pset.addPrimitive(dsl.ofcolor,[tuple,int],frozenset,name="ofcolor")
pset.addPrimitive(dsl.paint,[tuple,frozenset],tuple,name="paint")
pset.addPrimitive(dsl.remove,[object,object],object,name="remove")
pset.addPrimitive(dsl.remove,[object,frozenset],frozenset,name="remove")
pset.addPrimitive(dsl.replace,[tuple,int,int],tuple,name="replace")
pset.addPrimitive(dsl.subtract,[int,int],int,name="subtract_int")
pset.addPrimitive(dsl.subtract,[tuple,tuple],tuple,name="subtract_tuple")
pset.addPrimitive(dsl.tojvec,[int],tuple,name="tojvec")
pset.addPrimitive(dsl.sfilter,[frozenset,Callable],frozenset,name="sfilter")
pset.addPrimitive(dsl.branch,[bool,object,object],object,name="branch")
pset.addPrimitive(dsl.colorfilter,[frozenset,int],frozenset,name="colorfilter")
pset.addPrimitive(dsl.compress,[tuple],tuple,name="compress")
pset.addPrimitive(dsl.crement,[int],int,name="crement_int")
pset.addPrimitive(dsl.crement,[tuple],tuple,name="crement_tuple")
pset.addPrimitive(dsl.decrement,[int],int,name="decrement_int")
pset.addPrimitive(dsl.decrement,[tuple],tuple,name="decrement_tuple")
pset.addPrimitive(dsl.either,[bool,bool],bool,name="either")
pset.addPrimitive(dsl.extract,[frozenset,Callable],object,name="extract")
pset.addPrimitive(dsl.first,[frozenset],object,name="first")
pset.addPrimitive(dsl.frontiers,[tuple],frozenset,name="frontiers")
pset.addPrimitive(dsl.hperiod,[frozenset],int,name="hperiod")
pset.addPrimitive(dsl.hupscale,[tuple,int],tuple,name="hupscale")
pset.addPrimitive(dsl.increment,[int],int,name="increment_int")
pset.addPrimitive(dsl.increment,[tuple],tuple,name="increment_tuple")
pset.addPrimitive(dsl.last,[frozenset],object,name="last")
pset.addPrimitive(dsl.mfilter,[frozenset,Callable],frozenset,name="mfilter")
pset.addPrimitive(dsl.pair,[tuple,tuple],tuple,name="pair")
pset.addPrimitive(dsl.portrait,[tuple],bool,name="portrait")
pset.addPrimitive(dsl.positive,[int],bool,name="positive")
pset.addPrimitive(dsl.shape,[tuple],tuple,name="shape")
pset.addPrimitive(dsl.sign,[int],int,name="sign_int")
pset.addPrimitive(dsl.sign,[tuple],tuple,name="sign_tuple")
pset.addPrimitive(dsl.sizefilter,[frozenset,int],frozenset,name="sizefilter")
pset.addPrimitive(dsl.totuple,[frozenset],tuple,name="totuple")
pset.addPrimitive(dsl.underfill,[tuple,int,frozenset],tuple,name="underfill")
pset.addPrimitive(dsl.underpaint,[tuple,frozenset],tuple,name="underpaint")
pset.addPrimitive(dsl.vperiod,[frozenset],int,name="vperiod")
pset.addPrimitive(dsl.vupscale,[tuple,int],tuple,name="vupscale")
pset.addPrimitive(dsl.width,[tuple],int,name="width")
pset.addPrimitive(dsl.width,[frozenset],int,name="width_obj")
pset.addPrimitive(dsl.asindices,[tuple],frozenset,name="asindices")
pset.addPrimitive(dsl.occurrences,[tuple,frozenset],frozenset,name="occurrences")
pset.addPrimitive(dsl.trim,[tuple],tuple,name="trim")
pset.addPrimitive(dsl.cover,[tuple,frozenset],tuple,name="cover")
pset.addPrimitive(dsl.switch,[tuple,int,int],tuple,name="switch")
pset.addPrimitive(dsl.center,[frozenset],tuple,name="center")
pset.addPrimitive(dsl.position,[frozenset,frozenset],tuple,name="position")
pset.addPrimitive(dsl.inbox,[frozenset],frozenset,name="inbox")
pset.addPrimitive(dsl.outbox,[frozenset],frozenset,name="outbox")
pset.addPrimitive(dsl.box,[frozenset],frozenset,name="box")
pset.addPrimitive(dsl.gravitate,[frozenset,frozenset],tuple,name="gravitate")
pset.addPrimitive(dsl.merge,[frozenset],frozenset,name="merge")
pset.addPrimitive(dsl.maximum,[frozenset],int,name="maximum")
pset.addPrimitive(dsl.minimum,[frozenset],int,name="minimum")
pset.addPrimitive(dsl.valmax,[frozenset,Callable],int,name="valmax")
pset.addPrimitive(dsl.valmin,[frozenset,Callable],int,name="valmin")
pset.addPrimitive(dsl.shoot,[tuple,tuple],frozenset,name="shoot")

Int = int
Bool = bool
Num = (int, tuple)
AnyObj = object
Container = (tuple, frozenset)
Func = Callable
class Numerical:
    pass

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
pset.addPrimitive(dsl.invert,   [int],    int,   name="invert_int")
pset.addPrimitive(dsl.invert,   [tuple],  tuple, name="invert_tuple")
pset.addPrimitive(dsl.contained, [object, object], bool, name="contained_bool")
pset.addTerminal(frozenset(), frozenset, name="EmptySet")
pset.addPrimitive(
    dsl.upscale,
    [tuple, int],
    tuple,
    name="upscale_grid"
)
pset.addPrimitive(
    dsl.corners,              
    [frozenset],
    frozenset,
    name="corners"
)

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

for name, val in [
    ("ZERO", 0), ("ONE", 1), ("TWO", 2), ("THREE", 3),
    ("NEG_ONE", -1), ("NEG_TWO", -2),
]:
    pset.addTerminal(val, int, name=name)
pset.addTerminal(False, bool, name="F")
pset.addTerminal(True,  bool, name="T")

for name, val in [
    ("DOWN",  (1,0)),
    ("RIGHT", (0,1)),
    ("UP",    (-1,0)),
    ("LEFT",  (0,-1)),
]:
    pset.addTerminal(val, tuple, name=name)

def rand_int():
    return np.random.randint(0, 10)
pset.addEphemeralConstant("randInt", rand_int, int)






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
            save_tree_and_outputs_dot(task_name, best, func, task["test"])
            results.append({
                "task_name": task_name,
                "best_program": str(best),
                "solution_found": correct
            })
            print(f"{task_name} -> {'Correct-Solution Found' if correct else 'Failed-No Solution Found'}")

    # save summary in to json
    with open("tasks_eval_results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done. Results saved.")

if __name__ == "__main__":
    main()
