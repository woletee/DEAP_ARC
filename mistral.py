#!/usr/bin/env python3
import argparse
import numpy as np
import sys, os, json, glob
import deap.creator as creator
import deap.gp as gp
import deap.base as base
import deap.algorithms as algorithms
import tools
import deap.cma as cma
import hodel_dsl as dsl
import operator
import plot_utils
from plot_utils import plot_history, save_tree_and_outputs_dot
from program_generator import save_population_outputs, save_programs_only
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dsl import _objects, _objects2
from build import build_pset

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser(description="Run genetic programming over all tasks in ./training/")
parser.add_argument("--max_height", "-H", type=int, default=70, help="maximum tree height")
parser.add_argument("--cx_rate", "-C", type=float, default=0.5, help="initial crossover probability")
parser.add_argument("--mut_rate", "-M", type=float, default=0.4, help="initial mutation probability")
parser.add_argument("--generations", "-G", type=int, default=100, help="number of generations")
parser.add_argument("--iterations", "-I", type=int, default=3, help="number of LLM-GP refinement iterations")
args = parser.parse_args()

MAX_HEIGHT = args.max_height
CX0 = args.cx_rate
MUT0 = args.mut_rate
GENERATIONS = args.generations
LLM_ITERATIONS = args.iterations

if "FitnessMax" in creator.__dict__:
    del creator.FitnessMax
if "Individual" in creator.__dict__:
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

def call_codestral_dsl_selector(input_grid, output_grid, dsl_library):
    prompt = f"""
    Given this input grid:
    {np.array2string(np.array(input_grid), separator=',')}

    And this output grid:
    {np.array2string(np.array(output_grid), separator=',')}

    Which of the following DSL functions would best help explain this transformation?
    DSL options: {', '.join(dsl_library)}

    List the top 5 relevant functions.
    """
    model_id = "codellama/CodeLlama-7b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    selected = [dsl for dsl in dsl_library if dsl in decoded][:5]
    print("\n=== DSLs Selected by LLM ===")
    print(selected)
    return selected

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

def _init_task(task):
    global current_task
    current_task = task

def main():
    training_folder = "./training/"
    task_files = glob.glob(os.path.join(training_folder, "*.json"))
    results = []

    for task_file in task_files:
        task_name = os.path.basename(task_file)
        print(f"Processing {task_name}")
        with open(task_file) as f:
            task = json.load(f)

        train_example = task["train"][0]
        input_grid = train_example["input"]
        output_grid = train_example["output"]
        # Replace the `all_dsl_names` list with this cleaned version:

        all_dsl_names = [
            "rot90", "rot180", "ic_compress2", "flipy", "mirrorX", "mirrorY", "overlay", "set_bg",
            "ic_composegrowing", "ic_splitall", "ic_connectX", "ic_connectY", "ic_compress3", "ic_erasecol",
            "rarestcol", "left_half", "right_half", "top_half", "repeatX", "flipx", "ic_pickunique",
            "countToXY", "gravity_right", "split8", "ic_makeborder", "ic_filtercol", "ic_invert",
            "logical_and", "fillobj", "topcol", "gravity_down", "setcol", "ic_embed", "rot270",
            "mapSplit8", "pickcommon", "swapxy", "get_bg", "ic_fill", "ic_center", "countToY",
            "countPixels", "ic_splitcols", "grid_split", "stack_no_crop", "_objects2", "_objects",
            "move_down", "draw_line", "draw_line_slant_up", "draw_line_slant_down"
        ]


        best_fitness = -1
        best_result = None

        for iteration in range(LLM_ITERATIONS):
            print(f"\n--- Iteration {iteration + 1}/{LLM_ITERATIONS} ---")
            selected_dsls = call_codestral_dsl_selector(input_grid, output_grid, all_dsl_names)
            global pset
            pset = build_pset(selected_dsls)

            global toolbox
            toolbox = base.Toolbox()
            toolbox.register("compile", gp.compile, pset=pset)
            toolbox.register("select", tools.selTournament, tournsize=5)
            toolbox.register("mate", gp.cxOnePoint)
            toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
            toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
            toolbox.decorate("mate", gp.staticLimit(operator.attrgetter("height"), MAX_HEIGHT))
            toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), MAX_HEIGHT))
            toolbox.register("evaluate", evaluate_task)
            toolbox.register("population", tools.initRepeat, list,
                             lambda: creator.Individual(gp.genHalfAndHalf(pset, min_=2, max_=8)), n=100)
            toolbox.register("map", map)
            _init_task(task)

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
                cxpb = CX0 * (1 - gen / GENERATIONS)
                mutpb = MUT0 + (1 - MUT0) * (gen / GENERATIONS)
                if cxpb + mutpb > 1.0:
                    mutpb = 1.0 - cxpb
                offspring = algorithms.varOr(pop, toolbox, lambda_=(100 - len(hof)), cxpb=cxpb, mutpb=mutpb)
                fitnesses = toolbox.map(toolbox.evaluate, offspring)
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit
                hof.update(offspring)
                pop = toolbox.select(offspring + list(hof), k=100)
                record = stats.compile(pop)
                history.append(record)
                print(f"Gen {gen} | Max {record['max']} | Avg {record['avg']:.1f}")

            if hof and hof[0].fitness.values[0] > best_fitness:
                best_fitness = hof[0].fitness.values[0]
                best_result = (task_name, hof[0], history, task)

        if best_result:
            task_name, best, history, task = best_result
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
            results.append({"task_name": task_name, "best_program": str(best), "solution_found": correct})
            print(f"{task_name} -> {'Correct-Solution Found' if correct else 'Failed-No Solution Found'}")
            plot_history(history, task_name)

    with open("tasks_eval_results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done. Results saved.")

if __name__ == "__main__":
    main()
