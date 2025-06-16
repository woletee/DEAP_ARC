#!/usr/bin/env python3
import argparse
import numpy as np
import sys, os, json, glob
import deap.creator as creator
import deap.gp as gp
import deap.base as base
import deap.algorithms as algorithms
import tools
import operator
import plot_utils
from plot_utils import plot_history, save_tree_and_outputs_dot
from program_generator import save_population_outputs, save_programs_only
from dsl import _objects, _objects2
from build import build_pset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser()
parser.add_argument("--max_height", type=int, default=70)
parser.add_argument("--cx_rate", type=float, default=0.5)
parser.add_argument("--mut_rate", type=float, default=0.4)
parser.add_argument("--generations", type=int, default=100)
parser.add_argument("--iterations", type=int, default=3)
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


def call_llm_mutation(original_code, input_grid, output_grid):
    prompt = f"""
You are optimizing a program that transforms a grid.
Input:
{np.array2string(np.array(input_grid), separator=',')}
Output:
{np.array2string(np.array(output_grid), separator=',')}

Current program:
{original_code}

Please provide an improved version of the program as Python code using the DSL functions.
Only return valid Python code (no explanation).
"""
    model_id = "codellama/CodeLlama-7b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=150)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    code = decoded.split("\n")[0]  # crude, better: regex or ast
    return code


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

        input_grid = task["train"][0]["input"]
        output_grid = task["train"][0]["output"]

        all_dsl_names = ["rot90", "rot180", "ic_compress2", "flipy", "mirrorX", "mirrorY", "overlay", "set_bg",
                         "ic_composegrowing", "ic_splitall", "ic_connectX", "ic_connectY", "ic_compress3", "ic_erasecol",
                         "rarestcol", "left_half", "right_half", "top_half", "repeatX", "flipx", "ic_pickunique",
                         "countToXY", "gravity_right", "split8", "ic_makeborder", "ic_filtercol", "ic_invert",
                         "logical_and", "fillobj", "topcol", "gravity_down", "setcol", "ic_embed", "rot270",
                         "mapSplit8", "pickcommon", "swapxy", "get_bg", "ic_fill", "ic_center", "countToY",
                         "countPixels", "ic_splitcols", "grid_split", "stack_no_crop", "_objects2", "_objects",
                         "move_down", "draw_line", "draw_line_slant_up", "draw_line_slant_down"]

        selected_dsls = all_dsl_names[:15]  # baseline DSLs (can replace with LLM selector)
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

        for gen in range(1, GENERATIONS + 1):
            offspring = algorithms.varOr(pop, toolbox, lambda_=(100 - len(hof)), cxpb=CX0, mutpb=MUT0)
            # LLM mutation on top 5% of population
            for ind in sorted(pop, key=lambda x: x.fitness.values[0], reverse=True)[:5]:
                try:
                    code_str = str(ind)
                    llm_code = call_llm_mutation(code_str, input_grid, output_grid)
                    # Skipping code-to-tree conversion for now: placeholder
                    # Add new mutated individuals to offspring here if parsed
                except Exception as e:
                    print("LLM mutation failed:", e)

            fitnesses = toolbox.map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit
            hof.update(offspring)
            pop = toolbox.select(offspring + list(hof), k=100)
            record = stats.compile(pop)
            print(f"Gen {gen} | Max {record['max']} | Avg {record['avg']:.1f}")

        best = hof[0]
        func = toolbox.compile(expr=best)
        correct = True
        for tc in task["test"]:
            try:
                inp = tuple(map(tuple, tc["input"]))
                tgt = np.array(tc["output"])
                out = np.array(func(inp))
                if not np.array_equal(out, tgt):
                    correct = False
                    break
            except Exception:
                correct = False
                break

        save_tree_and_outputs_dot(task_name, best, func, task["test"])
        results.append({"task_name": task_name, "best_program": str(best), "solution_found": correct})

    with open("tasks_eval_results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done. Results saved.")


if __name__ == "__main__":
    main()
