# plot_utils.py
import matplotlib.pyplot as plt
import os
import json
import matplotlib.pyplot as plt
import networkx as nx
from deap import gp
import os
import json
import matplotlib.pyplot as plt     
from deap import gp
import graphviz           
import graphviz
from deap import gp

def plot_history(history, task_name):
    """
    Plots and saves the evolution of fitness over generations.
    
    :param history: aList of stat-dicts with keys 'avg' and 'max'.
    :param task_name: Name of the current task (used in plot title and filename).
    """
    generations = list(range(1, len(history) + 1))
    max_fit = [rec['max'] for rec in history]
    avg_fit = [rec['avg'] for rec in history]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, max_fit, label="Best Fitness")
    plt.plot(generations, avg_fit, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"Fitness Evolution: {task_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    filename = f"{task_name}_evolution.png"
    plt.savefig(filename)
    print(f"Saved fitness plot to {filename}")

def save_tree_and_outputs_dot(task_name, best_individual, compiled_func,
                              test_cases, output_dir="trees"):
    os.makedirs(output_dir, exist_ok=True)
    nodes, edges, labels = gp.graph(best_individual)
    dot = graphviz.Digraph(name=task_name, format="png")
    dot.attr("node", shape="box", fontname="Helvetica")
    dot.attr("edge", arrowhead="vee")
    for n in nodes:
        dot.node(str(n), label=labels[n])
    for u, v in edges:
        dot.edge(str(u), str(v))
    tree_path = os.path.join(output_dir, f"{task_name}_tree")
    dot.render(filename=tree_path, cleanup=True)
    print(f"Saved tree diagram to {tree_path}.png")
    prog_path = os.path.join(output_dir, f"{task_name}_program.txt")
    with open(prog_path, "w") as f:
        f.write(str(best_individual))
    print(f"Saved program text to {prog_path}")
    out_path = os.path.join(output_dir, f"{task_name}_outputs.json")
    outputs = []
    for tc in test_cases:
        inp = tuple(map(tuple, tc["input"]))
        try:
            res = compiled_func(inp)
        except Exception as e:
            res = f"<error: {e}>"
        outputs.append({"input": tc["input"], "output": res})
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Saved test outputs to {out_path}")
