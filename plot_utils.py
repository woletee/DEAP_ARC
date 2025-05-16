# plot_utils.py
import matplotlib.pyplot as plt

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
    
    # Show and save
    plt.show()
    filename = f"{task_name}_evolution.png"
    plt.savefig(filename)
    print(f"Saved fitness plot to {filename}")
