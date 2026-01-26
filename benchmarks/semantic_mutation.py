#!/usr/bin/env python3
"""
Quality Diversity GP + LLM for ARC (Qwen version)

Env vars (DashScope / Qwen):
  QWEN_API_KEY   (required)
  QWEN_MODEL     (default: qwen-plus)
  QWEN_API_URL   (default: DashScope generation endpoint)

Install:
  pip install deap python-dotenv requests
"""

import argparse
import copy
import datetime
import glob
import json
import os
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Check imports early
try:
    from deap import gp, tools, creator, base
except ImportError as e:
    print("ERROR: DEAP not installed. Run: pip install deap")
    print(f"Details: {e}")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("WARNING: python-dotenv not installed. Run: pip install python-dotenv")
    load_dotenv = lambda **kwargs: None

try:
    import requests
except ImportError as e:
    print("ERROR: requests not installed. Run: pip install requests")
    print(f"Details: {e}")
    sys.exit(1)

try:
    import gp as original_gp
    import primitives  # noqa: F401
except ImportError as e:
    print("ERROR: Cannot import gp.py or primitives.py")
    print("Make sure you have gp.py and primitives.py in the same directory")
    print(f"Details: {e}")
    sys.exit(1)

print("✓ All imports successful")

# ============================================================
# SETTINGS - OPTIMIZED FOR QUALITY DIVERSITY
# ============================================================
random.seed(0)
llm_call_count = 0
DEBUG_LLM = False

# Evolution - QD Parameters
POP_SIZE = 150
ELITISM_KEEP = 3
GP_MUTPB = 0.65
TOURNAMENT_SIZE = 3

# Quality Diversity Settings
QD_ENABLED = True
QD_ARCHIVE_SIZE = 200
QD_NOVELTY_WEIGHT = 0.3
QD_BEHAVIOR_DIMS = 3

# LLM Settings
LLM_ELITE_COUNT = 3
LLM_PROG_CANDS = 15
LLM_MAX_TOKENS = 600
THROTTLE_SECONDS = 0.10

# Prompts
PROMPT_TRAIN_EXAMPLES = 3
GRID_PREVIEW_SIZE = 100
PROMPT_INCLUDE_DSL = True
PROMPT_MAX_PRIMS = 25
PROMPT_MAX_TERMS = 15

# Acceptance
ACCEPT_EQUAL = True
ACCEPT_SMALL_DROP = True
SMALL_DROP_THRESHOLD = 0.90

# Diversity
PLATEAU_GENS = 5
INJECT_COUNT = 25
DIVERSITY_INJECT_EVERY = 3

# Post refinement
POST_LLM_ENABLED = True
POST_LLM_ITERS = 30
POST_LLM_CANDS = 15
POST_LLM_TOKENS = 700
POST_LLM_TEMP = 0.7
POST_LLM_PATIENCE = 8

# Cache
FITNESS_CACHE: Dict[str, Tuple[float, List[float]]] = {}
SEEN_PROGRAMS = set()

# Save
SAVE_NDJSON = "./results/results_qd_gp_llm.ndjson"

DSL_REF_CACHE = None


# ============================================================
# QUALITY DIVERSITY STRUCTURES
# ============================================================
class BehaviorArchive:
    """MAP-Elites style archive for quality diversity"""

    def __init__(self, dims: int = 3, bins_per_dim: int = 5):
        self.dims = dims
        self.bins_per_dim = bins_per_dim
        self.archive = {}  # niche -> (individual, raw_fitness, behavior)

    def get_niche_key(self, behavior: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(min(int(b), self.bins_per_dim - 1) for b in behavior)

    def add(self, individual, behavior: Tuple[int, ...], fitness: float):
        niche = self.get_niche_key(behavior)
        if niche not in self.archive or fitness > self.archive[niche][1]:
            self.archive[niche] = (toolbox.clone(individual), fitness, behavior)
            return True
        return False

    def get_diverse_sample(self, n: int):
        if not self.archive:
            return []
        niches = list(self.archive.keys())
        random.shuffle(niches)
        return [toolbox.clone(self.archive[k][0]) for k in niches[:n]]

    def coverage(self) -> float:
        total = self.bins_per_dim ** self.dims
        return len(self.archive) / total

    def size(self) -> int:
        return len(self.archive)


qd_archive: Optional[BehaviorArchive] = None


# ============================================================
# UTILITIES
# ============================================================
def now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def short(s, n=250):
    s = str(s)
    return s if len(s) <= n else s[:n] + "..." + s[-60:]


def history_reset():
    global FITNESS_CACHE, SEEN_PROGRAMS, qd_archive
    FITNESS_CACHE = {}
    SEEN_PROGRAMS = set()
    if QD_ENABLED:
        qd_archive = BehaviorArchive(dims=QD_BEHAVIOR_DIMS, bins_per_dim=5)


def save_ndjson(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


# ============================================================
# GRID HELPERS
# ============================================================
def is_grid(x) -> bool:
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], list))


def grid_shape(x) -> Tuple[int, int]:
    if not is_grid(x):
        return (0, 0)
    h = len(x)
    w = len(x[0]) if h else 0
    return (h, w)


def grid_to_string(g) -> str:
    if not is_grid(g):
        return "<not_grid>"
    h, w = grid_shape(g)
    if h == 0 or w == 0:
        return "[]"
    if h * w <= GRID_PREVIEW_SIZE:
        return json.dumps(g, separators=(",", ":"))
    preview = [row[:min(w, 10)] for row in g[:min(h, 10)]]
    return f"<{h}x{w} preview: {json.dumps(preview, separators=(',',':'))}>"


def analyze_transformation(inp, out) -> str:
    if not is_grid(inp) or not is_grid(out):
        return "invalid"
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)

    parts = []
    if (ih, iw) == (oh, ow):
        parts.append("same_size")
    else:
        parts.append(f"{ih}x{iw}→{oh}x{ow}")

    inp_colors = set()
    out_colors = set()
    for row in inp:
        inp_colors.update(row)
    for row in out:
        out_colors.update(row)

    if inp_colors != out_colors:
        new = out_colors - inp_colors
        if new:
            parts.append(f"+{tuple(sorted(new))}")

    return " | ".join(parts)


def raw_correct_pixels(pred, target) -> int:
    if not is_grid(pred) or not is_grid(target):
        return 0

    ph, pw = grid_shape(pred)
    th, tw = grid_shape(target)
    min_h = min(ph, th)
    min_w = min(pw, tw)

    correct = 0
    for r in range(min_h):
        for c in range(min_w):
            if pred[r][c] == target[r][c]:
                correct += 1
    return correct


def count_correct_pixels(pred, target) -> float:
    if not is_grid(pred) or not is_grid(target):
        return 0.0

    ph, pw = grid_shape(pred)
    th, tw = grid_shape(target)
    min_h = min(ph, th)
    min_w = min(pw, tw)

    correct = 0.0
    for r in range(min_h):
        for c in range(min_w):
            if pred[r][c] == target[r][c]:
                correct += 1.0

    if (ph, pw) == (th, tw):
        correct *= 1.15
    else:
        size_diff = abs(ph - th) + abs(pw - tw)
        correct -= size_diff * 1.2

    return max(0.0, correct)


def grids_equal(a, b) -> bool:
    if not is_grid(a) or not is_grid(b):
        return False
    if grid_shape(a) != grid_shape(b):
        return False
    h, w = grid_shape(a)
    for r in range(h):
        for c in range(w):
            if a[r][c] != b[r][c]:
                return False
    return True


def format_examples(task: dict, k: int) -> str:
    lines = []
    for i, ex in enumerate(task.get("train", [])[:k]):
        inp = ex["input"]
        out = ex["output"]
        trans = analyze_transformation(inp, out)
        lines.append(f"\nExample {i} ({trans}):")
        lines.append(f"  In:  {grid_to_string(inp)}")
        lines.append(f"  Out: {grid_to_string(out)}")
    return "\n".join(lines)


# ============================================================
# BEHAVIORAL CHARACTERIZATION
# ============================================================
def characterize_behavior(individual, task: dict) -> Tuple[int, ...]:
    try:
        func = toolbox.compile(expr=individual)
    except:
        return (0, 0, 0)

    # Dimension 1: output size ratio (binned)
    size_ratios = []
    for ex in task.get("train", []):
        target = ex["output"]
        if not is_grid(target):
            continue
        try:
            pred = func(ex["input"])
        except:
            continue
        if is_grid(pred):
            th, tw = grid_shape(target)
            ph, pw = grid_shape(pred)
            if th * tw > 0:
                size_ratios.append((ph * pw) / (th * tw))
    avg_size_ratio = sum(size_ratios) / len(size_ratios) if size_ratios else 0
    size_bin = min(4, int(avg_size_ratio * 2))

    # Dimension 2: color diversity (binned)
    all_colors = set()
    for ex in task.get("train", []):
        try:
            pred = func(ex["input"])
            if is_grid(pred):
                for row in pred:
                    all_colors.update(row)
        except:
            pass
    color_diversity = min(4, len(all_colors))

    # Dimension 3: complexity (binned by tree height)
    tree_depth = individual.height
    complexity_bin = min(4, tree_depth // 3)

    return (size_bin, color_diversity, complexity_bin)


def compute_novelty_score(individual, population, k: int = 5) -> float:
    ind_behavior = getattr(individual, "_behavior", None)
    if ind_behavior is None:
        return 0.0

    distances = []
    for other in population:
        if other is individual:
            continue
        ob = getattr(other, "_behavior", None)
        if ob is None:
            continue
        dist = sum((a - b) ** 2 for a, b in zip(ind_behavior, ob)) ** 0.5
        distances.append(dist)

    if len(distances) < k:
        return sum(distances) / len(distances) if distances else 0.0

    distances.sort()
    return sum(distances[:k]) / k


# ============================================================
# DEAP SETUP
# ============================================================
print("Setting up DEAP...")

pset: gp.PrimitiveSetTyped = original_gp.pset
toolbox = getattr(original_gp, "toolbox", None) or tools.Toolbox()

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox.register("clone", copy.deepcopy)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
toolbox.register(
    "mutate_raw",
    gp.mutUniform,
    expr=partial(gp.genFull, min_=1, max_=3),
    pset=pset,
)


def gp_mutate_safe(ind):
    try:
        child, = toolbox.mutate_raw(ind)
        child._origin = "GP_MUT"
        try:
            del child.fitness.values
        except:
            pass
        return child
    except:
        return toolbox.clone(ind)


def gp_mutate_diverse(ind):
    if random.random() < 0.3:
        try:
            child, = gp.mutUniform(
                ind,
                expr=partial(gp.genFull, min_=2, max_=5),
                pset=pset
            )
            child._origin = "DIV_MUT"
            try:
                del child.fitness.values
            except:
                pass
            return child
        except:
            pass
    return gp_mutate_safe(ind)


def init_individual():
    expr = gp.genHalfAndHalf(pset=pset, min_=2, max_=5)
    ind = creator.Individual(expr)
    ind._origin = "INIT"
    return ind


toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

OUTPUT_TYPE = getattr(pset, "ret", None)

print("✓ DEAP setup complete")


# ============================================================
# DSL REFERENCE
# ============================================================
def _type_name(t) -> str:
    return getattr(t, "__name__", str(t))


def build_dsl_reference() -> str:
    lines = ["DSL FUNCTIONS:"]
    for ret_t, prims in pset.primitives.items():
        ret_name = _type_name(ret_t)
        for p in prims[:PROMPT_MAX_PRIMS]:
            args = ", ".join(_type_name(a) for a in p.args)
            lines.append(f"  {p.name}({args}) → {ret_name}")
        if len(prims) > PROMPT_MAX_PRIMS:
            lines.append(f"  ... +{len(prims)-PROMPT_MAX_PRIMS} more")

    lines.append("\nTERMINALS:")
    for ret_t, terms in pset.terminals.items():
        ret_name = _type_name(ret_t)
        for t in terms[:PROMPT_MAX_TERMS]:
            nm = getattr(t, "name", str(t)).replace("\n", " ").strip()
            lines.append(f"  {nm} → {ret_name}")
        if len(terms) > PROMPT_MAX_TERMS:
            lines.append(f"  ... +{len(terms)-PROMPT_MAX_TERMS} more")

    return "\n".join(lines)


def get_dsl_ref() -> str:
    global DSL_REF_CACHE
    if DSL_REF_CACHE is None:
        DSL_REF_CACHE = build_dsl_reference()
    return DSL_REF_CACHE


# ============================================================
# QWEN (DashScope) SETUP
# ============================================================
print("Setting up Qwen (DashScope REST)...")

dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_API_URL = os.getenv(
    "QWEN_API_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
)

if not QWEN_API_KEY:
    print("ERROR: QWEN_API_KEY not found!")
    print("Set it in .env or environment variables.")
    sys.exit(1)

print(f"✓ Using Qwen model: {QWEN_MODEL}")
print(f"✓ Qwen API URL: {QWEN_API_URL}")


class LLMEngine:
    def __init__(self, model: str, api_key: str, api_url: str):
        self.model = model
        self.api_key = api_key
        self.api_url = api_url
        self.session = requests.Session()

    def _extract_text(self, data: dict) -> str:
        try:
            out = data.get("output", {})
            if isinstance(out, dict):
                choices = out.get("choices")
                if isinstance(choices, list) and choices:
                    ch0 = choices[0] or {}
                    msg = ch0.get("message")
                    if isinstance(msg, dict) and "content" in msg:
                        return (msg.get("content") or "").strip()
                    if "text" in ch0:
                        return (ch0.get("text") or "").strip()
                if "text" in out:
                    return (out.get("text") or "").strip()
        except Exception:
            pass
        return ""

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        global llm_call_count

        messages = [
            {
                "role": "system",
                "content": (
                    "You solve ARC puzzles by generating programs.\n"
                    "Output ONLY program expressions, one per line.\n"
                    "NO explanations, NO markdown, NO comments.\n"
                ),
            },
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": self.model,
            "input": {"messages": messages},
            "parameters": {
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, 6):
            try:
                llm_call_count += 1
                resp = self.session.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=45,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    return (self._extract_text(data) or "").strip()

                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(2.0, 0.1 * (2 ** attempt)) + random.random() * 0.05)
                    continue

                if DEBUG_LLM:
                    print(f"[QWEN] HTTP {resp.status_code}: {resp.text[:500]}")
                return ""

            except (requests.Timeout, requests.ConnectionError):
                time.sleep(min(2.0, 0.15 * attempt))
            except Exception as e:
                if DEBUG_LLM:
                    print(f"[QWEN] Exception: {e}")
                time.sleep(min(2.0, 0.15 * attempt))

        return ""


llm = LLMEngine(QWEN_MODEL, QWEN_API_KEY, QWEN_API_URL)


# ============================================================
# PROGRAM PARSING
# ============================================================
def extract_programs(raw: str, max_chars: int) -> List[str]:
    raw = raw.replace("```python", "").replace("```", "").replace("`", "").strip()

    programs = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or len(line) < 5:
            continue

        for prefix in ["Program:", "Solution:", ">>>", "//", "#", "-", "*"]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()

        if "(" not in line:
            continue
        if line.count("(") != line.count(")"):
            continue
        if len(line) > max_chars:
            continue

        programs.append(line)

    seen = set()
    unique = []
    for p in programs:
        if p not in seen:
            unique.append(p)
            seen.add(p)

    return unique


def tree_return_type(tree: gp.PrimitiveTree):
    try:
        return tree[0].ret
    except:
        return None


# ============================================================
# FITNESS EVALUATION WITH QD
# ============================================================
def eval_program_on_task(individual, task: dict) -> Tuple[float, List[float]]:
    prog_str = str(individual)

    if prog_str in FITNESS_CACHE:
        return FITNESS_CACHE[prog_str]

    try:
        func = toolbox.compile(expr=individual)
    except:
        result = (0.0, [0.0] * len(task.get("train", [])))
        FITNESS_CACHE[prog_str] = result
        SEEN_PROGRAMS.add(prog_str)
        return result

    total = 0.0
    per_ex = []

    for ex in task.get("train", []):
        inp = ex["input"]
        target = ex["output"]
        try:
            pred = func(inp)
        except:
            per_ex.append(0.0)
            continue

        score = count_correct_pixels(pred, target)
        per_ex.append(score)
        total += score

    result = (total, per_ex)
    FITNESS_CACHE[prog_str] = result
    SEEN_PROGRAMS.add(prog_str)
    return result


def eval_individual(individual, task: dict, population: List = None):
    total, per_ex = eval_program_on_task(individual, task)

    behavior = characterize_behavior(individual, task)
    individual._behavior = behavior
    individual._per_ex = per_ex

    fitness = total
    if QD_ENABLED and population is not None and len(population) > 1:
        novelty = compute_novelty_score(individual, population)
        fitness = total * (1 - QD_NOVELTY_WEIGHT) + novelty * QD_NOVELTY_WEIGHT * 100

    individual.fitness.values = (fitness,)

    if QD_ENABLED and qd_archive is not None:
        qd_archive.add(individual, behavior, total)


def is_exact_solution_on_train(individual, task: dict) -> bool:
    try:
        func = toolbox.compile(expr=individual)
    except:
        return False

    for ex in task.get("train", []):
        inp = ex["input"]
        target = ex["output"]
        try:
            pred = func(inp)
        except:
            return False
        if not grids_equal(pred, target):
            return False
    return True


def task_raw_correct_and_total(individual, task: dict) -> Tuple[int, int]:
    try:
        func = toolbox.compile(expr=individual)
    except:
        total_pixels = sum(
            len(ex["output"]) * len(ex["output"][0])
            for ex in task.get("train", [])
            if is_grid(ex.get("output"))
        )
        return 0, total_pixels

    raw_correct = 0
    total_pixels = 0

    for ex in task.get("train", []):
        target = ex["output"]
        if not is_grid(target):
            continue
        th, tw = grid_shape(target)
        total_pixels += th * tw

        try:
            pred = func(ex["input"])
        except:
            continue

        raw_correct += raw_correct_pixels(pred, target)

    return raw_correct, total_pixels


# ============================================================
# ERROR ANALYSIS
# ============================================================
def get_error_report(individual, task: dict, max_ex: int = 2) -> str:
    try:
        func = toolbox.compile(expr=individual)
    except Exception as e:
        return f"COMPILE ERROR: {str(e)[:100]}"

    lines = []
    for i, ex in enumerate(task.get("train", [])[:max_ex]):
        target = ex["output"]
        try:
            pred = func(ex["input"])
        except Exception:
            lines.append(f"Ex{i}: RUNTIME ERROR")
            continue

        if not is_grid(pred):
            lines.append(f"Ex{i}: NOT GRID")
            continue

        ph, pw = grid_shape(pred)
        th, tw = grid_shape(target)

        if (ph, pw) != (th, tw):
            lines.append(f"Ex{i}: WRONG SHAPE {ph}x{pw} vs {th}x{tw}")
        else:
            wrong = sum(
                1 for r in range(th) for c in range(tw)
                if pred[r][c] != target[r][c]
            )
            if wrong == 0:
                lines.append(f"Ex{i}: PERFECT ✓")
            else:
                pct = 100 * wrong / (th * tw)
                lines.append(f"Ex{i}: {wrong}/{th*tw} wrong ({pct:.0f}%)")

    return " | ".join(lines) if lines else "No errors"


# ============================================================
# LLM PROGRAM GENERATION WITH DIVERSITY
# ============================================================
def llm_generate_programs(parent, task: dict, target_fit: float, gen: int,
                          elite_idx: int, use_diverse_context: bool = False):
    parent_fit = float(parent.fitness.values[0]) if parent.fitness.valid else 0.0
    parent_prog = str(parent)

    examples = format_examples(task, k=PROMPT_TRAIN_EXAMPLES)
    errors = get_error_report(parent, task, max_ex=2)
    dsl_ref = get_dsl_ref() if PROMPT_INCLUDE_DSL else ""

    context_programs = []
    if use_diverse_context and QD_ENABLED and qd_archive is not None and qd_archive.size() > 3:
        diverse_inds = qd_archive.get_diverse_sample(3)
        context_programs = [str(ind) for ind in diverse_inds]

    progress = (parent_fit / target_fit * 100) if target_fit > 0 else 0

    prompt = f"""ARC Puzzle - Generate {LLM_PROG_CANDS} DIVERSE solutions exploring different approaches.

Progress: {progress:.0f}% ({int(parent_fit)}/{int(target_fit)})

{dsl_ref}

{examples}

Current program:
{parent_prog}

Errors: {errors}
"""

    if context_programs:
        prompt += "\nOther successful approaches:\n"
        for i, prog in enumerate(context_programs[:2]):
            prompt += f"{i+1}. {prog[:150]}\n"

    prompt += f"\nOutput {LLM_PROG_CANDS} DIFFERENT programs (one per line, NO explanations):\n"

    if THROTTLE_SECONDS > 0:
        time.sleep(THROTTLE_SECONDS)

    raw = llm.generate(prompt, max_tokens=LLM_MAX_TOKENS, temperature=0.8)
    progs = extract_programs(raw, max_chars=600)

    if DEBUG_LLM:
        print(f"  [LLM] Elite {elite_idx}: {len(progs)} candidates")

    best_child = None
    best_fit = None

    for prog_str in progs:
        try:
            tree = gp.PrimitiveTree.from_string(prog_str, pset)
        except:
            continue

        if OUTPUT_TYPE and tree_return_type(tree) != OUTPUT_TYPE:
            continue

        child = creator.Individual(tree)

        if str(child) in SEEN_PROGRAMS:
            continue

        eval_individual(child, task)
        fit = float(child.fitness.values[0])

        if best_fit is None or fit > best_fit:
            best_fit = fit
            best_child = child

    if best_child is None:
        return toolbox.clone(parent), False, 0.0

    delta = best_fit - parent_fit

    accept = (
        best_fit > parent_fit or
        (ACCEPT_EQUAL and best_fit == parent_fit) or
        (ACCEPT_SMALL_DROP and best_fit >= parent_fit * SMALL_DROP_THRESHOLD)
    )

    if accept:
        best_child._origin = "LLM"
        return best_child, True, delta
    else:
        return toolbox.clone(parent), False, delta


# ============================================================
# QUALITY DIVERSITY EVOLUTION
# ============================================================
def run_evolution(task: dict, pop_size: int, generations: int):
    target_fit = sum(
        len(ex["output"]) * len(ex["output"][0])
        for ex in task.get("train", [])
        if is_grid(ex.get("output"))
    )

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        eval_individual(ind, task, pop)

    hof = tools.HallOfFame(1)
    hof.update(pop)

    best_curve = []
    llm_acc_curve = []
    diversity_curve = []
    coverage_curve = []

    plateau = 0
    last_best = -1.0

    print(f"\n{'='*70}")
    print(f"QD Evolution: pop={pop_size} gens={generations} target={target_fit}")
    if QD_ENABLED:
        print(f"Quality Diversity: ENABLED (novelty_weight={QD_NOVELTY_WEIGHT})")
    print(f"{'='*70}\n")

    for gen in range(1, generations + 1):
        pop.sort(key=lambda x: x.fitness.values[0], reverse=True)

        elites = [toolbox.clone(ind) for ind in pop[:ELITISM_KEEP]]

        if QD_ENABLED and gen % 2 == 0 and qd_archive and qd_archive.size() > pop_size // 4:
            archive_sample = qd_archive.get_diverse_sample(pop_size // 4)
            regular_selected = toolbox.select(pop, pop_size - ELITISM_KEEP - len(archive_sample))
            offspring = [toolbox.clone(ind) for ind in regular_selected] + archive_sample
        else:
            offspring = toolbox.select(pop, pop_size - ELITISM_KEEP)
            offspring = [toolbox.clone(ind) for ind in offspring]

        mut_count = 0
        for i in range(len(offspring)):
            if random.random() < GP_MUTPB:
                if QD_ENABLED and random.random() < 0.4:
                    offspring[i] = gp_mutate_diverse(offspring[i])
                else:
                    offspring[i] = gp_mutate_safe(offspring[i])
                mut_count += 1

        llm_count = min(LLM_ELITE_COUNT, len(pop))
        llm_att = 0
        llm_acc = 0
        llm_children = []

        use_diverse = QD_ENABLED and gen % 3 == 0

        for j in range(llm_count):
            llm_att += 1
            parent_fit = float(pop[j].fitness.values[0])

            child, accepted, delta = llm_generate_programs(
                pop[j], task, target_fit, gen, j, use_diverse_context=use_diverse
            )
            llm_children.append(child)

            if accepted and delta > 0:
                llm_acc += 1
                print(f"  [LLM #{j}] ✓ {parent_fit:.0f} → {child.fitness.values[0]:.0f} (+{delta:.0f})")

        for i in range(min(len(llm_children), len(offspring))):
            offspring[i] = llm_children[i]

        pop = (offspring + elites)[:pop_size]

        for ind in pop:
            if not ind.fitness.valid:
                eval_individual(ind, task, pop)

        hof.update(pop)

        best = hof[0]
        best_fit = float(best.fitness.values[0])
        best_curve.append(best_fit)
        llm_acc_curve.append(llm_acc)

        if QD_ENABLED and qd_archive is not None:
            coverage = qd_archive.coverage()
            coverage_curve.append(coverage)

            behaviors = [getattr(ind, "_behavior", (0, 0, 0)) for ind in pop]
            unique_behaviors = len(set(behaviors))
            diversity = unique_behaviors / len(pop) if pop else 0
            diversity_curve.append(diversity)
        else:
            coverage_curve.append(0)
            diversity_curve.append(0)

        raw_corr, raw_total = task_raw_correct_and_total(best, task)
        raw_pct = (100.0 * raw_corr / raw_total) if raw_total else 0.0

        print(
            f"Gen {gen:02d} | Fit: {best_fit:.0f} | "
            f"Raw: {raw_corr}/{raw_total} ({raw_pct:.1f}%) | "
            f"LLM: {llm_acc}/{llm_att} | Mut: {mut_count}"
        )

        if QD_ENABLED and qd_archive is not None:
            print(f"         QD Coverage: {coverage*100:.1f}% | Diversity: {diversity*100:.1f}%")

        if best_fit <= last_best + 0.1:
            plateau += 1
        else:
            plateau = 0
            last_best = best_fit

        if plateau >= PLATEAU_GENS:
            print(f"  🔄 Plateau → inject {INJECT_COUNT} random")
            if QD_ENABLED and qd_archive and qd_archive.size() > INJECT_COUNT // 2:
                fresh = qd_archive.get_diverse_sample(INJECT_COUNT // 2)
                fresh += toolbox.population(n=INJECT_COUNT - len(fresh))
            else:
                fresh = toolbox.population(n=min(INJECT_COUNT, pop_size))

            for ind in fresh:
                eval_individual(ind, task, pop)

            pop.sort(key=lambda x: x.fitness.values[0])
            pop[:len(fresh)] = fresh
            hof.update(pop)
            plateau = 0

        if QD_ENABLED and gen % DIVERSITY_INJECT_EVERY == 0 and qd_archive and qd_archive.size() > 5:
            diverse_boost = qd_archive.get_diverse_sample(5)
            for ind in diverse_boost:
                eval_individual(ind, task, pop)
            pop.extend(diverse_boost)
            pop.sort(key=lambda x: x.fitness.values[0], reverse=True)
            pop = pop[:pop_size]

        if is_exact_solution_on_train(best, task):
            print(f"\n🎉 EXACT SOLVED at gen {gen}!")
            break

    return hof[0], best_curve, llm_acc_curve, diversity_curve, coverage_curve


# ============================================================
# POST-REFINEMENT
# ============================================================
def post_refine(best, task: dict, target_fit: float):
    if not POST_LLM_ENABLED:
        return best

    if not best.fitness.valid:
        eval_individual(best, task)

    if is_exact_solution_on_train(best, task):
        return best

    start_fit = float(best.fitness.values[0])

    print(f"\n{'='*70}")
    print(f"POST-REFINEMENT: {start_fit:.0f}/{target_fit}")
    print(f"{'='*70}\n")

    current = toolbox.clone(best)
    current_fit = start_fit
    best_seen = toolbox.clone(current)
    no_improve = 0

    examples = format_examples(task, k=PROMPT_TRAIN_EXAMPLES)
    dsl_ref = get_dsl_ref() if PROMPT_INCLUDE_DSL else ""

    for it in range(1, POST_LLM_ITERS + 1):
        errors = get_error_report(current, task, max_ex=2)

        context = ""
        if QD_ENABLED and qd_archive is not None and qd_archive.size() > 2:
            diverse_inds = qd_archive.get_diverse_sample(2)
            context = "\nAlternative approaches:\n"
            for i, ind in enumerate(diverse_inds):
                context += f"{i+1}. {str(ind)[:120]}\n"

        prompt = f"""Refine ARC solution. Generate {POST_LLM_CANDS} improved programs.

{dsl_ref}

{examples}

Current: {str(current)}
Errors: {errors}

{context}

Output {POST_LLM_CANDS} programs (one per line):
"""

        if THROTTLE_SECONDS > 0:
            time.sleep(THROTTLE_SECONDS)

        raw = llm.generate(prompt, POST_LLM_TOKENS, POST_LLM_TEMP)
        progs = extract_programs(raw, max_chars=800)

        iter_best = None
        iter_best_fit = None

        for prog_str in progs:
            try:
                tree = gp.PrimitiveTree.from_string(prog_str, pset)
            except:
                continue

            if OUTPUT_TYPE and tree_return_type(tree) != OUTPUT_TYPE:
                continue

            ind = creator.Individual(tree)

            if str(ind) in SEEN_PROGRAMS:
                continue

            eval_individual(ind, task)
            fit = float(ind.fitness.values[0])

            if iter_best_fit is None or fit > iter_best_fit:
                iter_best_fit = fit
                iter_best = ind

        if iter_best is None:
            no_improve += 1
            print(f"  Post {it:02d}: no valid ({no_improve}/{POST_LLM_PATIENCE})")
        else:
            if iter_best_fit > current_fit:
                delta = iter_best_fit - current_fit
                current = iter_best
                current_fit = iter_best_fit
                no_improve = 0
                if current_fit > float(best_seen.fitness.values[0]):
                    best_seen = toolbox.clone(current)
                print(f"  Post {it:02d}: ✓ {current_fit-delta:.0f} → {current_fit:.0f} (+{delta:.0f})")
            else:
                no_improve += 1

        if is_exact_solution_on_train(current, task):
            print("\n🎉 EXACT SOLVED in post-refine!")
            return current

        if no_improve >= POST_LLM_PATIENCE:
            break

    return best_seen


# ============================================================
# TASK RUNNER
# ============================================================
def run_single_task(task_file: str, pop_size: int, generations: int) -> dict:
    global llm_call_count

    history_reset()
    llm_before = llm_call_count
    start_time = time.time()

    with open(task_file, "r", encoding="utf-8") as f:
        task = json.load(f)

    original_gp.current_task = task

    target_fit = sum(
        len(ex["output"]) * len(ex["output"][0])
        for ex in task.get("train", [])
        if is_grid(ex.get("output"))
    )

    print(f"\n{'='*70}")
    print(f"TASK: {os.path.basename(task_file)}")
    print(f"Target pixels: {target_fit} | Model: {QWEN_MODEL}")
    print(f"{'='*70}")

    best, best_curve, llm_acc, div_curve, cov_curve = run_evolution(task, pop_size, generations)
    final = post_refine(best, task, target_fit)

    duration = time.time() - start_time
    llm_calls = llm_call_count - llm_before
    final_fit = float(final.fitness.values[0])

    raw_corr, raw_total = task_raw_correct_and_total(final, task)
    raw_pct = (100.0 * raw_corr / raw_total) if raw_total else 0.0

    solved = is_exact_solution_on_train(final, task)

    qd_coverage = qd_archive.coverage() * 100 if qd_archive else 0
    qd_size = qd_archive.size() if qd_archive else 0

    print(f"\n{'='*70}")
    print(f"FINAL Fitness: {final_fit:.0f}")
    print(f"FINAL Raw pixels: {raw_corr}/{raw_total} ({raw_pct:.1f}%)")
    print(f"Solved (exact train): {'YES ✓' if solved else 'NO'}")
    if QD_ENABLED:
        print(f"QD Archive: {qd_size} niches | Coverage: {qd_coverage:.1f}%")
    print(f"Time: {duration:.0f}s | LLM calls: {llm_calls}")
    print(f"Program: {short(str(final), 200)}")
    print(f"{'='*70}\n")

    result = {
        "timestamp": now_iso(),
        "task_file": os.path.basename(task_file),
        "target_fit": int(target_fit),
        "best_fit": float(final_fit),
        "raw_correct_pixels": int(raw_corr),
        "raw_total_pixels": int(raw_total),
        "raw_pixel_accuracy": float(raw_pct),
        "solved": bool(solved),
        "duration_seconds": float(duration),
        "llm_calls": int(llm_calls),
        "best_program": str(final),
        "qd_enabled": QD_ENABLED,
        "qd_archive_size": qd_size,
        "qd_coverage_pct": float(qd_coverage),
        "curves": {
            "best_fit": best_curve,
            "llm_accepts": llm_acc,
            "diversity": div_curve,
            "coverage": cov_curve,
        },
    }

    save_ndjson(SAVE_NDJSON, result)
    return result


def run_multiple_tasks(task_glob: str, pop_size: int, gens: int, max_tasks: int = 0):
    files = sorted(glob.glob(task_glob))
    if max_tasks > 0:
        files = files[:max_tasks]

    if not files:
        print(f"\n❌ ERROR: No files found matching: {task_glob}")
        print(f"Current directory: {os.getcwd()}")
        return []

    print(f"\n{'#'*70}\n✓ Found {len(files)} tasks\n{'#'*70}\n")

    results = []
    solved_count = 0
    start = time.time()

    for i, f in enumerate(files, 1):
        print(f"\n{'='*70}\nTASK {i}/{len(files)}: {f}\n{'='*70}")
        try:
            r = run_single_task(f, pop_size, gens)
            results.append(r)
            if r.get("solved"):
                solved_count += 1
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted")
            break
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start

    print(f"\n{'#'*70}\nSUMMARY\n{'#'*70}")
    print(f"Completed: {len(results)}/{len(files)}")
    print(f"Solved (exact): {solved_count}/{len(results)} ({100*solved_count/len(results) if results else 0:.0f}%)")

    if results and QD_ENABLED:
        avg_coverage = sum(r.get("qd_coverage_pct", 0) for r in results) / len(results)
        avg_archive = sum(r.get("qd_archive_size", 0) for r in results) / len(results)
        print(f"Avg QD Coverage: {avg_coverage:.1f}% | Avg Archive: {avg_archive:.0f} niches")

    print(f"Time: {total_time:.0f}s (avg {total_time/len(results) if results else 0:.0f}s/task)")
    print(f"Results: {SAVE_NDJSON}")
    print(f"{'#'*70}\n")

    return results


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Quality Diversity GP + LLM for ARC (Qwen)")
    p.add_argument("--tasks", default="./training/*.json", help="Glob pattern for task files")
    p.add_argument("--pop", type=int, default=150, help="Population size")
    p.add_argument("--gens", type=int, default=50, help="Max generations")
    p.add_argument("--max_tasks", type=int, default=0, help="Limit number of tasks (0 = all)")
    p.add_argument("--model", default="", help="Override Qwen model name")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--debug", action="store_true", help="Enable debug output")
    p.add_argument("--no_post", action="store_true", help="Disable post-refinement")
    p.add_argument("--no_qd", action="store_true", help="Disable quality diversity")
    p.add_argument("--novelty_weight", type=float, default=0.3, help="Weight for novelty in fitness (0-1)")
    return p.parse_args()


def main():
    global QWEN_MODEL, llm, DEBUG_LLM, POST_LLM_ENABLED, QD_ENABLED, QD_NOVELTY_WEIGHT

    print("\n" + "=" * 70)
    print("QUALITY DIVERSITY GP + LLM FOR ARC (QWEN)")
    print("=" * 70)

    args = parse_args()
    random.seed(args.seed)
    DEBUG_LLM = args.debug

    if args.model:
        QWEN_MODEL = args.model
        llm = LLMEngine(QWEN_MODEL, QWEN_API_KEY, QWEN_API_URL)

    if args.no_post:
        POST_LLM_ENABLED = False
    if args.no_qd:
        QD_ENABLED = False

    QD_NOVELTY_WEIGHT = args.novelty_weight

    print(f"Model: {QWEN_MODEL}")
    print(f"Pop: {args.pop} | Gens: {args.gens}")
    print(f"Quality Diversity: {'ON' if QD_ENABLED else 'OFF'}")
    if QD_ENABLED:
        print(f"Novelty Weight: {QD_NOVELTY_WEIGHT}")
    print(f"Post-refine: {'ON' if POST_LLM_ENABLED else 'OFF'}")
    print(f"Task pattern: {args.tasks}")
    print(f"Max tasks: {args.max_tasks if args.max_tasks > 0 else 'ALL'}")
    print("=" * 70 + "\n")

    run_multiple_tasks(args.tasks, args.pop, args.gens, args.max_tasks)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"FATAL ERROR: {e}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
