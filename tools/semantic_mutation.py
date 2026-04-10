import re
import os
import random
import requests
import numpy as np
import deap.gp as gp
import deap.creator as creator


DSL_REFERENCE = """
You are a program synthesizer for ARC-AGI tasks.
You write programs using a strictly typed DSL.
All programs must be valid Python expressions using ONLY the primitives below.

=== TYPES ===
  tuple       : a 2-D grid represented as a tuple of tuples of ints
  int         : a colour value in range 0-9
  bool        : True or False
  frozenset   : a set of objects (cells/patches)

=== PRIMITIVES (name, inputs -> output) ===
  objects(grid:tuple, univ:bool, diag:bool, wall:bool) -> frozenset
  replace(grid:tuple, old_color:int, new_color:int) -> tuple
  leastcolor(grid:tuple) -> int
  fill(grid:tuple, color:int, obj:frozenset) -> tuple
  vmirror(grid:tuple) -> tuple
  lefthalf(grid:tuple) -> tuple
  righthalf(grid:tuple) -> tuple
  cellwise(a:tuple, b:tuple, fallback:int) -> tuple
  hline_pred(obj:frozenset) -> bool

=== TERMINALS ===
  0,1,2,3,4,5,6,7,8,9
  T  (True)
  F  (False)
  EmptySet  (empty frozenset)

=== RULES ===
  - Single expression tree only.
  - Root must return type tuple.
  - Input grid is always ARG0.
  - No Python built-ins, loops, or imports.
  - Output ONLY the expression, nothing else.
"""


def _grid_to_str(grid):
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def _build_prompt(task, elite_programs, elite_fitnesses):
    lines = [DSL_REFERENCE, "\n=== TRAINING EXAMPLES ==="]
    for i, ex in enumerate(task["train"]):
        lines.append(f"\n-- Example {i+1} --")
        lines.append("INPUT:")
        lines.append(_grid_to_str(ex["input"]))
        lines.append("OUTPUT:")
        lines.append(_grid_to_str(ex["output"]))
    lines.append("\n=== CURRENT ELITE PROGRAMS ===")
    for rank, (prog, fit) in enumerate(zip(elite_programs, elite_fitnesses)):
        lines.append(f"Rank {rank+1} (fitness={fit:.1f}): {prog}")
    lines.append(
        "\n=== YOUR TASK ===\n"
        "Propose 3 NEW programs that are semantically different from the elites above.\n"
        "Each program must:\n"
        "  - Be a single valid DSL expression\n"
        "  - Return type tuple\n"
        "  - Use ARG0 as the input grid\n"
        "  - Try to fix the mistakes in the elite programs\n\n"
        "Output exactly 3 lines. Each line is one complete DSL expression.\n"
        "No numbering, no markdown, no extra text."
    )
    return "\n".join(lines)


def _call_llm(prompt, model="mistral-large-latest", max_tokens=1024):
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY environment variable not set.")
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.9,
        "top_p": 0.95,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


_PSET_NAME_MAP = {
    "objects":    "objects",
    "replace":    "replace",
    "leastcolor": "leastcolor",
    "fill":       "fill",
    "vmirror":    "vmirror",
    "lefthalf":   "lefthalf",
    "righthalf":  "righthalf",
    "cellwise":   "cellwise",
    "hline_pred": "hline_pred",
    "ARG0":       "ARG0",
    "EmptySet":   "EmptySet",
    "IDENTITY":   "IDENTITY",
    "T":          "T",
    "F":          "F",
    **{str(i): str(i) for i in range(10)},
}


def _tokenize(expr_str):
    expr_str = expr_str.strip()
    tokens = []
    i = 0
    while i < len(expr_str):
        ch = expr_str[i]
        if ch in (" ", "\n", "\r", "\t"):
            i += 1
        elif ch in ("(", ")", ","):
            tokens.append(ch)
            i += 1
        else:
            j = i
            while j < len(expr_str) and expr_str[j] not in (" ", "(", ")", ",", "\n"):
                j += 1
            tokens.append(expr_str[i:j])
            i = j
    return tokens


def _expr_to_prefix(tokens):
    prefix = []

    def parse():
        if not tokens:
            return
        tok = tokens.pop(0)
        if tok in ("(", ")", ","):
            return parse()
        prefix.append(tok)
        if tokens and tokens[0] == "(":
            tokens.pop(0)
            while tokens and tokens[0] != ")":
                if tokens[0] == ",":
                    tokens.pop(0)
                else:
                    parse()
            if tokens:
                tokens.pop(0)

    parse()
    return prefix


def _prefix_to_deap_tree(prefix_tokens, pset):
    all_prims = {}
    for type_list in pset.primitives.values():
        for p in type_list:
            all_prims[p.name] = p

    all_terms = {}
    for type_list in pset.terminals.values():
        for t in type_list:
            all_terms[t.name] = t

    tree_nodes = []
    token_queue = list(prefix_tokens)

    def build():
        if not token_queue:
            raise ValueError("Unexpected end of token stream")
        tok = token_queue.pop(0)
        mapped = _PSET_NAME_MAP.get(tok, tok)
        if mapped in all_prims:
            prim = all_prims[mapped]
            tree_nodes.append(prim)
            for _ in prim.args:
                build()
        elif mapped in all_terms:
            tree_nodes.append(all_terms[mapped])
        else:
            raise ValueError(f"Unknown DSL token: '{tok}'")

    build()
    if token_queue:
        raise ValueError(f"Leftover tokens: {token_queue}")
    return creator.Individual(tree_nodes)


def parse_dsl_expression(expr_str, pset):
    try:
        tokens = _tokenize(expr_str)
        prefix = _expr_to_prefix(tokens)
        return _prefix_to_deap_tree(prefix, pset)
    except Exception:
        return None


def type_check(individual, pset):
    try:
        gp.compile(individual, pset)
        return True
    except Exception:
        return False


def evaluate_candidate(individual, toolbox, task):
    try:
        func = toolbox.compile(expr=individual)
        total = 0
        for ex in task["train"]:
            inp = tuple(map(tuple, ex["input"]))
            tgt = np.array(ex["output"])
            try:
                out = np.array(func(inp))
                if out.shape == tgt.shape:
                    total += int(np.sum(out == tgt))
            except Exception:
                pass
        return float(total)
    except Exception:
        return 0.0


class SemanticMutationOperator:

    def __init__(self, pset, toolbox, task, g_plateau=5, n_cand=3, n_elites=3,
                 llm_model="mistral-large-latest", verbose=True):
        self.pset        = pset
        self.toolbox     = toolbox
        self.task        = task
        self.g_plateau   = g_plateau
        self.n_cand      = n_cand
        self.n_elites    = n_elites
        self.llm_model   = llm_model
        self.verbose     = verbose
        self._best_fitness   = -1.0
        self._stagnant_gens  = 0
        self._seen_programs  = set()
        self._trigger_count  = 0

    def step(self, population, generation, offspring):
        current_best = max(
            ind.fitness.values[0]
            for ind in population
            if ind.fitness.valid
        )
        if current_best > self._best_fitness:
            self._best_fitness  = current_best
            self._stagnant_gens = 0
        else:
            self._stagnant_gens += 1

        if self._stagnant_gens >= self.g_plateau:
            if self.verbose:
                print(f"\n[SemanticMutation] Gen {generation}: "
                      f"{self._stagnant_gens} stagnant gens "
                      f"(best={self._best_fitness:.1f}). Triggering LLM...")

            candidates = self._run_llm_mutation(population)

            if candidates:
                offspring = self._inject(offspring, candidates)
                self._stagnant_gens = 0
                self._trigger_count += 1
                if self.verbose:
                    print(f"[SemanticMutation] Injected {len(candidates)} candidates. "
                          f"Total LLM calls: {self._trigger_count}")
            else:
                if self.verbose:
                    print("[SemanticMutation] No valid candidates survived filtering.")

            return offspring, True

        return offspring, False

    def _get_elites(self, population):
        valid = [ind for ind in population if ind.fitness.valid]
        return sorted(valid, key=lambda ind: ind.fitness.values[0], reverse=True)[:self.n_elites]

    def _run_llm_mutation(self, population):
        elites = self._get_elites(population)
        if not elites:
            return []

        elite_strs      = [str(e) for e in elites]
        elite_fitnesses = [e.fitness.values[0] for e in elites]
        prompt          = _build_prompt(self.task, elite_strs, elite_fitnesses)

        try:
            raw_response = _call_llm(prompt, model=self.llm_model)
        except Exception as e:
            if self.verbose:
                print(f"[SemanticMutation] LLM call failed: {e}")
            return []

        if self.verbose:
            print(f"[SemanticMutation] LLM response:\n{raw_response}\n")

        candidate_lines = self._extract_candidates(raw_response)
        surviving = []

        for expr_str in candidate_lines:
            individual = parse_dsl_expression(expr_str, self.pset)
            if individual is None:
                if self.verbose:
                    print(f"  [Filter] PARSE FAIL: {expr_str[:80]}")
                continue

            if not type_check(individual, self.pset):
                if self.verbose:
                    print(f"  [Filter] TYPE FAIL: {expr_str[:80]}")
                continue

            prog_key = str(individual)
            if prog_key in self._seen_programs:
                if self.verbose:
                    print(f"  [Filter] DUPLICATE: {expr_str[:80]}")
                continue
            self._seen_programs.add(prog_key)

            fitness = evaluate_candidate(individual, self.toolbox, self.task)
            individual.fitness.values = (fitness,)
            surviving.append(individual)

            if self.verbose:
                print(f"  [Filter] ACCEPTED (fitness={fitness:.1f}): {expr_str[:80]}")

        return surviving

    def _extract_candidates(self, raw_response):
        raw_response = re.sub(r"```[a-zA-Z]*", "", raw_response).replace("```", "")
        lines = [l.strip() for l in raw_response.splitlines()]
        candidates = []
        for line in lines:
            if not line:
                continue
            line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if not line:
                continue
            if "(" in line or line in _PSET_NAME_MAP:
                candidates.append(line)
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique[:self.n_cand * self.n_elites]

    def _inject(self, offspring, candidates):
        if not candidates:
            return offspring
        offspring_sorted = sorted(
            offspring,
            key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else -1,
        )
        n_replace = min(len(candidates), len(offspring_sorted))
        for i in range(n_replace):
            offspring_sorted[i] = candidates[i]
        return offspring_sorted

    def summary(self):
        print(f"[SemanticMutation Summary] "
              f"Triggered {self._trigger_count} time(s). "
              f"Unique programs seen: {len(self._seen_programs)}. "
              f"Best fitness achieved: {self._best_fitness:.1f}.")
