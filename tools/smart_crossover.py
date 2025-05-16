import numpy as np
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
