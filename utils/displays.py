import networkx as nx
import re


def show_deps(file_path):
    G = nx.DiGraph()
    roots = []

    with open(file_path, 'r') as f:
        for l in f.readlines():
            fun = l.split(' : ')[0]
            deps = l.split(' : ')[1][:-1].split(' ')
            try:
                deps.remove('')
            except:
                pass
            if len(deps) == 0:
                roots.append(fun)
            else:
                [G.add_edge(d, fun) for d in deps]

    for s in roots:
        print(s)
        spacer = {s: 4}
        if s not in G.nodes:
            print('')
            continue
        for prereq, target in nx.dfs_edges(G, s):
            spacer[target] = spacer[prereq] + 4
            print ('{spacer}+ {t}'.format(
                                         spacer=' ' * spacer[prereq],
                                         t=target))
        print('')
    
    return