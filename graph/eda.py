from laion_graph import load_graph_and_keymap, get_neighbors



if __name__ == "__main__":


    g, k = load_graph_and_keymap(graph_file="results/laion400M_graph_malefemale_neighbors_only_500000_ALL.pkl", keymap_file="results/key_map_malefemale_neighbors_only_500000_ALL.pkl.gz")

    nw = get_neighbors(g, k, "woman", threshold=5)
    nm = get_neighbors(g, k, "man", threshold=5)

    m_words = [t[0] for t in nm]
    m_words = set(m_words)
    w_words = [t[0] for t in nw]
    w_words = set(w_words)

    m_not_in_women = [m for m in nm if m[0] not in w_words]
    mbias, _ = zip(*m_not_in_women)

    w_not_in_men = [w for w in nw if w[0] not in m_words]
    wbias, _ = zip(*w_not_in_men)

    print(mbias[0:30])
    print(wbias[0:30])



    g2, k2 = load_graph_and_keymap(graph_file="results/laion400M_graph_full_20000.pkl", keymap_file="results/key_map_full_20000.pkl.gz")

    nd = get_neighbors(g2, k2, "donut")
    nc = get_neighbors(g2, k2, "salad")

    d_words = [t[0] for t in nd]
    d_words = set(d_words)
    c_words = [t[0] for t in nc]
    c_words = set(c_words)

    d_not_in_salad = [d for d in nd if d[0] not in c_words]
    dbias, _ = zip(*d_not_in_salad)
    
    c_not_in_donut = [c for c in nc if c[0] not in d_words]
    cbias, _ = zip(*c_not_in_donut)

    print()

    print(dbias[0:15])
    print(cbias[0:15])


