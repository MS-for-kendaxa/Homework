def load_data(fname):
    """
    Loads data and labels the entities in the data.
    :param fname:
    :return: List of lines (lists of single words), list of labels.
    """

    data_raw = [l.strip().split() for l in open(fname).readlines()]
    data_clean = []
    labels = []
    for l in data_raw:
        current_clean_l = []
        current_labels = []
        current_label = "o"
        for i, w in enumerate(l):
            if w.startswith("<START:"):
                ent_tp = w[len("<START:"):-1]
                if l[i + 2] == "<END>":
                    current_label = ent_tp + "-U"
                else:
                    current_label = ent_tp + "-S"
            elif w == "<END>":
                if current_label.endswith("-I"):
                    current_labels[-1] = current_labels[-1][:-2] + "-E"
                current_label = "o"
            else:
                current_clean_l.append(w)
                current_labels.append(current_label)
                if current_label.endswith("-S"):
                    current_label = current_label[:-2] + "-I"
        if len(current_clean_l)>0:
            data_clean.append(current_clean_l)
            labels.append(current_labels)
    return data_clean, labels