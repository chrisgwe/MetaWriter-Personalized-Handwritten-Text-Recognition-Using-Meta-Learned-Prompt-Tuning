def LM_str_to_ind(labels, str, oov_symbol=None):
    oov_index = labels.index(oov_symbol) if oov_symbol in labels else None
    indices = []
    for c in str:
        try:
            indices.append(labels.index(c))
        except ValueError:
            if oov_index is not None:
                indices.append(oov_index)
            else:
                raise ValueError(f"'{c}' is not in list and no OOV symbol provided")
    return indices



def LM_ind_to_str(labels, ind, oov_symbol=None):
    if oov_symbol is not None:
        res = []
        for i in ind:
            if i < len(labels):
                res.append(labels[i])
            else:
                res.append(oov_symbol)
    else:
        res = [labels[i] for i in ind]
    return "".join(res)


