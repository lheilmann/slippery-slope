import numpy as np

from src.utils import glove_dir


def get_embeddings(n_dim=300) -> dict:
    embeddings = {}
    with open(glove_dir() / "glove.840B.{}d.txt".format(n_dim), "r") as f:
        for line in f:
            values = line.split(" ")
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings[word] = vector

    return embeddings
