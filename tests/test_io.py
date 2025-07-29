import numpy as np
import pickle
from scipy.sparse import csr_matrix

def test_pickle_matrix(tmp_path):
    M = csr_matrix([[1,0],[0,1]])
    path = tmp_path/"mat.pkl"
    with open(path, "wb") as f: pickle.dump(M, f)
    with open(path, "rb") as f: loaded = pickle.load(f)
    assert (loaded != M).nnz == 0
