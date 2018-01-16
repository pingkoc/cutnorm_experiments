import numpy as np

def sbm(community_sizes, prob_mat):
    community_sizes = np.array(community_sizes)
    prob_mat = np.array(prob_mat)
    if prob_mat.shape[0] != prob_mat.shape[1]:
        raise ValueError("prob_mat needs to be square matrix.")
    if not np.all(community_sizes > 0):
        raise ValueError("Each community size in community_sizes "
                         "needs to be greater than 1.")
    if len(community_sizes) != len(prob_mat):
        raise ValueError("community_sizes needs to be of size n if "
                         "prob_mat is nxn")
    if not (np.all(prob_mat >= 0) and np.all(prob_mat <= 1)):
        raise ValueError("Needs to be a valid probability matrix.")

    n = np.sum(community_sizes)
    sbm = np.zeros((n, n))
    for i, size_i in enumerate(community_sizes):
        for j, size_j in enumerate(community_sizes):
            prev_sum_i = np.sum(community_sizes[:i])
            prev_sum_j = np.sum(community_sizes[:j])
            prob = prob_mat[i][j]
            sample = np.random.choice(2, size=size_i*size_j, p=[1-prob, prob]).reshape((size_i, size_j))
            sbm[prev_sum_i:prev_sum_i + size_i, prev_sum_j:prev_sum_j + size_j] = sample

    return sbm

def sbm_prob(community_sizes, prob_mat):
    community_sizes = np.array(community_sizes)
    prob_mat = np.array(prob_mat)
    if prob_mat.shape[0] != prob_mat.shape[1]:
        raise ValueError("prob_mat needs to be square matrix.")
    if not np.all(community_sizes > 0):
        raise ValueError("Each community size in community_sizes "
                         "needs to be greater than 1.")
    if len(community_sizes) != len(prob_mat):
        raise ValueError("community_sizes needs to be of size n if "
                         "prob_mat is nxn")
    if not (np.all(prob_mat >= 0) and np.all(prob_mat <= 1)):
        raise ValueError("Needs to be a valid probability matrix.")

    n = np.sum(community_sizes)
    sbm = np.zeros((n, n))
    for i, size_i in enumerate(community_sizes):
        for j, size_j in enumerate(community_sizes):
            prev_sum_i = np.sum(community_sizes[:i])
            prev_sum_j = np.sum(community_sizes[:j])
            prob = prob_mat[i][j]
            prob_block = prob*np.ones((size_i, size_j))
            sbm[prev_sum_i:prev_sum_i + size_i,
                prev_sum_j:prev_sum_j + size_j] = prob_block

    return sbm

def _sbm_autoregressive_gen_prob_matrix(prob_list):
    n_probs = len(prob_list)
    prob_matrix = np.zeros((n_probs, n_probs))
    for i in range(n_probs):
        for j in range(n_probs):
            if i == j:
                prob_matrix[i, i] = prob_list[i]
            else:
                prob_matrix[i, j] = (prob_list[i] * prob_list[j])**(abs(i - j))
    return prob_matrix

def sbm_autoregressive(community_sizes, prob_list):
    # Construct the prob matrix from prob list using
    # idea similar to that of autoregressive model
    # Each element in the prob matrix is
    # (p_i * p_j)^(abs(i - j))
    prob_matrix = _sbm_autoregressive_gen_prob_matrix(prob_list)
    return sbm(community_sizes, prob_matrix)

def sbm_autoregressive_prob(community_sizes, prob_list):
    # Construct the prob matrix from prob list using
    # idea similar to that of autoregressive model
    # Each element in the prob matrix is
    # (p_i * p_j)^(abs(i - j))
    prob_matrix = _sbm_autoregressive_gen_prob_matrix(prob_list)
    return sbm_prob(community_sizes, prob_matrix)

def make_symmetric_triu(mat):
    mat = np.triu(mat)
    mat = np.maximum(mat, mat.T)
    return mat
