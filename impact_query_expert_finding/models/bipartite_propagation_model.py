import numpy as np
import scipy.sparse
import impact_query_expert_finding.language_models.wrapper
from sklearn.preprocessing import normalize
import os

# Normalize by setting negative scores to zero and
# dividing by the norm 2 value
def numpy_norm2(in_arr):
    in_arr.clip(min=0)
    norm = np.linalg.norm(in_arr)
    if norm > 0:
        return in_arr / norm
    return in_arr

class BiPropagationModel:

    def __init__(self, config, limit = 100, min_error = 0.0001, **kargs):
        self.type = kargs["language_model"]
        self.limit = limit
        self.min_error = min_error
        self.config = config
        self.dataset = None
        self.language_model = None
        self.bigraph = None
        self.eta = kargs["eta"]
        self.input_dir = kargs["input_dir"]

    def fit(self, x, Y, dataset = None ,mask = None):
        print("LM:", self.type, "  eta:", self.eta, "  limit:", self.limit, "  min_error:", self.min_error)
        doc_rep_dir = os.path.join(self.input_dir, "documents_representations")
        self.language_model =  impact_query_expert_finding.language_models.wrapper.LanguageModel(doc_rep_dir, type=self.type)
        self.dataset = dataset

        D = self.dataset.ds.associations.shape[0]
        C = self.dataset.ds.associations.shape[1]

        sparse_zeros_candidates = scipy.sparse.csr_matrix((C, C))
        sparse_zeros_documents = scipy.sparse.csr_matrix((D, D))

        left_side = scipy.sparse.vstack([sparse_zeros_candidates, self.dataset.ds.associations])
        right_side = scipy.sparse.vstack([self.dataset.ds.associations.T, sparse_zeros_documents])

        self.bigraph = scipy.sparse.hstack([left_side, right_side])
        self.bigraph = normalize(self.bigraph, axis=0, norm='l1')

    def predict(self, query, leave_one_out = None):
        C = self.dataset.ds.associations.shape[1]
        # Create jumping vector
        Pd = self.language_model.compute_similarity(query)

        if leave_one_out is not None:
            Pd[leave_one_out] = 0

        if Pd.sum() > 0:
            Pd = Pd / Pd.sum()
        Pc = np.zeros(self.dataset.ds.associations.shape[1])
        P = np.vstack([Pc.reshape(Pc.shape[0],1), Pd.reshape(Pd.shape[0],1)])
        if P.sum() > 0:
            P = P / P.sum()
        #P = scipy.sparse.crs_matrix(P)

        #  Build x init
        x = np.vstack([np.zeros(C).reshape(C,1), Pd.reshape(Pd.shape[0],1)])
        if x.sum() > 0:
            x = x / x.sum()

        # eta
        eta = self.eta

        # Q
        Q = self.bigraph.copy()
        if leave_one_out is not None:
            C = self.dataset.ds.associations.shape[1]
            M = scipy.sparse.lil_matrix(Q)
            M[C+leave_one_out] = 0
            M[:,C + leave_one_out] = 0
            Q = scipy.sparse.csr_matrix(M)

        error = self.min_error
        for i in range(self.limit):
            xprev = x
            x = (1-eta) * Q.dot(x)
            x = (1 - eta) * Q.dot(x) + eta * P # double walk
            error = np.linalg.norm(x - xprev)
            if error < self.min_error:
                break

        #print("Ended propagation at iter =", i+1," with error =", error)

        # Remove query node
        x = Q.dot(x)
        x = x.reshape((len(x),))
        candidates_scores = x[:C]

        return candidates_scores

