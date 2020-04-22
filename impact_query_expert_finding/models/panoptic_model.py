import numpy as np
import  scipy.spatial.distance
import scipy.sparse.linalg
import impact_query_expert_finding.language_models.wrapper
import scipy.sparse
import os

def cos_sparse(A,B):
    M = A.dot(B.T)
    normA = scipy.sparse.linalg.norm(A, axis=1)
    normA[normA == 0] = 1
    normB = scipy.sparse.linalg.norm(B, axis=1)
    normB[normB == 0] = 1
    if len(normA) > 1:
        print("norm not implemented for len(A) > 1 !")
    else:
        M = M / (normA*normB)
    return M[0]

# Normalize by setting negative scores to zero and
# dividing by the norm 2 value
def numpy_norm2(in_arr):
    in_arr.clip(min=0)
    norm = np.linalg.norm(in_arr)
    if norm > 0:
        return in_arr / norm
    return in_arr

class PanopticModel:

    def __init__(self, config, type = "tfidf", **kargs):
        self.type = kargs["language_model"]
        self.config = config
        self.dataset = None
        self.authors_metadocs = []
        self.language_model = None
        self.docs_vectors = None
        self.input_dir = kargs["input_dir"]

    def fit(self, x, Y, dataset = None, mask = None):
        print("LM:", self.type)
        doc_rep_dir = os.path.join(self.input_dir, "documents_representations")
        self.language_model =  impact_query_expert_finding.language_models.wrapper.LanguageModel(doc_rep_dir, type=self.type)
        self.dataset = dataset
        #self.corpus = language_models.wrapper.load_corpus(self.type, self.config)
        for i in range(self.dataset.ds.associations.shape[1]):
            self.authors_metadocs.append(" ".join([self.dataset.ds.documents[j] for j in self.dataset.ds.associations[:,i].nonzero()[0] ]))
        self.docs_vectors = self.language_model.vectorize(self.authors_metadocs)

    def predict(self, query, leave_one_out = None):
        metadocs_vectors = self.docs_vectors.copy()
        if leave_one_out is not None:
            candiates_to_recompute = self.dataset.ds.associations[leave_one_out, :].nonzero()[1]
            for c in candiates_to_recompute:
                metadocs = " ".join([self.dataset.ds.documents[j] for j in self.dataset.ds.associations[:, c].nonzero()[0]  if c != j ])
                metadocs_vectors[c] = self.language_model.vectorize(metadocs)[0]

        # Compute documents scores and normalize them
        query_vector = self.language_model.vectorize([query])

        metadocs_scores = None
        if type(query_vector) is np.ndarray:
            metadocs_scores =  scipy.spatial.distance.cdist(query_vector, metadocs_vectors, metric='cosine').flatten()
        else:
            metadocs_scores = cos_sparse(query_vector,metadocs_vectors)
        candidates_scores = np.ravel(numpy_norm2(metadocs_scores))
        return candidates_scores

    def predict_deprecated(self, query, leave_one_out = None):
        metadocs_vectors = self.docs_vectors.copy()
        if leave_one_out is not None:
            candiates_to_recompute = self.dataset.ds.associations[leave_one_out, :].nonzero()[1]
            for c in candiates_to_recompute:
                metadocs = " ".join([self.dataset.ds.documents[j] for j in self.dataset.ds.associations[:, c].nonzero()[0]  if c != j ])
                metadocs_vectors[c] = self.language_model.vectorize(metadocs)[0]

        # Compute documents scores and normalize them
        query_vector = self.language_model.vectorize([query])

        metadocs_scores = None
        if type(query_vector) is np.ndarray:
            metadocs_scores =  scipy.spatial.distance.cdist(query_vector, metadocs_vectors, metric='cosine').flatten()
        else:
            metadocs_scores = cos_sparse(query_vector,metadocs_vectors)
        candidates_scores = np.ravel(numpy_norm2(metadocs_scores))
        return candidates_scores