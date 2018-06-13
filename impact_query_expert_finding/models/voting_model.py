import numpy as np
import impact_query_expert_finding.language_models.wrapper
import scipy.sparse
import os

# Normalize by setting negative scores to zero and
# dividing by the norm 2 value
def numpy_norm2(in_arr):
    in_arr.clip(min=0)
    norm = np.linalg.norm(in_arr)
    if norm > 0:
        return in_arr / norm
    return in_arr

class VotingModel:

    def __init__(self, config,type = "tfidf", vote="rr", **kargs):
        self.type = kargs["language_model"]
        if "vote_technique" in kargs:
            self.vote = kargs["vote_technique"]
        else:
            self.vote = "panoptic"
        self.config = config
        self.dataset = None
        self.language_model = None
        self.input_dir = kargs["input_dir"]

    def fit(self, x, Y, dataset = None, mask = None):
        print("LM:", self.type, "  vote:", self.vote)
        doc_rep_dir = os.path.join(self.input_dir, "documents_representations")
        self.language_model =  impact_query_expert_finding.language_models.wrapper.LanguageModel(doc_rep_dir, type=self.type)
        self.dataset = dataset

    def predict(self, query, leave_one_out = None):
        # Compute documents scores and normalize them
        documents_scores = self.language_model.compute_similarity(query)
        #documents_scores = numpy_norm2(documents_scores)

        if leave_one_out is not None:
            documents_scores[leave_one_out] = 0

        documents_sorting_indices = documents_scores.argsort()[::-1]
        document_ranks = documents_sorting_indices.argsort() + 1

        if self.vote == "rr":
            # Sort scores and get ranks
            candidates_scores = np.ravel(
                self.dataset.ds.associations.T.dot(scipy.sparse.diags(1 / document_ranks, 0)).T.sum(
                    axis=0))  # A.T.dot(np.diag(b)) multiply each column of A element-wise by b
            return candidates_scores
        elif self.vote == "log_rr":
            candidates_scores = self.dataset.ds.associations.T.dot(scipy.sparse.diags(1/np.log(document_ranks,0))).T.sum(axis=0)
            return candidates_scores
        elif self.vote == "panoptic":
            candidates_scores = np.ravel(
                self.dataset.ds.associations.T.dot(scipy.sparse.diags(documents_scores, 0)).T.sum(
                    axis=0))
            return candidates_scores
        else:
            print("Voting technique ", self.vote, "doesn't exists !!")





