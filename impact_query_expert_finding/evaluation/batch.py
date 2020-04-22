import impact_query_expert_finding.evaluation.metrics
import numpy as np
import impact_query_expert_finding.data.io

class EvalBatch:
    def __init__(self, dataset, dump_dir = None,  max_queries = None, topics_as_queries = False):
        self.seed = 0
        np.random.seed(seed=self.seed)
        self.dataset = None
        self.dump_dir = dump_dir
        self.dataset = dataset
        self.queries = list() # queries (doc_ids)
        self.queries_experts = list() # experts that wrote a query
        self.labels = list() # list of true topic indices for each queries
        self.labels_y_true = list() # list of ytrue experts boolean vectors per labels
        self.topics_as_queries = topics_as_queries
        if self.topics_as_queries:
            self.build_topics_evaluations()
        else:
            self.max_queries = max_queries
            self.build_individual_evaluations()

    def run_individual_evaluations(self, model):
        eval_batches = list()
        model.fit(None, None, dataset=self.dataset)
        N = len(self.queries)
        step = N / 10
        next = 0
        k = 0
        for i,d in enumerate(self.queries):
            if self.topics_as_queries:
                query = d
                leave_one_out = None
            else:
                query = self.dataset.ds.documents[d]
                leave_one_out = d
            y_score_candidates = model.predict(
                query,
                leave_one_out = leave_one_out)
            y_score_experts = y_score_candidates[self.dataset.gt.experts_mask]
            y_true_experts = self.labels_y_true[self.labels[i]]
            eval = self.eval_all(y_true_experts, y_score_experts)
            eval["info"] = {
                "topic": self.dataset.gt.topics[self.labels[i]],
                "query_number": i,
                "experts": self.queries_experts[i].tolist(),
                "query": query
            }
            eval_batches.append(eval)

            k += 1
            if k >= next:
                next += step
                print("Query", i + 1, "/", len(self.queries), " (", '{:.2%}'.format(k / N), ")")
        return eval_batches

    def build_topics_evaluations(self):
        np.random.seed(0)
        for i, t in enumerate(self.dataset.gt.topics):
            experts_indices = self.dataset.gt.associations[i, self.dataset.gt.experts_mask].nonzero()[1]
            experts_booleans = np.zeros(len(self.dataset.gt.experts_mask))
            experts_booleans[experts_indices] = 1
            self.labels_y_true.append(experts_booleans)
            query = " ".join(t.split("_"))
            self.queries.append(query)
            self.labels.append(i)
            self.queries_experts.append(experts_indices)
            print(
                "Topic",
                self.dataset.gt.topics[i],
                "=>",
                int(self.labels_y_true[i].sum()),
                "/",
                len(self.dataset.gt.experts_mask),
                "experts and",
                1,
                "query",
                "  (the topic itself)"
            )

    def build_individual_evaluations(self):
        np.random.seed(0)
        for i, t in enumerate(self.dataset.gt.topics):
            experts_indices = self.dataset.gt.associations[i, self.dataset.gt.experts_mask].nonzero()[1]
            experts_booleans = np.zeros(len(self.dataset.gt.experts_mask))
            experts_booleans[experts_indices] = 1
            self.labels_y_true.append(experts_booleans)
            documents_indices = np.unique(self.dataset.ds.associations[:, self.dataset.gt.experts_mask[experts_indices]].nonzero()[0])
            maxq = len(documents_indices)
            if self.max_queries is not None:
                np.random.shuffle(documents_indices)
                maxq = min(self.max_queries, maxq)
            for j in range(maxq):
                d = documents_indices[j]
                self.queries.append(d)
                self.labels.append(i)
                self.queries_experts.append( self.dataset.ds.associations[d,:].nonzero()[1] )
            print(
                "Topic",
                self.dataset.gt.topics[i],
                 "=>",
                 int(self.labels_y_true[i].sum()),
                 "/",
                 len(self.dataset.gt.experts_mask),
                 "experts and",
                 len(documents_indices),
                 "queries",
                 "  (max_queries = ", maxq ,")"
                 )




    """
    Merge evaluations given overall and per topics metrics
    """
    def merge_evaluations(self, eval_batches):
        all_eval = self.empty_eval()
        all_eval["info"]["topic"] = "all"
        topics_evals = dict()
        topic_count = dict()
        all_count = 0
        for t in self.dataset.gt.topics:
            topics_evals[t] = self.empty_eval()
            topics_evals[t]["info"]["topic"] = t
            topic_count[t] = 0
        for eval in eval_batches:
            t = eval["info"]["topic"]
            topic_count[t] += 1
            all_count += 1
            for key, value in eval["metrics"].items():
                topics_evals[t]["metrics"][key] += value
                all_eval["metrics"][key] += value
            for key, value in eval["curves"].items():
                all_eval["curves"][key].append(value)
                topics_evals[t]["curves"][key].append(value)

        for t in self.dataset.gt.topics:
            for key, value in topics_evals[t]["metrics"].items():
                topics_evals[t]["metrics"][key] = value / topic_count[t]
            print(topic_count[t], " requests done for topic: '" , t, "'")
        for key, value in all_eval["metrics"].items():
            all_eval["metrics"][key] = value / all_count
        print(all_count, " requests done for all topics.")
        eval = {
            "all": all_eval,
            "topics": topics_evals
        }

        if self.dump_dir is not None:
            impact_query_expert_finding.data.io.save_as_json(self.dump_dir, "eval_batches.json", eval_batches)
            impact_query_expert_finding.data.io.save_as_json(self.dump_dir, "eval_merged.json", eval)
        return eval

    def eval_all(self, y_true, y_score):
        precision, recall, thresholds_pr = impact_query_expert_finding.evaluation.metrics.get_precision_recall_curve(y_true, y_score)
        fpr, tpr, thresholds_roc = impact_query_expert_finding.evaluation.metrics.get_roc_curve(y_true, y_score)

        metrics = {
            "AP": impact_query_expert_finding.evaluation.metrics.get_average_precision(y_true, y_score).item(),
            "RR": impact_query_expert_finding.evaluation.metrics.get_reciprocal_rank(y_true, y_score),
            "P@5": impact_query_expert_finding.evaluation.metrics.get_precision_at_k(y_true, y_score, 5).item(),
            "P@10": impact_query_expert_finding.evaluation.metrics.get_precision_at_k(y_true, y_score, 10).item(),
            "P@50": impact_query_expert_finding.evaluation.metrics.get_precision_at_k(y_true, y_score, 50).item(),
            "P@100": impact_query_expert_finding.evaluation.metrics.get_precision_at_k(y_true, y_score, 100).item(),
            "P@200": impact_query_expert_finding.evaluation.metrics.get_precision_at_k(y_true, y_score, 200).item(),
            "ROC AUC": impact_query_expert_finding.evaluation.metrics.get_roc_auc_score(y_true, y_score).item()
        }

        curves = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds_pr": thresholds_pr.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds_roc": thresholds_roc.tolist()
        }

        return {"metrics": metrics, "curves": curves, "info":dict()}

    def empty_eval(self):
        return {
            "metrics": {
                "AP": 0,
                "RR": 0,
                "P@5": 0,
                "P@10": 0,
                "P@50": 0,
                "P@100": 0,
                "P@200": 0,
                "ROC AUC": 0
            },
            "curves": {
                "precision": list(),
                "recall": list(),
                "thresholds_pr": list(),
                "fpr": list(),
                "tpr": list(),
                "thresholds_roc": list()
            },
            "info":{

            }
        }
