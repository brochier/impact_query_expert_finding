import collections
import numpy as np
import sklearn.preprocessing

class RandomChoice(object):
    def __init__(self, cograph_choices, cograph_prob, depth=100):
        C = len(cograph_choices)
        self.depth = depth
        self.cograph_choices = cograph_choices
        self.cograph_prob = cograph_prob
        self.choices = [None] * C
        for i in range(C):
            self.choices[i] = collections.deque(np.random.choice(self.cograph_choices[i],
                                                                   size=self.depth,
                                                                   p=self.cograph_prob[i]))

    def __getitem__(self, arg):

        if len(self.choices[arg]) == 0:
           self.choices[arg] = collections.deque(np.random.choice(self.cograph_choices[arg],
                                                                  size=self.depth,
                                                                  p=self.cograph_prob[arg]))
        return self.choices[arg].popleft()

def extract_experts_citations_subgraph(dataset, length_walks = 10, number_of_walks = 1000):
    experts_documents_mask = np.unique(dataset.ds.associations[:, dataset.gt.experts_mask].nonzero()[0])
    citation_graph = dataset.ds.citations + dataset.ds.citations.T
    citation_graph[citation_graph > 1] = 1

    doc_to_doc = sklearn.preprocessing.normalize(citation_graph, norm='l1')
    nonzero_doc_to_doc = doc_to_doc.nonzero()
    data_doc_to_doc = doc_to_doc.data
    K_doc_to_doc = len(data_doc_to_doc)
    choices_doc_to_doc = list()
    prob_doc_to_doc = list()
    C_doc_to_doc = doc_to_doc.shape[0]
    k = 0
    for i in range(C_doc_to_doc):
        choices_doc_to_doc.append(list())
        prob_doc_to_doc.append(list())
        if nonzero_doc_to_doc[0][k] > i:
            choices_doc_to_doc[i].append(i)
            prob_doc_to_doc[i].append(1.0)
        else:
            while k < K_doc_to_doc and nonzero_doc_to_doc[0][k] == i:
                choices_doc_to_doc[i].append(nonzero_doc_to_doc[1][k])
                prob_doc_to_doc[i].append(data_doc_to_doc[k])
                k += 1
        choices_doc_to_doc[i] = np.array(choices_doc_to_doc[i], dtype=np.uint32)
        prob_doc_to_doc[i] = np.array(prob_doc_to_doc[i], dtype=np.float_)
    randomChoice_doc_to_doc = RandomChoice(choices_doc_to_doc, prob_doc_to_doc, depth=10)

    documents_set = set(experts_documents_mask)
    docs_experts_set = set(experts_documents_mask)
    N = number_of_walks
    step = N / 10
    next = 0
    K = 0

    print("Starting random walks...")
    for i in range(number_of_walks):
        for start_node in docs_experts_set:
            docs = list([start_node])
            # Random walks
            for j in range(length_walks):
                docs.append(randomChoice_doc_to_doc[docs[j]])
                if docs[j + 1] != start_node and docs[j + 1] in docs_experts_set:
                    documents_set.update(docs)
        K += 1
        if K >= next:
            next += step
            print("Walk ", K, "/", N, " (", '{:.2%}'.format(K / N), ")")
            print("Len document_set: ", len(documents_set))

    candidates_set = np.sort(np.unique(dataset.ds.associations[list(documents_set)].nonzero()[1]))

    return np.sort(np.array(list(documents_set))), candidates_set

# Operate random walks from experts to extract a connected subgraph
def extract_experts_associations_subgraph(dataset, length_walks = 10, number_of_walks = 1000):
    doc_to_can = sklearn.preprocessing.normalize(dataset.ds.associations, norm='l1')
    nonzero_doc_to_can = doc_to_can.nonzero()
    data_doc_to_can = doc_to_can.data
    K_doc_to_can = len(data_doc_to_can)
    choices_doc_to_can = list()
    prob_doc_to_can = list()
    C_doc_to_can = doc_to_can.shape[0]
    k = 0
    for i in range(C_doc_to_can):
        choices_doc_to_can.append(list())
        prob_doc_to_can.append(list())
        if nonzero_doc_to_can[0][k] > i:
            choices_doc_to_can[i].append(i)
            prob_doc_to_can[i].append(1.0)
        else:
            while k < K_doc_to_can and nonzero_doc_to_can[0][k] == i:
                choices_doc_to_can[i].append(nonzero_doc_to_can[1][k])
                prob_doc_to_can[i].append(data_doc_to_can[k])
                k += 1
        choices_doc_to_can[i] = np.array(choices_doc_to_can[i], dtype=np.uint32)
        prob_doc_to_can[i] = np.array(prob_doc_to_can[i], dtype=np.float_)
    randomChoice_doc_to_can = RandomChoice(choices_doc_to_can, prob_doc_to_can, depth=10)

    can_to_doc = sklearn.preprocessing.normalize(dataset.ds.associations.T, norm='l1')
    nonzero_can_to_doc = can_to_doc.nonzero()
    data_can_to_doc = can_to_doc.data
    K_can_to_doc = len(data_can_to_doc)
    choices_can_to_doc = list()
    prob_can_to_doc = list()
    C_can_to_doc = can_to_doc.shape[0]
    k = 0
    for i in range(C_can_to_doc):
        choices_can_to_doc.append(list())
        prob_can_to_doc.append(list())
        if nonzero_can_to_doc[0][k] > i:
            choices_can_to_doc[i].append(i)
            prob_can_to_doc[i].append(1.0)
        else:
            while k < K_can_to_doc and nonzero_can_to_doc[0][k] == i:
                choices_can_to_doc[i].append(nonzero_can_to_doc[1][k])
                prob_can_to_doc[i].append(data_can_to_doc[k])
                k += 1
        choices_can_to_doc[i] = np.array(choices_can_to_doc[i], dtype=np.uint32)
        prob_can_to_doc[i] = np.array(prob_can_to_doc[i], dtype=np.float_)
    randomChoice_can_to_doc = RandomChoice(choices_can_to_doc, prob_can_to_doc, depth=10)

    documents_set = set()
    candidates_set = set(dataset.gt.experts_mask)
    expert_set = set(dataset.gt.experts_mask)

    N = number_of_walks
    step = N / 10
    next = 0
    K = 0

    same_category = np.array([0] * len(dataset.gt.experts_mask))
    other_category = np.array([0] * len(dataset.gt.experts_mask))

    print("Starting random walks...")
    for i in range(number_of_walks):
        for start_node in dataset.gt.experts_mask:
            cans = list([start_node])
            docs = list()
            # Random walks
            labels = set(dataset.gt.associations[:,start_node].nonzero()[0])
            for j in range(length_walks):
                docs.append(randomChoice_can_to_doc[cans[j]])
                cans.append(randomChoice_doc_to_can[docs[j]])
                if cans[j+1] != start_node and cans[j+1] in expert_set:
                    documents_set.update(docs)
                    candidates_set.update(cans)
                    if any(i in labels for i in dataset.gt.associations[:,cans[j+1]].nonzero()[0]):
                        same_category[list(labels)] += 1
                    else:
                        other_category[list(labels)] += 1

        K += 1
        if K >= next:
            next += step
            print("Walk ", K, "/", N, " (", '{:.2%}'.format(K / N), ")")
            print("Len document_set: ", len(documents_set))
            print("Len candidates_set: ", len(candidates_set))

    print("-STATS-RW-")
    for i, c in enumerate(dataset.gt.topics):
        print(c, "same=", same_category[i], "other=", other_category[i], "(", same_category[i] * 100/ (same_category[i]+other_category[i]) ,"%)")
    print("-STATS-RW-")

    return np.sort(np.array(list(documents_set))), np.sort(np.array(list(candidates_set)))
