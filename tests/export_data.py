from context import impact_query_expert_finding
import impact_query_expert_finding.data.sets
import numpy as np
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_folder,'test_data/','data/', "V2", "dataset_cleaned")
output_filename = os.path.join(dataset_path,'dblp.npz')

dataset = impact_query_expert_finding.data.sets.DataSet("aminer")
dataset.load(dataset_path)
dataset.print_stats()


A_da = dataset.ds.associations.astype(np.bool)
A_dd = dataset.ds.citations.astype(np.bool)
T = dataset.ds.documents
L_a = dataset.gt.associations[:, dataset.gt.experts_mask].astype(np.bool)
L_a_mask = dataset.gt.experts_mask
tags = dataset.gt.topics
print(A_da.shape, dataset.gt.associations.T.shape)
L_d = A_da.dot(dataset.gt.associations.T).astype(np.bool)
L_d_mask = L_d.sum(axis=1).nonzero()[0]
L_d = L_d[L_d_mask]

data_to_save = {
    "A_da":A_da,
    "A_dd":A_dd,
    "T":T,
    "L_d":L_d,
    "L_d_mask":L_d_mask,
    "L_a":L_a,
    "L_a_mask": L_a_mask,
    "tags":tags
}
np.savez_compressed(output_filename, **data_to_save)