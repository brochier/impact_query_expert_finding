## Python package for the paper [ Impact of the Query Set on the Evaluation of Expert Finding Systems](https://arxiv.org/pdf/1806.10813.pdf) (BIRNDL18@SIGIR18)


You can install the code as a python package and run the experiments shown in the paper "Impact of the Query Set on the Evaluation of Expert Finding Systems" presented in the 3rd Joint Workshop on Bibliometric-enhanced Information Retrieval and Natural Language Processing for Digital Libraries (BIRNDL 2018) hosted at SIGIR 2018. 

How to reproduce the results:
-----------------------------

If you work with conda, you can create a dedicated environment. Make sure you have python 3.6 installed (other version of python 3 might work as well but were not tested).  

**Install:**

- sudo apt-get install unrar (for non-linux distrib, make sure you have rar or unrar installed)
- (optional) conda create -n impact_query_expert_finding
- (optional) source activate impact_query_expert_finding
- (optional) conda install python=3.6
- pip install --upgrade git+https://github.com/brochier/impact_query_expert_finding


**Fetch the data**:

Run this python script in a dedicated folder. Data will be fetched in the *data* directory and some statistics will be reported in the *data_info* directory. Two datasets are fetched (V1 and V2). A cleaned version using our preprocessing described in the paper is generated in the directory *dataset_cleaned* and a small version is generated in the *dataset_associations*. The name *associations* is used because the small dataset is generated using random walks starting from the ground truth expert-candidates social associations. 


```python
# fetch_data.py

import impact_query_expert_finding.main.fetch_data
import impact_query_expert_finding.data.io
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_folder,'output/','data/')
dump_dir = os.path.join(current_folder,'output/', 'data_info')

parameters = {
    'output_dir': output_dir,  # directory where the data will be stored
    'dump_dir': dump_dir       # dump directory where statistics will be produced
}
impact_query_expert_finding.main.fetch_data.run(parameters)
```


**Compute documents representations**:

Run this script to precompute the documents representations (TF, TF-IDF and LSA). You can specify several *input_dir* depending on the dataset (V1 or V2) you want to work with and if you want to use the entire dataset (dataset_cleaned) or a small version (dataset_association). The following script create 4 documents_representation for the 2 datasets and their cleaned and small version. 


```python
# documents_representations.py

import impact_query_expert_finding.main.language_models
import impact_query_expert_finding.data.config
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
datasets_versions = ["V1","V2"]
datasets_types = ["dataset_associations","dataset_cleaned"]

for dv in datasets_versions:
    for dt in datasets_types:
        input_dir = os.path.join(current_folder,'output/','data/', dv, dt) 
        output_dir = os.path.join(input_dir,'documents_representations')
        parameters = {
            'output_dir': output_dir,
            'input_dir': input_dir
        }
        impact_query_expert_finding.main.language_models.run(parameters)

```

**Run the experiments**:

This script runs the experiments on the small datasets (uncomment dataset_type to run on full data). The experiments are done with the three baseline algorithms, the three documents representations, the two datasets using parameter *eta=0.1* for the propagation model and subsampling the queries to a maximum of 50 per topic. The experiments are done with the topic_query and the document_query approaches.


```python
# experiments.py

import impact_query_expert_finding.main.evaluate
import impact_query_expert_finding.main.topics
import os

current_folder = os.path.dirname(os.path.abspath(__file__))

algorithms = ["panoptic", "vote", "bipartite_propagation"]
documents_representations = ["tf","tfidf","lsa"]
datasets_version = ["V1", "V2"]
#dataset_type = "dataset_cleaned"
dataset_type = "dataset_associations"
query_types = ["documents", "topics"]

for qt in query_types:
    for dv in datasets_version:
        input_dir = os.path.join(current_folder,'output/','data/', dv, dataset_type)
        for a in algorithms:
            for dr in documents_representations:
                output_dir = os.path.join(current_folder,'output/', "xp_"+qt+"_"+dv+"_"+dataset_type+"_"+a+"_"+dr)
                parameters = {
                    'output_dir': output_dir,
                    'input_dir': input_dir,
                    'algorithm': a,
                    'language_model': dr,
                    'vote_technique': 'rr',
                    'eta': 0.1,
                    'seed': 0,
                    'max_queries': 50,
                    'dump_dir': output_dir
                }
                if qt is "documents":
                    impact_query_expert_finding.main.evaluate.run(parameters)
                if qt is "topics":
                    impact_query_expert_finding.main.topics.run(parameters)


```

### Citing

If you use this code, please consider citing the paper:

	@inproceedings{brochier2018impact,
		title={Impact of the Query Set on the Evaluation of Expert Finding Systems},
		author={Brochier, Robin and Guille, Adrien and Rothan, Benjamin and Velcin, Julien},
		booktitle={3rd Joint Workshop on Bibliometric-enhanced Information Retrieval and Natural Language Processing for Digital Libraries (BIRNDL) at SIGIR},
		year={2018}
	}
