Python code of the paper "On the Impact of the Query Set on the Evaluation of Expert Finding Systems" presented in the workshop BIRNDL hosted at SIGIR 2018. 
************************************************************************************************************************************************************

You can install the code as a python package and run the experiments shown in the paper "On the Impact of the Query Set on the Evaluation of Expert Finding Systems" presented in the 3rd Joint Workshop on Bibliometric-enhanced Information Retrieval and Natural Language Processing for Digital Libraries (BIRNDL 2018) hosted at SIGIR 2018. 

How to reproduce the results:
-----------------------------

If you work with conda, you can create a dedicated environment. Make sure you have install python 3.6 installed (other version of python 3 might work as well but were not tested).  

**Install:**

- sudo apt-get install unrar (for non-linux distrib, make sure you have rar or unrar installed)
- (optional) conda create -n impact_query_expert_finding
- (optional) source activate impact_query_expert_finding
- (optional) conda install python=3.6
- pip install --upgrade git+https://github.com/brochier/impact_query_expert_finding


**Fetch the data**:

Run this python script in a dedicated folder. Data will be fetched in the *data* directory and some statistics will be reported in the *data_info* directory.


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

Run this script to precompute the documents representations (TF, TF-IDF and LSA). You can specify several *input_dir* depending on the dataset (V1 or V2) you want to work with and if you want to use the entire dataset (dataset_cleaned) or a small version (dataset_association). 


```python
# documents_representations.py

import impact_query_expert_finding.main.language_models
import impact_query_expert_finding.data.config
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
# set "V1", "V2" to select the dataset you want to work with
# set "dataset_cleaned" to use the entire dataset or "dataset_association" to use a reduced version
input_dir = os.path.join(current_folder,'output/','data/', "V2", "dataset_cleaned") 
output_dir = os.path.join(input_dir,'documents_representations')

parameters = {
    'output_dir': output_dir,
    'input_dir': input_dir
}
impact_query_expert_finding.main.language_models.run(parameters)

```
