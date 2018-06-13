#!/bin/bash

source deactivate

conda update conda

source activate impact_query_expert_finding

pip install --upgrade pip

pip install -r requirements.txt
