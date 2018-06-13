#!/bin/bash

conda update conda

cd ..                                                           # avoids automatic env file loading
conda create --name impact_query_expert_finding python=3.6
cd impact_query_expert_finding                                  # avoids automatic env file loading 
source activate impact_query_expert_finding

bash env/update.sh


