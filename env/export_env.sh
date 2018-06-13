#!/bin/bash

source activate impact_query_expert_finding

pip freeze --local | grep -v '^\-e' | cut -d = -f 1 > requirements.txt
