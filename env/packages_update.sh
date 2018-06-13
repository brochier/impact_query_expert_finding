source deactivate

conda update conda

source activate impact_query_expert_finding

pip install --upgrade pip

pip freeze --local | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U
