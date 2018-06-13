from context import impact_query_expert_finding
import impact_query_expert_finding.main.evaluate
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_folder,'test_data/','data/', "V2", "dataset_associations")
output_dir = os.path.join(current_folder,'test_data/', 'test_random')

print("Testing random model...")
parameters = {
     'output_dir': output_dir,
    'input_dir': input_dir,
    'algorithm': 'random',
    'seed': 0,
    'max_queries': 50,
}
impact_query_expert_finding.main.evaluate.run(parameters)