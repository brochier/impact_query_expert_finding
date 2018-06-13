from context import impact_query_expert_finding
import impact_query_expert_finding.main.evaluate
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_folder,'test_data/','data/', "V2", "dataset_associations")
output_dir = os.path.join(current_folder,'test_data/', 'test_panoptic')

print("Testing p@noptic model...")
parameters = {
    'output_dir': output_dir,
    'input_dir': input_dir,
    'algorithm': 'panoptic',
    'language_model': 'lsa',
    'vote_technique': 'panoptic',
    'max_queries': 50,
    'dump_dir': output_dir
}
impact_query_expert_finding.main.evaluate.run(parameters)