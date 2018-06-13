from context import impact_query_expert_finding
import impact_query_expert_finding.main.language_models
import impact_query_expert_finding.data.config
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_folder,'test_data/','data/', "V2", "dataset_associations")
output_dir = os.path.join(input_dir,'documents_representations')
dump_dir = os.path.join(current_folder,'test_data/', 'test_language_models_dump')
print("Testing language_models...")
parameters = {
    'output_dir': output_dir,
    'input_dir': input_dir,
    'dump_dir': dump_dir,
}
impact_query_expert_finding.main.language_models.run(parameters)
