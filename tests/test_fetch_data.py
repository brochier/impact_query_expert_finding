from context import impact_query_expert_finding
import impact_query_expert_finding.main.fetch_data
import os
import impact_query_expert_finding.data.io

current_folder = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.join(current_folder,'test_data/','data/')
dump_dir = os.path.join(current_folder,'test_data/', 'test_fetch_data_dump')

print("Testing data download...")
parameters = {
    'output_dir': working_dir,
    'dump_dir': dump_dir
}
impact_query_expert_finding.main.fetch_data.run(parameters)