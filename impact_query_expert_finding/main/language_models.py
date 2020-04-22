import impact_query_expert_finding.data.config
import impact_query_expert_finding.language_models.wrapper
import impact_query_expert_finding.data.sets
import impact_query_expert_finding.data.io
import impact_query_expert_finding.tools.graphs
import impact_query_expert_finding.evaluation.visual

def run(parameters):
    output_dir = parameters["output_dir"]
    input_dir = parameters["input_dir"]
    impact_query_expert_finding.data.io.check_and_create_dir(output_dir)
    impact_query_expert_finding.data.io.check_and_create_dir(input_dir)

    print("Building language model")
    dataset = impact_query_expert_finding.data.sets.DataSet("aminer")
    dataset.load(input_dir)
    impact_query_expert_finding.language_models.wrapper.build_all(dataset.ds.documents, output_dir)

