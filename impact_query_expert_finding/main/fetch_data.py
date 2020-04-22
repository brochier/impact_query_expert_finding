import impact_query_expert_finding.data.config
import impact_query_expert_finding.data.io
import impact_query_expert_finding.data.sets
import impact_query_expert_finding.evaluation.visual
import impact_query_expert_finding.tools.graphs

import os
import patoolib
import zipfile
import pkg_resources
import urllib.request

def run(parameters):
    config_path = pkg_resources.resource_filename("impact_query_expert_finding", 'conf.yml')
    config = impact_query_expert_finding.data.config.load_from_yaml(config_path)
    impact_query_expert_finding.data.io.check_and_create_dir(parameters["output_dir"])
    impact_query_expert_finding.data.io.check_and_create_dir(parameters["dump_dir"])

    #  Dataset V1
    print("Downloading and build dataset V1... (may take a few minutes)")
    data_folder = os.path.join(parameters["output_dir"], "V1")
    impact_query_expert_finding.data.io.check_and_create_dir(data_folder)
    dest_file = os.path.join(data_folder, config["data_citation_network_rar_file_name_v1"])
    text_file = os.path.join(data_folder, config["data_citation_network_text_file_name_v1"])
    
    try:
        os.remove(dest_file)
        os.remove(text_file)
    except OSError:
        pass
    urllib.request.urlretrieve(config["data_citation_network_url_v1"], dest_file)
    patoolib.extract_archive(dest_file, outdir=data_folder)


    full_path = os.path.join(data_folder, "dataset_full")
    full_stats_path = os.path.join(parameters["dump_dir"], "dataset_v1_stats_full")
    impact_query_expert_finding.data.io.check_and_create_dir(full_path)
    impact_query_expert_finding.data.io.check_and_create_dir(full_stats_path)
    input_file = os.path.join(data_folder, config["data_citation_network_text_file_name_v1"])
    dataset = impact_query_expert_finding.data.sets.DataSet("aminer", input_file, config["data_experts_folder"], version = "V1")
    impact_query_expert_finding.evaluation.visual.plot_stats(dataset,
                                                full_stats_path,
                                                min_documents=100,
                                                min_in_citations=100,
                                                min_out_citations=100)
    dataset.save(full_path)

    print("Creating cleaned dataset")
    cleaned_path = os.path.join(data_folder, "dataset_cleaned")
    cleaned_stats_path = os.path.join(parameters["dump_dir"], "dataset_v1_stats_cleaned")
    impact_query_expert_finding.data.io.check_and_create_dir(cleaned_path)
    impact_query_expert_finding.data.io.check_and_create_dir(cleaned_stats_path)
    dataset.clean_associations(max_documents_per_candidates=100, min_documents_per_candidates=1)
    impact_query_expert_finding.evaluation.visual.plot_stats(dataset,
                                                             cleaned_stats_path,
                                                             min_documents=10,
                                                             min_in_citations=10,
                                                             min_out_citations=10
                                                             )
    dataset.save(cleaned_path)

    # Build subgraph for associations
    print("Creating associations sub graph dataset")
    associations_path = os.path.join(data_folder, "dataset_associations")
    associations_stats_path = os.path.join(parameters["dump_dir"], "dataset_v1_stats_associations")
    impact_query_expert_finding.data.io.check_and_create_dir(associations_path)
    impact_query_expert_finding.data.io.check_and_create_dir(associations_stats_path)
    dataset.clean_associations(max_documents_per_candidates=100, min_documents_per_candidates=2)
    documents_set, candidates_set = impact_query_expert_finding.tools.graphs.extract_experts_associations_subgraph(
        dataset,
        length_walks=5,
        number_of_walks=50
        )
    dataset.reduce(documents_set, candidates_set)
    impact_query_expert_finding.evaluation.visual.plot_stats(dataset,
                                                             associations_stats_path,
                                                             min_documents=10,
                                                             min_in_citations=10,
                                                             min_out_citations=10
                                                             )
    dataset.save(associations_path)



    #  Dataset V2
    print("Downloading and build dataset V2... (may take a few minutes)")
    data_folder = os.path.join(parameters["output_dir"], "V2")
    impact_query_expert_finding.data.io.check_and_create_dir(data_folder)
    dest_file = os.path.join(data_folder, config["data_citation_network_rar_file_name_v2"])
    text_file = os.path.join(data_folder, config["data_citation_network_text_file_name_v2"])

    try:
        os.remove(dest_file)
        os.remove(text_file)
    except OSError:
        pass
    urllib.request.urlretrieve(config["data_citation_network_url_v2"], dest_file)
    zip_ref = zipfile.ZipFile(dest_file, 'r')
    zip_ref.extractall(data_folder)
    zip_ref.close()

    full_path = os.path.join(data_folder, "dataset_full")
    full_stats_path = os.path.join(parameters["dump_dir"], "dataset_v2_stats_full")
    impact_query_expert_finding.data.io.check_and_create_dir(full_path)
    impact_query_expert_finding.data.io.check_and_create_dir(full_stats_path)
    input_file = os.path.join(data_folder, config["data_citation_network_text_file_name_v2"])
    dataset = impact_query_expert_finding.data.sets.DataSet("aminer", input_file, config["data_experts_folder"], version = "V2")
    impact_query_expert_finding.evaluation.visual.plot_stats(dataset,
                                                full_stats_path,
                                                min_documents=100,
                                                min_in_citations=100,
                                                min_out_citations=100)
    dataset.save(full_path)

    print("Creating cleaned dataset")
    cleaned_path = os.path.join(data_folder, "dataset_cleaned")
    cleaned_stats_path = os.path.join(parameters["dump_dir"], "dataset_v2_stats_cleaned")
    impact_query_expert_finding.data.io.check_and_create_dir(cleaned_path)
    impact_query_expert_finding.data.io.check_and_create_dir(cleaned_stats_path)
    dataset.clean_associations(max_documents_per_candidates=100, min_documents_per_candidates=1)
    impact_query_expert_finding.evaluation.visual.plot_stats(dataset,
                                                             cleaned_stats_path,
                                                             min_documents=10,
                                                             min_in_citations=10,
                                                             min_out_citations=10
                                                             )
    dataset.save(cleaned_path)

    # Build subgraph for associations
    print("Creating associations sub graph dataset")
    associations_path = os.path.join(data_folder, "dataset_associations")
    associations_stats_path = os.path.join(parameters["dump_dir"], "dataset_v2_stats_associations")
    impact_query_expert_finding.data.io.check_and_create_dir(associations_path)
    impact_query_expert_finding.data.io.check_and_create_dir(associations_stats_path)
    dataset.clean_associations(max_documents_per_candidates=100, min_documents_per_candidates=2)
    documents_set, candidates_set = impact_query_expert_finding.tools.graphs.extract_experts_associations_subgraph(
        dataset,
        length_walks=5,
        number_of_walks=50
    )
    dataset.reduce(documents_set, candidates_set)
    impact_query_expert_finding.evaluation.visual.plot_stats(dataset,
                                                             associations_stats_path,
                                                             min_documents=10,
                                                             min_in_citations=10,
                                                             min_out_citations=10
                                                             )
    dataset.save(associations_path)




    

