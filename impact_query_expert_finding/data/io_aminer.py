import itertools
import os
import pkg_resources

def load_papers(infname):
    print("Loading papers from '", infname,"'")
    papers = list()
    with open(infname, 'r') as f:
        for key, group in itertools.groupby(f, key=lambda l: l.strip(' \n\r') == ''):
            if not key:
                refs = []
                authors = []
                title, venue, year, idx, abstract = [''] * 5
                for item in group:
                    item = item.strip(' \r\n')
                    if item.startswith('#*'):
                        title = item[2:].strip()
                    elif item.startswith('#@'):
                        authors = item[2:].split(',')
                        authors = [a.strip() for a in authors]
                    elif item.startswith('#t'):
                        year = item[2:].strip()
                    elif item.startswith('#c'):
                        venue = item[2:].strip()
                    elif item.startswith('#index'):
                        idx = int(item[6:].strip())
                    elif item.startswith('#!'):
                        abstract = item[2:].strip()
                    elif item.startswith('#%'):
                        refs.append(int(item[2:].strip()))
                if len(title+abstract) > 50:
                    papers.append({
                        "idx":idx,
                        "title":title,
                        "venue":venue,
                        "authors":authors,
                        "year":year,
                        "refs":refs,
                        "abstract":abstract
                    })
    return papers

def load_experts(foldername, version = "V1"):
    print("Loading experts from '", foldername, "'")
    list_of_files = {
        "boosting": "Boosting.txt",
        "data_mining": "Data-Mining.txt",
        "information_extraction": "Information-Extraction.txt",
        "intelligent_agents": "Intelligent-Agents.txt",
        "machine_learning": "Machine-Learning.txt",
        "natural_language_processing": "Natural-Language-Processing.txt",
        "ontology_alignment": "Ontology-Alignment.txt",
        "planning": "Planning.txt",
        "semantic_web": "Semantic-Web.txt",
        "support_vector_machine": "Support-Vector-Machine.txt",
        "computer_vision": "Computer-Vision.txt",
        "cryptography": "Cryptography.txt",
        "neural_networks": "Neural-Networks.txt"
    }
    list_of_new_files = {
        "information_extraction": "New-Information-Extraction.txt",
        "intelligent_agents": "New-Intelligent-Agents.txt",
        "machine_learning": "New-Machine-Learning.txt",
        "natural_language_processing": "New-Natural-Language-Processing.txt",
        "planning": "New-Planning.txt",
        "semantic_web": "New-Semantic-Web.txt",
        "support_vector_machine": "New-Support-Vector-Machine.txt",
    }
    authors = dict()

    if version == "V1":
        for topic, filename in list_of_files.items():
            authors[topic] = list()
            file = pkg_resources.resource_filename("impact_query_expert_finding", os.path.join(foldername, filename))
            print("Reading file: ",filename)
            with open(file, 'rb') as f:
                for line in f:
                    try:
                        string = line.decode('utf-8').strip()
                        authors[topic].append(string)
                    except:
                        print("Can't decode:",line)
    elif version == "V2":
        for topic, filename in list_of_new_files.items():
            authors[topic] = list()
            file = pkg_resources.resource_filename("impact_query_expert_finding", os.path.join(foldername, filename))
            print("Reading file: ",filename)
            with open(file, 'rb') as f:
                for line in f:
                    try:
                        string = line.decode('utf-8').strip()
                        authors[topic].append(string)
                    except:
                        print("Can't decode:",line)
    else:
        print("Unknown version provided !")
        return None

    return authors
