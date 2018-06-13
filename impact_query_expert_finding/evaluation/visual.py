#  http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
# https://github.com/reiinakano/scikit-plot/blob/master/scikitplot/plotters.py

import numpy as np
import matplotlib 
#matplotlib.use('Agg') # Avoids graphical non support on remote server
import matplotlib.pyplot as plt
from scipy import interp
import os
from sklearn import manifold

path_fig1 = "alltopics"
path_fig2 = "roc_topics"
path_fig3 = "pr_topics"
path_fig4 = "allmodels"

fig_dim = (15.0,12.0)

def plot_evaluations(evals, prefix = "no_prefix", path_visuals = None):
    i = 0
    cmap = plt.cm.jet
    colors = [cmap(float(i) / len(evals)) for i in range(len(evals))]
    plt.figure(figsize=fig_dim)
    f, axarr = plt.subplots(1, 2, figsize=fig_dim)
    for t, eval in evals.items():
        precision, recall, thresholds_pr = eval["all"]["curves"]["precision"], eval["all"]["curves"]["recall"], eval["all"]["curves"]["thresholds_pr"]
        fpr, tpr, thresholds_roc = eval["all"]["curves"]["fpr"], eval["all"]["curves"]["tpr"], eval["all"]["curves"]["thresholds_roc"]
        ###############
        # PRECISION CURVE
        ###############

        mean_rec = np.linspace(0, 1, 100)
        pres = []
        for pre, rec in zip(precision, recall):
            pres.append(interp(mean_rec, rec[::-1], pre[::-1]))
            pres[-1][0] = 1.0
            #axarr[0].plot(rec, pre, lw=1, alpha=0.05)

        mean_pre = np.mean(pres, axis=0)
        mean_pre[-1] = 0.0
        axarr[0].plot(mean_rec, mean_pre, color=matplotlib.colors.to_hex(colors[i]), label=r'Mean PR for model "{0}" (AP={1:.3f})'.format(t,float(eval["all"]["metrics"]["AP"])))

        std_pre = np.std(pres, axis=0)
        pre_upper = np.minimum(mean_pre + std_pre, 1)
        pre_lower = np.maximum(mean_pre - std_pre, 0)
        axarr[0].fill_between(mean_rec, pre_lower, pre_upper, alpha=.2, color=matplotlib.colors.to_hex(colors[i]))

        axarr[0].set_xlabel('Recall')
        axarr[0].set_ylabel('Precision')
        axarr[0].set_ylim([-0.05, 1.05])
        axarr[0].set_xlim([-0.05, 1.05])
        axarr[0].set_title("Precision Recall Curve")
        axarr[0].legend(loc="lower right")

        ###############
        # ROC CURVE
        ###############

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        for fpr, tpr in zip(fpr, tpr):
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            #axarr[1].plot(fpr, tpr, lw=1, alpha=0.05)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        axarr[1].plot(mean_fpr, mean_tpr, color=matplotlib.colors.to_hex(colors[i]), label=r'Mean ROC for topic "{0}" (Mean AUC={1:.3f})'.format(t,float(eval["all"]["metrics"]["ROC AUC"])))

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        axarr[1].fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.2, color=matplotlib.colors.to_hex(colors[i]))

        axarr[1].set_xlim([-0.05, 1.05])
        axarr[1].set_ylim([-0.05, 1.05])
        axarr[1].set_xlabel('False Positive Rate')
        axarr[1].set_ylabel('True Positive Rate')
        axarr[1].set_title('Receiver operating characteristic')
        axarr[1].legend(loc="lower right")
        lw = 2
        axarr[1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

        i+=1

    title = "ROC curve and PR curve of several models"
    plt.suptitle(title)
    path = os.path.join(path_visuals, path_fig4+"_"+prefix+".png")
    plt.savefig(path)

def plot_evaluation(eval, prefix = "no_prefix", path_visuals=None):
    precision, recall, thresholds_pr = eval["curves"]["precision"], eval["curves"]["recall"], eval["curves"]["thresholds_pr"]
    fpr, tpr, thresholds_roc = eval["curves"]["fpr"], eval["curves"]["tpr"], eval["curves"]["thresholds_roc"]

    f, axarr = plt.subplots(1, 2, figsize=fig_dim)

    ###############
    # PRECISION CURVE
    ###############

    mean_rec = np.linspace(0, 1, 100)
    pres = []
    for pre, rec in zip(precision, recall):
        pres.append(interp(mean_rec, rec[::-1], pre[::-1]))
        pres[-1][0] = 1.0
        #axarr[0].plot(rec, pre, lw=1, alpha=0.05)

    mean_pre = np.mean(pres, axis=0)
    mean_pre[-1] = 0.0
    axarr[0].plot(mean_rec, mean_pre, color='b', label=r'Mean PR')

    std_pre = np.std(pres, axis=0)
    pre_upper = np.minimum(mean_pre + std_pre, 1)
    pre_lower = np.maximum(mean_pre - std_pre, 0)
    axarr[0].fill_between(mean_rec, pre_lower, pre_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    axarr[0].set_xlabel('Recall')
    axarr[0].set_ylabel('Precision')
    axarr[0].set_ylim([-0.05, 1.05])
    axarr[0].set_xlim([-0.05, 1.05])
    axarr[0].set_title("Precision Recall Curve")
    axarr[0].legend(loc="lower right")

    ###############
    # ROC CURVE
    ###############

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for fpr, tpr in zip(fpr, tpr):
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        #axarr[1].plot(fpr, tpr, lw=1, alpha=0.05)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    axarr[1].plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC')

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    axarr[1].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    axarr[1].set_xlim([-0.05, 1.05])
    axarr[1].set_ylim([-0.05, 1.05])
    axarr[1].set_xlabel('False Positive Rate')
    axarr[1].set_ylabel('True Positive Rate')
    axarr[1].set_title('Receiver operating characteristic')
    axarr[1].legend(loc="lower right")
    lw = 2
    axarr[1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')



    title = " - ".join([i + ": " + str(j) for i, j in eval["info"].items()]) + "\n"
    title += " - ".join([i + "={0:.3f}".format(j) for i, j in eval["metrics"].items()])
    plt.suptitle(title)
    path = os.path.join(path_visuals, path_fig1+"_"+prefix+".png")
    plt.savefig(path)

def plot_ROC_topics(evals, prefix = "no_prefix", path_visuals=None):
    i = 0
    plt.figure(figsize=fig_dim)
    ax = plt.subplot(111)
    cm = plt.get_cmap('tab20')
    ax.set_color_cycle([cm(1. * i / len(evals)) for i in range(len(evals))])
    for t,eval in evals.items():
        fpr, tpr, thresholds_roc = eval["curves"]["fpr"], eval["curves"]["tpr"], eval["curves"]["thresholds_roc"]
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        for fpr, tpr in zip(fpr, tpr):
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            #plt.plot(fpr, tpr, lw=1, alpha=0.1)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        plt.plot(mean_fpr, mean_tpr, label=r'Mean ROC for topic "{0}" (Mean AUC={1:.3f})'.format(t,float(eval["metrics"]["ROC AUC"])) )
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.1)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        lw = 2
        i+=1

    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.title("Mean ROC curve for each topic")
    path = os.path.join(path_visuals, path_fig2+"_"+prefix+".png")
    plt.savefig(path)

def plot_PreRec_topics(evals, prefix = "no_prefix", path_visuals=None):
    i = 0
    plt.figure(figsize=fig_dim)
    ax = plt.subplot(111)
    cm = plt.get_cmap('tab20')
    ax.set_color_cycle([cm(1. * i / len(evals)) for i in range(len(evals))])
    for t, eval in evals.items():
        precision, recall, thresholds_pr = eval["curves"]["precision"], eval["curves"]["recall"], eval["curves"]["thresholds_pr"]
        mean_rec = np.linspace(0, 1, 100)
        pres = []
        for pre, rec in zip(precision, recall):
            pres.append(interp(mean_rec, rec[::-1], pre[::-1]))
            pres[-1][0] = 1.0
            #plt.plot(rec, pre, lw=1, alpha=0.05)

        mean_pre = np.mean(pres, axis=0)
        mean_pre[-1] = 0.0
        plt.plot(mean_rec, mean_pre, label=r'Mean PR for topic "{0}" (AP={1:.3f})'.format(t,float(eval["metrics"]["AP"])))

        std_pre = np.std(pres, axis=0)
        pre_upper = np.minimum(mean_pre + std_pre, 1)
        pre_lower = np.maximum(mean_pre - std_pre, 0)
        plt.fill_between(mean_rec, pre_lower, pre_upper, alpha=.1)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([-0.05, 1.05])
        plt.xlim([-0.05, 1.05])
        plt.title("Precision Recall Curve")
        plt.legend(loc="upper right")
        i+=1

    plt.title("Mean precision-recall for each topic")
    path = os.path.join(path_visuals, path_fig3+"_"+prefix+".png")
    plt.savefig(path)

def drop_zeros(a_list):
    return [i for i in a_list if i > 0]


def log_binning(counter_dict, bin_count=100):
    max_x = np.log10(max(counter_dict.keys()))
    max_y = np.log10(max(counter_dict.values()))
    max_base = max([max_x, max_y])

    min_x = np.log10(min(drop_zeros(counter_dict.keys())))

    bins = np.logspace(min_x, max_base, num=bin_count)

    # Based off of: http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
    bin_means_y = (np.histogram(list(counter_dict.keys()), bins, weights=list(counter_dict.values()))[0] /
                   np.histogram(list(counter_dict.keys()), bins)[0])
    bin_means_x = (np.histogram(list(counter_dict.keys()), bins, weights=list(counter_dict.keys()))[0] /
                   np.histogram(list(counter_dict.keys()), bins)[0])

    return bin_means_x, bin_means_y

def plot_stats(dataset, dir_path, min_documents = 1, min_in_citations = 1, min_out_citations = 1):

    plt.figure(figsize=fig_dim)
    y = np.sort(np.squeeze(np.asarray(dataset.ds.associations.sum(axis=0))))
    #y = y[0:len(y)-1] # the author "Hyun-Kuk Kim" has 67251 documents...
    y = y[y >= min_documents]
    x = np.arange(len(y))
    plt.plot(x, y)
    plt.xlabel('Candidates indices')
    plt.ylabel('Number of linked documents')
    plt.title('Ordered number of documents per candidate')
    plt.savefig(os.path.join(dir_path, "documents_per_candidates.png"))

    plt.figure(figsize=fig_dim)
    y = np.sort(np.squeeze(np.asarray(dataset.ds.associations[:,dataset.gt.experts_mask].sum(axis=0))))
    x = np.arange(len(y))
    plt.plot(x, y)
    plt.xlabel('Experts indices')
    plt.ylabel('Number of linked documents')
    plt.title('Ordered number of documents per experts')
    plt.savefig(os.path.join(dir_path, "documents_per_experts.png"))

    plt.figure(figsize=fig_dim)
    y = np.sort(np.squeeze(np.asarray(dataset.ds.citations.sum(axis=0))))
    y = y[y >= min_in_citations]
    x = np.arange(len(y))
    plt.plot(x, y)
    plt.xlabel('Documents indices')
    plt.ylabel('Number of documents citing')
    plt.title('Ordered number of in citations per documents')
    plt.savefig(os.path.join(dir_path, "in_citations_per_documents.png"))

    plt.figure(figsize=fig_dim)
    y = np.sort(np.squeeze(np.asarray(dataset.ds.citations.sum(axis=1))))
    y = y[y >= min_out_citations]
    x = np.arange(len(y))
    plt.plot(x, y)
    plt.xlabel('Experts indices')
    plt.ylabel('Number of documents cited')
    plt.title('Ordered number of out citations per documents')
    plt.savefig(os.path.join(dir_path, "out_citations_per_documents.png"))


    # Degress distribution for dataset association
    degrees_documents = np.squeeze(np.asarray(dataset.ds.associations.sum(axis=1)))
    degrees_candidates = np.squeeze(np.asarray(dataset.ds.associations.sum(axis=0)))
    degrees_all = np.hstack((degrees_documents, degrees_candidates))
    N_doc = len(degrees_candidates)
    N_can = len(degrees_documents)
    degrees_documents = degrees_documents / N_doc
    degrees_candidates = degrees_candidates / N_can
    degrees_all = degrees_all / (N_doc + N_can)

    plt.figure(figsize=fig_dim)
    doc_counter = {v: 0 for v in set(degrees_documents)}
    for v in degrees_documents:
        doc_counter[v] += 1
    if 0 in doc_counter:
        del doc_counter[0]
    min_k = min(doc_counter.keys())
    max_k = max(doc_counter.keys())
    min_v = min(doc_counter.values())
    max_v = max(doc_counter.values())
    print("doc_counter keys: ", "    min", min_k, "    max", max_k)
    print("doc_counter values: ", "    min", min_v, "    max",max_v)
    plt.xscale('log')
    plt.yscale('log')
    ba_x, ba_y = log_binning(doc_counter)
    plt.scatter(list(doc_counter.keys()), list(doc_counter.values()), c='b', marker='x')
    plt.scatter(ba_x, ba_y, c='r', marker='s', s=50)
    plt.xlim( (10**int(np.log10(min_k)), 10**int(np.log10(max_k))) )
    plt.ylim( (.9, 10**(1+int(np.log10(max_v)))) )
    plt.xlabel('Degrees (normalized)')
    plt.ylabel('Number of nodes')
    plt.title('Power law of degrees distribution of documents_to_candidates bipartite graph.')
    plt.savefig(os.path.join(dir_path, "power_law_documents_to_candidates.png"))

    plt.figure(figsize=fig_dim)
    can_counter = {v: 0 for v in set(degrees_candidates)}
    for v in degrees_candidates:
        can_counter[v] += 1
    if 0 in can_counter:
        del can_counter[0]
    min_k = min(can_counter.keys())
    max_k = max(can_counter.keys())
    min_v = min(can_counter.values())
    max_v = max(can_counter.values())
    print("can_counter keys: ", "    min", min_k, "    max", max_k)
    print("can_counter values: ", "    min", min_v, "    max",max_v)
    plt.xscale('log')
    plt.yscale('log')
    ba_x, ba_y = log_binning(can_counter)
    plt.scatter(list(can_counter.keys()), list(can_counter.values()), c='b', marker='x')
    plt.scatter(ba_x, ba_y, c='r', marker='s', s=50)
    plt.xlim((10 ** int(np.log10(min_k)), 10 ** int(np.log10(max_k))))
    plt.ylim( (.9, 10**(1+int(np.log10(max_v)))) )
    plt.xlabel('Degrees (normalized)')
    plt.ylabel('Number of nodes')
    plt.title('Power law of degrees distribution of candidates_to_documents bipartite graph.')
    plt.savefig(os.path.join(dir_path, "power_law_candidates_to_documents.png"))

    plt.figure(figsize=fig_dim)
    counter = {v: 0 for v in set(degrees_all)}
    for v in degrees_all:
        counter[v] += 1
    if 0 in counter:
        del counter[0]
    min_k = min(counter.keys())
    max_k = max(counter.keys())
    min_v = min(counter.values())
    max_v = max(counter.values())
    print("counter keys: ", "    min", min_k, "    max", max_k)
    print("counter values: ", "    min", min_v, "    max", max_v)
    plt.xscale('log')
    plt.yscale('log')
    ba_x, ba_y = log_binning(counter)
    plt.scatter(list(counter.keys()), list(counter.values()), c='b', marker='x')
    plt.scatter(ba_x, ba_y, c='r', marker='s', s=50)
    plt.xlim((10 ** int(np.log10(min_k)), 10 ** int(np.log10(max_k))))
    plt.ylim( (.9, 10**(1+int(np.log10(max_v)))) )
    plt.xlabel('Degrees (normalized)')
    plt.ylabel('Number of nodes')
    plt.title('Power law of degrees distribution of candidates_documents bipartite graph.')
    plt.savefig(os.path.join(dir_path, "power_law_candidates_documents.png"))

    #  Degress distribution for dataset citations
    degrees_out = np.squeeze(np.asarray(dataset.ds.citations.sum(axis=1)))
    degrees_in = np.squeeze(np.asarray(dataset.ds.citations.sum(axis=0)))
    degrees = degrees_out + degrees_in
    N = dataset.ds.citations.shape[0]
    degrees_out = degrees_out / N
    degrees_in = degrees_in / N
    degrees = degrees / N

    plt.figure(figsize=fig_dim)
    out_counter = {v: 0 for v in set(degrees_out)}
    for v in degrees_out:
        out_counter[v] += 1
    if 0 in out_counter:
        del out_counter[0]
    min_k = min(out_counter.keys())
    max_k = max(out_counter.keys())
    min_v = min(out_counter.values())
    max_v = max(out_counter.values())
    print("out_counter keys: ", "    min", min_k, "    max", max_k)
    print("out_counter values: ", "    min", min_v, "    max", max_v)
    plt.xscale('log')
    plt.yscale('log')
    ba_x, ba_y = log_binning(out_counter)
    plt.scatter(list(out_counter.keys()), list(out_counter.values()), c='b', marker='x')
    plt.scatter(ba_x, ba_y, c='r', marker='s', s=50)
    plt.xlim((10 ** int(np.log10(min_k)), 10 ** int(np.log10(max_k))))
    plt.ylim( (.9, 10**(1+int(np.log10(max_v)))) )
    plt.xlabel('Degrees (normalized)')
    plt.ylabel('Number of nodes')
    plt.title('Power law of degrees distribution of out citations graph.')
    plt.savefig(os.path.join(dir_path, "power_law_out_citations.png"))

    plt.figure(figsize=fig_dim)
    in_counter = {v: 0 for v in set(degrees_in)}
    for v in degrees_in:
        in_counter[v] += 1
    if 0 in in_counter:
        del in_counter[0]
    min_k = min(in_counter.keys())
    max_k = max(in_counter.keys())
    min_v = min(in_counter.values())
    max_v = max(in_counter.values())
    print("in_counter keys: ", "    min", min_k, "    max", max_k)
    print("in_counter values: ", "    min", min_v, "    max", max_v)
    plt.xscale('log')
    plt.yscale('log')
    ba_x, ba_y = log_binning(in_counter)
    plt.scatter(list(in_counter.keys()), list(in_counter.values()), c='b', marker='x')
    plt.scatter(ba_x, ba_y, c='r', marker='s', s=50)
    plt.xlim((10 ** int(np.log10(min_k)), 10 ** int(np.log10(max_k))))
    plt.ylim( (.9, 10**(1+int(np.log10(max_v)))) )
    plt.xlabel('Degrees (normalized)')
    plt.ylabel('Number of nodes')
    plt.title('Power law of degrees distribution of in citations graph.')
    plt.savefig(os.path.join(dir_path, "power_law_in_citations.png"))

    plt.figure(figsize=fig_dim)
    counter = {v: 0 for v in set(degrees)}
    for v in degrees:
        counter[v] += 1
    if 0 in counter:
        del counter[0]
    min_k = min(counter.keys())
    max_k = max(counter.keys())
    min_v = min(counter.values())
    max_v = max(counter.values())
    print("counter keys: ", "    min", min_k, "    max", max_k)
    print("counter values: ", "    min", min_v, "    max", max_v)
    plt.xscale('log')
    plt.yscale('log')
    ba_x, ba_y = log_binning(counter)
    plt.scatter(list(counter.keys()), list(counter.values()), c='b', marker='x')
    plt.scatter(ba_x, ba_y, c='r', marker='s', s=50)
    plt.xlim((10 ** int(np.log10(min_k)), 10 ** int(np.log10(max_k))))
    plt.ylim( (.9, 10**(1+int(np.log10(max_v)))) )
    plt.xlabel('Degrees (normalized)')
    plt.ylabel('Number of nodes')
    plt.title('Power law of degrees distribution of in+out citations graph.')
    plt.savefig(os.path.join(dir_path, "power_law_in_out_citations.png"))


def tsne( embeddings, labels, legend, output_path, file_name, no_label = False):
    print("Computing TSNE with ", len(embeddings) ,"samples...")
    for p in [8]:
        plt.figure()
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=p)
        X_tsne = tsne.fit_transform(embeddings)
        plot_embedding(X_tsne, labels,legend, no_label = no_label, title = "t-SNE embedding" + file_name)
        plt.savefig(os.path.join(output_path,file_name), bbox_inches='tight', format='png', dpi=1000)

def plot( embeddings, labels, legend, output_path, file_name, no_label = False):
    print("Plot with ", len(embeddings) ,"samples...")
    plt.figure()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=8)
    X_tsne = tsne.fit_transform(embeddings)
    plot_embedding(X_tsne, labels,legend, no_label = no_label, title = file_name)
    plt.savefig(os.path.join(output_path,file_name), bbox_inches='tight', format='png', dpi=1000)

# Scale and visualize the embedding vectors
def plot_embedding(X, labels, legend, no_label = False, title=None):
    M = len(legend)
    if no_label is True:
        legend.append("no_label")
        labels = np.array(labels)
        labels[labels == -1] = M

    ax = plt.subplot(111)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    perturbation = X.mean(axis=0) / 100

    cm = plt.get_cmap('tab20')
    ax.set_color_cycle([cm(1. * i / len(legend)) for i in range(len(legend))])

    points_x = list()
    points_y = list()
    for l in legend:
        points_x.append(list())
        points_y.append(list())
    for i in range(X.shape[0]):
        for j,l in enumerate(labels[i]):
            add = perturbation
            points_x[l].append(X[i, 0] + 2*np.cos(j)*add[0])
            points_y[l].append(X[i, 1] + 2*np.sin(j)*add[1])
    for i,l in enumerate(legend):
        ax.scatter(points_x[i], points_y[i], s=1, label=l, alpha=1 - (i == M)*0.95)
    plt.xticks([]), plt.yticks([])

    ax.legend(loc="lower left", prop={'size': 3})

    if title is not None:
        plt.title(title)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
