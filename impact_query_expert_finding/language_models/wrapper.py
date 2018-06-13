import pkg_resources
import impact_query_expert_finding.data.config
import impact_query_expert_finding.data.io
import os
import gensim
import numpy as np
import scipy.sparse
import time
import re


stop_words = {'an', 'through', 'own', 'too', 'again', 'nor', 'doesn', "didn't", 'whom', 'couldn', 'shouldn', 'him',
              'after', "weren't", 'or', 'itself', "shouldn't", "isn't", 'to', 'needn', "hasn't", 'of', 'her', 'why',
              'against', 'should', 'themselves', 'didn', "won't", 'the', 'our', 'here', 'we', 'does', 'at', 'during',
              'it', 'such', 'for', 'about', 'wasn', 'once', 'did', 'out', 'but', 'their', 'myself', 'will', 'herself',
              'below', "needn't", "should've", 'my', 'no', 'down', 'his', 'they', 'is', 'so', "shan't", "aren't",
              "it's", 'each', 'into', "you've", 'himself', 'above', 'aren', 'wouldn', 'most', 'this', "wasn't",
              'theirs', "mustn't", 'm', 'both', "wouldn't", 'mustn', 'because', 'all', "couldn't", "mightn't", "you'd",
              'with', 'other', 'am', 's', 'll', "hadn't", 'won', 're', "don't", 'were', 'me', 'these', 'off', "you'll",
              'as', 'over', 'while', 'haven', 'on', 'then', "haven't", 'are', 'she', 'having', 'mightn', 'yourself',
              'have', 't', 'some', 'isn', 'shan', 've', 'can', 'had', "you're", 'few', 'under', 'up', "doesn't",
              'yourselves', 'he', 'from', 'those', 'only', 'hadn', 'that', 'don', 'which', 'not', 'doing', 'yours',
              "that'll", 'been', 'in', 'where', 'hasn', 'weren', 'very', 'being', 'more', 'be', 'your', 'there', 'do',
              "she's", 'now', 'ain', 'a', 'how', 'you', 'has', 'further', 'who', 'o', 'before', 'just', 'same',
              'ourselves', 'until', 'and', 'between', 'ma', 'ours', 'what', 'them', 'y', 'hers', 'was', 'when', 'its',
              'if', 'by', 'd', 'i', 'than', 'any'}

config_path = pkg_resources.resource_filename("impact_query_expert_finding", 'conf.yml')
config = impact_query_expert_finding.data.config.load_from_yaml(config_path)



def build_all(texts, output_dir, min_count=3, cpu_workers=16, window_size = 10, filter_min = 20, filter_max = 400):
    output_dir_temp = output_dir

    # Create Texts
    train_phraser(texts, output_dir)
    texts = tokenize_stream(texts, output_dir)
    for i, t in enumerate(texts):
        if len(t) == 0:
            t.append("[EMPTY]")

    impact_query_expert_finding.data.io.save_as_json(output_dir_temp, config["texts_filename"], texts)

    print("Size of corpus: ", len(texts))

    # Create Dictionary
    print("Create dictionary...")
    dictionary = gensim.corpora.dictionary.Dictionary(texts, prune_at=None)
    dictionary.filter_extremes(no_below=min_count, no_above=0.5, keep_n=None)
    dictionary.compactify()
    dictionary.save(os.path.join(output_dir_temp, "dictionary"))

    # Create corpus
    print("Create corpus...")
    corpus = [dictionary.doc2bow(text) for text in texts]
    count = 0
    for i, c in enumerate(corpus):
        if len(c) == 0:
            count += 1
    print("Number of empty texts", count)

    gensim.corpora.MmCorpus.serialize(os.path.join(output_dir_temp, "corpus"), corpus)
    corpus_index = gensim.similarities.docsim.Similarity(
        os.path.join(output_dir_temp, "corpus_index"), corpus, num_features=len(dictionary))
    corpus_index.save(os.path.join(output_dir_temp, "corpus_index"))

    # Create length_filter
    corpus_length_filter = [i for i, c in enumerate(corpus) if len(c) <= filter_min or len(c) >= filter_max]
    impact_query_expert_finding.data.io.save_as_json(output_dir_temp, config["length_filter_filename"], corpus_length_filter)

    #  tf-idf
    print("Compute Tfidf...")
    tfidf = gensim.models.TfidfModel(
        corpus,
        wlocal = np.log1p,
        normalize=True
    )
    corpus_tfidf = tfidf[corpus]
    gensim.corpora.MmCorpus.serialize(os.path.join(output_dir_temp, "corpus_tfidf"), corpus_tfidf)
    tfidf.save(os.path.join(output_dir_temp, "tfidf"))
    corpus_tfidf_index = gensim.similarities.docsim.Similarity(
        os.path.join(output_dir_temp, "corpus_tfidf_index"), corpus_tfidf,
        num_features=len(dictionary))
    corpus_tfidf_index.save(os.path.join(output_dir_temp, "corpus_tfidf_index"))

    # lsa
    lsa_num_topics = 500
    print("Compute LSA...")
    lsa = gensim.models.LsiModel(
        corpus_tfidf,
        num_topics=lsa_num_topics
    )
    corpus_lsa = lsa[corpus_tfidf]
    gensim.corpora.MmCorpus.serialize(os.path.join(output_dir_temp, "corpus_lsa"), corpus_lsa)
    lsa.save(os.path.join(output_dir_temp, "lsa"))
    corpus_lsa_index = gensim.similarities.docsim.Similarity(
        os.path.join(output_dir_temp, "corpus_lsa_index"), corpus_lsa, num_features=lsa_num_topics)
    corpus_lsa_index.save(os.path.join(output_dir_temp, "corpus_lsa_index"))

    # We create a stamp
    stamp = {"datetime": time.time()}
    impact_query_expert_finding.data.io.save_as_json(output_dir_temp, config["stamp_filename"], stamp)

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def preprocess_text(stream):
    tokens_stream = [
        gensim.utils.simple_preprocess(cleanhtml(gensim.utils.decode_htmlentities(t))
        ,deacc=True, min_len=1, max_len=100) for t in
        stream]
    for i, tokens in enumerate(tokens_stream):
        tokens_stream[i] = [j for j in tokens if j not in stop_words]
    return tokens_stream

def train_phraser(stream, output_dir, max_num_words=3):
    tokens_stream = preprocess_text(stream)
    grams = None
    for i in range(max_num_words - 1):
        print(i + 2, "-grams")
        phrases = gensim.models.phrases.Phrases(tokens_stream)
        grams = gensim.models.phrases.Phraser(phrases)
        tokens_stream.extend(list(grams[tokens_stream]))
    grams.save(os.path.join(output_dir, "phraser"))


def tokenize(text, input_dir, max_num_words=3, grams=None):
    tokens = preprocess_text([text])[0]
    if grams == None:
        grams = gensim.models.phrases.Phraser.load(os.path.join(input_dir, "phraser"))
    for i in range(max_num_words - 1):
        tokens = grams[tokens]
    return tokens


def tokenize_stream(stream, input_dir, max_num_words=3):
    tokens_stream = preprocess_text(stream)
    grams = gensim.models.phrases.Phraser.load(os.path.join(input_dir, "phraser"))
    for i in range(max_num_words - 1):
        tokens_stream = list(grams[tokens_stream])
    return tokens_stream


#  Reuse the pipeline in the similarity function but keep
# all loaded data in memory for serial call to similarity
# Use Similarity class instead of MatrixSimilarity if data doesn't fit in memory
class LanguageModel:
    def __init__(self, input_dir, type="tfidf"):
        self.type = type
        self.input_dir = input_dir
        self.grams = gensim.models.phrases.Phraser.load(os.path.join(input_dir, "phraser"))
        if self.type == "tf":
            self.dictionary = gensim.corpora.Dictionary.load(
                os.path.join(input_dir, "dictionary"))
            self.corpus_index = gensim.similarities.docsim.SparseMatrixSimilarity.load(
                os.path.join(input_dir, "corpus_index"))
            self.corpus_index.output_prefix = os.path.join(input_dir,"corpus_index")
            self.corpus_index.check_moved()
        elif self.type == "tfidf":
            self.dictionary = gensim.corpora.Dictionary.load(
                os.path.join(input_dir, "dictionary"))
            self.tfidf = gensim.models.TfidfModel.load(os.path.join(input_dir, "tfidf"))
            self.corpus_tfidf_index = gensim.similarities.docsim.SparseMatrixSimilarity.load(
                os.path.join(input_dir, "corpus_tfidf_index"))
            self.corpus_tfidf_index.output_prefix = os.path.join(input_dir,"corpus_tfidf_index")
            self.corpus_tfidf_index.check_moved()
        elif self.type == "tfidf_lsa":
            self.dictionary = gensim.corpora.Dictionary.load(
                os.path.join(input_dir, "dictionary"))
            self.tfidf = gensim.models.TfidfModel.load(os.path.join(input_dir, "tfidf"))
            self.corpus_tfidf_index = gensim.similarities.docsim.SparseMatrixSimilarity.load(
                os.path.join(input_dir, "corpus_tfidf_index"))
            self.corpus_tfidf_index.output_prefix = os.path.join(input_dir,"corpus_tfidf_index")
            self.corpus_tfidf_index.check_moved()
            self.lsa = gensim.models.LsiModel.load(os.path.join(input_dir, "lsa"))
            self.corpus_lsa_index = gensim.similarities.docsim.MatrixSimilarity.load(
                os.path.join(input_dir, "corpus_lsa_index"))
            self.corpus_lsa_index.output_prefix = os.path.join(input_dir,"corpus_lsa_index")
            self.corpus_lsa_index.check_moved()
        elif self.type == "lsa":
            self.dictionary = gensim.corpora.Dictionary.load(
                os.path.join(input_dir, "dictionary"))
            self.tfidf = gensim.models.TfidfModel.load(os.path.join(input_dir, "tfidf"))
            self.lsa = gensim.models.LsiModel.load(os.path.join(input_dir, "lsa"))
            self.corpus_lsa_index = gensim.similarities.docsim.MatrixSimilarity.load(
                os.path.join(input_dir, "corpus_lsa_index"))
            self.corpus_lsa_index.output_prefix = os.path.join(input_dir, "corpus_lsa_index")
            self.corpus_lsa_index.check_moved()
        else:
            raise ValueError(
                'model type "' + self.type + '" is not an option. Available names are ["tf, tfidf, lsa, tfidf_lsa, word2vec_mean, word2vec_tfidf"].')

    def get_word_from_dictionary(self, index):
        return self.dictionary[index]

    def compute_similarity(self, input_text):
        if self.type == "tf":
            return np.array(
                self.corpus_index[self.dictionary.doc2bow(tokenize(input_text, self.input_dir, grams=self.grams))])
        elif self.type == "tfidf":
            return np.array(self.corpus_tfidf_index[self.tfidf[
                self.dictionary.doc2bow(tokenize(input_text, self.input_dir, grams=self.grams))]])
        elif self.type == "lsa":
            return np.array(self.corpus_lsa_index[self.lsa[
                self.tfidf[self.dictionary.doc2bow(tokenize(input_text, self.input_dir, grams=self.grams))]]])
        elif self.type == "tfidf_lsa":
            v1 = np.array(self.corpus_tfidf_index[
                              self.tfidf[self.dictionary.doc2bow(tokenize(input_text, self.input_dir, grams=self.grams))]])
            v2 = np.array(self.corpus_lsa_index[self.lsa[
                self.tfidf[self.dictionary.doc2bow(tokenize(input_text, self.input_dir, grams=self.grams))]]])
            return v1 + v2
        else:
            raise ValueError(
                'model type "' + self.type + '" is not an option. Available names are ["tf, tfidf, lsa, lda, word2vec_mean"].')

    def vectorize(self, docs):
        if self.type == "tf":
            sp = np.array( [ self.dictionary.doc2bow(tokenize(input_text, self.input_dir, grams = self.grams)) for input_text in docs ] )
            M = len(docs)
            N = len(self.dictionary)
            dat = list()
            row_ind = list()
            col_ind = list()
            for i, d in enumerate(sp):
                for (j,v) in d:
                    dat.append(v)
                    row_ind.append(i)
                    col_ind.append(j)
            return scipy.sparse.csr_matrix((np.array(dat), (np.array(row_ind), np.array(col_ind))), shape=(M, N))
        elif self.type == "tfidf":
            sp =  np.array( [ self.tfidf[ self.dictionary.doc2bow(tokenize(input_text, self.input_dir, grams = self.grams)) ] for input_text in docs ] )
            M = len(docs)
            N = len(self.dictionary)
            dat = list()
            row_ind = list()
            col_ind = list()
            for i, d in enumerate(sp):
                for (j, v) in d:
                    dat.append(v)
                    row_ind.append(i)
                    col_ind.append(j)
            return scipy.sparse.csr_matrix((np.array(dat), (np.array(row_ind), np.array(col_ind))), shape=(M, N))
        elif self.type == "lsa":
            mat = np.zeros((len(docs),self.lsa.num_topics))
            vec = [self.lsa[self.tfidf[self.dictionary.doc2bow(tokenize(input_text, self.input_dir, grams=self.grams))]]
                   for input_text in docs]
            for i,v in enumerate(vec):
                for t in v:
                    mat[i,t[0]] = t[1]
            return mat
        else:
            raise ValueError(
                'model type "' + self.type + '" is not an option. Available names are ["tf, tfidf, lsa, lda, word2vec_mean"].')


