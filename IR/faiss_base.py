import pickle
import faiss
import numpy as np
import Levenshtein
import torch
import pandas as pd

from transformers import RobertaTokenizer, RobertaModel

from IR.bert_whitening import sents_to_vecs, transform_and_normalize

tokenizer = RobertaTokenizer.from_pretrained("D:\\new_idea\\Final\\model\\codebert")
model = RobertaModel.from_pretrained("D:\\new_idea\\Final\\model\\codebert")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)

def sim_jaccard(s1, s2):
    """jaccard相似度"""
    s1, s2 = set(s1), set(s2)
    ret1 = s1.intersection(s2)  # 交集
    ret2 = s1.union(s2)  # 并集
    sim = 1.0 * len(ret1) / len(ret2)
    return sim

class Retrieval(object):
    def __init__(self, dim, whitening_file, kernel_file, bias_file, train_code_list, train_ast_list, train_nl_list):
        f = open(whitening_file, 'rb')
        self.bert_vec = pickle.load(f)
        f.close()
        f = open(kernel_file, 'rb')
        self.kernel = pickle.load(f)
        f.close()
        f = open(bias_file, 'rb')
        self.bias = pickle.load(f)
        f.close()

        self.dim = dim
        df = pd.read_csv(train_code_list, header=None)
        self.train_code_list = df[0].tolist()
        df = pd.read_csv(train_ast_list, header=None)
        self.train_ast_list = df[0].tolist()
        df = pd.read_csv(train_nl_list, header=None)
        self.train_nl_list = df[0].tolist()
        self.id2text = None
        self.vecs = None
        self.ids = None
        self.index = None

    def encode_file(self):
        all_texts = []
        all_ids = []
        all_vecs = []
        for i in range(len(self.train_code_list)):
            all_texts.append(self.train_code_list[i])
            all_ids.append(i)
            all_vecs.append(self.bert_vec[i].reshape(1,-1))
        all_vecs = np.concatenate(all_vecs, 0)
        id2text = {idx: text for idx, text in zip(all_ids, all_texts)}
        self.id2text = id2text
        self.vecs = np.array(all_vecs, dtype="float32")
        self.ids = np.array(all_ids, dtype="int64")

    def build_index(self, n_list):
        quant = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFFlat(quant, self.dim, min(n_list, self.vecs.shape[0]), faiss.METRIC_INNER_PRODUCT)
        index.train(self.vecs)
        index.add_with_ids(self.vecs, self.ids)
        self.index = index

    def single_query(self, code, ast, topK=5, lam=0.7, code_len = 256):
        body = sents_to_vecs([code], tokenizer, model, ir=True, code_len = 256)
        body = transform_and_normalize(body, self.kernel, self.bias)
        vec = body[[0]].reshape(1, -1).astype('float32')
        _, sim_idx = self.index.search(vec, topK)
        sim_idx = sim_idx[0].tolist()
        max_score = 0
        max_idx = 0
        code_score_list = []
        ast_score_list = []
        for j in sim_idx:
            code_score = sim_jaccard(self.train_code_list[j].split(), code.split())
            ast_score = Levenshtein.seqratio(str(self.train_ast_list[j]).split(), str(ast).split())
            code_score_list.append(code_score)
            ast_score_list.append(ast_score)
        for i in range(len(sim_idx)):
            code_score = code_score_list[i]
            ast_score = ast_score_list[i]
            score = lam * code_score + (1-lam) * ast_score
            if score > max_score:
                max_score = score
                max_idx = sim_idx[i]
        code_vec = self.vecs[max_idx].reshape(1, -1).astype('float32')
        inner_score = np.dot(code_vec[0], vec[0])
        return self.train_nl_list[max_idx], sim_jaccard(self.train_code_list[max_idx].split(), code.split()), Levenshtein.seqratio(str(self.train_code_list[j]).split(), str(ast).split()), inner_score
