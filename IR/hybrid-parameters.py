import string

import torch
from nlgeval import compute_metrics
from tqdm import tqdm

from IR.faiss_base import Retrieval
import pandas as pd

scsir = Retrieval(dim=256,whitening_file='D:\\new_idea\\Final\\model\\PCSD\\code_vector.pkl', kernel_file='D:\\new_idea\\Final\\model\\PCSD\\kernel.pkl',bias_file='D:\\new_idea\\Final\\model\\PCSD\\bias.pkl',
train_code_list = 'D:\\new_idea\\Final\\data\\PCSD\\train_function_clean.csv', train_ast_list = 'D:\\new_idea\\Final\\data\\PCSD\\train_ast_clean.csv',train_nl_list='D:\\new_idea\\Final\\data\\PCSD\\train_comment_clean.csv')
scsir.encode_file()
scsir.build_index(n_list=1)
scsir.index.nprob = 1

def get_sim_info(scsir, code, sbt):
    sim_nl, code_score, ast_score, inner_score = scsir.single_query(code, sbt, topK=5, lam =0.7, code_len = 32)
    return {'sim_nl':sim_nl, 'code_score':code_score, 'ast_score':ast_score, 'inner_score':inner_score}

df = pd.read_csv("D:\\new_idea\\Final\\data\\PCSD\\test_function_clean.csv", header=None)
code_list = df[0].tolist()
df = pd.read_csv("D:\\new_idea\\Final\\data\\PCSD\\test_ast_clean.csv", header=None)
ast_list = df[0].tolist()
df = pd.read_csv("D:\\new_idea\\Final\\data\\PCSD\\test_comment_clean.csv", header=None)
comment_list = df[0].tolist()

data_list = []
for i in tqdm(range(len(code_list))):
    result = get_sim_info(scsir, code_list[i], ast_list[i])
    ir_nl = result['sim_nl']
    data_list.append(ir_nl)

df = pd.DataFrame(data_list)
df.to_csv("python_len_32.csv", index=False, header=None)

compute_metrics("python_len_32.csv", ["python_ref.csv"])