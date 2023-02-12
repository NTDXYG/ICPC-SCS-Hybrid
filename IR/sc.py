import string

import torch
from nlgeval import compute_individual_metrics
from tqdm import tqdm

from IR.faiss_base import Retrieval
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizerFast

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

java_scsir = Retrieval(dim=256,whitening_file='D:\\new_idea\\Final\\model\\SCSD\\code_vector.pkl', kernel_file='D:\\new_idea\\Final\\model\\SCSD\\kernel.pkl',bias_file='D:\\new_idea\\Final\\model\\SCSD\\bias.pkl',
train_code_list = 'D:\\new_idea\\Final\\data\\SCSD\\train_function_clean.csv', train_ast_list = 'D:\\new_idea\\Final\\data\\SCSD\\train_ast_clean.csv',train_nl_list='D:\\new_idea\\Final\\data\\SCSD\\train_comment_clean.csv')
java_scsir.encode_file()
java_scsir.build_index(n_list=1)
java_scsir.index.nprob = 1


def get_sim_info(scsir, code, sbt):
    sim_nl, code_score, ast_score, inner_score = scsir.single_query(code, sbt, topK=5)
    return {'sim_nl':sim_nl, 'code_score':code_score, 'ast_score':ast_score, 'inner_score':inner_score}

df = pd.read_csv("D:\\new_idea\\Final\\data\\SCSD\\test_function_clean.csv", header=None)
code_list = df[0].tolist()
df = pd.read_csv("D:\\new_idea\\Final\\data\\SCSD\\test_ast_clean.csv", header=None)
ast_list = df[0].tolist()
df = pd.read_csv("D:\\new_idea\\Final\\data\\SCSD\\test_comment_clean.csv", header=None)
comment_list = df[0].tolist()

df = pd.read_csv("CodeBert.csv", header=None)
dl_list = df[0].tolist()
data_list = []
for i in tqdm(range(len(code_list))):
    result = get_sim_info(java_scsir, code_list[i], ast_list[i])
    ir_nl = result['sim_nl']
    dl_nl = dl_list[i]
    ir_score = compute_individual_metrics(comment_list[i],ir_nl)['Bleu_4']
    dl_score = compute_individual_metrics(comment_list[i],dl_nl)['Bleu_4']
    if(dl_score > ir_score):
        data_list.append([result['code_score'], result['ast_score'], result['inner_score'], 1])
    else:
        data_list.append([result['code_score'], result['ast_score'], result['inner_score'], 0])

df = pd.DataFrame(data_list)
df.to_csv("sc.csv", index=False, header=None)