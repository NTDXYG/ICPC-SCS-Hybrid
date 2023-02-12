import string

import torch
from nlgeval import compute_individual_metrics
from tqdm import tqdm

from IR.faiss_base import Retrieval
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizerFast

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BartForConditionalGeneration.from_pretrained("D:\\new_idea\\NTUFinal\\model\\ComFormer_Java")
tokenizer = BartTokenizerFast.from_pretrained("D:\\new_idea\\NTUFinal\\model\\ComFormer_Java")
model.to(DEVICE)

java_scsir = Retrieval(dim=256,whitening_file='D:\\new_idea\\Final\\model\\JCSD\\code_vector.pkl', kernel_file='D:\\new_idea\\Final\\model\\JCSD\\kernel.pkl',bias_file='D:\\new_idea\\Final\\model\\JCSD\\bias.pkl',
train_code_list = 'D:\\new_idea\\Final\\data\\JCSD\\train_function_clean.csv', train_ast_list = 'D:\\new_idea\\Final\\data\\JCSD\\train_ast_clean.csv',train_nl_list='D:\\new_idea\\Final\\data\\JCSD\\train_comment_clean.csv')
java_scsir.encode_file()
java_scsir.build_index(n_list=1)
java_scsir.index.nprob = 1

def get_dl_comment(code, ast):
    input_text = ' '.join(code.split()[:256]) + " <ast> " + ' '.join(ast.split()[:256])
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True)
    input_ids = input_ids.to(DEVICE)
    summary_text_ids = model.generate(
        input_ids=input_ids,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=1.2,
        top_k=5,
        top_p=0.95,
        max_length=52,
        min_length=2,
        num_beams=6,
    )
    comment = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    if (comment[-1] in string.punctuation):
        comment = comment[:-1] + " " + comment[-1]
    return comment

def get_sim_info(scsir, code, sbt):
    sim_nl, code_score, ast_score, inner_score = scsir.single_query(code, sbt, topK=5)
    return {'sim_nl':sim_nl, 'code_score':code_score, 'ast_score':ast_score, 'inner_score':inner_score}

df = pd.read_csv("D:\\new_idea\\Final\\data\\JCSD\\test_function_clean.csv", header=None)
code_list = df[0].tolist()
df = pd.read_csv("D:\\new_idea\\Final\\data\\JCSD\\test_ast_clean.csv", header=None)
ast_list = df[0].tolist()
df = pd.read_csv("D:\\new_idea\\Final\\data\\JCSD\\test_comment_clean.csv", header=None)
comment_list = df[0].tolist()


import joblib
model_decide = joblib.load("../model/JCSD/java_decide.model")

data_list = []
for i in tqdm(range(len(code_list))):
    result = get_sim_info(java_scsir, code_list[i], ast_list[i])
    ir_nl = result['sim_nl']
    if (model_decide.predict([[result['code_score'], result['ast_score'], result['inner_score']]])[0] == 0):
        data_list.append(ir_nl)
    else:
        dl_nl = get_dl_comment(code_list[i], ast_list[i])
        data_list.append(dl_nl)

df = pd.DataFrame(data_list)
df.to_csv("SCS-Hybrid.csv", index=False, header=None)