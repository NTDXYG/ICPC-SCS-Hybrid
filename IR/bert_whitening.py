import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import pandas as pd

MODEL_NAME = "D:\\new_idea\\Final\\model\\codebert" # 本地模型文件

POOLING = 'first_last_avg'
# POOLING = 'last_avg'
# POOLING = 'last2avg'

USE_WHITENING = True
N_COMPONENTS = 256
MAX_LENGTH = 512

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(name):
    tokenizer = RobertaTokenizer.from_pretrained(name)
    model = RobertaModel.from_pretrained(name)
    model = model.to(DEVICE)
    model = model.get_encoder()
    return tokenizer, model

def sents_to_vecs(sents, tokenizer, model, ir=False, code_len = 256):
    vecs = []
    with torch.no_grad():
        if(ir==True):
            for sent in sents:
                inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True,  max_length=code_len)
                inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
                inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)

                hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

                if POOLING == 'first_last_avg':
                    output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
                elif POOLING == 'last_avg':
                    output_hidden_state = (hidden_states[-1]).mean(dim=1)
                elif POOLING == 'last2avg':
                    output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
                else:
                    raise Exception("unknown pooling {}".format(POOLING))

                vec = output_hidden_state.cpu().numpy()[0]
                vecs.append(vec)
        else:
            for sent in tqdm(sents):
                inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True, max_length=code_len)
                inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
                inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)

                hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

                if POOLING == 'first_last_avg':
                    output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
                elif POOLING == 'last_avg':
                    output_hidden_state = (hidden_states[-1]).mean(dim=1)
                elif POOLING == 'last2avg':
                    output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
                else:
                    raise Exception("unknown pooling {}".format(POOLING))

                vec = output_hidden_state.cpu().numpy()[0]
                vecs.append(vec)
    assert len(sents) == len(vecs)
    vecs = np.array(vecs)
    return vecs


def compute_kernel_bias(vecs, n_components):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    W = W[:, :n_components]
    return W, -mu


def transform_and_normalize(vecs, kernel, bias):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def normalize(vecs):
    """标准化
    """
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def main():
    print(f"Configs: {MODEL_NAME}-{POOLING}-{USE_WHITENING}-{N_COMPONENTS}.")
    tokenizer, model = build_model(MODEL_NAME)
    print("Building {} tokenizer and model successfuly.".format(MODEL_NAME))
    # 修改
    df = pd.read_csv("../data/java/train.spl.src", delimiter='\n', header=None)
    code_list = df[0].tolist()
    df = pd.read_csv("../data/java/train.ast.src", delimiter='\n', header=None)
    ast_list = df[0].tolist()
    data_list = []
    for i in range(len(code_list)):
        data_list.append(code_list[i] +' <ast> '+ast_list[i])

    print("Transfer sentences to BERT vectors.")
    vecs_func_body = sents_to_vecs(data_list, tokenizer, model, ir=False, code_len = MAX_LENGTH)
    if USE_WHITENING:
        print("Compute kernel and bias.")
        kernel, bias = compute_kernel_bias([
            vecs_func_body
        ], n_components=N_COMPONENTS)
        vecs_func_body = transform_and_normalize(vecs_func_body, kernel, bias)
    else:
        vecs_func_body = normalize(vecs_func_body)
    print(vecs_func_body.shape)
    import pickle

    # 修改
    f = open('../model/scsdl_code_vector_whitening.pkl', 'wb')
    pickle.dump(vecs_func_body, f)
    f.close()
    f = open('../model/scsdl_kernel.pkl', 'wb')
    pickle.dump(kernel, f)
    f.close()
    f = open('../model/scsdl_bias.pkl', 'wb')
    pickle.dump(bias, f)
    f.close()

if __name__ == "__main__":
    main()