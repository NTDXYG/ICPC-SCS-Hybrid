# coding=utf8
import string

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import joblib

from IR.faiss_base import Retrieval
from flask.json import JSONEncoder as _JSONEncoder

import ast_utils_java, ast_utils_python, ast_utils_sc
import sbt_utils_java, sbt_utils_python, sbt_utils_sc

from transformers import BartForConditionalGeneration, BartTokenizerFast

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, supports_credentials=True)

class JSONEncoder(_JSONEncoder):
    def default(self, o):
        import decimal
        if isinstance(o, decimal.Decimal):
            return float(o)
        super(JSONEncoder, self).default(o)

app.json_encoder = JSONEncoder

java_scsir = Retrieval(dim=256,whitening_file='D:\\new_idea\\Final\\model\\JCSD\\code_vector.pkl', kernel_file='D:\\new_idea\\Final\\model\\JCSD\\kernel.pkl',bias_file='D:\\new_idea\\Final\\model\\JCSD\\bias.pkl',
train_code_list = 'D:\\new_idea\\Final\\data\\JCSD\\train_function_clean.csv', train_ast_list = 'D:\\new_idea\\Final\\data\\JCSD\\train_ast_clean.csv',train_nl_list='D:\\new_idea\\Final\\data\\JCSD\\train_comment_clean.csv')
java_scsir.encode_file()
java_scsir.build_index(n_list=1)
java_scsir.index.nprob = 1
java_scsdl = BartForConditionalGeneration.from_pretrained("D:\\new_idea\\NTUFinal\\model\\ComFormer_Java")
tokenizer = BartTokenizerFast.from_pretrained("D:\\new_idea\\NTUFinal\\model\\ComFormer_Java")
java_scsdl.to(torch.device('cpu'))
java_model_decide = joblib.load("./model/JCSD/java_decide.model")

python_scsir = Retrieval(dim=256,whitening_file='D:\\new_idea\\Final\\model\\PCSD\\code_vector.pkl', kernel_file='D:\\new_idea\\Final\\model\\PCSD\\kernel.pkl',bias_file='D:\\new_idea\\Final\\model\\PCSD\\bias.pkl',
train_code_list = 'D:\\new_idea\\Final\\data\\PCSD\\train_function_clean.csv', train_ast_list = 'D:\\new_idea\\Final\\data\\PCSD\\train_ast_clean.csv',train_nl_list='D:\\new_idea\\Final\\data\\PCSD\\train_comment_clean.csv')
python_scsir.encode_file()
python_scsir.build_index(n_list=1)
python_scsir.index.nprob = 1
python_scsdl = BartForConditionalGeneration.from_pretrained("D:\\new_idea\\NTUFinal\\model\\ComFormer_Python")
python_scsdl.to(torch.device('cpu'))
python_model_decide = joblib.load("./model/PCSD/python_decide.model")

sc_scsir = Retrieval(dim=256,whitening_file='D:\\new_idea\\Final\\model\\SCSD\\code_vector.pkl', kernel_file='D:\\new_idea\\Final\\model\\SCSD\\kernel.pkl',bias_file='D:\\new_idea\\Final\\model\\SCSD\\bias.pkl',
train_code_list = 'D:\\new_idea\\Final\\data\\SCSD\\train_function_clean.csv', train_ast_list = 'D:\\new_idea\\Final\\data\\SCSD\\train_ast_clean.csv',train_nl_list='D:\\new_idea\\Final\\data\\SCSD\\train_comment_clean.csv')
sc_scsir.encode_file()
sc_scsir.build_index(n_list=1)
sc_scsir.index.nprob = 1

def get_sim_info(scsir, code, sbt):
    sim_nl, code_score, ast_score, inner_score = scsir.single_query(code, sbt, topK=5)
    return {'sim_nl':sim_nl, 'code_score':code_score, 'ast_score':ast_score, 'inner_score':inner_score}

def get_dl_comment(model, code, ast):
    input_text = ' '.join(code.split()[:256]) + " <ast> " + ' '.join(ast.split()[:256])
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True)
    # input_ids = input_ids.to(DEVICE)
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

@app.route('/get_java_comment',methods=['POST'])
def get_java_comment():
    get_data = json.loads(request.get_data(as_text=True))
    code = get_data['code']
    processed_code = ast_utils_java.process_source(code)
    ast = ast_utils_java.get_ast(processed_code)
    code_seq, sbt = sbt_utils_java.transformer(code)
    result = get_sim_info(java_scsir, code_seq, sbt)
    ir_nl = result['sim_nl']
    if (java_model_decide.predict([[result['code_score'], result['ast_score'], result['inner_score']]])[0] == 0):
        return jsonify({'trans': ir_nl, 'ast_tree': json.loads(ast)})
    else:
        dl_nl = get_dl_comment(java_scsdl, code_seq, sbt)
        return jsonify({'trans': dl_nl, 'ast_tree': json.loads(ast)})

@app.route('/get_python_comment',methods=['POST'])
def get_python_comment():
    get_data = json.loads(request.get_data(as_text=True))
    code = get_data['code']
    ast = ast_utils_python.parse_ast(code)
    code_seq, sbt = sbt_utils_python.transformer(code)
    result = get_sim_info(python_scsir, code_seq, sbt)
    ir_nl = result['sim_nl']
    if (python_model_decide.predict([[result['code_score'], result['ast_score'], result['inner_score']]])[0] == 0):
        return jsonify({'trans': ir_nl, 'ast_tree': json.loads(ast)})
    else:
        dl_nl = get_dl_comment(python_scsdl, code_seq, sbt)
        return jsonify({'trans': dl_nl, 'ast_tree': json.loads(ast)})

@app.route('/get_sc_comment',methods=['POST'])
def get_sc_comment():
    get_data = json.loads(request.get_data(as_text=True))
    code = get_data['code']
    ast = ast_utils_sc.parse_ast(code)
    code_seq, sbt = sbt_utils_sc.transformer(code)
    sim_nl = get_sim_info(sc_scsir, code_seq, sbt)['sim_nl']
    result = {'trans':sim_nl,'ast_tree':json.loads(ast)}
    return jsonify(result)

@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run(port=5000)
