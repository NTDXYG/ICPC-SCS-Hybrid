import pandas as pd

df = pd.read_csv("python_score.csv", header=None)
score_list = df[0].tolist()
df = pd.read_csv("python_ir_bleu4.csv", header=None)
ir_bleu4_list = df[0].tolist()
df = pd.read_csv("python_ir_meteor.csv", header=None)
ir_meteor_list = df[0].tolist()
df = pd.read_csv("python_ir_rouge.csv", header=None)
ir_rouge_list = df[0].tolist()
df = pd.read_csv("python_ir_cider.csv", header=None)
ir_cider_list = df[0].tolist()

df = pd.read_csv("python_dl_bleu4.csv", header=None)
dl_bleu4_list = df[0].tolist()
df = pd.read_csv("python_dl_meteor.csv", header=None)
dl_meteor_list = df[0].tolist()
df = pd.read_csv("python_dl_rouge.csv", header=None)
dl_rouge_list = df[0].tolist()
df = pd.read_csv("python_dl_cider.csv", header=None)
dl_cider_list = df[0].tolist()

def calculate_group(score_list, measure_list):
    score_dict = {}
    for i in range(0, 11, 1)[:-1]:
        temp_index = i*0.1

        temp_score_sum = 0
        temp_count = 0
        for j in range(len(score_list)):
            if(temp_index <= score_list[j] <=temp_index+0.1):
                temp_count += 1
                temp_score_sum += measure_list[j]
        if(temp_count == 0):
            avg_score = 0
        else:
            avg_score = temp_score_sum/temp_count
        score_dict[format(temp_index, '.1f')+"-"+format(temp_index+0.1, '.1f')] = avg_score
    return score_dict

def calculate_lambda(score_dict_ir, score_dict_dl):
    for i in list(score_dict_ir.keys()):
        if(score_dict_ir[i] > score_dict_dl[i]):
            return float(i[:3])

score_dict_ir = calculate_group(score_list, ir_bleu4_list)
score_dict_dl = calculate_group(score_list, dl_bleu4_list)
bleu4_lambda = calculate_lambda(score_dict_ir, score_dict_dl)

score_dict_ir = calculate_group(score_list, ir_meteor_list)
score_dict_dl = calculate_group(score_list, dl_meteor_list)
meteor_lambda = calculate_lambda(score_dict_ir, score_dict_dl)

score_dict_ir = calculate_group(score_list, ir_rouge_list)
score_dict_dl = calculate_group(score_list, dl_rouge_list)
rouge_lambda = calculate_lambda(score_dict_ir, score_dict_dl)

score_dict_ir = calculate_group(score_list, ir_cider_list)
score_dict_dl = calculate_group(score_list, dl_cider_list)
cider_lambda = calculate_lambda(score_dict_ir, score_dict_dl)

print((bleu4_lambda + meteor_lambda + rouge_lambda + cider_lambda)/4)