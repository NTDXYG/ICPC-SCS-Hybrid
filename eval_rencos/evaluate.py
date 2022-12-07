from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from meteor.meteor import Meteor
import numpy as np
import pandas as pd
import sys


def main(hyp, ref, len=None):
    if(hyp.endswith("csv")):
        df = pd.read_csv(hyp, header=None)
    else:
        df = pd.read_csv(hyp, delimiter='\n', header=None)
    hypothesis = df[0].tolist()
    if(len is None):
        res = {k: [" ".join(v.strip().lower().split())] for k, v in enumerate(hypothesis)}
    else:
        res = {k: [" ".join(v.strip().lower().split()[:len])] for k, v in enumerate(hypothesis)}
    if(ref.endswith("csv")):
        df = pd.read_csv(ref, header=None)
    else:
        df = pd.read_csv(ref, delimiter='\n', header=None)
    references = df[0].tolist()
    gts = {k: [v.strip().lower()] for k, v in enumerate(references)}

    score_Bleu, scores_Bleu = Bleu(4).compute_score(gts, res)
    # df = pd.DataFrame(scores_Bleu[3])
    # df.to_csv("python_dl_bleu4.csv",index=False,header=None)
    print("Bleu_1: "), np.mean(scores_Bleu[0])
    print("Bleu_2: "), np.mean(scores_Bleu[1])
    print("Bleu_3: "), np.mean(scores_Bleu[2])
    print("Bleu_4: "), np.mean(scores_Bleu[3])

    score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    # df = pd.DataFrame(scores_Meteor)
    # df.to_csv("python_dl_meteor.csv",index=False,header=None)
    print("Meteor: "), score_Meteor

    score_Rouge, scores_Rouge,score_list = Rouge().compute_score(gts, res)
    # df = pd.DataFrame(score_list)
    # df.to_csv("python_dl_rouge.csv",index=False,header=None)
    print("ROUGE-L: "), score_Rouge
    #
    score_Cider, scores_Cider = Cider().compute_score(gts, res)
    # df = pd.DataFrame(scores_Cider)
    # df.to_csv("python_dl_cider.csv",index=False,header=None)
    print("Cider: "), score_Cider
    return [np.mean(scores_Bleu[3]), score_Meteor, score_Rouge, score_Cider]

if __name__ == '__main__':
    main("java_ir_dl.csv", "test.txt.tgt", 30)
