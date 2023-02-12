from nlgeval import compute_metrics

corpus_list = ['JCSD', 'PCSD', 'SCSD']

for c in corpus_list:
    print(c)
    metrics_dict = compute_metrics(hypothesis=c+'-SCS-Hybrid.csv',
                                references=['../data/'+c+'/test_comment_clean.csv'],no_skipthoughts=True, no_glove=True)