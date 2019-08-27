import numpy as np
from bert_serving.client import BertClient
from termcolor import colored
import pandas as pd

if __name__ == '__main__':
    topk = 5

    questions = pd.read_csv('Q&A.csv').Q.tolist()
    print('%d questions loaded, avg. len of %d' % (len(questions), np.mean([len(d) for d in questions])))

    with BertClient() as bc:
        doc_vecs = bc.encode(questions)

        while True:
            query = input(colored('your question: ', 'green'))
            query_vec = bc.encode([query])[0]
            # compute normalized dot product as score
            score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
            topk_idx = np.argsort(score)[::-1][:topk]
            print('top %d questions similar to "%s"' % (topk, colored(query, 'green')))
            for idx in topk_idx:
                print('> %s\t%s' % (colored('%.1f' % score[idx], 'cyan'), colored(questions[idx], 'yellow')))
