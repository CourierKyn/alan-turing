from flask import Flask, Response
from flask import request
import pandas as pd
import json
from backend.extract_keras_bert_feature import KerasBertVector
import numpy as np
from flask_cors import *

app = Flask(__name__)

CORS(app, resources=r'/*')
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    ques_1 = request.get_json(force=True)['query']
    def cosine_distance(v1, v2): # 余弦距离
        if v1 and v2:
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        else:
            return 0
    vector_1 = bert_vector.bert_encode([ques_1])
    # q_and_a = pd.read_csv('Q_and_A.csv')
    # queses_2 = q_and_a.Q
    answers = q_and_a.A.tolist()
    comp = []
    for vector_2 in queses_2_vec:
        sim = cosine_distance(vector_1[0], vector_2)
        comp.append(sim)
    comp = np.array(comp)
    argmax_comp = np.argmax(comp)
    print('相似度：')
    print(comp[argmax_comp])
    if comp.max() < 0.85:
        most_sim = '相关功能正在更新'
    else:
        most_sim = answers[argmax_comp]
    return_value = json.dumps(
        {
            'answer': most_sim,
        }, ensure_ascii=False
    )
    # print(most_sim)
    # return Response(response=return_value, status=200, mimetype='text/html')
    # return_value.headers['Access-Control-Allow-Origin'] = '*'
    return return_value

if __name__ == '__main__':
    bert_vector = KerasBertVector()
    q_and_a = pd.read_csv('Q&A.csv')
    queses_2 = q_and_a.Q
    answers = q_and_a.A.tolist()
    queses_2_vec = []
    for ques_2 in queses_2:
        vector_2 = bert_vector.bert_encode([ques_2])
        queses_2_vec.append(vector_2[0])
    app.run()
    # print('请输入问题：')
    # hello_world()
    # print(a)
