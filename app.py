from flask import Flask,jsonify
from flask import request
import numpy as np
import pandas as pd
from backend.tet_bert_keras_sim import sim_two_question
# app = Flask(__name__)


# @app.route('/', methods=['GET', 'POST'])
def hello_world():
    # ques_1 = request.args.get("name") or ""
    ques_1 = input()
    # q_and_a = pd.read_csv('Q_and_A.csv')
    # queses_2 = q_and_a.Q
    # answers = q_and_a.A.tolist()
    # comp = []
    # for ques_2 in queses_2:
    #     sim = sim_two_question(ques_1, ques_2)
    #     comp.append(sim)
    # comp = np.array(comp)
    # most_sim = answers[np.argmax(comp)]
    most_sim = sim_two_question(ques_1)
    print(most_sim)
    # return jsonify({'answer': most_sim})


if __name__ == '__main__':
    # app.run()
    print('请输入问题：')
    hello_world()
    # print(answer)
