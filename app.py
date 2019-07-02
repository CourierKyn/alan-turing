from flask import Flask, jsonify
from flask import request
import numpy as np
import pandas as pd
from backend.tet_bert_keras_sim import sim_two_question
import json

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    ques_1 = request.args.get("name") or ""
    # ques_1 = input()
    most_sim = sim_two_question(ques_1)
    # print(most_sim)
    return json.dumps(
        {
            'answer': most_sim
        }, ensure_ascii=False
    )


if __name__ == '__main__':
    app.run()
    # print('请输入问题：')
    # a = hello_world()
    # print(a)
