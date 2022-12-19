import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
from flask_cors import CORS

with open('evidence.json', 'r') as f:
    evi = f.read()
evidence_json = json.loads(evi)

model = SentenceTransformer('bert-base-nli-mean-tokens')


def similar_cal(sen_list1, sen_list2, threshold=0.8):
    embeddings1 = model.encode(sen_list1, convert_to_tensor=True)
    embeddings2 = model.encode(sen_list2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    #print(cosine_scores)
    index = [i for _, i in np.argwhere(np.array(cosine_scores) > threshold)]
    return index


def map_value(relevant_fact):
    y = relevant_fact['relevant_facts']
    support = [k for i in y for k in i['sentences']]
    support_fact = [i['describe'] for i in y]
    support_l = [i['sentences'] for i in y]
    mapping = {v: [support.index(i) for i in k] for v, k in enumerate(support_l)}#{0:[0,1],1:[2,3],2:[4,5,6]}

    return support, support_fact, mapping


def match(value, mapping, support_fact, support):
    fact_lack = support_fact
    fact_present = []
    for item in value:
        index = [i for i in mapping if item in mapping[i]][0]
        print(support_fact[index])
        print(support[item])
        print("----------------")
        if support_fact[index] not in fact_present:
            fact_present.append(support_fact[index])
            fact_lack.remove(support_fact[index])
    return fact_present, fact_lack


def cal(lesson_id, sentences):
    lesson_type = evidence_json[lesson_id]['prompt_type']
    if lesson_type == 1:
        relevant_fact_y = evidence_json[lesson_id]['support_yes']
        support_y, support_fact_y, mapping_y = map_value(relevant_fact_y)
        value_y = similar_cal(sentences, support_y)
        fact_present_y, fact_lack_y = match(value_y, mapping_y, support_fact_y,support_y)
        result_y = {'support_type': 'yes', 'num_support': len(value_y), 'fact_present': fact_present_y,
                    'fact_lack': fact_lack_y}
        relevant_fact_n = evidence_json[lesson_id]['support_no']
        support_n, support_fact_n, mapping_n = map_value(relevant_fact_n)
        value_n = similar_cal(sentences, support_n)
        fact_present_n, fact_lack_n = match(value_n, mapping_n, support_fact_n,support_n)
        result_n = {'support_type': 'no', 'num_support': len(value_n), 'facts_present': fact_present_n,
                    'fact_lack': fact_lack_n}
        if result_y['num_support'] == 0 and result_n['num_support'] == 0: #  find no direct evidence
            return {'support_type': 'no direct evidence', 'num_support': 0, 'facts_present': [],
                    'fact_lack': []}
        elif result_y['num_support'] > result_n['num_support']:
            return result_y
        elif result_y['num_support'] < result_n['num_support']:
            return result_n
        else:
            return {'support_type': 'not sure', 'num_support': 0, 'facts_present': [],
                    'fact_lack': []}
    else:
        return {'general_info': 0}


app = Flask(__name__)  ####
CORS(app)


@app.route("/predict", methods=['GET', 'POST'])
def run():
    if request.method == 'POST':  ####POST
        # data_fz = json.loads(request.get_data().decode('utf-8')) ####get data
        data_fz = request.get_json()
        # print(data_fz)

        if data_fz is not None:
            # data_fz = request.to_dict()
            lesson_id = data_fz['lesson_id']
            content = data_fz['content']

        else:
            return jsonify({'Bj': -1, 'Mess': '', 'type': 'Error'})  ####return -1 if no data
    else:
        return jsonify({'Bj': -2, 'Mess': '', 'type': 'Error'})  #### return -2 if not right format

    label = cal(lesson_id, content)

    return jsonify(label)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8008, debug=False)