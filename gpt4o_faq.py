import openai
import json
import re

def make_query(query, prompt_name, top3_ind, api_key):
    client = openai.OpenAI(api_key=api_key)

    def make_req(_input_msg, task):
        model_name = "gpt-4o-2024-05-13"
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"content": task, "role": "system"}, {"content": _input_msg, "role": "user"}]
        )
        l = response.choices[0].message.content
        return l, response.usage.total_tokens

    with open(prompt_name + ".json", 'r') as fl:
        prompt = json.load(fl)["prompt"]

    with open("data/questions.json", 'r') as fl:
        Q = json.load(fl)
    with open("data/answers.json", 'r') as fl:
        A = json.load(fl)

    def make_input(q, t3):
        _input_msg = "Example FAQs:\n"
        for _i in range(3):
            _input_msg += str(_i + 1) + '. Question: ' + Q[t3[_i]]['question'] + '\n'
            _input_msg += 'Answers: ' + A[str(t3[_i])] + '\n\n'

        _input_msg += 'Additional Question: ' + q
        return _input_msg

    _im = make_input(query, top3_ind)
    answer, num_tokens = make_req(_im, prompt)

    if prompt_name == 'prompt_choose':
        if len(re.findall('^\d+', answer)) != 0:
            num = int(re.findall('^\d+', answer)[0]) - 1
            if num in [0, 1, 2]:
                answer = A[str(top3_ind[num])]
    return answer