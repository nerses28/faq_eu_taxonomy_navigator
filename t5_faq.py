import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_dir = "./fine_tuned_t5"

tokenizer = T5Tokenizer.from_pretrained(model_dir, truncation_side='left')
model = T5ForConditionalGeneration.from_pretrained(model_dir)

def make_query(query, top3_ind):
    with open("data/questions.json", 'r') as fl:
        Q = json.load(fl)

    with open("data/answers.json", 'r') as fl:
        A = json.load(fl)

    context = 'CONTEXT:\n'
    for ci in top3_ind:
        context += 'Q: ' + Q[ci]["question"] + '\n'
        context += 'A: ' + A[str(ci)] + '\n\n'

    input_text = context + ' TASK: \nQ: ' + query + '\nA: '
    with torch.no_grad():
        input_ids = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_new_tokens=512)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text