from llm2vec import LLM2Vec

import torch
import json
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

# Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
tokenizer = AutoTokenizer.from_pretrained(
    "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"
)
config = AutoConfig.from_pretrained(
    "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
)
model = PeftModel.from_pretrained(
    model,
    "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
)
model = model.merge_and_unload()
model = PeftModel.from_pretrained(
    model, "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised"
)
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)


def make_query0(query):
    instruction = (
        "Given a web search query, retrieve relevant passages that answer the query:"
    )
    with open("data/questions.json", 'r') as fl:
        DATA = json.load(fl)

    with open("data/answers.json", 'r') as fl:
        ANSW = json.load(fl)

    queries = [[instruction, query]]
    q_reps = l2v.encode(queries)
    q_reps_norm = q_reps / q_reps.norm(dim=1, keepdim=True)
    emb_norm = torch.load('emb/questions.EMB_NORM.pt')

    cossim_matrix = torch.matmul(emb_norm, q_reps_norm.T)
    closest_index = torch.argmax(cossim_matrix)

    answ_index = DATA[closest_index]['answer']
    answer = ANSW[answ_index]
    return answer, answ_index

def make_query1(query):
    instruction = (
        "Given a web search query, retrieve relevant passages that answer the query:"
    )
    with open("data/questions-AUG.train_dev.json", 'r') as fl:
        DATA = json.load(fl)

    with open("data/answers.json", 'r') as fl:
        ANSW = json.load(fl)

    queries = [[instruction, query]]
    q_reps = l2v.encode(queries)
    q_reps_norm = q_reps / q_reps.norm(dim=1, keepdim=True)
    emb_norm = torch.load('emb/questions-AUG.train_dev.EMB_NORM.pt')

    cossim_matrix = torch.matmul(emb_norm, q_reps_norm.T)
    closest_index = torch.argmax(cossim_matrix)

    answ_index = DATA[closest_index]['answer']
    answer = ANSW[answ_index]
    return answer, answ_index

def make_query2(query):
    instruction = (
        "Given a web search query, retrieve relevant passages that answer the query:"
    )
    with open("data/answers.json", 'r') as fl:
        ANSW = json.load(fl)

    queries = [[instruction, query]]
    q_reps = l2v.encode(queries)
    q_reps_norm = q_reps / q_reps.norm(dim=1, keepdim=True)
    emb_norm = torch.load('emb/ANSWERS.EMB_NORM.pt')

    cossim_matrix = torch.matmul(emb_norm, q_reps_norm.T)
    closest_index = torch.argmax(cossim_matrix)

    answ_index = str(closest_index.item())
    answer = ANSW[answ_index]
    return answer, answ_index






