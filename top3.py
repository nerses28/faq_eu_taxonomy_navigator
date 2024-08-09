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

def make_query(query):
    instruction = (
        "Given a web search query, retrieve relevant passages that answer the query:"
    )

    queries = [[instruction, query]]
    q_reps = l2v.encode(queries)
    q_reps_norm = q_reps / q_reps.norm(dim=1, keepdim=True)
    emb_norm = torch.load('emb/questions.EMB_NORM.pt')

    cossim_matrix = torch.matmul(emb_norm, q_reps_norm.T)
    _, closest_index = torch.topk(cossim_matrix, 3, dim=0)
    answs_indexes = []
    for ci in closest_index:
        answs_indexes.append(ci.item())
    return answs_indexes
