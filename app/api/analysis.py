from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModel, XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained("skt/kobert-base-v1")
model = AutoModel.from_pretrained("skt/kobert-base-v1")
model.eval()

analysis = APIRouter(prefix="/analysis")


@analysis.get("/get-score", tags=["Monotony Score"])
async def get_monotony(user_id: int):
    return {"user_id": user_id, "monotony_score": 70}

class SentenceRequest(BaseModel) :
    sentences: List[str]

def get_sentence_embedding(sentence, tokenizer, model):
    inputs = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        return_tensors="pt"
    )

    # KoBERT 일부 버전 호환 → token_type_ids 제거
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state

    return last_hidden[:, 0, :].squeeze(0)

@analysis.post("/make-score", tags=["Monotony Score"])
async def calculate_similarity(req: SentenceRequest):

    sentences = req.sentences

    # 문장 임베딩 생성
    embeddings = []
    for text in sentences:
        emb = get_sentence_embedding(text, tokenizer, model)
        embeddings.append(emb)

    embedding_matrix = torch.stack(embeddings)

    # 코사인 유사도 계산
    normalized = F.normalize(embedding_matrix, p=2, dim=1)
    sim_matrix = torch.matmul(normalized, normalized.transpose(0, 1))

    # numpy 변환
    sim = sim_matrix.cpu().numpy()

    # ---------------------------
    # 🔥 평균 유사도 계산
    # (자기 자신 i==j 는 제외)
    # ---------------------------
    n = len(sentences)
    total = 0
    count = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                total += sim[i][j]
                count += 1

    avg_similarity = total / count if count > 0 else 0
    avg_similarity = float(f"{avg_similarity:.2f}")

    return {
        "sentence_count": n,
        "average_similarity": avg_similarity * 100
    }
