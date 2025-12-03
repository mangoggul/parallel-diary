from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

import torch
import torch.nn.functional as F
# KoBERT 사용을 위해 AutoTokenizer 대신 XLNetTokenizer를 사용하고 계시지만, 
# KoBERT는 보통 KoBERTTokenizer (또는 AutoTokenizer with specific model name)를 사용합니다. 
# XLNetTokenizer를 KoBERT 모델과 함께 사용하면 호환성 문제가 발생할 수 있지만, 
# 기존 코드를 유지하고 KoBERT 호환성 문제 처리 코드를 남겨둡니다.
# KoBERT 사용 시에는 'skt/kobert-base-v1' 모델과 함께 AutoTokenizer를 사용하는 것이 일반적입니다.
from transformers import AutoModel, XLNetTokenizer 

# KoBERT 모델 로드
tokenizer = XLNetTokenizer.from_pretrained("skt/kobert-base-v1")
model = AutoModel.from_pretrained("skt/kobert-base-v1")
model.eval()

analysis = APIRouter(prefix="/analysis")

# --------------------------
# 1. 요청 및 응답 Pydantic 모델 정의
# --------------------------

# /get-score 엔드포인트의 응답 모델
class MonotonyScoreResponse(BaseModel):
    user_id: int
    monotony_score: int

# /make-score 엔드포인트의 요청 모델
class SentenceRequest(BaseModel) :
    sentences: List[str]

# /make-score 엔드포인트의 응답 모델
class CalculateMonotonyResponse(BaseModel):
    sentence_count: int
    average_similarity: float

# --------------------------
# 2. 임베딩 함수 (변경 없음)
# --------------------------

def get_sentence_embedding(sentence, tokenizer, model):
    """주어진 문장의 KoBERT 임베딩을 추출합니다."""
    inputs = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        return_tensors="pt"
    )

    # KoBERT 일부 버전 호환을 위해 token_type_ids 제거 (필요 없는 경우도 있음)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        # [CLS] 토큰 임베딩 (첫 번째 토큰)을 문장 임베딩으로 사용
        last_hidden = outputs.last_hidden_state 

    return last_hidden[:, 0, :].squeeze(0)

# --------------------------
# 3. API 라우터 (response_model 추가)
# --------------------------



@analysis.post("/make-score", tags=["Monotony Score"], response_model=CalculateMonotonyResponse)
async def calculate_monotony(req: SentenceRequest):
    """
    주어진 문장들의 평균 코사인 유사도를 계산하여 단조로움 점수를 측정합니다.
    (유사도가 높을수록 단조롭다고 간주)
    """
    sentences = req.sentences

    # 문장 임베딩 생성
    embeddings = []
    for text in sentences:
        emb = get_sentence_embedding(text, tokenizer, model)
        embeddings.append(emb)

    if not embeddings:
        return {
            "sentence_count": 0,
            "average_similarity": 0.0
        }

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
    # 소수점 둘째 자리까지 반올림
    avg_similarity = float(f"{avg_similarity:.2f}")

    # 코사인 유사도 값 (0.00 ~ 1.00)을 0~100 스케일로 반환
    return {
        "sentence_count": n,
        "average_similarity": avg_similarity * 100
    }