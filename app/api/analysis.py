from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModel, XLNetTokenizer 
from fastapi import HTTPException

# ----------------------------------------------
# 1. KoBERT 모델 로드 및 전역 변수 설정
#    * 모델 로딩 실패 시 API 요청 처리에서 예외 발생
# ----------------------------------------------
try:
    print("KoBERT 모델 로딩 중...")
    # KoBERT 모델 로드
    tokenizer = XLNetTokenizer.from_pretrained("skt/kobert-base-v1")
    model = AutoModel.from_pretrained("skt/kobert-base-v1")
    model.eval() # 평가 모드로 설정
    print("KoBERT 모델 로딩 완료.")
except Exception as e:
    # 모델 로딩 실패 시 API 호출 시 오류를 반환하기 위해 None으로 설정
    print(f"KoBERT 모델 로딩 중 오류 발생. (메모리 부족 가능성): {e}")
    tokenizer = None
    model = None


analysis = APIRouter(prefix="/analysis")

# --------------------------
# 2. 요청 및 응답 Pydantic 모델 정의
# --------------------------


# /make-score 및 /classify-diary 엔드포인트의 요청 모델
class SentenceRequest(BaseModel) :
    sentences: List[str]

# /make-score 엔드포인트의 응답 모델
class CalculateMonotonyResponse(BaseModel):
    sentence_count: int
    average_similarity: float

# /classify-diary 엔드포인트의 응답 모델 (새로 추가됨)
class DiaryTypeClassificationResponse(BaseModel):
    diary_type: str  # 새로운 시도형, 흐름형, 루틴 충실형 중 하나
    

# --------------------------
# 3. 임베딩 및 유사도 계산 핵심 로직 함수
# --------------------------

def get_sentence_embedding(sentence, tokenizer, model):
    """주어진 문장의 KoBERT 임베딩 (CLS 토큰)을 추출합니다."""
    if tokenizer is None or model is None:
        raise RuntimeError("KoBERT 모델이 로드되지 않았습니다. 서버 메모리 상태를 확인하세요.")

    inputs = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        return_tensors="pt"
    )

    # KoBERT 일부 버전 호환을 위해 token_type_ids 제거 
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        # [CLS] 토큰 임베딩 (첫 번째 토큰)을 문장 임베딩으로 사용
        last_hidden = outputs.last_hidden_state 

    return last_hidden[:, 0, :].squeeze(0)


def calculate_average_similarity(sentences: List[str], tokenizer, model) -> float:
    """문장 리스트의 평균 코사인 유사도(단조로움 점수)를 계산합니다."""
    
    if len(sentences) < 2:
        return 0.0

    # 문장 임베딩 생성
    embeddings = []
    try:
        for text in sentences:
            emb = get_sentence_embedding(text, tokenizer, model)
            embeddings.append(emb)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Model error: {e}")

    embedding_matrix = torch.stack(embeddings)

    # 코사인 유사도 계산
    normalized = F.normalize(embedding_matrix, p=2, dim=1)
    sim_matrix = torch.matmul(normalized, normalized.transpose(0, 1))

    # numpy 변환 후 평균 계산
    sim = sim_matrix.cpu().numpy()
    n = len(sentences)
    total = 0
    count = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                total += sim[i][j]
                count += 1

    avg_similarity = total / count if count > 0 else 0
    # 0.00 ~ 1.00 스케일로 반올림
    return float(f"{avg_similarity:.2f}")

# --------------------------
# 4. API 라우터 정의
# --------------------------




@analysis.post("/make-score", tags=["Monotony Score"], response_model=CalculateMonotonyResponse)
async def calculate_monotony(req: SentenceRequest):
    """
    주어진 문장들의 평균 코사인 유사도를 계산하여 단조로움 점수를 측정합니다.
    (유사도가 높을수록 단조롭다고 간주)
    """
    avg_similarity = calculate_average_similarity(req.sentences, tokenizer, model)

    # 코사인 유사도 값 (0.00 ~ 1.00)을 0~100 스케일로 반환
    return {
        "sentence_count": len(req.sentences),
        "average_similarity": avg_similarity * 100
    }


@analysis.post("/classify-Type", tags=["Monotony Score"], response_model=DiaryTypeClassificationResponse)
async def classify_diary_type(req: SentenceRequest):
    """
    일기 문장들의 평균 유사도(단조로움 점수)를 기반으로 일기 타입을 분류합니다.
    - 새로운 시도형: 낮은 유사도 (다양함)
    - 적응형: 중간 유사도 (유연함)
    - 루틴 충실형: 높은 유사도 (단조로움)
    """
    sentences = req.sentences

    if not sentences or len(sentences) < 2:
        return DiaryTypeClassificationResponse(
            diary_type="분류 불가"
            )

    # 평균 유사도 계산 (0.0 ~ 1.0)
    avg_similarity = calculate_average_similarity(sentences, tokenizer, model)
    # 0.00 ~ 100.00% 스케일로 변환
    avg_similarity_percent = avg_similarity * 100 

    # 2. 유사도 점수를 기준으로 일기 타입 분류 (임계값 예시: 40, 70)
    
    # 0.0% ~ 40.0% (낮은 유사도, 다양함)
    if avg_similarity_percent <= 40.0:
        diary_type = "새로운 시도형"
        message = "기록의 다양성이 매우 높습니다. 일상에 새로운 변화와 경험을 많이 시도하고 계시는군요!"
    # 40.0% 초과 ~ 70.0% 이하 (중간 유사도, 유연함)
    elif avg_similarity_percent <= 70.0:
        diary_type = "흐름형 (적응형)"
        message = "유연하게 상황과 감정의 흐름에 적응하며 기록하고 있습니다. 균형 잡힌 일상입니다."
    # 70.0% 초과 (높은 유사도, 단조로움)
    else: 
        diary_type = "루틴 충실형"
        message = "일관된 패턴과 루틴을 충실하게 따르고 있습니다. 매우 안정적이고 예측 가능한 일상입니다."

    # 3. 결과 반환
    return DiaryTypeClassificationResponse(
        diary_type=diary_type,
        average_similarity=float(f"{avg_similarity_percent:.2f}"),
        message=message
    )