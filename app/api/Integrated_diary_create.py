from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import json
import os
from dotenv import load_dotenv

from openai import OpenAI

# ------------------------------------------------------------------
# 라우터 정의
# ------------------------------------------------------------------
# 일기 관련 기능(추천, 평행일기)을 담당하는 라우터
recommend_and_parallel_diary = APIRouter(prefix="/diary")

# ----------------------
# Upstage API 로딩 및 클라이언트 초기화
# ----------------------
load_dotenv()
try:
    # API 키는 환경 변수에서 가져옵니다.
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://api.upstage.ai/v1"
    )
except Exception as e:
    # 클라이언트 초기화 실패 시 경고 출력 및 client를 None으로 설정
    print(f"Warning: Failed to initialize OpenAI client (Upstage) due to: {e}")
    client = None

# ================================
# 1) 요청 모델 정의 (original_diary_id 제거)
# ================================
class IntegratedRecommendRequest(BaseModel):
    """
    일기 내용과 단조로움 지수를 받아 AI 처리를 요청하는 모델입니다.
    """
    content: str = Field(description="원본 일기 내용")
    monotony_score: int = Field(description="단조로움 지수")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "content": "오늘은 팀 프로젝트 마감일이라 야근을 했고, 집 와서 바로 잠들었다. 일주일 내내 반복된 일상이라 지루하다.",
                    "monotony_score": 85,
                }
            ]
        }
    }

# ================================
# 2) 개별 항목 및 통합 응답 모델 정의
# ================================
class RecommendItem(BaseModel):
    """개별 일상 추천 항목의 세부 정보 모델"""
    recommendation: str = Field(description="새로운 일상 추천 내용")
    description: str = Field(description="해당 추천 내용에 대한 간결한 설명")
    emoji: str = Field(description="추천 내용을 대표하는 이모지")

class IntegratedResponse(BaseModel):
    """
    일상 추천 리스트와 평행일기 내용을 포함하는 최종 통합 응답 모델입니다. (original_diary_id 제거)
    """
    parallel_content: str = Field(description="원본 일기 내용을 바탕으로 '~다' 형식으로 작성된 평행일기 본문")
    recommendations: List[RecommendItem] = Field(description="평행일기에서 추출된 대조적인 활동 리스트 (최대 4개)")


# ================================
# 3) 통합 API 엔드포인트
# ================================
@recommend_and_parallel_diary.post(
    "/recommend-and-parallel-diary", 
    tags=["Diary"], 
    response_model=IntegratedResponse
)
async def recommend_and_parallel(req: IntegratedRecommendRequest): 
    """
    사용자의 일기 내용을 받아 평행일기를 생성하고, 평행일기에서 발생한 새로운 일상 활동을 4가지 추천합니다.
    (단조로움 지수(monotony_score)에 따라 추천의 성격이 달라집니다.)
    """
    if not client:
        raise HTTPException(status_code=503, detail="API Client is not initialized. Check API Key.")
    
    diary_content = req.content
    monotony_score = req.monotony_score # 단조로움 지수 값 사용
    
    # 두 가지 결과를 저장할 딕셔너리 (original_diary_id 제거)
    final_result = {
        "parallel_content": "",
        "recommendations": []
    }

    # -----------------------------------------------------
    # A. 단조로움 지수 기반 추천 가이드라인 생성
    # -----------------------------------------------------
    if monotony_score >= 70:
        # 단조로움이 매우 높을 때 (예: 70% 이상)
        score_guideline = "사용자의 단조로움 지수가 매우 높습니다. 평행일기에는 일상과 **비슷하면서 적당히 재밌는** 활동을 포함해야 하며, 추천 항목도 이에 맞춰 **높은 변화**를 유도해야 한다."
    elif monotony_score <= 30:
        # 단조로움이 매우 낮을 때 (예: 30% 이하)
        score_guideline = "사용자의 단조로움 지수가 낮습니다. 평행일기에는 이미 다양한 활동을 즐기고 있다는 점을 고려하여, **특정 취향을 깊게 파고들거나(마스터) 섬세한 차이**를 즐길 수 있는 활동을 대조적으로 포함해야 한다."
    else:
        # 단조로움이 중간일 때 (31% ~ 69%)
        score_guideline = "사용자의 단조로움 지수가 중간 수준입니다. 평행일기에는 현재 일상과 적절한 대비를 이루면서도 **쉽게 시도할 수 있는 새로운 경험**을 포함해야 한다."

    # -----------------------------------------------------
    # B. 평행일기 생성 및 추천 추출 로직 (통합)
    # -----------------------------------------------------
    try:
        # AI가 반환해야 할 JSON 스키마 (평행일기 + 추천을 한번에 받도록 통합)
        combined_schema = {
            "parallel_content": "string",
            "recommendations": [
                {"recommendation": "string", "description": "string", "emoji": "string"}
            ]
        }
        parallel_diary_schema = json.dumps(combined_schema, ensure_ascii=False)
        
        # 시스템 프롬프트 정의: 평행일기 작성과 추천 추출을 동시에 요구 (가이드라인 포함)
        parallel_diary_system_prompt = (
            "너는 사용자의 원본 일기 내용을 완전히 대조되는 방향의 '평행 세계' 일기('~다' 형식)로 상상하여 작성하는 AI 작가이자 큐레이터이다. "
            f"**[단조로움 지수 기반 생성 가이드라인]:** {score_guideline} "
            "1. **평행일기 내용(parallel_content) 작성:** 원본 일기 내용과 대조적이고, 가이드라인에 맞는 톤으로 작성한다."
            "2. **일상 추천(recommendations) 작성:** 평행일기에서 일어난 '새롭고 대조적인 활동' 4가지를 추출하여 추천 항목으로 정리한다. 이 추천은 평행일기 내용 그 자체여야 한다. "
            "너의 **유일하고 최종적인 응답은 반드시 하나의 JSON 객체**여야 하며, 어떠한 설명, 안내 문장, 추가적인 텍스트도 절대 포함되어서는 안 된다. "
            f"**반드시 이 스키마를 따를 것:** {parallel_diary_schema}"
        )
        
        parallel_messages = [
            {"role": "system", "content": parallel_diary_system_prompt},
            {"role": "user", "content": f"원본 일기: {diary_content}"}
        ]

        parallel_response = client.chat.completions.create(
            model="solar-pro2",
            messages=parallel_messages,
            stream=False,
            response_format={"type": "json_object"},
        )
        
        raw_response_json = parallel_response.choices[0].message.content
        ai_response_dict = json.loads(raw_response_json)
        
        # 평행일기 내용 및 추천 내용 저장
        final_result["parallel_content"] = ai_response_dict.get("parallel_content", "평행일기 생성 실패.")
        final_result["recommendations"] = ai_response_dict.get("recommendations", [])
        
    except Exception as e:
        print(f"AI 처리 중 오류 발생: {e}")
        error_detail = f"AI 처리 중 오류 발생: {e}"
        final_result["parallel_content"] = error_detail
        final_result["recommendations"] = [{"recommendation": "오류 발생", "description": error_detail, "emoji": "⚠️"}]

    # -----------------------------------------------------
    # C. 최종 결과 반환
    # -----------------------------------------------------
    try:
        return IntegratedResponse(**final_result)
    except Exception as e:
        # 통합 모델 검증 실패 시 (JSON 스키마가 안 맞을 경우)
        raise HTTPException(status_code=500, detail=f"최종 응답 모델 검증 실패: {e}. AI 출력 데이터: {final_result}")