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
# 일기 관련 키워드 추출 및 추천 기능을 담당하는 라우터
diary_recommend = APIRouter(prefix="/diary")

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
    print(f"Warning: Failed to initialize OpenAI client (Upstage) due to: {e}")
    client = None


# ================================
# 1) 요청 모델 정의
# ================================
class RecommendRequest(BaseModel):
    """
    일기 내용과 단조로움 지수(monotony_score)를 받아 추천을 생성하는 요청 모델입니다.
    """
    user_id: int = Field(description="사용자 ID (로그 목적으로 포함)")
    content: str = Field(description="일기나 긴 텍스트 내용")
    monotony_score: int = Field(description="단조로움 지수 (%)")

    # Swagger UI에 표시될 예시를 커스터마이징합니다.
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": 101,
                    "content": "오늘은 정말 오랜만에 새로운 취미인 목공을 시작했다. 서툰 솜씨지만 작은 나무 수납함을 만들었다. 내일은 회사에 휴가를 내고 동네 맛집 투어를 계획 중이다.",
                    "monotony_score": 80
                }
            ]
        }
    }



# ================================
# 2) 응답 모델 정의
# ================================


class RecommendItem(BaseModel):
    """개별 일상 추천 항목의 세부 정보 모델"""
    recommendation: str = Field(description="새로운 일상 추천 내용 (예: 동네 산책로 탐방, 낯선 음식 도전하기)")
    description: str = Field(description="해당 추천 내용에 대한 간결한 설명")
    emoji: str = Field(description="추천 내용을 대표하는 이모지 (예: 🚶, 🍜)")


class RecommendResponse(BaseModel):
    """AI가 반환해야 할 일상 추천 리스트를 정의하는 모델입니다."""
    recommendations: List[RecommendItem] = Field(description="텍스트에서 추출된 일상과 다른 새로운 일상 추천 리스트 (최대 4개)")


# ================================
# 3) 일상 추천 API 엔드포인트
# ================================
@diary_recommend.post("/recommend", tags=["Diary"], response_model=RecommendResponse)
async def recommend_new_routine_based_on_monotony(req: RecommendRequest): 
    """
    사용자의 일기 내용과 단조로움 지수 user_id를 받아 새로운 일상을 4가지 추천합니다.
    """
    if not client:
        raise HTTPException(status_code=503, detail="API Client is not initialized. Check API Key.")
    
    # 요청 객체에서 단조로움 지수만 사용
    monotony_score = req.monotony_score
    
    # AI에게 전달할 목표 JSON 스키마 예시
    json_schema_example = json.dumps({
        "recommendations": [
            {
                "recommendation": "동네 북카페에서 새로운 책 읽기",
                "description": "익숙한 곳을 벗어나 독서를 통해 새로운 영감을 얻어보세요.",
                "emoji": "📚"
            }
        ]
    }, ensure_ascii=False)

    # -----------------------------------------------------
    # 단조로움 지수 기반 추천 가이드라인 생성
    # -----------------------------------------------------
    
    if monotony_score >= 70:
        # 단조로움이 매우 높을 때 (예: 70% 이상)
        score_guideline = "사용자의 단조로움 지수가 매우 높습니다. 일상과 **가장 파격적이고 대조적인** 활동을 추천하여 삶에 충격을 주세요."
    elif monotony_score <= 30:
        # 단조로움이 매우 낮을 때 (예: 30% 이하)
        score_guideline = "사용자의 단조로움 지수가 낮습니다. 이미 다양한 활동을 즐기고 있습니다. **특정 취향을 깊게 파고들거나(마스터) 섬세한 차이**를 즐길 수 있는 활동을 추천하세요."
    else:
        # 단조로움이 중간일 때 (31% ~ 69%)
        score_guideline = "사용자의 단조로움 지수가 중간 수준입니다. 현재 일상과 적절한 대비를 이루면서도 쉽게 시도할 수 있는 활동을 추천하세요."
        
    system_prompt_content = (
        "너는 사용자의 일상 기록(일기)을 분석하여, 현재 일상 패턴에서 벗어나 **삶에 활력을 줄 수 있는 새로운 일상 활동**을 최대 4가지 추천하는 전문 큐레이터 AI이다. "
        f"**[사용자 단조로움 지수 분석]:** 단조로움 지수: {monotony_score}%. "
        f"이에 대한 추천 가이드라인: {score_guideline} "
        "사용자의 일기를 분석하여 그와 **가장 대조되는 활동**을 추천해야 한다. "
        "너의 **유일하고 최종적인 응답은 반드시 JSON 객체**여야 하며, 어떠한 설명, 안내 문장, 추가적인 텍스트도 절대 포함되어서는 안 된다. "
        "추천 내용은 명확한 제목, 간결한 설명, 그리고 적절한 이모지를 포함해야 한다. "
        f"**반드시 이 JSON 스키마를 따를 것:** {json_schema_example}" 
    )

    messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": f"다음 일기 내용을 분석하여 새로운 일상을 4가지 추천해줘:\n\n{req.content}"}
    ]

    try:
        # ===========================
        # Solar-Pro2에게 요청 보내기
        # ===========================
        response = client.chat.completions.create(
            model="solar-pro2",
            messages=messages,
            stream=False,
            # API 자체에 JSON 응답을 강제함
            response_format={"type": "json_object"},
        )

        # AI 응답에서 JSON 문자열 추출
        raw_recommendation_json = response.choices[0].message.content

        # 응답이 유효한지 Pydantic 모델로 검증 및 반환
        return RecommendResponse.model_validate_json(raw_recommendation_json)

    except Exception as e:
        # API 호출 또는 JSON 파싱 중 오류 발생 시
        print(f"AI 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"AI 처리 중 오류 발생: {e}")

