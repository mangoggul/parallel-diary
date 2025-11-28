from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import json
import os
from dotenv import load_dotenv

from openai import OpenAI

diary_keyword = APIRouter(prefix="/diary")

# ----------------------
# Upstage API 로딩 및 클라이언트 초기화
# ----------------------
# .env 파일에서 환경 변수(API_KEY)를 로드합니다.
load_dotenv()
try:
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://api.upstage.ai/v1"
    )
except Exception as e:
    # API 키 로딩 실패 시 임시 클라이언트 (실제 API 호출은 실패함)
    print(f"Warning: Failed to initialize OpenAI client (Upstage) due to: {e}")
    client = None

# 라우터 정의
diary_keyword = APIRouter(prefix="/keywords")


# ================================
# 1) 요청 Body 모델 및 응답 모델 정의
# ================================
class KeywordRequest(BaseModel):
    """키워드를 추출할 텍스트 내용을 담는 요청 모델입니다."""
    content: str = Field(description="키워드 추출 대상이 될 일기나 긴 텍스트 내용")

class KeywordResponse(BaseModel):
    """AI가 반환해야 할 키워드 리스트를 정의하는 모델입니다."""
    keywords: List[str] = Field(description="텍스트에서 추출된 최대 4개의 핵심 키워드 리스트")


# ================================
# 2) 키워드 추출 API (Solar-Pro2)
# ================================
@diary_keyword.post("/extract", tags=["Keywords"], response_model=KeywordResponse)
async def extract_keywords_api(req: KeywordRequest):
    if not client:
        raise HTTPException(status_code=503, detail="API Client is not initialized. Check API Key.")

    # AI에게 전달할 목표 JSON 스키마 예시
    json_schema_example = json.dumps({
        "keywords": ["키워드1", "키워드2", "키워드3", "키워드4"]
    }, ensure_ascii=False)

    system_prompt_content = (
        "너는 입력된 텍스트 내용을 분석하여 핵심 키워드를 추출하는 전문 AI 분석가다. "
        "너의 **유일하고 최종적인 응답은 반드시 JSON 객체**여야 하며, 어떠한 설명, 안내 문장, 추가적인 텍스트도 절대 포함되어서는 안 된다. "
        "텍스트를 요약하는 가장 중요한 4개의 키워드를 **반드시 리스트 형식**으로 반환해야 한다. "
        "키워드는 명사 또는 명사구 형태여야 한다. 최대 4개만 추출한다."
        f"**반드시 이 JSON 스키마를 따를 것:** {json_schema_example}" 
    )

    messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": f"다음 텍스트에서 키워드를 추출해줘:\n\n{req.content}"}
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
        raw_keyword_json = response.choices[0].message.content

        # 응답이 유효한지 Pydantic 모델로 검증 및 반환
        return KeywordResponse.model_validate_json(raw_keyword_json)

    except Exception as e:
        # API 호출 또는 JSON 파싱 중 오류 발생 시
        print(f"AI 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"AI 처리 중 오류 발생: {e}")