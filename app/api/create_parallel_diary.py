from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict

from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# ----------------------
# Upstage API 로딩 (기존 설정 유지)
# ----------------------
load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL", "https://api.upstage.ai/v1")
)

create_parallel_diary = APIRouter(prefix="/diary")

# --------------------------------
# 1) 요청 Body 모델 및 응답 모델 정의 (ID 제거)
# --------------------------------

# 사용자의 원본 일기 내용과 ID를 받는 요청 모델
class ParallelDiaryRequest(BaseModel):
    # 원본 일기의 고유 ID 추가
    original_diary_id: str = Field(description="원본 일기의 고유 식별자 (ID)")
    original_content: str = Field(description="사용자가 작성한 원본 일기 내용")

# AI가 반드시 반환해야 할 JSON 구조를 정의 (평행일기 결과)
class ParallelDiaryResponseContent(BaseModel):
    # 생성된 평행일기의 고유 ID == 원본 일기의 고유 ID 와 같음
    parallel_diary_id: str = Field(description="새로 생성된 평행일기의 고유 식별자 (ID) == 원본 일기 ID 와 같음")
    parallel_content: str = Field(description="원본 일기 내용을 바탕으로 '~다' 형식으로 작성된 새로운 평행일기 본문")

# --------------------------------
# 2) 평행일기 생성 API (Solar-Pro2) (프롬프트 수정)
# --------------------------------
@create_parallel_diary.post(
    "/make-parallel-diary", 
    tags=["Diary"], 
    response_model=ParallelDiaryResponseContent
)
async def make_parallel_diary(req: ParallelDiaryRequest):
    
    # AI가 반환해야 할 JSON 스키마 예시 (content만 남음)
    json_schema_example = '{"parallel_content": "string"}'

    # 시스템 프롬프트 정의 (ID 관련 내용 제거)
    system_prompt_content = (
        "너는 사용자의 일기 내용을 듣고, 그 내용과는 완전히 다른 방향의 '평행 세계'에서의 일기를 상상하여 작성해주는 전문 AI 작가다. "
        "예를 들어, 사용자가 '나는 오늘 시험에 붙었다'고 하면, 너는 '나는 오늘 시험을 보러 가지 않았다'와 같은 다른 내용으로 일기를 작성해야 한다. "
        "적당히 긍정적인 일기를 작성하는 게 좋다."
        "너의 **유일하고 최종적인 응답은 반드시 하나의 JSON 객체**여야 하며, 어떠한 설명, 안내 문장, 추가적인 텍스트도 절대 포함되어서는 안 된다. "
        "일기는 반드시 하나만 작성하고, 말투는 '~다' 형식으로 작성한다. "
        "**반드시 이 스키마를 따를 것:** "
        f'{json_schema_example}'
    )
    
    # AI에게 전달할 전체 메시지 구성
    all_messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": f"원본 일기: {req.original_content}"}
    ]

    try:
        # ===========================
        # Solar-Pro2에게 요청 보내기
        # ===========================
        response = client.chat.completions.create(
            model="solar-pro2",
            messages=all_messages,
            stream=False,
            response_format={"type": "json_object"},
        )

        # AI 응답에서 JSON 문자열 추출
        raw_diary_json = response.choices[0].message.content

        # 1. AI 응답 JSON 문자열을 딕셔너리로 변환
        ai_response_dict = json.loads(raw_diary_json)
        
        # 2. 요청받은 original_diary_id를 parallel_diary_id 키로 추가
        ai_response_dict["parallel_diary_id"] = req.original_diary_id

        # 3. 최종 딕셔너리를 Pydantic 모델로 검증 및 반환
        return ParallelDiaryResponseContent(**ai_response_dict)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 처리 중 오류 발생: {e}")