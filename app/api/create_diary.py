from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict

from openai import OpenAI
from dotenv import load_dotenv
import os

# ----------------------
# Upstage API 로딩
# ----------------------
load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.upstage.ai/v1"
)

create_diary = APIRouter(prefix="/diary")


# ================================
# 1) 요청 Body 모델 및 응답 모델 정의
# ================================
# 요청 모델은 List[dict]로 유지 (이전 대화 기록)
class DiaryRequest(BaseModel):
    messages: List[Dict[str, str]]  # [{"role": "user", "content": "..."}]

# AI가 반드시 반환해야 할 JSON 구조를 정의 (Pydantic)
class DiaryResponseContent(BaseModel):
    diary_id: str = Field(description="일기 고유 ID (예: d_20251124_001)")
    content: str = Field(description="사용자의 대화 내용을 요약하여 '~다' 형식으로 작성된 일기 본문")


# ================================
# 2) 일기 생성 API (Solar-Pro2)
# ================================
@create_diary.post("/make-diary", tags=["Diary"], response_model=DiaryResponseContent)
async def make_diary(req: DiaryRequest):
    
    json_schema_example = '{"diary_id": "string", "content": "string"}'

    system_prompt = {
        "role": "system",
        "content": (
            "너는 사용자의 하루 일과를 듣고 요약하여 일기를 작성해주는 전문 AI 비서다. "
            "너의 **유일하고 최종적인 응답은 반드시 하나의 JSON 객체**여야 하며, 어떠한 설명, 안내 문장, 추가적인 텍스트도 절대 포함되어서는 안 된다. "
            "일기는 반드시 하나만 작성하고, 말투는 '~다' 형식으로 작성한다. "
            "**반드시 이 스키마를 따를 것:** "
            f'{json_schema_example}' 
        )
    }

    all_messages = [system_prompt] + req.messages

    try:
        # ===========================
        # Solar-Pro2에게 요청 보내기
        # ===========================
        response = client.chat.completions.create(
            model="solar-pro2",
            messages=all_messages,
            stream=False,
            # 💡 가장 중요한 부분: API 자체에 JSON 응답을 강제함
            response_format={"type": "json_object"},
        )

        # AI 응답에서 JSON 문자열 추출
        raw_diary_json = response.choices[0].message.content

        # 응답이 유효한지 Pydantic 모델로 검증 및 반환
        return DiaryResponseContent.model_validate_json(raw_diary_json)

    except Exception as e:
        # API 호출 또는 JSON 파싱 중 오류 발생 시
        raise HTTPException(status_code=500, detail=f"AI 처리 중 오류 발생: {e}")