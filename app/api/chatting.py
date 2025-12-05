from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Generator, Any
import json
import os
from dotenv import load_dotenv

from openai import OpenAI

# ----------------------
# 1. 환경 설정 및 클라이언트 초기화
# ----------------------
load_dotenv()
# 환경 변수에서 API 키와 기본 URL을 로드합니다.
# API_BASE_URL 환경 변수가 설정되지 않은 경우 기본값으로 Upstage API를 사용합니다.
try:
    client = OpenAI(
        api_key=os.getenv('API_KEY'),
        base_url=os.getenv("API_BASE_URL", "https://api.upstage.ai/v1")
    )
except Exception as e:
    # 클라이언트 초기화 오류 시 로그를 남기고 None으로 설정
    print(f"Warning: Failed to initialize OpenAI client: {e}")
    client = None

# ------------------------------------------------------------------
# 라우터 정의
# ------------------------------------------------------------------
chat_history_router = APIRouter(prefix="/chat")

# --------------------------------
# 2. 요청 Body 모델 정의: 대화 기록 유지의 핵심
# --------------------------------
class ChatRequest(BaseModel):
    """
    대화 기록 전체를 담는 요청 모델.
    messages 리스트는 AI와의 문맥을 유지하는 데 사용됩니다.
    """
    messages: List[Dict[str, str]] = Field(
        ...,
        description="**[대화 기록 유지의 핵심]** 이전 대화와 현재 사용자 메시지를 포함하는 메시지 리스트입니다. (role: 'user' 또는 'assistant', content: '대화 내용')"
    )

    # Swagger UI에 표시될 예시를 커스터마이징하여 프론트엔드 개발자가 이해하기 쉽도록 합니다.
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {"role": "assistant", "content": "오늘 일상은 어떠셨나요?"},
                        {"role": "user", "content": "오늘 점심은 초밥 먹었어."},
                        {"role": "assistant", "content": "초밥 맛있으셨겠어요. 또 무슨 일을 하셨나요?"},
                        {"role": "user", "content": "AI 가 개발자 일자리 다 없앨 거 같애서 러다이트 운동을 일으켰어"}
                    ]
                }
            ]
        }
    }


# --------------------------------
# 3. 스트리밍 응답을 위한 제너레이터 함수
# --------------------------------

def stream_response_generator(messages: List[Dict[str, str]]) -> Generator[str, Any, None]:
    """
    제공된 messages 리스트(대화 기록 포함)를 AI에 전달하고, 
    응답을 SSE 형식의 문자열로 스트리밍하는 제너레이터 함수입니다.
    """
    if client is None:
        raise Exception("API Client is not initialized.")

    try:
        # 💡 시스템 프롬프트: AI의 역할을 정의합니다.
        # 기존 코드를 개선하여, system prompt가 messages 리스트에 포함되지 않았을 경우에만 추가합니다.
        system_prompt_content = (
            "당신은 사용자가 하루를 되돌아보고 경험을 상세하게 기록하도록 돕는 전문적인 일기 작성 AI 도우미입니다."
            "당신의 주된 역할은 사용자의 감정과 행동에 깊이 공감하며 긍정적으로 인정한 후, 먼저 해당 행동에 대한 구체적인 후속 질문을 던져 기록의 깊이를 더하도록 유도합니다. 그 후, 자연스러운 대화 형태로 그다음 시간대나 행동에 대해 질문하여 일기의 흐름을 시간 순서대로 이끌어가는 것입니다."
            "사용자가 하루의 한 장면을 기록하면, 그 행동에 공감하고 해당 행동의 세부 사항을 물은 다음, 즉시 하루의 다음 장면을 묻는 질문을 던져야 합니다."
            "답변 스타일은 따뜻하고 격려하는 어조를 유지하되, 이모지나 마크업(굵게, 기울임 등)은 사용하지 않아야 합니다."
            "최대한 짧게 답변합니다. () 와 같은 어구들은 사용하지 않습니다."
        )
        all_messages = messages[:] 

        # messages 리스트의 첫 요소가 'system'이 아니거나 리스트가 비어있으면 시스템 프롬프트를 맨 앞에 추가합니다.
        if not all_messages or all_messages[0].get("role") != "system":
            system_prompt = {"role": "system", "content": system_prompt_content}
            all_messages.insert(0, system_prompt)

        # Solar-Pro2에게 요청 보내기 (stream=True)
        stream = client.chat.completions.create(
            model="solar-pro2",
            messages=all_messages,
            stream=True,
        )

        # 스트림 청크를 읽고 클라이언트로 전송 (SSE 포맷)
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                # Server-Sent Events (SSE) 포맷: data: [내용]\n\n
                yield f"data: {content}\n\n"

    except Exception as e:
        # 오류 발생 시 클라이언트에게 오류 메시지를 전송하고 서버 로그에 기록
        error_message = f"AI 처리 중 오류 발생: {type(e).__name__} - {str(e)}"
        print(error_message) # 서버 로그에 오류 출력
        yield f"data: [ERROR] {error_message}\n\n"
        # 스트리밍 함수 내부에서 HTTPException을 직접 발생시키기보다, 
        # 에러 메시지를 클라이언트에 전달하고 함수를 종료하는 것이 스트리밍의 일반적인 처리 방식입니다.


# --------------------------------
# 4. API 엔드포인트 정의
# --------------------------------
@chat_history_router.post("/chat-sse", tags=["SSE Chat"], response_class=StreamingResponse)
async def stream_chat_with_history(req: ChatRequest):
    """
    대화 기록(messages)을 받아 AI 비서의 응답을 실시간으로 스트리밍합니다 (SSE).
    """
    if client is None:
        raise HTTPException(status_code=503, detail="AI Service is unavailable. Check API initialization.")

    return StreamingResponse(
        stream_response_generator(req.messages),
        media_type="text/event-stream"
    )

