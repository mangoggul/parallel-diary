from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Generator, Any

from openai import OpenAI
from dotenv import load_dotenv
import os

# ----------------------
# 1. Upstage API 로딩 및 클라이언트 초기화
# ----------------------
load_dotenv()
# 환경 변수에서 API 키와 기본 URL을 로드합니다.
client = OpenAI(
    api_key=os.getenv('API_KEY'),
    base_url=os.getenv("API_BASE_URL", "https://api.upstage.ai/v1")
)

chat_history_router = APIRouter(prefix="/chat-history")

# --------------------------------
# 2. 요청 Body 모델 정의: 대화 기록 유지의 핵심
# --------------------------------
class ChatRequest(BaseModel):
    # 이 리스트에 이전 대화 내용이 모두 담겨서 API로 전송됩니다.
    messages: List[Dict[str, str]] = Field(
        ...,
        description="**[대화 기록 유지의 핵심]** 이전 대화와 현재 사용자 메시지를 포함하는 메시지 리스트입니다. (예: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}])"
    )

# --------------------------------
# 3. 스트리밍 응답을 위한 제너레이터 함수
# --------------------------------

def stream_response_generator(messages: List[Dict[str, str]]) -> Generator[str, Any, None]:
    """
    제공된 messages 리스트(대화 기록 포함)를 AI에 전달하고, 
    응답을 SSE 형식의 문자열로 스트리밍하는 제너레이터 함수입니다.
    """
    try:
        # 💡 시스템 프롬프트: 항상 대화의 첫 부분에 위치하여 AI의 역할을 정의합니다.
        system_prompt = {
            "role": "system",
            "content": "너는 사용자의 하루 일과가 궁금한 친절한 AI 비서야. 사용자에게 하루 일과를 계속 질문해줘. 이전 대화 내용을 참고하여 문맥에 맞는 대답을 딱 하나만 해."
        }
        
        # 메시지 리스트의 첫 요소가 'system'이 아니거나 비어있으면 시스템 프롬프트를 추가합니다.
        if not messages or messages[0].get("role") != "system":
            all_messages = [system_prompt] + messages
        else:
            all_messages = messages

        # Solar-Pro2에게 요청 보내기 (stream=True)
        # all_messages 리스트 전체가 AI에게 전달되어 문맥이 유지됩니다.
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
        # 오류 발생 시 클라이언트에게 오류 메시지를 전송
        error_message = f"AI 처리 중 오류 발생: {e}"
        yield f"data: [ERROR] {error_message}\n\n"
        # API 오류가 발생하면 HTTP 500 예외를 발생시켜 로그를 남깁니다.
        raise HTTPException(status_code=500, detail=error_message)


# --------------------------------
# 4. API 엔드포인트 정의
# --------------------------------
@chat_history_router.post("/stream-chat-with-history", tags=["Chat History"], response_class=StreamingResponse)
async def stream_chat_with_history(req: ChatRequest):
    """
    대화 기록(messages)을 받아 AI 비서의 응답을 실시간으로 스트리밍합니다 (SSE).
    """
    return StreamingResponse(
        stream_response_generator(req.messages),
        media_type="text/event-stream"
    )

# --------------------------------
# 5. 사용 예시
# --------------------------------
# 이 라우터를 FastAPI 앱에 등록하여 사용하세요:
#
# from fastapi import FastAPI
# app = FastAPI()
# app.include_router(chat_history_router)
#
# 요청 예시 (두 번째 대화):
# {
#   "messages": [
#     {"role": "user", "content": "오늘 점심은 뭐 먹었는지 기억해 줄래?"},
#     {"role": "assistant", "content": "저는 AI라서 식사를 하지 않아요. 사용자님은 점심으로 무엇을 드셨나요?"},
#     {"role": "user", "content": "저는 샌드위치를 먹었는데 별로 맛이 없었어요."}
#   ]
# }