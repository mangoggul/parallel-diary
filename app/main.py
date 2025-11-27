from typing import Union

from fastapi import FastAPI
from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from api.analysis import analysis
from api.create_diary import create_diary
from api.diary_keyword import diary_keyword
from api.create_parallel_diary import create_parallel_diary
from api.chatting import chat_history_router



app = FastAPI()

origins = [
    # Vercel 배포 주소 (HTTPS 사용)
    "https://parallel-diary-frontend.vercel.app",
    
    # 로컬 개발 환경 주소 (HTTP 사용)
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins= origins,
    allow_credentials=True,         # 쿠키를 포함한 인증 정보 교환 허용
    allow_methods=["*"],            # 모든 HTTP 메서드 (GET, POST, PUT, DELETE 등) 허용
    allow_headers=["*"],            # 모든 HTTP 헤더 허용
)



app.include_router(analysis)

app.include_router(create_diary)
app.include_router(create_parallel_diary)

app.include_router(diary_keyword)

app.include_router(chat_history_router)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}