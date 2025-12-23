from typing import Union

from fastapi import FastAPI
from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from api.create_diary import create_diary
from api.diary_keyword import diary_keyword
from api.create_parallel_diary import create_parallel_diary
from api.chatting import chat_history_router
from api.recommendation import diary_recommend
from api.Integrated_diary_create import recommend_and_parallel_diary
from api.integrated_path import cctv_router, start_cctv_streams

app = FastAPI()

origins = [
    # Vercel 배포 주소 (HTTPS 사용)
    "https://parallel-diary-frontend.vercel.app",
    "http://3.105.9.139:3000",
    "https://dappy-sw-ai-hack.vercel.app/"
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

@app.on_event("startup")
async def startup_event():
    start_cctv_streams()


app.include_router(create_diary)
app.include_router(create_parallel_diary)

app.include_router(diary_keyword)

app.include_router(chat_history_router)

app.include_router(diary_recommend)
app.include_router(recommend_and_parallel_diary)

app.include_router(cctv_router)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

