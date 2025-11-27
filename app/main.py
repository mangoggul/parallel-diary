from typing import Union

from fastapi import FastAPI
from fastapi import APIRouter
from api.analysis import analysis
from api.create_diary import create_diary
from api.diary_keyword import diary_keyword
from api.create_parallel_diary import create_parallel_diary
from api.chatting import chat_history_router

app = FastAPI()
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