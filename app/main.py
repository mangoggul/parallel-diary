from typing import Union

from fastapi import FastAPI
from fastapi import APIRouter
from api.analysis import analysis
from api.create_diary import create_diary

app = FastAPI()
app.include_router(analysis)
app.include_router(create_diary)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}