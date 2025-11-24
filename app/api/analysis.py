from fastapi import APIRouter

analysis = APIRouter(prefix="/analysis")


monotony_score = 42

@analysis.get("/", tags = ["analysis"])
async def root_analysis():
    return {"Hello": "Analysis"}


@analysis.get("/monotony_score", tags=["monotony score"])
async def get_monotony(user_id: int):
    return {"user_id": user_id, "monotony_score": monotony_score}