# pip install openai
 
from openai import OpenAI # openai==1.52.2
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
    api_key= os.getenv('API_KEY'),
    base_url="https://api.upstage.ai/v1"
)
 
stream = client.chat.completions.create(
    model="solar-pro2",
    messages=[
        
        {
            "role": "system",
            "content": "너는 사용자의 하루 일과를 듣고 요약하여 일기를 작성해주는 친절한 AI 비서야. 너의 response 는 json 데이터로 응답해야되고 diary_id 필드랑 content 필드로 답해줘. "
            "그리고 일기는 반드시 하나만 작성해. 말투는 ~다 식의 말투 그리고 json data 이외의 응답은 절대 작성 금지"
        },
        {
            "role": "user",
            "content": "오늘 하루가 너무 길었어. 아침 7시에 일어나서 바로 출근 준비 했지."
        },
        {
            "role": "assistant",
            "content": "7시면 조금 일찍 시작하셨네요! 출근길은 어떠셨어요? 피곤하지 않으셨나요?"
        },
        {
            "role": "user",
            "content": "오늘은 지하철에 사람이 너무 많아서 힘들었어. 회사 도착하자마자 중요한 회의가 있어서 정신없이 발표하고."
        },
        {
            "role": "assistant",
            "content": "힘든 출근길에 바로 중요한 일정을 소화하셨군요. 발표는 잘 마무리되셨나요? 점심은 뭐 드셨어요?"
        },
        {
            "role": "user",
            "content": "점심은 그냥 회사 근처 김치찌개 식당에서 빠르게 먹었어. 오후에는 밀린 서류 작업 처리하고. 제일 좋았던 건 퇴근하고 집에 와서 따뜻한 물로 샤워한 거야."
        },
    

    ],
    stream=True,
)
 
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

# print("스트림" , stream)
# Use with stream=False
# print(stream.choices[0].message.content)