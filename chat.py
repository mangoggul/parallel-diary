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
            "content": "너는 사용자의 하루 일과가 궁금한 친절한 AI 비서야."
            "사용자에게 하루 일과를 게속 질문해줘"
        },
       
        {
            "role": "user",
            "content": "오늘 하루는 굉장히 피곤했어"
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