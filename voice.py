import os
import urllib.request
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')

encText = urllib.parse.quote("오늘 하루는 어떠셨나요?")
data = "speaker=vara&volume=0&speed=0&pitch=0&format=mp3&text=" + encText
url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
request = urllib.request.Request(url)
request.add_header("X-NCP-APIGW-API-KEY-ID",CLIENT_ID)
request.add_header("X-NCP-APIGW-API-KEY",CLIENT_SECRET)
response = urllib.request.urlopen(request, data=data.encode('utf-8'))
rescode = response.getcode()
if(rescode==200):
    print("TTS mp3 저장")
    response_body = response.read()
    with open('1111.mp3', 'wb') as f:
        f.write(response_body)
else:
    print("Error Code:" + rescode)
