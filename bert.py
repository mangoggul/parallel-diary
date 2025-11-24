import torch
from transformers import AutoModel, AutoTokenizer, XLNetTokenizer

# 이 방법이 BertTokenizer를 강제로 사용하는 것보다 훨씬 안정적입니다.
tokenizer = XLNetTokenizer.from_pretrained("skt/kobert-base-v1")

# 2. 모델은 AutoModel로 로드 (이것은 보통 문제가 없음)
model = AutoModel.from_pretrained("skt/kobert-base-v1")

# 1. 임의 일기 데이터 생성
diary_data = [
    "2025년 11월 20일 목요일. 날씨가 맑아서 기분이 정말 좋았다.",
    "아침에 따뜻한 커피 한 잔을 마시며 여유로운 시간을 보냈다.",
    "오후에는 친구를 만나서 재미있는 영화를 봤다. 내용은 조금 슬펐지만 좋았다.",
    "저녁에는 오랜만에 가족들과 함께 맛있는 저녁 식사를 했다. 행복한 하루였다.",
    "내일은 더 신나는 일이 생길 것 같은 예감이 든다."
]

print("## 📓 생성된 일기 데이터")
for idx, text in enumerate(diary_data):
    print(f"[{idx+1}] {text}")

print("\n" + "="*50 + "\n")

print("## ✨ XLNetTokenizer를 이용한 토큰화 (Subword 분석)")

# 3. 각 일기 문장에 대해 토큰화 및 분석 수행
for i, sentence in enumerate(diary_data):
    # 특수 토큰([CLS], [SEP])을 포함하여 인코딩
    tokenized_output = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True, # [CLS]와 [SEP] 추가
        return_tensors='pt'       # PyTorch 텐서로 반환
    )

    if 'token_type_ids' in tokenized_output:
        del tokenized_output['token_type_ids']

    # 토큰 ID를 실제 텍스트 토큰으로 변환
    tokens = tokenizer.convert_ids_to_tokens(tokenized_output['input_ids'][0])

    print(f"\n--- [일기 {i+1}] 원본: {sentence} ---")
    print(f"🔍 토큰 시퀀스: {tokens}")
    print(f"📏 토큰 개수 (특수 토큰 포함): {len(tokens)}")

    # 4. BERT 임베딩 벡터 생성
    with torch.no_grad():
        outputs = model(**tokenized_output)
        last_hidden_states = outputs.last_hidden_state

    # 임베딩 벡터의 크기 확인 (토큰 개수 x 임베딩 차원)
    print(f"📊 임베딩 벡터 크기: {last_hidden_states.shape} (토큰수, 임베딩_차원)")