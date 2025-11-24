import torch
from transformers import AutoModel, XLNetTokenizer
import torch.nn.functional as F
import random
import datetime

# 1. 토크나이저 및 모델 로드
print("KoBERT 토크나이저 및 모델 로드 중...")
tokenizer = XLNetTokenizer.from_pretrained("skt/kobert-base-v1")
model = AutoModel.from_pretrained("skt/kobert-base-v1")
print("로드 완료.")

# 2. 임의 일기 데이터 100개 생성 함수
def generate_random_diary_entry(entry_id):
    templates = [
        f"{entry_id}번 일기. 오늘은 날씨가 [날씨]해서 기분이 [기분] 좋았다.",
        f"{entry_id}번 일기. [시간]에 [활동]을 하며 [느낌] 시간을 보냈다.",
        f"{entry_id}번 일기. [장소]에서 [누구]를 만나 [경험]했다.",
        f"{entry_id}번 일기. [저녁]에는 [대상]과 함께 [행동]했다. [결론].",
        f"{entry_id}번 일기. 내일은 [미래_기대] 일이 생길 것 같은 예감이 든다.",
        f"{entry_id}번 일기. [오늘의_생각]에 대해 깊이 생각해보는 시간이었다.",
        f"{entry_id}번 일기. 오랜만에 [무엇]을 해서 [감정]을 느꼈다.",
        f"{entry_id}번 일기. [주말]에는 [계획]을 세워볼 생각이다.",
        f"{entry_id}번 일기. [날짜]의 일기. [사건]이 있어서 [반응]했다.",
        f"{entry_id}번 일기. [경험_구체화]하는 하루였다. [총평]."
    ]

    weather = ["맑", "흐림", "비", "구름", "눈"]
    mood = ["정말", "매우", "조금", "꽤", "그냥"]
    time = ["아침", "오후", "저녁", "밤늦게", "점심시간"]
    activity = ["따뜻한 커피 한 잔", "산책", "책 읽기", "음악 감상", "운동"]
    feeling = ["여유로운", "평화로운", "지루한", "즐거운", "힘든"]
    place = ["집 근처 카페", "공원", "도서관", "친구 집", "회사 앞"]
    who = ["친구", "가족", "동료", "혼자", "연인"]
    experience = ["재미있는 영화를 봤", "맛있는 식사를 했", "이야기를 나눴", "쇼핑을 했", "새로운 것을 배웠"]
    evening_activities = ["맛있는 저녁 식사", "늦잠", "드라마 시청", "청소", "게임"]
    target = ["가족들", "친구들", "애인", "나 자신", "반려동물"]
    action = ["즐거운 시간을 보냈", "휴식을 취했", "의견을 교환했", "추억을 만들었", "하루를 정리했"]
    conclusion = ["행복한 하루였다", "피곤했지만 보람 있었다", "생각이 많아졌다", "내일을 기약했다", "아쉬움이 남는다"]
    future_expectation = ["신나는", "흥미로운", "새로운", "특별한", "어려운"]
    todays_thought = ["인생의 의미", "직업의 가치", "인간관계", "미래 계획", "과거의 추억"]
    what = ["새로운 음식", "오래된 영화", "친구와의 통화", "취미 활동", "여행 계획"]
    emotion = ["즐거움", "편안함", "아쉬움", "기대감", "만족감"]
    weekend_plan = ["여행 준비", "밀린 잠 자기", "운동하기", "친구 만나기", "영화 보기"]
    event = ["예상치 못한 소식", "작은 성공", "어려운 문제", "새로운 만남", "오래된 친구와의 재회"]
    response = ["놀랐다", "기뻤다", "고민에 빠졌다", "즐거웠다", "반가웠다"]
    detailed_experience = ["새로운 아이디어를 떠올리", "예상치 못한 행운이 찾아오", "소소한 일상에서 행복을 찾", "하루 종일 생각에 잠기", "즐거운 대화를 나누"]
    summary = ["즐거운 하루였다", "생각이 깊어진 날이었다", "평범했지만 소중한 하루였다", "다음이 기대되는 하루였다", "피곤했지만 알찬 하루였다"]


    replacements = {
        "[날씨]": random.choice(weather),
        "[기분]": random.choice(mood),
        "[시간]": random.choice(time),
        "[활동]": random.choice(activity),
        "[느낌]": random.choice(feeling),
        "[장소]": random.choice(place),
        "[누구]": random.choice(who),
        "[경험]": random.choice(experience),
        "[저녁]": random.choice(evening_activities),
        "[대상]": random.choice(target),
        "[행동]": random.choice(action),
        "[결론]": random.choice(conclusion),
        "[미래_기대]": random.choice(future_expectation),
        "[오늘의_생각]": random.choice(todays_thought),
        "[무엇]": random.choice(what),
        "[감정]": random.choice(emotion),
        "[주말]": random.choice(weekend_plan),
        "[사건]": random.choice(event),
        "[반응]": random.choice(response),
        "[경험_구체화]": random.choice(detailed_experience),
        "[총평]": random.choice(summary),
        "[날짜]": (datetime.date(2025, 1, 1) + datetime.timedelta(days=random.randint(0, 364))).strftime("%Y년 %m월 %d일")
    }

    template = random.choice(templates)
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template

# 100개의 임의 일기 데이터 생성
diary_data_100 = [generate_random_diary_entry(i+1) for i in range(100)]

print("## 📓 생성된 100개의 일기 데이터 (일부만 출력)")
for idx, text in enumerate(diary_data_100[:5]): # 처음 5개만 예시로 출력
    print(f"[{idx+1}] {text}")
if len(diary_data_100) > 5:
    print(f"... (총 {len(diary_data_100)}개 문장)")

print("\n" + "="*50 + "\n")

print("## ✨ 100개 문장의 임베딩 추출 중...")

# 문장 임베딩을 추출하는 함수
def get_sentence_embedding(sentence, tokenizer, model):
    tokenized_output = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        return_tensors='pt'
    )
    
    # KoBERT 호환성 문제 해결: token_type_ids 제거
    if 'token_type_ids' in tokenized_output:
        del tokenized_output['token_type_ids']

    with torch.no_grad():
        outputs = model(**tokenized_output)
        last_hidden_states = outputs.last_hidden_state
        
    # [CLS] 토큰의 임베딩만 추출 (문장 임베딩)
    return last_hidden_states[:, 0, :].squeeze(0)

# 2. 100개 문장 모두의 임베딩 벡터 생성
embeddings = []
for i, sentence in enumerate(diary_data_100):
    embedding = get_sentence_embedding(sentence, tokenizer, model)
    embeddings.append(embedding)
    if (i + 1) % 10 == 0:
        print(f"  - {i+1}/{len(diary_data_100)} 문장 임베딩 완료.")

# PyTorch 텐서로 변환: [100, 768]
embedding_matrix = torch.stack(embeddings) 
print(f"총 {embedding_matrix.shape[0]}개 문장의 임베딩 매트릭스 생성 완료: {embedding_matrix.shape}")

print("\n" + "="*50 + "\n")
print("## ✨ 모든 쌍의 코사인 유사도 행렬 계산 중...")

# 3. 모든 쌍의 코사인 유사도 계산
# 벡터 정규화: 각 임베딩 벡터를 단위 길이(크기 1)로 만듭니다.
normalized_embeddings = F.normalize(embedding_matrix, p=2, dim=1)

# 코사인 유사도 행렬 계산: 행렬 * 행렬의 전치 (A * A_T)
# 결과는 [100x100] 행렬이 됩니다.
similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(0, 1))

# 결과 출력 (행렬이 너무 커서 전체 출력은 어렵습니다. 일부만 출력)
print(f"총 {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} 코사인 유사도 행렬 생성 완료.")
print("행렬의 (i, j) 원소는 문장 i와 문장 j의 유사도를 나타냅니다.")

print("\n## 📊 코사인 유사도 행렬 (일부 출력: 0~4번 문장 vs 0~4번 문장)")
# 처음 5x5 부분만 출력하여 예시로 보여줍니다.
similarity_matrix_np = similarity_matrix[:5, :5].cpu().numpy()
formatted_matrix_partial = [[f"{val:.4f}" for val in row] for row in similarity_matrix_np]

header_partial = [""] + [f"Sent {i+1}" for i in range(5)]
row_format_partial = "{:<10}" * (len(header_partial))
print(row_format_partial.format(*header_partial))
print("-" * (10 * len(header_partial)))

for i, row in enumerate(formatted_matrix_partial):
    print(row_format_partial.format(f"Sent {i+1}", *row))

# 모든 문장 쌍의 평균 유사도 (단조로움 지수) 계산
# 대각선 (자기 자신과의 유사도)을 제외한 값들의 평균
# 상삼각 행렬 또는 하삼각 행렬의 원소만 사용합니다.
upper_triangle_indices = torch.triu_indices(similarity_matrix.shape[0], similarity_matrix.shape[1], offset=1)
pairwise_similarities = similarity_matrix[upper_triangle_indices[0], upper_triangle_indices[1]]
average_similarity = torch.mean(pairwise_similarities).item()

print("\n" + "="*50)
print(f"✅ 총 {len(diary_data_100)}개 문장에 대한 모든 쌍의 코사인 유사도 계산 완료.")
print(f"**전체 문장 쌍 개수:** {len(diary_data_100) * (len(diary_data_100) - 1) // 2}개")
print(f"**모든 문장 쌍의 평균 코사인 유사도 (의미적 단조로움 지수):** {average_similarity:.4f}")
print("평균 유사도 값이 1.0에 가까울수록 문장들이 전반적으로 단조롭다는 의미입니다.")
print("="*50)