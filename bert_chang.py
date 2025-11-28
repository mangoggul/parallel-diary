import torch
from transformers import AutoModel, XLNetTokenizer
import torch.nn.functional as F

# 1. 토크나이저 및 모델 로드
tokenizer = XLNetTokenizer.from_pretrained("skt/kobert-base-v1")
model = AutoModel.from_pretrained("skt/kobert-base-v1")

# 1. 임의 일기 데이터 (7개 문장 가정)
# 사용자님의 기존 5개 문장에 2개를 추가했습니다.
diary_data = [
    "2025년 11월 20일 목요일. 날씨가 맑아서 기분이 정말 좋았다.", # 1번
    "아침에 따뜻한 커피 한 잔을 마시며 여유로운 시간을 보냈다.", # 2번
    "오후에는 친구를 만나서 재미있는 영화를 봤다. 내용은 조금 슬펐지만 좋았다.", # 3번
    "오늘은 바로 학교에 갔다. 매우 힘든 수업이었다. 그리디콘에서 흥미로운 강연을 들었다. 재밌었다", # 4번
    "오후에 일어나서 친구랑 영화를 봤다. 내용이 참 재밌었다." #5번
]

# 문장 임베딩을 추출하는 함수 (이전 답변에서 사용한 함수와 동일)
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

# 2. 7개 문장 모두의 임베딩 벡터 생성
embeddings = []
for sentence in diary_data:
    embedding = get_sentence_embedding(sentence, tokenizer, model)
    embeddings.append(embedding)

# PyTorch 텐서로 변환: [7, 768] (7개 문장, 768은 KoBERT 임베딩 차원)
embedding_matrix = torch.stack(embeddings) 

# 3. 모든 쌍의 코사인 유사도 계산
# F.cosine_similarity는 일반적으로 두 텐서의 요소를 쌍으로 묶어 계산합니다.
# 모든 쌍의 유사도를 구하기 위해 행렬 곱셈을 사용합니다.

# 벡터 정규화: 각 임베딩 벡터를 단위 길이(크기 1)로 만듭니다.
# 코사인 유사도는 정규화된 벡터의 내적과 같습니다.
normalized_embeddings = F.normalize(embedding_matrix, p=2, dim=1)

# 코사인 유사도 행렬 계산: 행렬 * 행렬의 전치 (A * A_T)
# 결과는 [7x7] 행렬이 되며, (i, j) 원소가 문장 i와 문장 j의 유사도입니다.
similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(0, 1))

# 결과 출력
print("## ✨ 4x4 코사인 유사도 행렬")
print("행렬의 (i, j) 원소는 문장 i와 문장 j의 유사도를 나타냅니다.")

# 출력 포맷 조정 (소수점 4자리까지)
similarity_matrix_np = similarity_matrix.cpu().numpy()
formatted_matrix = [[f"{val:.4f}" for val in row] for row in similarity_matrix_np]

# 표 형태로 출력
header = [""] + [f"Sent {i+1}" for i in range(len(diary_data))]
row_format = "{:<10}" * (len(header))
print(row_format.format(*header))
print("-" * (10 * len(header)))

for i, row in enumerate(formatted_matrix):
    print(row_format.format(f"Sent {i+1}", *row))

print("\n" + "="*50)
print(f"**총 문장 쌍 개수:** {len(diary_data) * (len(diary_data) - 1) // 2}개")
print("유사도 값은 1.0에 가까울수록 의미적으로 유사함을 의미합니다.")