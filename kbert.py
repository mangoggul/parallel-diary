from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import numpy as np

# 1. 비교할 4개의 일기 데이터
diary_data = [
    "2025년 11월 20일 목요일. 날씨가 맑아서 기분이 정말 좋았다.", # 1번
    "아침에 따뜻한 커피 한 잔을 마시며 여유로운 시간을 보냈다.", # 2번
    "오후에는 친구를 만나서 재미있는 영화를 봤다. 내용은 조금 슬펐지만 좋았다.", # 3번
    "오늘은 바로 학교에 갔다. 매우 힘든 수업이었다. 그리디콘에서 흥미로운 강연을 들었다. 재밌었다", # 4번
    "오후에 일어나서 친구랑 영화를 봤다. 내용이 참 재밌었다." #5번
]
N = len(diary_data)

# 2. KoBERT 기반 SBERT 모델 로드
# 한국어 문장 유사도에 최적화된 모델을 사용합니다.
print("SBERT 모델 로드 중...")
# 모델 이름: KR-SBERT-V40K-klueNLI-512 (한국어 특화)
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
print("로드 완료.")

# 3. 임베딩 생성
print(f"총 {N}개 문장 임베딩 생성 시작...")
# SBERT는 내부적으로 배치 처리 및 토크나이징을 자동 수행합니다.
embeddings = model.encode(diary_data, convert_to_tensor=True)
print(f"임베딩 매트릭스 생성 완료: {embeddings.shape}") # [4, 512] (512는 임베딩 차원)

# 4. 모든 쌍의 코사인 유사도 행렬 계산
# F.cosine_similarity를 사용하여 행렬을 계산합니다.
similarity_matrix = F.cosine_similarity(
    embeddings.unsqueeze(1), # [4, 1, 512]
    embeddings.unsqueeze(0), # [1, 4, 512]
    dim=2 # 마지막 차원(임베딩 차원)을 따라 비교
)

# 5. 모든 문장 쌍의 평균 유사도 (의미적 단조로움 지수) 계산
# 대각선 (자기 자신과의 유사도)을 제외한 값들의 평균을 구합니다.
upper_triangle_indices = torch.triu_indices(N, N, offset=1)
pairwise_similarities = similarity_matrix[upper_triangle_indices[0], upper_triangle_indices[1]]
average_similarity = torch.mean(pairwise_similarities).item()

# 6. 결과 출력
print("\n" + "="*50)
print(f"## ✨ {N}x{N} 코사인 유사도 행렬")
print("행렬의 (i, j) 원소는 문장 i와 문장 j의 유사도를 나타냅니다.")

# 표 형태로 출력
similarity_matrix_np = similarity_matrix.cpu().numpy()
formatted_matrix = [[f"{val:.4f}" for val in row] for row in similarity_matrix_np]

header = [""] + [f"Sent {i+1}" for i in range(N)]
row_format = "{:<10}" * (N + 1)
print(row_format.format(*header))
print("-" * (10 * (N + 1)))

for i, row in enumerate(formatted_matrix):
    print(row_format.format(f"Sent {i+1}", *row))

print("\n" + "="*50)
print(f"✅ 총 {N}개 문장에 대한 모든 쌍의 코사인 유사도 계산 완료.")
print(f"**전체 문장 쌍 개수:** {N * (N - 1) // 2}개")
print(f"**모든 문장 쌍의 평균 코사인 유사도 (의미적 단조로움 지수):** {average_similarity:.4f}")
print("="*50)