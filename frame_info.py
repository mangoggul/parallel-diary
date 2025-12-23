import cv2
import torch
import json
from ultralytics import YOLO

# --- [1. 설정] ---
VIDEO_SOURCES = [f"video/video_{i}.mp4" for i in range(1, 17)] 
THRESHOLDS = {'fire': 0.20, 'smoke': 0.30, 'knife': 0.2, 'person': 0.2}

device = "cuda" if torch.cuda.is_available() else "cpu"
fire_model = YOLO("best.pt")
weapon_model = YOLO("yolo_small_weights.pt")

def get_zone_status(frame, zone_id):
    """이미지 한 장을 분석하여 JSON 데이터 반환"""
    h, w, _ = frame.shape
    total_area = w * h
    
    data = {
        "zoneId": zone_id,
        "fireLevel": 0.0,
        "smokeLevel": 0.0,
        "knife": False,
        "people_cnt": 0
    }

    # 1. 화재/연기 추론
    f_res = fire_model.predict(frame, imgsz=320, verbose=False, device=device)[0]
    fire_sum = 0
    smoke_sum = 0
    for box in f_res.boxes:
        cls = fire_model.names[int(box.cls)].lower()
        if box.conf >= THRESHOLDS.get(cls, 0.25):
            b = box.xyxy[0].cpu().numpy()
            area = (b[2] - b[0]) * (b[3] - b[1])
            if cls == 'fire': fire_sum += area
            elif cls == 'smoke': smoke_sum += area

    # 2. 사람/칼 추론
    w_res = weapon_model.predict(frame, imgsz=320, verbose=False, device=device)[0]
    for box in w_res.boxes:
        cls_idx = int(box.cls)
        conf = float(box.conf)
        if cls_idx == 0 and conf >= THRESHOLDS['person']:
            data["people_cnt"] += 1
        elif cls_idx == 43 and conf >= THRESHOLDS['knife']:
            data["knife"] = True

    data["fireLevel"] = round(min(fire_sum / total_area, 1.0), 4)
    data["smokeLevel"] = round(min(smoke_sum / total_area, 1.0), 4)
    
    return data

# --- [2. 16번 순차 실행 세션] ---
final_results = []

print("🚀 16개 구역 개별 분석 시작...")

for i, src in enumerate(VIDEO_SOURCES):
    cap = cv2.VideoCapture(src)
    success, frame = cap.read()
    
    zone_id = f"zone_{i+1:02d}"
    
    if success:
        # 분석 수행 및 결과 즉시 출력
        status = get_zone_status(frame, zone_id)
        print(f"[{zone_id}] 결과: {json.dumps(status, ensure_ascii=False)}")
        final_results.append(status)
    else:
        print(f"[{zone_id}] 오류: 영상을 불러올 수 없습니다. ({src})")
    
    cap.release() # 분석 후 즉시 리소스 해제

print("\n✅ 모든 분석이 완료되었습니다.")
# 필요 시 final_results를 한꺼번에 반환하거나 저장할 수 있습니다.