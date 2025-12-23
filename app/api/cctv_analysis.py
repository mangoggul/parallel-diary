import cv2
import torch
import threading
import time
from fastapi import APIRouter
from ultralytics import YOLO

cctv_router = APIRouter()

# --- [1. 설정 및 모델 로드] ---
VIDEO_SOURCES = [f"../video/video_{i}.mp4" for i in range(1, 17)]
THRESHOLDS = {'fire': 0.20, 'smoke': 0.30, 'knife': 0.2, 'person': 0.2}
TARGET_SIZE = 224 

# OpenVINO 모델 로드
try:
    fire_model = YOLO("../best_openvino_model/") 
    weapon_model = YOLO("../yolo_small_weights_openvino_model/")
    print("✅ OpenVINO 모델 로드 성공")
except:
    print("⚠️ OpenVINO 모델 로드 실패, .pt 사용")
    fire_model = YOLO("../best.pt")
    weapon_model = YOLO("../yolo_small_weights.pt")

latest_frames = {f"zone_{i+1:02d}": None for i in range(16)}

def video_streamer(index, src):
    zone_id = f"zone_{index+1:02d}"
    cap = cv2.VideoCapture(src)
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        latest_frames[zone_id] = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE))
        time.sleep(0.01)

def start_cctv_streams():
    for i, src in enumerate(VIDEO_SOURCES):
        thread = threading.Thread(target=video_streamer, args=(i, src), daemon=True)
        thread.start()

@cctv_router.get("/cctv/analyze-all")
async def get_cctv_status():
    start_time = time.time()
    final_response = []
    total_area = TARGET_SIZE * TARGET_SIZE

    # 16개 구역을 순차적으로 분석 (OpenVINO Batch 1 호환성 해결)
    for zone_id, frame in latest_frames.items():
        if frame is None:
            continue

        data = {"zoneId": zone_id, "fireLevel": 0.0, "smokeLevel": 0.0, "knife": False, "people_cnt": 0}

        # [수정] 한 장씩 추론 (OpenVINO RuntimeError 방지)
        f_res_list = fire_model.predict(frame, imgsz=TARGET_SIZE, verbose=False, device='cpu')
        w_res_list = weapon_model.predict(frame, imgsz=TARGET_SIZE, verbose=False, device='cpu')
        
        f_res = f_res_list[0]
        w_res = w_res_list[0]

        # 화재 결과 파싱
        if f_res.boxes:
            fire_sum = 0
            smoke_sum = 0
            for b in f_res.boxes:
                cls_name = fire_model.names[int(b.cls)].lower()
                if b.conf >= THRESHOLDS.get(cls_name, 0.25):
                    box = b.xyxy[0].cpu().numpy()
                    area = (box[2]-box[0]) * (box[3]-box[1])
                    if cls_name == 'fire': fire_sum += area
                    elif cls_name == 'smoke': smoke_sum += area
            data["fireLevel"] = round(min(float(fire_sum) / total_area, 1.0), 4)
            data["smokeLevel"] = round(min(float(smoke_sum) / total_area, 1.0), 4)

        # 무기/사람 결과 파싱
        if w_res.boxes:
            for b in w_res.boxes:
                cls_idx = int(b.cls)
                if cls_idx == 0 and b.conf >= THRESHOLDS['person']:
                    data["people_cnt"] += 1
                elif cls_idx == 43 and b.conf >= THRESHOLDS['knife']:
                    data["knife"] = True
        
        final_response.append(data)

    print(f"⏱️ 분석 소요 시간: {time.time() - start_time:.2f}초")
    return final_response