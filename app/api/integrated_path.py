import cv2
import torch
import threading
import time
import heapq
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO

cctv_router = APIRouter()

# --- [1. 그래프 데이터 및 설정] ---
# 기본 그래프 데이터
BASE_FACILITY_GRAPH = {
    '복도1': {'복도2': 3.75, '203호': 1.50, '204호': 1.83, '계단1': 2.0},
    '복도2': {'복도1': 3.75, '복도3': 3.75, '202호': 1.50, '205호': 1.54},
    '복도3': {'복도2': 3.75, '복도4': 3.70, '201호': 1.65, '206호': 1.56, '계단3': 2.0},
    '복도4': {'복도3': 3.70, '복도5': 2.34, '계단2': 2.0},
    '복도5': {'복도4': 2.34, '복도6': 3.50, '207호': 1.68, '208호': 2.12, '211호': 2.12, '계단4': 2.0},
    '복도6': {'복도5': 3.50, '209호': 1.58, '210호': 1.58},
    '201호': {'복도3': 1.65}, '202호': {'복도2': 1.50}, '203호': {'복도1': 1.50},
    '204호': {'복도1': 1.83}, '205호': {'복도2': 1.54}, '206호': {'복도3': 1.56},
    '207호': {'복도5': 1.68}, '208호': {'복도5': 2.12}, '209호': {'복도6': 1.58},
    '210호': {'복도6': 1.58}, '211호': {'복도5': 2.12},
    '계단1': {'복도1': 2.0}, '계단2': {'복도4': 2.0}, '계단3': {'복도3': 2.0}, '계단4': {'복도5': 2.0}
}

# CCTV ID와 실제 위치 매핑 (필요에 따라 수정)
ZONE_MAP = {
    "zone_01": "201호", "zone_02": "202호", "zone_03": "203호", "zone_04": "204호",
    "zone_05": "205호", "zone_06": "206호", "zone_07": "207호", "zone_08": "208호",
    "zone_09": "209호", "zone_10": "210호", "zone_11": "211호", "zone_12": "복도1",
    "zone_13": "복도2", "zone_14": "복도3", "zone_15": "복도4", "zone_16": "복도5"
}

VIDEO_SOURCES = [f"../video/video_{i}.mp4" for i in range(1, 17)]
TARGET_SIZE = 224 

# 모델 로드
try:
    fire_model = YOLO("../best_openvino_model/") 
    weapon_model = YOLO("../yolo_small_weights_openvino_model/")
    print("✅ OpenVINO 모델 로드 성공")
except:
    fire_model = YOLO("../best.pt")
    weapon_model = YOLO("../yolo_small_weights.pt")

latest_frames = {f"zone_{i+1:02d}": None for i in range(16)}
analysis_results = {} # 최신 분석 결과 저장용

class LocationRequest(BaseModel):
    current_location: str

# --- [2. 영상 스트리밍 스레드] ---
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

# --- [3. 분석 및 경로 탐색 API] ---
@cctv_router.post("/get-escape-path")
async def get_escape_path(request: LocationRequest):
    start_time = time.time()
    current_zone_results = []
    
    # 가변 그래프 생성 (위험 구역 가중치 조절용)
    dynamic_graph = {node: neighbors.copy() for node, neighbors in BASE_FACILITY_GRAPH.items()}
    
    # 1. 16개 구역 분석 및 위험 구역 판단
    total_area = TARGET_SIZE * TARGET_SIZE
    for zone_id, frame in latest_frames.items():
        if frame is None: continue
        
        # 분석 결과 초기화
        res_data = {"zoneId": ZONE_MAP.get(zone_id, zone_id), "fireLevel": 0.0, "smokeLevel": 0.0, "knife": False, "people_cnt": 0}
        
        # YOLO 추론
        f_res = fire_model.predict(frame, imgsz=TARGET_SIZE, verbose=False, device='cpu')[0]
        w_res = weapon_model.predict(frame, imgsz=TARGET_SIZE, verbose=False, device='cpu')[0]
        
        if f_res.boxes:
            f_sum, s_sum = 0, 0
            for b in f_res.boxes:
                cls = fire_model.names[int(b.cls)].lower()
                if b.conf >= 0.25:
                    box = b.xyxy[0].cpu().numpy()
                    area = (box[2]-box[0]) * (box[3]-box[1])
                    if cls == 'fire': f_sum += area
                    elif cls == 'smoke': s_sum += area
            res_data["fireLevel"] = float(round(min(f_sum / total_area, 1.0), 4))
            res_data["smokeLevel"] = float(round(min(s_sum / total_area, 1.0), 4))

        if w_res.boxes:
            for b in w_res.boxes:
                if int(b.cls) == 0 and b.conf >= 0.2: res_data["people_cnt"] += 1
                elif int(b.cls) == 43 and b.conf >= 0.2: res_data["knife"] = True

        current_zone_results.append(res_data)

        # 2. 위험 구역 간선 가중치 수정 (길이 999로 변경)
        if res_data["fireLevel"] > 0.1 or res_data["smokeLevel"] > 0.1 or res_data["knife"]:
            danger_node = res_data["zoneId"]
            if danger_node in dynamic_graph:
                # 해당 노드로 들어오거나 나가는 모든 길을 막음
                for neighbor in dynamic_graph[danger_node]:
                    dynamic_graph[danger_node][neighbor] = 999.0
                for node in dynamic_graph:
                    if danger_node in dynamic_graph[node]:
                        dynamic_graph[node][danger_node] = 999.0

    # 3. 다익스트라 경로 계산
    start_node = request.current_location
    if start_node not in dynamic_graph:
        raise HTTPException(status_code=404, detail="Location not found")

    stairs = ['계단1', '계단2', '계단3', '계단4']
    distances = {node: float('inf') for node in dynamic_graph}
    predecessors = {node: None for node in dynamic_graph}
    distances[start_node] = 0
    queue = [(0, start_node)]

    while queue:
        d, curr = heapq.heappop(queue)
        if d > distances[curr]: continue
        for neighbor, weight in dynamic_graph[curr].items():
            new_dist = d + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = curr
                heapq.heappush(queue, (new_dist, neighbor))

    # 최단 거리 계단 탐색
    nearest_stair = min(stairs, key=lambda s: distances[s])
    
    # 경로 역추적
    path = []
    curr = nearest_stair
    while curr is not None:
        path.append(curr)
        curr = predecessors[curr]

    print(f"⏱️ 통합 처리 시간: {time.time() - start_time:.2f}초")
    
    return {
        "analysis": current_zone_results,
        "escape_path": {
            "start": start_node,
            "destination": nearest_stair,
            "path": path[::-1],
            "total_distance": float(round(distances[nearest_stair], 2)),
            "is_safe": distances[nearest_stair] < 999.0
        }
    }