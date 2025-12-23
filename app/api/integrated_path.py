import cv2
import torch
import threading
import time
import heapq
import traceback
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO

cctv_router = APIRouter()

# --- [1. 설정 및 그래프 데이터] ---
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

ZONE_MAP = {f"zone_{i+1:02d}": name for i, name in enumerate([
    "201호", "202호", "203호", "204호", "205호", "206호", "207호", "208호",
    "209호", "210호", "211호", "복도1", "복도2", "복도3", "복도4", "복도5", "복도6"
])}

N_ZONES = len(ZONE_MAP)
VIDEO_SOURCES = [f"../video/video_{i}.mp4" for i in range(1, N_ZONES + 1)]
TARGET_SIZE = 224 

try:
    fire_model = YOLO("../best_openvino_model/", task='detect') 
    weapon_model = YOLO("../yolo_small_weights_openvino_model/", task='detect')
    print("✅ OpenVINO 모델 로드 성공")
except Exception as e:
    fire_model = YOLO("../best.pt", task='detect')
    weapon_model = YOLO("../yolo_small_weights.pt", task='detect')
    print(f"⚠️ PyTorch (.pt) 로드: {e}")

latest_frames = {f"zone_{i+1:02d}": None for i in range(N_ZONES)}

cache_lock = threading.Lock()
cached_zone_results = None
cached_dynamic_graph = None
cache_timestamp = 0.0
cache_ttl = 10.0
analysis_in_progress = False

class LocationRequest(BaseModel):
    current_location: str

# --- [2. 영상 스트리밍] ---
def video_streamer(index, src):
    zone_id = f"zone_{index+1:02d}"
    cap = cv2.VideoCapture(src)
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        latest_frames[zone_id] = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE))
        time.sleep(0.1)

def start_cctv_streams():
    for i, src in enumerate(VIDEO_SOURCES):
        threading.Thread(target=video_streamer, args=(i, src), daemon=True).start()

# --- [3. 분석 함수: 대소문자 및 라벨 수정] ---
def analyze_single_frame(zone_id, frame):
    real_name = ZONE_MAP.get(zone_id, zone_id)
    res_data = {"zoneId": real_name, "fireLevel": 0.0, "smokeLevel": 0.0, "knife": False, "people_cnt": 0}
    total_area = float(TARGET_SIZE * TARGET_SIZE)
    
    try:
        f_results = fire_model.predict(frame, imgsz=TARGET_SIZE, verbose=False, device='cpu', conf=0.15)
        if f_results and len(f_results) > 0:
            f_res, f_sum, s_sum = f_results[0], 0.0, 0.0
            if f_res.boxes is not None:
                for b in f_res.boxes:
                    c_name = str(fire_model.names[int(b.cls.item())]).lower()
                    box = b.xyxy[0].cpu().numpy()
                    area = float((box[2]-box[0]) * (box[3]-box[1]))
                    if 'fire' in c_name: f_sum += area
                    if 'smoke' in c_name: s_sum += area
            res_data["fireLevel"] = float(round(min(f_sum / total_area, 1.0), 4))
            res_data["smokeLevel"] = float(round(min(s_sum / total_area, 1.0), 4))

        w_results = weapon_model.predict(frame, imgsz=TARGET_SIZE, verbose=False, device='cpu', conf=0.2)
        if w_results and len(w_results) > 0:
            w_res = w_results[0]
            if w_res.boxes is not None:
                for b in w_res.boxes:
                    c_idx = int(b.cls.item())
                    if c_idx == 0: res_data["people_cnt"] += 1
                    elif c_idx == 1: res_data["knife"] = True # violence 라벨 대응
    except Exception as e:
        print(f"❌ {real_name} 분석 에러: {e}")
    return res_data

# --- [4. 백그라운드 분석 로직] ---
def _run_full_analysis_and_update_cache():
    global cached_zone_results, cached_dynamic_graph, cache_timestamp, analysis_in_progress
    try:
        with cache_lock: analysis_in_progress = True
        current_frames = latest_frames.copy()
        current_zone_results = []
        dynamic_graph = {node: neighbors.copy() for node, neighbors in BASE_FACILITY_GRAPH.items()}

        for i in range(N_ZONES):
            zone_id = f"zone_{i+1:02d}"
            frame = current_frames.get(zone_id)
            res = analyze_single_frame(zone_id, frame) if frame is not None else \
                  {"zoneId": ZONE_MAP[zone_id], "fireLevel": 0.0, "smokeLevel": 0.0, "knife": False, "people_cnt": 0}
            current_zone_results.append(res)

            if res["fireLevel"] > 0.1 or res["smokeLevel"] > 0.1 or res["knife"]:
                d_node = res["zoneId"]
                if d_node in dynamic_graph:
                    for neighbor in list(dynamic_graph[d_node].keys()): dynamic_graph[d_node][neighbor] = 999.0
                    for node in dynamic_graph:
                        if d_node in dynamic_graph[node]: dynamic_graph[node][d_node] = 999.0

        with cache_lock:
            cached_zone_results = current_zone_results
            cached_dynamic_graph = dynamic_graph
            cache_timestamp = time.time()
    finally:
        with cache_lock: analysis_in_progress = False

# --- [5. 경로 탐색 API (Response Model 유지 + 사용자 구제 추가)] ---
@cctv_router.post("/get-escape-path")
async def get_escape_path(request: LocationRequest):
    start_time = time.time()
    try:
        now = time.time()
        with cache_lock:
            cache_age = now - cache_timestamp if cache_timestamp else None
            cache_valid = (cached_zone_results is not None) and (cache_age is not None and cache_age <= cache_ttl)
            currently_running = analysis_in_progress

        if not cache_valid and not currently_running:
            threading.Thread(target=_run_full_analysis_and_update_cache, daemon=True).start()
            with cache_lock: currently_running = True

        with cache_lock:
            local_zone_results = cached_zone_results
            local_dynamic_graph = cached_dynamic_graph

        if local_zone_results is None:
            return {
                "analysis": [],
                "escape_path": {"start": request.current_location, "destination": "처리중", "path": [], "total_distance": 999.0, "is_safe": False},
                "processing_time": round(time.time() - start_time, 2), "status": "processing"
            }

        # --- 사용자 위치 구제 로직 ---
        start_node = request.current_location
        # 원본 캐시 그래프를 복사하여 사용 (start_node의 간선만 임시 복구)
        dynamic_graph = {node: neighbors.copy() for node, neighbors in local_dynamic_graph.items()}
        
        if start_node in dynamic_graph:
            for neighbor in dynamic_graph[start_node]:
                if dynamic_graph[start_node][neighbor] >= 999.0:
                    # 원본 거리 데이터에서 가져와서 길을 열어줌
                    dynamic_graph[start_node][neighbor] = BASE_FACILITY_GRAPH[start_node].get(neighbor, 1.0)

        # 다익스트라
        stairs = ['계단1', '계단2', '계단3', '계단4']
        distances = {node: float('inf') for node in dynamic_graph}
        predecessors = {node: None for node in dynamic_graph}
        distances[start_node] = 0
        pq = [(0, start_node)]

        while pq:
            d, curr = heapq.heappop(pq)
            if d > distances.get(curr, float('inf')): continue
            for neighbor, weight in dynamic_graph.get(curr, {}).items():
                if d + weight < distances[neighbor]:
                    distances[neighbor] = d + weight
                    predecessors[neighbor] = curr
                    heapq.heappush(pq, (distances[neighbor], neighbor))

        reachable_stairs = [s for s in stairs if distances.get(s, float('inf')) < 900.0]

        if reachable_stairs:
            nearest_stair = min(reachable_stairs, key=lambda s: distances[s])
            path, curr = [], nearest_stair
            while curr is not None:
                path.append(curr); curr = predecessors[curr]
            final_path, dist = path[::-1], float(round(distances[nearest_stair], 2))
        else:
            nearest_stair, final_path, dist = "탈출 불가", [start_node], 999.0

        return {
            "analysis": local_zone_results,
            "escape_path": {
                "start": start_node,
                "destination": nearest_stair,
                "path": final_path,
                "total_distance": dist,
                "is_safe": bool(dist < 900.0)
            },
            "processing_time": round(time.time() - start_time, 2),
            "status": "ok",
            "cache_age": round(now - cache_timestamp, 2)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))