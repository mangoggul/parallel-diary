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
    "209호", "210호", "211호", "복도1", "복도2", "복도3", "복도4", "복도5","복도6"
])}

# 사용되는 zone 수를 ZONE_MAP 기준으로 동적으로 설정 (17개로 자동 적용)
N_ZONES = len(ZONE_MAP)

VIDEO_SOURCES = [f"../video/video_{i}.mp4" for i in range(1, N_ZONES + 1)]
TARGET_SIZE = 224 

# 모델 로드 (전역변수)
try:
    fire_model = YOLO("../best_openvino_model/", task='detect') 
    weapon_model = YOLO("../yolo_small_weights_openvino_model/", task='detect')
    print("✅ OpenVINO 모델 로드 성공")
    USE_OPENVINO = True
except Exception as e:
    print(f"⚠️ OpenVINO 로드 실패: {e}")
    fire_model = YOLO("../best.pt", task='detect')
    weapon_model = YOLO("../yolo_small_weights.pt", task='detect')
    print("⚠️ PyTorch (.pt) 모델 로드")
    USE_OPENVINO = False

latest_frames = {f"zone_{i+1:02d}": None for i in range(N_ZONES)}

# --- 캐시 및 동기화 변수 추가 ---
cache_lock = threading.Lock()
cached_zone_results = None          # 최근 분석(프레임->zone 결과) 캐시
cached_dynamic_graph = None         # 위험 반영된 그래프 캐시
cache_timestamp = 0.0
cache_ttl = 10.0                    # 초 단위: 캐시 유효 시간
analysis_in_progress = False

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
        time.sleep(0.1)

def start_cctv_streams():
    for i, src in enumerate(VIDEO_SOURCES):
        thread = threading.Thread(target=video_streamer, args=(i, src), daemon=True)
        thread.start()

# --- 새로운 함수: 프레임 분석을 백그라운드에서 수행해 캐시 갱신 ---
def _run_full_analysis_and_update_cache():
    global cached_zone_results, cached_dynamic_graph, cache_timestamp, analysis_in_progress
    try:
        with cache_lock:
            analysis_in_progress = True

        # 스냅샷 프레임 복사
        current_frames = latest_frames.copy()
        zone_ids = [f"zone_{i+1:02d}" for i in range(N_ZONES)]

        # 순차 분석 (기존 로직 재사용)
        current_zone_results = []
        dynamic_graph = {node: neighbors.copy() for node, neighbors in BASE_FACILITY_GRAPH.items()}

        for zone_id in zone_ids:
            frame = current_frames.get(zone_id)
            if frame is None:
                res_data = {
                    "zoneId": ZONE_MAP.get(zone_id, zone_id),
                    "fireLevel": 0.0,
                    "smokeLevel": 0.0,
                    "knife": False,
                    "people_cnt": 0
                }
            else:
                # 기존의 무거운 분석 함수 호출
                res_data = analyze_single_frame(zone_id, frame)

            current_zone_results.append(res_data)

            # 위험 노드 무력화 (기존 로직)
            if res_data["fireLevel"] > 0.1 or res_data["smokeLevel"] > 0.1 or res_data["knife"]:
                danger_node = res_data["zoneId"]
                if danger_node in dynamic_graph:
                    for neighbor in list(dynamic_graph[danger_node].keys()):
                        dynamic_graph[danger_node][neighbor] = 999.0
                    for node in dynamic_graph:
                        if danger_node in dynamic_graph[node]:
                            dynamic_graph[node][danger_node] = 999.0

        # 캐시 업데이트
        with cache_lock:
            cached_zone_results = current_zone_results
            cached_dynamic_graph = dynamic_graph
            cache_timestamp = time.time()
    except Exception as e:
        print(f"❌ 백그라운드 분석 실패: {e}")
        traceback.print_exc()
    finally:
        with cache_lock:
            analysis_in_progress = False

# --- [3. 분석 함수 (개선됨)] ---
def analyze_single_frame(zone_id, frame):
    """단일 프레임 분석 - 에러 처리 강화"""
    res_data = {
        "zoneId": ZONE_MAP.get(zone_id, zone_id),
        "fireLevel": 0.0, 
        "smokeLevel": 0.0, 
        "knife": False, 
        "people_cnt": 0
    }
    
    total_area = float(TARGET_SIZE * TARGET_SIZE)
    
    try:
        # 화재 분석
        f_res = fire_model.predict(
            frame, 
            imgsz=TARGET_SIZE, 
            verbose=False, 
            device='cpu',
            conf=0.25
        )[0]
        
        if f_res.boxes is not None and len(f_res.boxes) > 0:
            f_sum, s_sum = 0.0, 0.0
            for b in f_res.boxes:
                c_idx = int(b.cls.item())
                c_name = fire_model.names[c_idx].lower()
                conf = float(b.conf.item())
                
                if conf >= 0.25:
                    box = b.xyxy[0].cpu().numpy()
                    area = float((box[2]-box[0]) * (box[3]-box[1]))
                    if 'fire' in c_name: 
                        f_sum += area
                    if 'smoke' in c_name: 
                        s_sum += area
            
            res_data["fireLevel"] = float(round(min(f_sum / total_area, 1.0), 4))
            res_data["smokeLevel"] = float(round(min(s_sum / total_area, 1.0), 4))

    except Exception as e:
        print(f"⚠️ {zone_id} 화재 분석 에러: {e}")
        traceback.print_exc()

    try:
        # 무기 분석
        w_res = weapon_model.predict(
            frame, 
            imgsz=TARGET_SIZE, 
            verbose=False, 
            device='cpu',
            conf=0.2
        )[0]
        
        if w_res.boxes is not None and len(w_res.boxes) > 0:
            for b in w_res.boxes:
                c_idx = int(b.cls.item())
                conf = float(b.conf.item())
                
                if c_idx == 0 and conf >= 0.2:  # person
                    res_data["people_cnt"] += 1
                elif c_idx == 43 and conf >= 0.2:  # knife
                    res_data["knife"] = True
                    
    except Exception as e:
        print(f"⚠️ {zone_id} 무기 분석 에러: {e}")
        traceback.print_exc()

    return res_data

# --- [4. 경로 탐색 API] ---
@cctv_router.post("/get-escape-path")
async def get_escape_path(request: LocationRequest):
    start_time = time.time()

    try:
        # 요청 시점에 캐시가 유효한지 확인
        now = time.time()
        with cache_lock:
            cache_age = now - cache_timestamp if cache_timestamp else None
            cache_valid = (cached_zone_results is not None) and (cache_age is not None and cache_age <= cache_ttl)
            currently_running = analysis_in_progress

        # 캐시가 없고 분석이 진행중이 아니면 백그라운드로 분석 시작
        if not cache_valid and not currently_running:
            thread = threading.Thread(target=_run_full_analysis_and_update_cache, daemon=True)
            thread.start()
            with cache_lock:
                currently_running = True  # 즉시 표시

        # 만약 캐시가 아직 없고 분석 중이면 빠르게 대기 상태 응답(프론트가 1초마다 요청하면 이후 캐시가 채워지면 결과를 받음)
        with cache_lock:
            local_zone_results = cached_zone_results
            local_dynamic_graph = cached_dynamic_graph

        if local_zone_results is None:
            # 초기 분석이 아직 끝나지 않았고 반환할 이전 결과도 없을 때
            return {
                "analysis": [],
                "escape_path": {
                    "start": request.current_location,
                    "destination": "처리중",
                    "path": [],
                    "total_distance": 999.0,
                    "is_safe": False
                },
                "processing_time": round(time.time() - start_time, 2),
                "status": "processing"
            }

        # 캐시는 존재 -> 빠르게 Dijkstra로 경로만 계산하여 반환 (이 부분은 가벼움)
        dynamic_graph = local_dynamic_graph.copy()
        start_node = request.current_location
        if start_node not in dynamic_graph:
            raise HTTPException(status_code=404, detail=f"위치를 찾을 수 없습니다: {start_node}")

        # 다익스트라 (요청당 가볍게 실행)
        stairs = ['계단1', '계단2', '계단3', '계단4']
        distances = {node: float('inf') for node in dynamic_graph}
        predecessors = {node: None for node in dynamic_graph}
        distances[start_node] = 0
        pq = [(0, start_node)]

        while pq:
            d, curr = heapq.heappop(pq)
            if d > distances[curr]:
                continue
            for neighbor, weight in dynamic_graph[curr].items():
                if d + weight < distances[neighbor]:
                    distances[neighbor] = d + weight
                    predecessors[neighbor] = curr
                    heapq.heappush(pq, (distances[neighbor], neighbor))

        reachable_stairs = [s for s in stairs if distances.get(s, float('inf')) < 999.0]

        if reachable_stairs:
            nearest_stair = min(reachable_stairs, key=lambda s: distances[s])
            path = []
            curr = nearest_stair
            while curr is not None:
                path.append(curr)
                curr = predecessors[curr]
            final_path = path[::-1]
            dist = float(round(distances[nearest_stair], 2))
        else:
            nearest_stair = "탈출 불가"
            final_path = [start_node]
            dist = 999.0

        elapsed_time = time.time() - start_time
        return {
            "analysis": local_zone_results,
            "escape_path": {
                "start": start_node,
                "destination": nearest_stair,
                "path": final_path,
                "total_distance": dist,
                "is_safe": bool(dist < 999.0)
            },
            "processing_time": round(elapsed_time, 2),
            "status": "ok",
            "cache_age": round(now - cache_timestamp, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 치명적 에러: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")