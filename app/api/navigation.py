from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import heapq

router = APIRouter()

# 1. 고정된 그래프 데이터
facility_graph = {
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

# Request 모델
class LocationRequest(BaseModel):
    current_location: str

@router.post("/get-escape-path")
async def get_escape_path(request: LocationRequest):
    start_node = request.current_location
    
    if start_node not in facility_graph:
        raise HTTPException(status_code=404, detail="Location not found in facility map")

    stairs = ['계단1', '계단2', '계단3', '계단4']
    
    # 2. 다익스트라 알고리즘
    distances = {node: float('inf') for node in facility_graph}
    predecessors = {node: None for node in facility_graph}
    distances[start_node] = 0
    queue = [(0, start_node)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if distances[current_node] < current_distance:
            continue
        for neighbor, weight in facility_graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    # 3. 최단 거리 계단 선택
    nearest_stair = None
    min_dist = float('inf')
    for stair in stairs:
        if distances[stair] < min_dist:
            min_dist = distances[stair]
            nearest_stair = stair

    if not nearest_stair or distances[nearest_stair] == float('inf'):
        return {"path": [], "message": "No escape path found"}

    # 4. 경로 역추적
    path = []
    curr = nearest_stair
    while curr is not None:
        path.append(curr)
        curr = predecessors[curr]
    
    return {
        "start": start_node,
        "destination": nearest_stair,
        "path": path[::-1],
        "total_distance": round(distances[nearest_stair], 2)
    }