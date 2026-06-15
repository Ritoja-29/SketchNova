import asyncio
import base64
import json
import time
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# ── CNN prediction ─────────────────────────────────────────────────────────
try:
    from predict_handwriting import predict_from_canvas
    _PREDICT_AVAILABLE= True
    print("[OK] predict_handwriting loaded.")
except ImportError:
    _PREDICT_AVAILABLE= False
    print("[WARN] predict_handwriting not found.")

# ── FastAPI app setup ──────────────────────────────────────────────────────
app = FastAPI(title="SketchNova Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MediaPipe setup ────────────────────────────────────────────────────────
mp_hands= mp.solutions.hands
hands= mp_hands.Hands(max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)
clients: dict= {}
PREDICT_HOLD_FRAMES= 12


# ── Finger / gesture helpers ───────────────────────────────────────────────
def get_fingers(hand) -> list:
    fingers= []
    fingers.append(1 if hand.landmark[4].x < hand.landmark[3].x else 0)
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(1 if hand.landmark[tip].y < hand.landmark[pip].y else 0)
    return fingers


def get_gesture(fingers: list) -> str:
    thumb, index, middle, ring, pinky= fingers
    if index and middle and ring and pinky:
        return "CLEAR"
    if index and middle and ring and not pinky:
        return "ERASE"
    if index and middle and not ring and not pinky:
        return "PREDICT"
    if index and not middle and not ring and not pinky:
        return "DRAW"
    return "HOVER"


def make_state() -> dict:
    return {
        "drawing_active":False,
        "predict_hold": 0,
        "predict_triggered":False,
        "saved_canvas": None,
        "fps_ts":time.time(),
        "fps_count":0,
        "fps":0,
    }


def process_video_frame(payload: dict, state: dict) -> dict:
    selected_mode= payload.get("selected_mode", "WRITE")
    state["fps_count"]+= 1
    now= time.time()
    elapsed= now - state["fps_ts"]
    if elapsed>= 1.0:
        x=3
        state["fps"]= x* round(state["fps_count"] / elapsed)
        state["fps_count"]= 0
        state["fps_ts"]= now

    img_b64= payload.get("image", "")
    raw= base64.b64decode(img_b64.split(",")[1])
    np_arr= np.frombuffer(raw, np.uint8)
    frame= cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    h, w, _= frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result= hands.process(rgb)

    gesture= "HOVER"
    hand_detected= False
    landmarks_out= []
    prediction= ""

    if result.multi_hand_landmarks:
        hand_detected= True
        hand= result.multi_hand_landmarks[0]
        fingers= get_fingers(hand)
        gesture= get_gesture(fingers)
        landmarks_out= [{"x": 1.0 - lm.x, "y": lm.y}
                         for lm in hand.landmark]

        if gesture== "DRAW":
            state["drawing_active"]= True
            state["predict_hold"]= 0
            state["predict_triggered"]= False

        elif gesture== "PREDICT" and selected_mode== "WRITE":
            if not state["predict_triggered"]:
                state["predict_hold"]+= 1
                if state["predict_hold"]>= PREDICT_HOLD_FRAMES:
                    letter = None
                    if state["saved_canvas"] and _PREDICT_AVAILABLE:
                        try:
                            canvas_raw= base64.b64decode(
                                state["saved_canvas"].split(",")[1])
                            canvas_arr= np.frombuffer(canvas_raw, np.uint8)
                            canvas_img= cv2.imdecode(
                                canvas_arr, cv2.IMREAD_UNCHANGED)
                            if canvas_img is not None:
                                letter= predict_from_canvas(canvas_img)
                                print(f"[PREDICT] Canvas: {canvas_img.shape}")
                            else:
                                print("[PREDICT] canvas_img decode failed")
                        except Exception as e:
                            print(f"[PREDICT ERR] {e}")
                    else:
                        if not state["saved_canvas"]:
                            print("[PREDICT] No canvas image saved yet!")

                    state["predict_triggered"]= True
                    state["predict_hold"]= 0
                    prediction = letter if letter else "PREDICT_FAIL"
            else:
                state["predict_hold"]= 0

        elif gesture == "ERASE":
            state["predict_hold"]= 0
            state["predict_triggered"]= False

        elif gesture == "CLEAR":
            state["drawing_active"]= False
            state["predict_hold"]= 0
            state["predict_triggered"]= False
            state["saved_canvas"]= None
            prediction = "CLEAR_UI"

        else:
            state["predict_hold"]= 0

    else:
        state["predict_hold"]= 0

    return {
        "hand_detected":hand_detected,
        "mode":gesture,
        "fps":state["fps"],
        "landmarks":landmarks_out,
        "prediction":prediction,
        "predict_hold":state["predict_hold"],
        "predict_frames":PREDICT_HOLD_FRAMES,
    }


# ── FastAPI REST endpoint (health check) ──────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "clients": len(clients)}


# ── FastAPI WebSocket endpoint ─────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = make_state()
    clients[id(websocket)] = state
    print(f"[+] Client connected  (total: {len(clients)})")
    try:
        while True:
            try:
                message= await websocket.receive_text()
                payload= json.loads(message)
                msg_type= payload.get("type", "frame")
                if msg_type== "canvas":
                    canvas_b64= payload.get("canvas_image", "")
                    if canvas_b64:
                        state["saved_canvas"]= canvas_b64
                        print(f"[CANVAS] Received ({len(canvas_b64)//1024}KB)")
                    await websocket.send_text(
                        json.dumps({"type": "canvas_ack"}))
                    continue
                response= process_video_frame(payload, state)
                if response:
                    await websocket.send_text(json.dumps(response))
                else:
                    await websocket.send_text(json.dumps({
                        "hand_detected": False, "mode": "HOVER",
                        "fps": 0, "landmarks": [], "prediction": "",
                        "predict_hold": 0, "predict_frames": 12,
                    }))
            except WebSocketDisconnect:
                break
            except Exception as exc:
                print(f"[ERR] {exc}")
                try:
                    await websocket.send_text(json.dumps({
                        "hand_detected": False, "mode": "HOVER",
                        "fps": 0, "landmarks": [], "prediction": "",
                        "predict_hold": 0, "predict_frames": 12,
                    }))
                except:
                    break
    finally:
        clients.pop(id(websocket), None)
        print(f"[-] Client disconnected (total: {len(clients)})")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__== "__main__":
    print("[*] FastAPI + WebSocket → ws://0.0.0.0:8000/ws")
    print("[*] Health check       → http://0.0.0.0:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)






