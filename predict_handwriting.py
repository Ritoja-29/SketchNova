import torch
import torch.nn as nn
import numpy as np
import cv2
DEVICE = "cpu"
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU()
        )
        self.fc= nn.Sequential(
            nn.Linear(128*3*3,128), nn.ReLU(),
            nn.Linear(128, 26)
        )

    def forward(self, x):
        x= self.conv(x)
        x= x.view(x.size(0), -1)
        return self.fc(x)


model= CNN().to(DEVICE)
try:
    model.load_state_dict(
        torch.load("model/handwriting_model.pth", map_location=DEVICE)
    )
    model.eval()
    print("[OK] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Model Load Error: {e}")


def predict_from_canvas(canvas_img):
    if canvas_img is None:
        return None
    # ── Step 1: Grayscale ──────────────────────────
    if len(canvas_img.shape)== 3:
        ch= canvas_img.shape[2]
        if ch== 4:
            alpha= canvas_img[:, :, 3:4].astype(np.float32) / 255.0
            rgb= canvas_img[:, :, :3].astype(np.float32)
            white= np.ones_like(rgb)*255.0
            blended= (alpha*rgb+(1 - alpha)*white).astype(np.uint8)
            gray= cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
        else:
            gray= cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
    else:
        gray= canvas_img.copy()

    # ── Step 2: Invert ─────────────────────────────
    inverted= cv2.bitwise_not(gray)

    # ── Step 3: Threshold ──────────────────────────
    _, thresh= cv2.threshold(inverted, 30, 255, cv2.THRESH_BINARY)

    # ── Step 4: Bounding box ───────────────────────
    coords= cv2.findNonZero(thresh)
    if coords is None or len(coords)< 30:
        return None
    x, y, w, h= cv2.boundingRect(coords)
    H, W= thresh.shape
    if w>W*0.92 or h>H*0.92:
        return None
    roi= thresh[y:y+h, x:x+w]

    # ── Step 5: Erosion ───────────────
    target= 28
    scale= max(w, h) / target

    erode_px= max(1, int(scale*0.3))  
    kernel= np.ones((erode_px, erode_px), np.uint8)
    thinned= cv2.erode(roi, kernel, iterations=1)
    if cv2.countNonZero(thinned)< 20:
        thinned= roi

    # ── Step 6: Centering ─────────────────────────
    coords2= cv2.findNonZero(thinned)
    if coords2 is None:
        return None
    x2, y2, w2, h2= cv2.boundingRect(coords2)
    roi2= thinned[y2:y2 + h2, x2:x2 + w2]
    pad= max(int(max(w2, h2)*0.20), 4)
    side= max(w2, h2)+2*pad
    square= np.zeros((side, side), dtype=np.uint8)
    yo= (side-h2)//2
    xo= (side-w2)//2
    square[yo:yo+h2, xo:xo+w2]= roi2

    # ── Step 7: Resize ────────────────────────────
    img= cv2.resize(square,(28, 28),interpolation=cv2.INTER_AREA)

    # ── Step 8: Normalize ─────────────────────────
    img= img.astype(np.float32)/255.0

    # ── Step 9: Prediction ────────────────────────
    tensor= torch.from_numpy(img).float().view(1, 1, 28, 28).to(DEVICE)

    with torch.no_grad():
        output= model(tensor)
        probs= torch.softmax(output, dim=1)[0]
        top3= torch.topk(probs, 3)
        best_idx= top3.indices[0].item()
        best_conf= top3.values[0].item()
        if best_conf< 0.25:
            return None
        letter= chr(best_idx+ord('A'))
        if letter== "Y":
            h,w= img.shape
            top_density= np.sum(img[:h//2, :])
            bottom_density= np.sum(img[h//2:, :])            
            if bottom_density> top_density*1.3:
                letter="V"
        return letter







        
