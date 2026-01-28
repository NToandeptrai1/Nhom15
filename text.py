import cv2 as cv
import numpy as np
import math
import random
import time

def heart_points(n=300):
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x = 16 * (math.sin(t) ** 3)
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((x, -y))
    return np.array(pts, dtype=np.float32)

def make_particles(base_pts, count=1200):
    particles = []
    N = len(base_pts)
    for i in range(count):
        idx = random.randrange(N)
        base = base_pts[idx].copy()
        base += np.array([random.uniform(-0.6,0.6), random.uniform(-0.6,0.6)], dtype=np.float32)
        particles.append({
            "base": base,
            "offset": random.uniform(0, 2*math.pi),
            "drift": random.uniform(-0.25, 0.25),
            "size": random.uniform(0.8, 3.2),
            "alpha": random.uniform(0.5, 1.0),
            "hue_shift": random.uniform(-10, 10)
        })
    return particles

def draw_particles(img, particles, center, scale, t):
    h, w = img.shape[:2]
    overlay = np.zeros_like(img, dtype=np.uint8)
    for p in particles:
        s = scale * (1 + 0.08 * math.sin(t * 6 + p["offset"]))
        px = int(center[0] + p["base"][0] * s + p["drift"] * math.sin(t*2 + p["offset"]) * 40)
        py = int(center[1] + p["base"][1] * s + math.sin(t*3 + p["offset"]) * 8)
        r = max(1, int(p["size"] * (1 + 0.5 * math.sin(t*5 + p["offset"]))))
        b = int(180 + 40 * p["alpha"] + p["hue_shift"])
        g = int(30 + 20 * p["alpha"])
        r_col = int(200 + 30 * p["alpha"])
        color = (b, g, r_col)
        if 0 <= px < w and 0 <= py < h:
            cv.circle(overlay, (px, py), r, color, -1, lineType=cv.LINE_AA)
    blurred = cv.GaussianBlur(overlay, (0,0), sigmaX=8, sigmaY=8)
    out = cv.addWeighted(img, 1.0, blurred, 0.95, 0)
    return out

def make_background(W, H):
    bg = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        t = y / H
        cb = int(6 + 18 * (1 - t))
        cg = int(6 + 18 * (1 - t))
        cr = int(10 + 30 * t)
        bg[y, :] = (cb, cg, cr)
    # subtle vignette
    Y, X = np.ogrid[:H, :W]
    cx, cy = W/2, H/2 - 60
    radius = max(W, H) * 0.8
    mask = ((X - cx)**2 + (Y - cy)**2) / (radius**2)
    mask = np.clip(mask, 0, 1)
    for c in range(3):
        bg[:, :, c] = (bg[:, :, c].astype(np.float32) * (1 - 0.5*mask)).astype(np.uint8)
    return bg

def main():
    W, H = 720, 1280
    center = (W // 2, H // 2 - 80)
    base = heart_points(300)
    base *= 7.5
    particles = make_particles(base, count=1400)
    bg = make_background(W, H)

    t0 = time.time()
    cv.namedWindow("Pulsing Heart", cv.WINDOW_NORMAL)
    cv.resizeWindow("Pulsing Heart", 360, 640)
    while True:
        now = time.time()
        t = now - t0
        heartbeat = 7.8 + 0.9 * (1 + math.sin(t * 2.2))
        frame = bg.copy()
        frame = draw_particles(frame, particles, center, heartbeat, t)
        core_r = int(18 + 8 * (1 + math.sin(t*2.2))/2)
        cv.circle(frame, center, core_r, (220, 40, 220), -1, lineType=cv.LINE_AA)
        cv.imshow("Pulsing Heart", frame)
        key = cv.waitKey(30) & 0xFF
        if key in (27, ord('q')):
            break
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()