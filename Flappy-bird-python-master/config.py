# =========================
# GAME CONSTANTS & SETTINGS
# =========================

SCREEN_WIDHT = 400
SCREEN_HEIGHT = 600

SPEED = 6
GRAVITY = 0.9
GAME_SPEED = 4

GROUND_WIDHT = 2 * SCREEN_WIDHT
GROUND_HEIGHT = 100

PIPE_WIDHT = 80
PIPE_HEIGHT = 500
PIPE_GAP = 190

FPS = 12

# =========================
# GESTURE DETECTION SETTINGS
# =========================

# Blink detection
BLINK_COOLDOWN = 0.35
EAR_THRESHOLD = 0.19
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Hand detection
HAND_COOLDOWN = 0.35

# Head detection
HEAD_COOLDOWN = 0.35
HEAD_UP_THRESHOLD = 18  # tăng nếu bị nhảy liên tục

# Face landmarks indices
NOSE_TIP = 1
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
