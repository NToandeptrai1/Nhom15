import os
import time
import math
import random
import pygame
import cv2
import mediapipe as mp
from pygame.locals import *

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def asset_path(*parts):
    return os.path.join(BASE_DIR, *parts)


# =========================
# GAME VARIABLES
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

wing = asset_path("assets", "audio", "wing.wav")
hit = asset_path("assets", "audio", "hit.wav")

pygame.init()
pygame.mixer.init()

# =========================
# MEDIAPIPE FACE + HAND
# =========================
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def open_camera():
    backends = [
        cv2.CAP_DSHOW,
        cv2.CAP_MSMF,
        cv2.CAP_ANY,
    ]

    for backend in backends:
        cam = cv2.VideoCapture(0, backend)
        time.sleep(1)
        if cam.isOpened():
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cam

    raise RuntimeError("Khong mo duoc camera!")


camera = open_camera()

# =========================
# CONTROL VARIABLES
# =========================
last_blink_time = 0
BLINK_COOLDOWN = 0.35
blink_state_closed = False

last_hand_time = 0
HAND_COOLDOWN = 0.35

last_head_time = 0
HEAD_COOLDOWN = 0.35

# Face landmarks
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.19

NOSE_TIP = 1
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
HEAD_UP_THRESHOLD = 18  # tăng nếu bị nhảy liên tục


# =========================
# GAME CLASSES
# =========================
class Bird(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()

        self.images = [
            pygame.image.load(
                asset_path("assets", "sprites", "bluebird-upflap.png")
            ).convert_alpha(),
            pygame.image.load(
                asset_path("assets", "sprites", "bluebird-midflap.png")
            ).convert_alpha(),
            pygame.image.load(
                asset_path("assets", "sprites", "bluebird-downflap.png")
            ).convert_alpha(),
        ]

        self.speed = SPEED
        self.current_image = 0
        self.image = self.images[0]
        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDHT // 6
        self.rect.y = SCREEN_HEIGHT // 2

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.mask = pygame.mask.from_surface(self.image)

        self.speed += GRAVITY
        self.rect.y += self.speed

    def bump(self):
        self.speed = -7

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.mask = pygame.mask.from_surface(self.image)


class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        super().__init__()

        self.image = pygame.image.load(
            asset_path("assets", "sprites", "pipe-green.png")
        ).convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDHT, PIPE_HEIGHT))

        self.rect = self.image.get_rect()
        self.rect.x = xpos

        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.y = -(self.rect.height - ysize)
        else:
            self.rect.y = SCREEN_HEIGHT - ysize

        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect.x -= GAME_SPEED


class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        super().__init__()

        self.image = pygame.image.load(
            asset_path("assets", "sprites", "base.png")
        ).convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDHT, GROUND_HEIGHT))

        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.x = xpos
        self.rect.y = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect.x -= GAME_SPEED


# =========================
# HELPER FUNCTIONS
# =========================
def is_off_screen(sprite):
    return sprite.rect.x < -sprite.rect.width


def get_random_pipes(xpos):
    size = random.randint(150, 320)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted


def play_sound(path):
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    except pygame.error:
        pass


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calc_ear(points):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 1.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def is_index_finger_up(hand_landmarks):
    lm = hand_landmarks.landmark

    index_up = lm[8].y < lm[6].y
    middle_down = lm[12].y > lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y

    return index_up and middle_down and ring_down and pinky_down


def is_head_up(points):
    nose = points[NOSE_TIP]
    left_eye = points[LEFT_EYE_CORNER]
    right_eye = points[RIGHT_EYE_CORNER]

    eye_center_y = (left_eye[1] + right_eye[1]) / 2

    return nose[1] < eye_center_y - HEAD_UP_THRESHOLD


def detect_command():
    """
    Return:
    1  -> detected command (blink / finger / head up)
    0  -> no command
    -1 -> ESC pressed
    """
    global last_blink_time, blink_state_closed
    global last_hand_time, last_head_time

    ret, frame = camera.read()
    if not ret:
        return 0

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    now = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_result = face_mesh.process(rgb)
    hand_result = hands.process(rgb)

    command = 0
    label_text = "No face / no hand"

    # ===== FACE DETECTION =====
    if face_result.multi_face_landmarks:
        face_landmarks = face_result.multi_face_landmarks[0]
        points = []

        for lm in face_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))

        # Vẽ điểm mắt trái
        eye_points = [points[i] for i in LEFT_EYE_IDX]
        for p in eye_points:
            cv2.circle(frame, p, 2, (0, 255, 255), -1)

        # Vẽ mũi + 2 mắt ngoài để debug head up
        cv2.circle(frame, points[NOSE_TIP], 3, (0, 0, 255), -1)
        cv2.circle(frame, points[LEFT_EYE_CORNER], 3, (255, 255, 0), -1)
        cv2.circle(frame, points[RIGHT_EYE_CORNER], 3, (255, 255, 0), -1)

        # ----- HEAD UP -----
        if is_head_up(points):
            label_text = "Head Up -> JUMP"
            if (now - last_head_time) > HEAD_COOLDOWN:
                command = 1
                last_head_time = now

        # ----- BLINK -----
        ear = calc_ear(eye_points)

        if ear < EAR_THRESHOLD:
            blink_state_closed = True
            if command == 0:
                label_text = f"Eye Closed | EAR={ear:.2f}"
        else:
            if blink_state_closed and (now - last_blink_time) > BLINK_COOLDOWN:
                label_text = "Blink -> JUMP"
                command = 1
                last_blink_time = now
                blink_state_closed = False
            else:
                blink_state_closed = False
                if command == 0:
                    label_text = f"Eye Open | EAR={ear:.2f}"

    # ===== HAND DETECTION =====
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

            if is_index_finger_up(hand_landmarks):
                label_text = "Index Finger Up -> JUMP"
                if (now - last_hand_time) > HAND_COOLDOWN:
                    command = 1
                    last_hand_time = now

    cv2.putText(
        frame,
        label_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        "Blink / Index finger / Head up to jump",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 0, 255),
        2,
    )
    cv2.putText(
        frame,
        "ESC to quit",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
    )

    cv2.imshow("Blink Control", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        return -1

    return command


def cleanup():
    camera.release()
    face_mesh.close()
    hands.close()
    cv2.destroyAllWindows()
    pygame.quit()


# =========================
# PYGAME SETUP
# =========================
screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird - Blink + Finger + Head Control")

BACKGROUND = pygame.image.load(
    asset_path("assets", "sprites", "background-day.png")
).convert()
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))
BEGIN_IMAGE = pygame.image.load(
    asset_path("assets", "sprites", "message.png")
).convert_alpha()

font = pygame.font.SysFont("Arial", 26, bold=True)

bird_group = pygame.sprite.Group()
bird = Bird()
bird_group.add(bird)

ground_group = pygame.sprite.Group()
for i in range(2):
    ground = Ground(GROUND_WIDHT * i)
    ground_group.add(ground)

pipe_group = pygame.sprite.Group()
for i in range(2):
    pipes = get_random_pipes(SCREEN_WIDHT * i + 800)
    pipe_group.add(pipes[0])
    pipe_group.add(pipes[1])

clock = pygame.time.Clock()
begin = True
running = True
score = 0
passed_pipes = set()

# =========================
# BEGIN SCREEN
# =========================
while begin:
    clock.tick(FPS)
    command = detect_command()

    if command == -1:
        cleanup()
        raise SystemExit

    for event in pygame.event.get():
        if event.type == QUIT:
            cleanup()
            raise SystemExit
        if event.type == KEYDOWN:
            if event.key == K_SPACE or event.key == K_UP:
                bird.bump()
                play_sound(wing)
                begin = False

    if command == 1:
        bird.bump()
        play_sound(wing)
        begin = False

    screen.blit(BACKGROUND, (0, 0))
    screen.blit(BEGIN_IMAGE, (120, 150))

    info1 = font.render("Nhay mat / gio ngon tro / nguoc dau", True, (255, 255, 255))
    info2 = font.render("Nhan SPACE / UP de choi bang ban phim", True, (255, 255, 255))
    screen.blit(info1, (8, 70))
    screen.blit(info2, (8, 105))

    if is_off_screen(ground_group.sprites()[0]):
        ground_group.remove(ground_group.sprites()[0])
        new_ground = Ground(GROUND_WIDHT - 20)
        ground_group.add(new_ground)

    bird.begin()
    ground_group.update()

    bird_group.draw(screen)
    ground_group.draw(screen)
    pygame.display.update()

# =========================
# MAIN GAME LOOP
# =========================
while running:
    clock.tick(FPS)
    command = detect_command()

    if command == -1:
        running = False

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        if event.type == KEYDOWN:
            if event.key == K_SPACE or event.key == K_UP:
                bird.bump()
                play_sound(wing)

    if command == 1:
        bird.bump()
        play_sound(wing)

    screen.blit(BACKGROUND, (0, 0))

    if is_off_screen(ground_group.sprites()[0]):
        ground_group.remove(ground_group.sprites()[0])
        new_ground = Ground(GROUND_WIDHT - 20)
        ground_group.add(new_ground)

    if len(pipe_group.sprites()) >= 2 and is_off_screen(pipe_group.sprites()[0]):
        pipe_group.remove(pipe_group.sprites()[0])
        pipe_group.remove(pipe_group.sprites()[0])
        pipes = get_random_pipes(SCREEN_WIDHT * 2)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])

    bird_group.update()
    ground_group.update()
    pipe_group.update()

    for pipe in pipe_group:
        if pipe.rect.y > 0:
            pipe_id = id(pipe)
            if pipe_id not in passed_pipes and pipe.rect.right < bird.rect.left:
                passed_pipes.add(pipe_id)
                score += 1

    bird_group.draw(screen)
    pipe_group.draw(screen)
    ground_group.draw(screen)

    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    pygame.display.update()

    if (
        pygame.sprite.groupcollide(
            bird_group, ground_group, False, False, pygame.sprite.collide_mask
        )
        or pygame.sprite.groupcollide(
            bird_group, pipe_group, False, False, pygame.sprite.collide_mask
        )
        or bird.rect.top < 0
    ):
        play_sound(hit)
        time.sleep(1)
        running = False

# =========================
# GAME OVER
# =========================
screen.blit(BACKGROUND, (0, 0))
bird_group.draw(screen)
pipe_group.draw(screen)
ground_group.draw(screen)

game_over_text = font.render(f"Game Over - Score: {score}", True, (255, 255, 255))
screen.blit(game_over_text, (35, SCREEN_HEIGHT // 2))

pygame.display.update()
time.sleep(2)

cleanup()