# =========================
# FLAPPY BIRD GAME - MAIN
# =========================

import time
import random
import pygame
from pygame.locals import *
from config import *
from utils import asset_path, is_off_screen, play_sound
from game_objects import Bird, Pipe, Ground
from gesture_control import GestureController, WING_SOUND, HIT_SOUND


def get_random_pipes(xpos):
    """Tạo ngẫu nhiên cặp ống trên-dưới"""
    size = random.randint(150, 320)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted


def init_pygame():
    """Khởi tạo Pygame"""
    pygame.init()
    pygame.mixer.init()
    
    screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - Blink + Finger + Head Control")
    
    return screen


def load_assets():
    """Tải tất cả hình ảnh nền"""
    background = pygame.image.load(
        asset_path("assets", "sprites", "background-day.png")
    ).convert()
    background = pygame.transform.scale(background, (SCREEN_WIDHT, SCREEN_HEIGHT))
    
    begin_image = pygame.image.load(
        asset_path("assets", "sprites", "message.png")
    ).convert_alpha()
    
    return background, begin_image


def begin_screen(screen, background, begin_image, font, gesture_controller):
    """
    Màn hình chơi trước khi bắt đầu
    Chờ người dùng nhấn SPACE, UP hoặc sử dụng cử chỉ để bắt đầu
    """
    bird = Bird()
    bird_group = pygame.sprite.Group()
    bird_group.add(bird)
    
    ground_group = pygame.sprite.Group()
    for i in range(2):
        ground = Ground(GROUND_WIDHT * i)
        ground_group.add(ground)
    
    clock = pygame.time.Clock()
    begun = False
    
    while not begun:
        clock.tick(FPS)
        command = gesture_controller.detect_command()
        
        # Nếu nhấn ESC
        if command == -1:
            gesture_controller.cleanup()
            pygame.quit()
            raise SystemExit

        for event in pygame.event.get():
            if event.type == QUIT:
                gesture_controller.cleanup()
                pygame.quit()
                raise SystemExit
            if event.type == KEYDOWN:
                if event.key == K_SPACE or event.key == K_UP:
                    bird.bump()
                    play_sound(WING_SOUND)
                    begun = True

        # Nếu phát hiện cử chỉ
        if command == 1:
            bird.bump()
            play_sound(WING_SOUND)
            begun = True

        # Vẽ
        screen.blit(background, (0, 0))
        screen.blit(begin_image, (120, 150))
        
        info1 = font.render("Nhay mat / gio ngon tro / nguoc dau", True, (255, 255, 255))
        info2 = font.render("Nhan SPACE / UP de choi bang ban phim", True, (255, 255, 255))
        screen.blit(info1, (8, 70))
        screen.blit(info2, (8, 105))
        
        # Cập nhật các sprite
        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            new_ground = Ground(GROUND_WIDHT - 20)
            ground_group.add(new_ground)
        
        bird.begin()
        ground_group.update()
        
        bird_group.draw(screen)
        ground_group.draw(screen)
        pygame.display.update()


def main_game_loop(screen, background, font, gesture_controller):
    """
    Vòng lặp game chính
    Quản lý va chạm, điểm số, cập nhật vị trí
    """
    # Khởi tạo các sprite
    bird = Bird()
    bird_group = pygame.sprite.Group()
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
    running = True
    score = 0
    passed_pipes = set()
    
    # Vòng lặp game
    while running:
        clock.tick(FPS)
        command = gesture_controller.detect_command()
        
        # Nếu nhấn ESC
        if command == -1:
            running = False

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_SPACE or event.key == K_UP:
                    bird.bump()
                    play_sound(WING_SOUND)

        # Nếu phát hiện cử chỉ
        if command == 1:
            bird.bump()
            play_sound(WING_SOUND)

        # Vẽ nền
        screen.blit(background, (0, 0))

        # Cập nhật mặt đất (tạo vòng lặp)
        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            new_ground = Ground(GROUND_WIDHT - 20)
            ground_group.add(new_ground)

        # Cập nhật ống (tạo vòng lặp)
        if len(pipe_group.sprites()) >= 2 and is_off_screen(pipe_group.sprites()[0]):
            pipe_group.remove(pipe_group.sprites()[0])
            pipe_group.remove(pipe_group.sprites()[0])
            pipes = get_random_pipes(SCREEN_WIDHT * 2)
            pipe_group.add(pipes[0])
            pipe_group.add(pipes[1])

        # Cập nhật vị trí tất cả sprite
        bird_group.update()
        ground_group.update()
        pipe_group.update()

        # Tính điểm (khi chim vượt qua ống)
        for pipe in pipe_group:
            if pipe.rect.y > 0:
                pipe_id = id(pipe)
                if pipe_id not in passed_pipes and pipe.rect.right < bird.rect.left:
                    passed_pipes.add(pipe_id)
                    score += 1

        # Vẽ tất cả sprite
        bird_group.draw(screen)
        pipe_group.draw(screen)
        ground_group.draw(screen)

        # Vẽ điểm số
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        pygame.display.update()

        # Kiểm tra va chạm
        if (
            pygame.sprite.groupcollide(
                bird_group, ground_group, False, False, pygame.sprite.collide_mask
            )
            or pygame.sprite.groupcollide(
                bird_group, pipe_group, False, False, pygame.sprite.collide_mask
            )
            or bird.rect.top < 0
        ):
            play_sound(HIT_SOUND)
            time.sleep(1)
            running = False
    
    return score, bird_group, pipe_group, ground_group


def game_over_screen(screen, background, font, score, bird_group, pipe_group, ground_group):
    """Màn hình kết thúc game"""
    screen.blit(background, (0, 0))
    bird_group.draw(screen)
    pipe_group.draw(screen)
    ground_group.draw(screen)

    game_over_text = font.render(f"Game Over - Score: {score}", True, (255, 255, 255))
    screen.blit(game_over_text, (35, SCREEN_HEIGHT // 2))

    pygame.display.update()
    time.sleep(2)


# =========================
# MAIN ENTRY POINT
# =========================
if __name__ == "__main__":
    try:
        # Khởi tạo
        screen = init_pygame()
        background, begin_image = load_assets()
        font = pygame.font.SysFont("Arial", 26, bold=True)
        gesture_controller = GestureController()
        
        # Màn hình chờ
        begin_screen(screen, background, begin_image, font, gesture_controller)
        
        # Vòng lặp game chính
        score, bird_group, pipe_group, ground_group = main_game_loop(
            screen, background, font, gesture_controller
        )
        
        # Màn hình game over
        game_over_screen(screen, background, font, score, bird_group, pipe_group, ground_group)
        
    finally:
        gesture_controller.cleanup()
        pygame.quit()
