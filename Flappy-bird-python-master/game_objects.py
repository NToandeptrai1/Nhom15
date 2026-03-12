# =========================
# GAME OBJECTS/SPRITES
# =========================

import pygame
from config import *
from utils import asset_path


class Bird(pygame.sprite.Sprite):
    """Chim chính của game"""
    
    def __init__(self):
        super().__init__()

        # Tải 3 hình ảnh cánh của chim
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
        """Cập nhật vị trí và hình ảnh chim"""
        # Xoay ảnh cánh
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.mask = pygame.mask.from_surface(self.image)

        # Áp dụng trọng lực
        self.speed += GRAVITY
        self.rect.y += self.speed

    def bump(self):
        """Nhảy - giảm vận tốc"""
        self.speed = -7

    def begin(self):
        """Xoay ảnh cánh lúc chờ game bắt đầu"""
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.mask = pygame.mask.from_surface(self.image)


class Pipe(pygame.sprite.Sprite):
    """Ống nước (chướng ngại vật)"""
    
    def __init__(self, inverted, xpos, ysize):
        super().__init__()

        # Tải hình ảnh ống
        self.image = pygame.image.load(
            asset_path("assets", "sprites", "pipe-green.png")
        ).convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDHT, PIPE_HEIGHT))

        self.rect = self.image.get_rect()
        self.rect.x = xpos

        # Nếu inverted = True, lật ống lên trên
        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.y = -(self.rect.height - ysize)
        else:
            self.rect.y = SCREEN_HEIGHT - ysize

        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        """Di chuyển ống sang trái"""
        self.rect.x -= GAME_SPEED


class Ground(pygame.sprite.Sprite):
    """Mặt đất ở dưới cùng"""
    
    def __init__(self, xpos):
        super().__init__()

        # Tải hình ảnh mặt đất
        self.image = pygame.image.load(
            asset_path("assets", "sprites", "base.png")
        ).convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDHT, GROUND_HEIGHT))

        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.x = xpos
        self.rect.y = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        """Di chuyển mặt đất sang trái"""
        self.rect.x -= GAME_SPEED
