# =========================
# HELPER UTILITIES
# =========================

import os
import math
import pygame


def asset_path(*parts):
    """Tạo đường dẫn đầy đủ từ BASE_DIR"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(BASE_DIR, "Flappy-bird-python-master", *parts)


def is_off_screen(sprite):
    """Kiểm tra sprite đã ra khỏi màn hình"""
    return sprite.rect.x < -sprite.rect.width


def distance(p1, p2):
    """Tính khoảng cách Euclidean giữa 2 điểm"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calc_ear(points):
    """
    Tính Eye Aspect Ratio (EAR) để phát hiện chớp mắt
    
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Args:
        points: 6 điểm từ LEFT_EYE_IDX
    
    Returns:
        float: EAR value
    """
    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 1.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def play_sound(path):
    """Phát âm thanh"""
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    except pygame.error:
        pass
