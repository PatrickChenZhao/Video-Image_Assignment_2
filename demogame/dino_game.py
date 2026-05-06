"""
Side-scrolling parkour shooter mini-game for Xbox controllers

How to run:
    python dino_game.py

Dependencies:
    pip install pygame

Description:
    - The window size is fixed at 800x400, and the game runs at 60 FPS.
    - The player is fixed on the left side of the screen, and obstacles in the world move from right to left.
    - Xbox controller button mapping:
        A / Button 0: Hold to crouch, release to stand
        Y / Button 3: Press to jump (only works when on the ground)
        B / Button 1: Press to shoot (with a short cooldown)
"""

import os
import random
import sys

import pygame


# =========================
# Basic Constant Configuration
# =========================

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
FPS = 60

# Set a fixed black horizon line. Both the player and the cacti on the ground stand on this line.
HORIZON_Y = 330

# The player is at a fixed x-coordinate on the left side of the screen.
PLAYER_X = 90
PLAYER_NORMAL_WIDTH = 58
PLAYER_NORMAL_HEIGHT = 82

# Physical parameters. The settings are optimized for 60 FPS, with an arcade-style feel and crisp responsiveness.
GRAVITY = 0.85
JUMP_VELOCITY = -17.0

# Obstacle and bullet parameters.
OBSTACLE_SPEED_START = 6.0
OBSTACLE_SPEED_MAX = 13.0
OBSTACLE_SPAWN_MIN_MS = 900
OBSTACLE_SPAWN_MAX_MS = 1600
BULLET_SPEED = 12
SHOOT_COOLDOWN_MS = 280
BULLET_CENTER_Y = HORIZON_Y - PLAYER_NORMAL_HEIGHT // 2

# Common Xbox button assignments. These may vary slightly depending on the driver, but the question specifies that this mapping should be used.
BUTTON_A = 0
BUTTON_B = 1
BUTTON_Y = 3

# Color.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (120, 120, 120)
RED = (210, 40, 40)
BLUE = (40, 120, 230)


def asset_path(filename):
    """Return the absolute path of the assets in the same directory as the script to prevent images from being missing when the script is run from a different directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def masks_overlap(first_rect, first_mask, second_rect, second_mask):
    """Use a pixel-level mask to determine whether the visible areas of two objects actually overlap."""
    offset = (second_rect.x - first_rect.x, second_rect.y - first_rect.y)
    return first_mask.overlap(second_mask, offset) is not None


class Player:
    """Player: Responsible for standing, jumping, crouching, gravity, and rendering."""

    NORMAL_WIDTH = PLAYER_NORMAL_WIDTH
    NORMAL_HEIGHT = PLAYER_NORMAL_HEIGHT
    DUCK_WIDTH = 74
    DUCK_HEIGHT = 42

    def __init__(self, filename="robotman.png"):
        # Player assets include an alpha channel; you must use `convert_alpha()` to preserve transparent areas.
        self.image = pygame.image.load(asset_path(filename)).convert_alpha()
        self.image = pygame.transform.scale(
            self.image, (self.NORMAL_WIDTH, self.NORMAL_HEIGHT)
        )
        self.normal_image = self.image
        self.duck_image = pygame.transform.scale(
            self.normal_image, (self.DUCK_WIDTH, self.DUCK_HEIGHT)
        )
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = pygame.Rect(
            PLAYER_X,
            HORIZON_Y - self.NORMAL_HEIGHT,
            self.NORMAL_WIDTH,
            self.NORMAL_HEIGHT,
        )
        self.velocity_y = 0.0
        self.on_ground = True
        self.ducking = False

    def reset(self):
        """Return to the starting standing position."""
        self.rect.size = (self.NORMAL_WIDTH, self.NORMAL_HEIGHT)
        self.rect.x = PLAYER_X
        self.rect.bottom = HORIZON_Y
        self.velocity_y = 0.0
        self.on_ground = True
        self.ducking = False

    def jump(self):
        """Jumping is only allowed when the player is standing on the ground."""
        if self.on_ground:
            self.velocity_y = JUMP_VELOCITY
            self.on_ground = False
            self.ducking = False
            self._apply_standing_size()

    def set_ducking(self, ducking):
        """
        Set the crouching position.

        Crouching only works on the ground; you cannot change your collision height while in the air to avoid obstacles by “crouching” mid-air.
        """
        self.ducking = ducking and self.on_ground
        if self.ducking:
            self._apply_duck_size()
        else:
            self._apply_standing_size()

    def _apply_standing_size(self):
        """Switch to the standing collision box, keeping the soles of your feet on the ground or in their current position in the air."""
        bottom = self.rect.bottom
        self.rect.size = (self.NORMAL_WIDTH, self.NORMAL_HEIGHT)
        self.rect.x = PLAYER_X
        self.rect.bottom = bottom
        self.image = self.normal_image
        self.mask = pygame.mask.from_surface(self.image)

    def _apply_duck_size(self):
        """Switch to the crouching collision box, which is about half the height of the standing position."""
        bottom = self.rect.bottom
        self.rect.size = (self.DUCK_WIDTH, self.DUCK_HEIGHT)
        self.rect.x = PLAYER_X
        self.rect.bottom = bottom
        self.image = self.duck_image
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        """Use gravity and handle the landing."""
        if not self.on_ground:
            self.velocity_y += GRAVITY
            self.rect.y += int(self.velocity_y)

            if self.rect.bottom >= HORIZON_Y:
                self.rect.bottom = HORIZON_Y
                self.velocity_y = 0.0
                self.on_ground = True

        # If you've just landed and are still holding down the A button, the main loop will continue to be called set_ducking(True)。
        if self.on_ground and not self.ducking:
            self._apply_standing_size()

    def draw(self, screen):
        """Draw a picture of a person standing or squatting based on the given pose."""
        screen.blit(self.image, self.rect)


class Bullet:
    """Player-fired bullets: small rectangles moving horizontally to the right."""

    WIDTH = 14
    HEIGHT = 6

    def __init__(self, x, y):
        self.image = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA).convert_alpha()
        self.image.fill(BLUE)
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = pygame.Rect(x, y, self.WIDTH, self.HEIGHT)

    def update(self):
        self.rect.x += BULLET_SPEED

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    @property
    def off_screen(self):
        return self.rect.left > SCREEN_WIDTH


class Obstacle:
    """Obstacle Base Class: Responsible for shared movement and rendering logic."""

    def __init__(self, kind, speed):
        self.kind = kind
        self.speed = speed

    def setup_collision(self):
        """Generate a rect and a pixel mask based on the current image."""
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.left = SCREEN_WIDTH + random.randint(20, 80)
        self.place_vertically()

    def place_vertically(self):
        """Each specific obstacle subclass determines its own Y-axis position."""
        raise NotImplementedError

    def update(self):
        self.rect.x -= int(self.speed)

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    @property
    def off_screen(self):
        return self.rect.right < 0


class Cactus(Obstacle):
    """Ground cacti: You can only jump over them; bullets will pass right through them."""

    def __init__(self, speed):
        super().__init__("cactus", speed)
        self.image = pygame.image.load(asset_path("cactus.png")).convert_alpha()
        self.image = pygame.transform.scale(self.image, (46, 72))
        self.setup_collision()

    def place_vertically(self):
        self.rect.bottom = HORIZON_Y


class Bird(Obstacle):
    """Flying Birds: You can crouch to dodge them, or they can be destroyed by bullets."""

    def __init__(self, speed):
        super().__init__("bird", speed)
        self.image = pygame.image.load(asset_path("bird.png")).convert_alpha()
        self.image = pygame.transform.scale(self.image, (64, 42))
        self.setup_collision()

    def place_vertically(self):
        # Fixed a bug related to bullet trajectory: The centerline of the bird's flight path now perfectly aligns with the centerline of the bullet when the player is standing and firing.
        self.rect.centery = BULLET_CENTER_Y


class FighterJet(Obstacle):
    """Mid-air fighter: Indestructible; when hit by a bullet, only the bullet disappears."""

    def __init__(self, speed):
        super().__init__("jet", speed)
        self.image = pygame.image.load(asset_path("jetflight.png")).convert_alpha()
        self.image = pygame.transform.scale(self.image, (78, 38))
        self.setup_collision()

    def place_vertically(self):
        # The fighter jet remains at a low altitude to evade detection.
        self.rect.bottom = HORIZON_Y - 50


class Game:
    """Game Controller: Initialization, Input, Update, Collision, Scoring, and Rendering."""

    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Xbox Dino Shooter")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24)
        self.big_font = pygame.font.SysFont("arial", 46, bold=True)

        self.player = Player()
        self.bullets = []
        self.obstacles = []
        self.game_over = False
        self.score = 0
        self.start_ticks = pygame.time.get_ticks()
        self.last_shot_ticks = -SHOOT_COOLDOWN_MS
        self.next_spawn_ticks = 0
        self.obstacle_speed = OBSTACLE_SPEED_START

        
        # This stores the first detected controller; if no controller is connected, the game will still display a prompt and allow for keyboard testing.
        self.joystick = None
        self._init_joystick()

        
        # In ViGEmBus / vgamepad scenarios, the virtual gamepad may repeatedly trigger device add/remove events.
        # Never respond to these events to reinitialize the joystick while the game is running, as this will cause a hardware polling storm.
        # Therefore, suppress hot-plug events before the main loop begins, retaining only standard button events and per-frame button state reads.
        pygame.event.set_blocked([pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED])
        self.reset()

    def _init_joystick(self):
        """Initialize the first available controller."""
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Controller connected：{self.joystick.get_name()}")
        else:
            print("No controller detected. Please connect an Xbox controller; you can also use the A/Y/B keys on your keyboard to test.")

    def reset(self):
        """Restart the game."""
        self.player.reset()
        self.bullets.clear()
        self.obstacles.clear()
        self.game_over = False
        self.score = 0
        self.start_ticks = pygame.time.get_ticks()
        self.last_shot_ticks = -SHOOT_COOLDOWN_MS
        self.obstacle_speed = OBSTACLE_SPEED_START
        self._schedule_next_obstacle()

    def _schedule_next_obstacle(self):
        """Randomize the timing of the next obstacle to create a more natural rhythm."""
        now = pygame.time.get_ticks()
        delay = random.randint(OBSTACLE_SPAWN_MIN_MS, OBSTACLE_SPAWN_MAX_MS)
        self.next_spawn_ticks = now + delay

    def shoot(self):
        """Fire a bullet, and use the cooldown to prevent Button 1 from firing too rapidly."""
        now = pygame.time.get_ticks()
        if now - self.last_shot_ticks < SHOOT_COOLDOWN_MS:
            return

        bullet_x = self.player.rect.right + 4
        bullet_y = BULLET_CENTER_Y - Bullet.HEIGHT // 2
        self.bullets.append(Bullet(bullet_x, bullet_y))
        self.last_shot_ticks = now

    def spawn_obstacle(self):
        """Generate three types of obstacles based on their weights."""
        kind = random.choices(
            ["cactus", "bird", "jet"],
            weights=[0.48, 0.32, 0.20],
            k=1,
        )[0]
        obstacle_classes = {
            "cactus": Cactus,
            "bird": Bird,
            "jet": FighterJet,
        }
        self.obstacles.append(obstacle_classes[kind](self.obstacle_speed))
        self._schedule_next_obstacle()

    def handle_events(self):
        """Handle exit, controller button, and keyboard test inputs."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if self.game_over:
                # Game Over 后，按手柄任意键或键盘任意键都重置游戏。
                if event.type in (pygame.JOYBUTTONDOWN, pygame.KEYDOWN):
                    self.reset()
                continue

            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == BUTTON_Y:
                    self.player.jump()
                elif event.button == BUTTON_B:
                    self.shoot()

            if event.type == pygame.KEYDOWN:
                # 键盘只是便于没有手柄时调试，不改变题目要求的手柄映射。
                if event.key == pygame.K_y:
                    self.player.jump()
                elif event.key == pygame.K_b:
                    self.shoot()

        if not self.game_over:
            self._handle_hold_inputs()

    def _handle_hold_inputs(self):
        """
        处理需要“按住期间持续生效”的输入。

        Xbox A 键必须按住下蹲、松开恢复站立，因此这里每帧读取按钮状态，
        而不是只依赖 JOYBUTTONDOWN / JOYBUTTONUP 事件。
        """
        duck_pressed = False

        if self.joystick is not None and self.joystick.get_init():
            try:
                duck_pressed = bool(self.joystick.get_button(BUTTON_A))
            except pygame.error:
                duck_pressed = False

        # 键盘 A 也支持按住测试。
        keys = pygame.key.get_pressed()
        duck_pressed = duck_pressed or keys[pygame.K_a]

        self.player.set_ducking(duck_pressed)

    def update(self):
        """更新整局游戏状态。"""
        if self.game_over:
            return

        now = pygame.time.get_ticks()
        elapsed_seconds = (now - self.start_ticks) / 1000.0

        # 分数随存活时间增加。
        self.score = int(elapsed_seconds * 10)

        # 随时间逐渐增加速度，但限制最大值，避免后期完全不可玩。
        self.obstacle_speed = min(
            OBSTACLE_SPEED_MAX,
            OBSTACLE_SPEED_START + elapsed_seconds * 0.12,
        )

        if now >= self.next_spawn_ticks:
            self.spawn_obstacle()

        self.player.update()

        for bullet in self.bullets:
            bullet.update()
        self.bullets = [bullet for bullet in self.bullets if not bullet.off_screen]

        for obstacle in self.obstacles:
            obstacle.speed = self.obstacle_speed
            obstacle.update()
        self.obstacles = [obstacle for obstacle in self.obstacles if not obstacle.off_screen]

        self._handle_bullet_collisions()
        self._handle_player_collisions()

    def _handle_bullet_collisions(self):
        """
        处理子弹和障碍物的碰撞。

        规则：
            - 子弹击中飞鸟：飞鸟被摧毁，子弹也消失。
            - 子弹击中战斗机：子弹消失，战斗机绝对不会被摧毁。
            - 子弹遇到仙人掌：完全穿透，不做任何碰撞判定。
        """
        bullets_to_remove = set()
        obstacles_to_remove = set()

        for bullet_index, bullet in enumerate(self.bullets):
            for obstacle_index, obstacle in enumerate(self.obstacles):
                if obstacle.kind == "cactus":
                    continue

                if not masks_overlap(bullet.rect, bullet.mask, obstacle.rect, obstacle.mask):
                    continue

                bullets_to_remove.add(bullet_index)

                if obstacle.kind == "bird":
                    obstacles_to_remove.add(obstacle_index)

                # 飞鸟和战斗机都会吞掉子弹；只有飞鸟会被同时移除。
                break

        self.bullets = [
            bullet
            for index, bullet in enumerate(self.bullets)
            if index not in bullets_to_remove
        ]
        self.obstacles = [
            obstacle
            for index, obstacle in enumerate(self.obstacles)
            if index not in obstacles_to_remove
        ]

    def _handle_player_collisions(self):
        """玩家碰到任意障碍物即 Game Over。"""
        for obstacle in self.obstacles:
            if masks_overlap(self.player.rect, self.player.mask, obstacle.rect, obstacle.mask):
                self.game_over = True
                break

    def draw(self):
        """绘制背景、地平线、角色、障碍、子弹、分数和提示。"""
        self.screen.fill(WHITE)

        # 固定黑色地平线。
        pygame.draw.line(self.screen, BLACK, (0, HORIZON_Y), (SCREEN_WIDTH, HORIZON_Y), 3)

        self.player.draw(self.screen)

        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        for bullet in self.bullets:
            bullet.draw(self.screen)

        self._draw_score()
        self._draw_controller_hint()

        if self.game_over:
            self._draw_game_over()

        pygame.display.flip()

    def _draw_score(self):
        """Display the score in the upper-right corner."""
        text = self.font.render(f"Score: {self.score}", True, BLACK)
        rect = text.get_rect(topright=(SCREEN_WIDTH - 18, 14))
        self.screen.blit(text, rect)

    def _draw_controller_hint(self):
        """Displays the current joystick status in the top-left corner, making it easy to verify that pygame.joystick is working."""
        if self.joystick is not None and self.joystick.get_init():
            hint = f"Controller: {self.joystick.get_name()}"
            color = GRAY
        else:
            hint = "No controller detected"
            color = RED

        text = self.font.render(hint, True, color)
        self.screen.blit(text, (14, 14))

    def _draw_game_over(self):
        """Draw the “Game Over” overlay and the reset prompt."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 190))
        self.screen.blit(overlay, (0, 0))

        title = self.big_font.render("GAME OVER", True, BLACK)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
        self.screen.blit(title, title_rect)

        prompt = self.font.render("Press any controller button to restart", True, BLACK)
        prompt_rect = prompt.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 22))
        self.screen.blit(prompt, prompt_rect)

    def run(self):
        """Main loop: Input -> Update -> Render -> Control frame rate."""
        while True:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw()


def main():
    """Program entry."""
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
