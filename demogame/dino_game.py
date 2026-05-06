"""
Xbox 手柄横版跑酷射击小游戏

运行方式：
    python dino_game.py

依赖：
    pip install pygame

说明：
    - 窗口大小固定为 800x400，游戏逻辑以 60 FPS 运行。
    - 玩家固定在屏幕左侧，世界中的障碍物从右向左移动。
    - Xbox 手柄按键映射：
        A / Button 0：按住下蹲，松开站立
        Y / Button 3：按下跳跃，仅在地面时生效
        B / Button 1：按下射击，带短暂冷却
"""

import os
import random
import sys

import pygame


# =========================
# 基础常量配置
# =========================

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
FPS = 60

# 固定黑色地平线。玩家和地面仙人掌都站在这条线上。
HORIZON_Y = 330

# 玩家在屏幕左侧的固定 x 坐标。
PLAYER_X = 90
PLAYER_NORMAL_WIDTH = 58
PLAYER_NORMAL_HEIGHT = 82

# 物理参数。数值按 60 FPS 调整，手感偏街机，响应清晰。
GRAVITY = 0.85
JUMP_VELOCITY = -17.0

# 障碍和子弹参数。
OBSTACLE_SPEED_START = 6.0
OBSTACLE_SPEED_MAX = 13.0
OBSTACLE_SPAWN_MIN_MS = 900
OBSTACLE_SPAWN_MAX_MS = 1600
BULLET_SPEED = 12
SHOOT_COOLDOWN_MS = 280
BULLET_CENTER_Y = HORIZON_Y - PLAYER_NORMAL_HEIGHT // 2

# Xbox 常见按键编号。不同驱动可能略有差异，但题目指定按此映射实现。
BUTTON_A = 0
BUTTON_B = 1
BUTTON_Y = 3

# 颜色。
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (120, 120, 120)
RED = (210, 40, 40)
BLUE = (40, 120, 230)


def asset_path(filename):
    """返回与脚本同目录的素材绝对路径，避免从其他目录运行时找不到图片。"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def masks_overlap(first_rect, first_mask, second_rect, second_mask):
    """使用像素级 mask 判断两个对象是否发生真实可见区域重叠。"""
    offset = (second_rect.x - first_rect.x, second_rect.y - first_rect.y)
    return first_mask.overlap(second_mask, offset) is not None


class Player:
    """玩家：负责站立、跳跃、下蹲、重力和绘制。"""

    NORMAL_WIDTH = PLAYER_NORMAL_WIDTH
    NORMAL_HEIGHT = PLAYER_NORMAL_HEIGHT
    DUCK_WIDTH = 74
    DUCK_HEIGHT = 42

    def __init__(self, filename="robotman.png"):
        # 玩家素材带 Alpha 通道，必须用 convert_alpha() 保留透明区域。
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
        """恢复到初始站立状态。"""
        self.rect.size = (self.NORMAL_WIDTH, self.NORMAL_HEIGHT)
        self.rect.x = PLAYER_X
        self.rect.bottom = HORIZON_Y
        self.velocity_y = 0.0
        self.on_ground = True
        self.ducking = False

    def jump(self):
        """仅当玩家站在地面上时允许跳跃。"""
        if self.on_ground:
            self.velocity_y = JUMP_VELOCITY
            self.on_ground = False
            self.ducking = False
            self._apply_standing_size()

    def set_ducking(self, ducking):
        """
        设置下蹲状态。

        下蹲只在地面上生效；空中不允许改变碰撞高度，避免空中“缩身”逃避障碍。
        """
        self.ducking = ducking and self.on_ground
        if self.ducking:
            self._apply_duck_size()
        else:
            self._apply_standing_size()

    def _apply_standing_size(self):
        """切换到站立碰撞盒，保持脚底仍在地平线上或当前空中位置。"""
        bottom = self.rect.bottom
        self.rect.size = (self.NORMAL_WIDTH, self.NORMAL_HEIGHT)
        self.rect.x = PLAYER_X
        self.rect.bottom = bottom
        self.image = self.normal_image
        self.mask = pygame.mask.from_surface(self.image)

    def _apply_duck_size(self):
        """切换到下蹲碰撞盒，高度约为站立时的一半。"""
        bottom = self.rect.bottom
        self.rect.size = (self.DUCK_WIDTH, self.DUCK_HEIGHT)
        self.rect.x = PLAYER_X
        self.rect.bottom = bottom
        self.image = self.duck_image
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        """应用重力并处理落地。"""
        if not self.on_ground:
            self.velocity_y += GRAVITY
            self.rect.y += int(self.velocity_y)

            if self.rect.bottom >= HORIZON_Y:
                self.rect.bottom = HORIZON_Y
                self.velocity_y = 0.0
                self.on_ground = True

        # 如果刚落地且仍按着 A 键，主循环会继续调用 set_ducking(True)。
        if self.on_ground and not self.ducking:
            self._apply_standing_size()

    def draw(self, screen):
        """根据状态绘制站立或下蹲图片。"""
        screen.blit(self.image, self.rect)


class Bullet:
    """玩家发射的子弹：水平向右移动的小矩形。"""

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
    """障碍物基类：负责共用的移动和绘制逻辑。"""

    def __init__(self, kind, speed):
        self.kind = kind
        self.speed = speed

    def setup_collision(self):
        """根据当前 image 生成 rect 和像素遮罩。"""
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.left = SCREEN_WIDTH + random.randint(20, 80)
        self.place_vertically()

    def place_vertically(self):
        """由具体障碍物子类决定自己的 Y 轴位置。"""
        raise NotImplementedError

    def update(self):
        self.rect.x -= int(self.speed)

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    @property
    def off_screen(self):
        return self.rect.right < 0


class Cactus(Obstacle):
    """地面仙人掌：只能跳过，子弹会直接穿透。"""

    def __init__(self, speed):
        super().__init__("cactus", speed)
        self.image = pygame.image.load(asset_path("cactus.png")).convert_alpha()
        self.image = pygame.transform.scale(self.image, (46, 72))
        self.setup_collision()

    def place_vertically(self):
        self.rect.bottom = HORIZON_Y


class Bird(Obstacle):
    """半空飞鸟：可下蹲躲避，也可被子弹击毁。"""

    def __init__(self, speed):
        super().__init__("bird", speed)
        self.image = pygame.image.load(asset_path("bird.png")).convert_alpha()
        self.image = pygame.transform.scale(self.image, (64, 42))
        self.setup_collision()

    def place_vertically(self):
        # 修复射击高度 BUG：飞鸟中心线与玩家站立射击时的子弹中心线完全一致。
        self.rect.centery = BULLET_CENTER_Y


class FighterJet(Obstacle):
    """半空战斗机：不可摧毁，子弹命中后只有子弹消失。"""

    def __init__(self, speed):
        super().__init__("jet", speed)
        self.image = pygame.image.load(asset_path("jetflight.png")).convert_alpha()
        self.image = pygame.transform.scale(self.image, (78, 38))
        self.setup_collision()

    def place_vertically(self):
        # 战斗机仍保持半空下蹲躲避高度。
        self.rect.bottom = HORIZON_Y - 50


class Game:
    """游戏总控：初始化、输入、更新、碰撞、计分和渲染。"""

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

        # 题目要求必须使用 pygame.joystick 初始化 Xbox 手柄。
        # 这里保存第一个检测到的手柄；若没插手柄，游戏仍显示提示并允许键盘测试。
        self.joystick = None
        self._init_joystick()

        # 关键修复：
        # ViGEmBus / vgamepad 场景下，虚拟手柄可能反复触发设备添加/移除事件。
        # 游戏运行中绝对不要响应这些事件重新初始化 joystick，否则会造成硬件轮询风暴。
        # 因此在主循环开始前屏蔽热插拔事件，只保留普通按键事件和每帧按钮状态读取。
        pygame.event.set_blocked([pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED])
        self.reset()

    def _init_joystick(self):
        """初始化第一个可用手柄。"""
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"已连接手柄：{self.joystick.get_name()}")
        else:
            print("未检测到手柄。请连接 Xbox 手柄；也可用键盘 A/Y/B 测试。")

    def reset(self):
        """重置一局游戏。"""
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
        """随机安排下一次障碍物生成时间，让节奏更自然。"""
        now = pygame.time.get_ticks()
        delay = random.randint(OBSTACLE_SPAWN_MIN_MS, OBSTACLE_SPAWN_MAX_MS)
        self.next_spawn_ticks = now + delay

    def shoot(self):
        """发射子弹，并使用冷却时间防止 Button 1 连发过快。"""
        now = pygame.time.get_ticks()
        if now - self.last_shot_ticks < SHOOT_COOLDOWN_MS:
            return

        bullet_x = self.player.rect.right + 4
        bullet_y = BULLET_CENTER_Y - Bullet.HEIGHT // 2
        self.bullets.append(Bullet(bullet_x, bullet_y))
        self.last_shot_ticks = now

    def spawn_obstacle(self):
        """按权重生成三种障碍。"""
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
        """处理退出、手柄按键和键盘测试输入。"""
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
        """在右上角显示分数。"""
        text = self.font.render(f"Score: {self.score}", True, BLACK)
        rect = text.get_rect(topright=(SCREEN_WIDTH - 18, 14))
        self.screen.blit(text, rect)

    def _draw_controller_hint(self):
        """在左上角显示当前手柄状态，方便确认 pygame.joystick 已工作。"""
        if self.joystick is not None and self.joystick.get_init():
            hint = f"Controller: {self.joystick.get_name()}"
            color = GRAY
        else:
            hint = "No controller detected"
            color = RED

        text = self.font.render(hint, True, color)
        self.screen.blit(text, (14, 14))

    def _draw_game_over(self):
        """绘制 Game Over 蒙层和重置提示。"""
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
        """主循环：输入 -> 更新 -> 绘制 -> 控制帧率。"""
        while True:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw()


def main():
    """程序入口。"""
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
