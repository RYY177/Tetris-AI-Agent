import pygame
import sys
import os
import pickle
import csv
import cv2
import numpy as np
from datetime import datetime
from tetris import Tetris
from config import *
# from config_for_infer import *
# from config import *
# GRID_ROW_COUNT = 20
# GRID_COL_COUNT = 10

from agent import GeneticAgent  # 假设您的Agent类在agent.py中
# 添加环境变量设置
os.environ["SDL_VIDEODRIVER"] = "dummy"  # 使用虚拟显示驱动
os.environ["SDL_AUDIODRIVER"] = "dummy"  # 禁用音频

# SAVE_DIR = "saved_agents"
SAVE_DIR = "/datasets_genie/yangyang.ren/TetrisAI-main/saved_agents_1000"

# 在文件顶部使用与config.py一致的ACTION映射
ACTION_MAPPING = {
    ACTION.NOTHING: 0,       # 无操作
    ACTION.L: 1,             # 左移
    ACTION.R: 2,             # 右移
    ACTION.L2: 3,            # 左移两格
    ACTION.R2: 4,            # 右移两格
    ACTION.ROTATE: 5,        # 旋转（统一处理，不分顺/逆时针）
    ACTION.SWAP: 6,          # 交换（原HOLD）
    ACTION.FAST_FALL: 7,     # 快速下落（原SOFT_DROP）
    ACTION.INSTANT_FALL: 8   # 立即下落（原HARD_DROP）
}

class AgentRecorder:
    def __init__(self, agent_path, output_dir="agent_data"):
        # 加载训练好的Agent
        with open(agent_path, "rb") as f:
            self.agent = pickle.load(f)
        
        self.tetris = Tetris()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化pygame用于渲染
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Tetris Agent Recorder')
        self.font = pygame.font.SysFont('Arial', 20)
        
        # 视频录制
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = None
        
        # 数据记录
        self.csv_file = None
        self.csv_writer = None
        
        # 游戏状态
        self.recording = False
        self.game_start_time = None
        self.game_count = 0

        # 修改屏幕尺寸为单个游戏
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Tetris Agent Recorder')
        
        # 添加游戏区域偏移量（居中显示）
        self.grid_offset_x = (SCREEN_WIDTH - GAME_WIDTH) // 2
        self.grid_offset_y = (SCREEN_HEIGHT - GAME_HEIGHT - INFO_HEIGHT) // 2 + INFO_HEIGHT
        # self.grid_offset_y = 0
        
        # 使用config.py中的颜色定义
        self.bg_color = COLORS.BACKGROUND_BLACK.value
        self.grid_color = COLORS.TRIANGLE_GRAY.value
        self.info_bg_color = COLORS.BACKGROUND_DARK.value
        self.text_color = COLORS.WHITE.value
        
        # 创建方块颜色映射
        self.block_color_map = {
            1: COLORS.PIECE_I.value,
            2: COLORS.PIECE_L.value,
            3: COLORS.PIECE_J.value,
            4: COLORS.PIECE_S.value,
            5: COLORS.PIECE_Z.value,
            6: COLORS.PIECE_T.value,
            7: COLORS.PIECE_O.value
        }
    
    def init_recording(self, game_idx):
        """初始化录制"""
        self.recording = True
        self.game_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.game_count = game_idx
        
        # 初始化视频录制
        video_file = os.path.join(self.output_dir, f"game_{self.game_start_time}.mp4")
        self.video_writer = cv2.VideoWriter(
            video_file,
            self.fourcc,
            # 30.0,
            6, #帧率设置为8
            (SCREEN_WIDTH, SCREEN_HEIGHT)
        )
        
        # 初始化CSV文件
        csv_file = os.path.join(self.output_dir, f"game_{self.game_start_time}.csv")
        self.csv_file = open(csv_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # 更新CSV表头以匹配新的动作类型
        self.csv_writer.writerow([
            'video_name', 'frame_index', 'score',
            'action_value',
            'L', 'R', 'L2', 'R2',
            'ROTATE', 'SWAP', 
            'FAST_FALL', 'INSTANT_FALL', 'NOTHING'
        ])
        
        print(f"🎬 Recording game {game_idx} started: {self.game_start_time}")
        
        # # 写入CSV表头
        # self.csv_writer.writerow([
        #     'frame_index', 'timestamp', 'score', 'action', 
        #     'grid_state', 'current_piece', 'next_piece',
        #     'piece_x', 'piece_y', 'piece_shape'
        # ])
        
        # print(f"🎬 Recording game {game_idx} started: {self.game_start_time}")
    
    def stop_recording(self):
        """停止录制并关闭文件"""
        if not self.recording:
            return
        
        self.recording = False
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
        
        print(f"⏹️ Recording stopped for game {self.game_count}")
    
    def record_frame(self, action, frame_idx):
        """录制当前帧并写入简化数据"""
        if not self.recording:
            return
        
        # 捕获当前帧（保持不变）
        data = pygame.surfarray.array3d(self.screen)
        data = data.transpose([1, 0, 2])
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        self.video_writer.write(data)
        
        # 记录简化数据到CSV
        # timestamp = datetime.now().strftime("%H:%M:%S.%f")
        
        # 获取动作值和动作类型
        action_value = ACTION_MAPPING.get(action, 0)  # 安全获取，避免KeyError
        # action_type = action.name if action else "NOTHING"
        score = f"{self.tetris.score:.4f}"

        # 更新动作标志以匹配新的动作类型
        action_flags = {
            'L': 1 if action == ACTION.L else 0,
            'R': 1 if action == ACTION.R else 0,
            'L2': 1 if action == ACTION.L2 else 0,
            'R2': 1 if action == ACTION.R2 else 0,
            'ROTATE': 1 if action == ACTION.ROTATE else 0,
            'SWAP': 1 if action == ACTION.SWAP else 0,
            'FAST_FALL': 1 if action == ACTION.FAST_FALL else 0,
            'INSTANT_FALL': 1 if action == ACTION.INSTANT_FALL else 0,
            'NOTHING': 1 if action == ACTION.NOTHING else 0,
        }


        self.csv_writer.writerow([
            f"game_{self.game_start_time}",  # 视频名称
            frame_idx,                       # 帧索引
            score,
            action_value,                    # 动作数值
            # action_type,                     # 动作类型名称
            action_flags['L'],
            action_flags['R'],
            action_flags['L2'],
            action_flags['R2'],
            action_flags['ROTATE'],
            action_flags['SWAP'],
            action_flags['FAST_FALL'],
            action_flags['INSTANT_FALL'],
            action_flags['NOTHING']
        ])


    def draw_game(self):
        """绘制游戏界面（单个游戏）"""
        # 绘制整个屏幕的背景
        self.screen.fill((0, 0, 0))  # 使用纯黑色背景
        
        # 绘制游戏区域背景（按列交替）
        for col in range(GRID_COL_COUNT):
            # 交替使用深色和浅色背景
            color = (40, 40, 40) if col % 2 == 0 else (50, 50, 50)  # 深灰和浅灰
            rect = pygame.Rect(
                self.grid_offset_x + col * BLOCK_SIZE,
                self.grid_offset_y,
                BLOCK_SIZE,
                GRID_ROW_COUNT * BLOCK_SIZE
            )
            pygame.draw.rect(self.screen, color, rect)
        
        # 绘制已落下的方块
        for y in range(GRID_ROW_COUNT):
            for x in range(GRID_COL_COUNT):
                cell_value = self.tetris.grid[y][x]
                if cell_value != 0 and cell_value in self.block_color_map:
                    rect_x = self.grid_offset_x + x * BLOCK_SIZE
                    rect_y = self.grid_offset_y + y * BLOCK_SIZE
                    
                    # 绘制方块主体
                    pygame.draw.rect(
                        self.screen, 
                        self.block_color_map[cell_value], 
                        (rect_x, rect_y, BLOCK_SIZE, BLOCK_SIZE)
                    )
                    
                    # 绘制方块边框
                    pygame.draw.rect(
                        self.screen, 
                        (0, 0, 0),  # 黑色边框
                        (rect_x, rect_y, BLOCK_SIZE, BLOCK_SIZE), 
                        1
                    )
                    
                    # 绘制左上角的小三角形
                    offset = BLOCK_SIZE / 10
                    pygame.draw.polygon(
                        self.screen,
                        (200, 200, 200),  # 浅灰色三角形
                        [
                            (rect_x + offset, rect_y + offset),
                            (rect_x + 3 * offset, rect_y + offset),
                            (rect_x + offset, rect_y + 3 * offset)
                        ]
                    )
        
        # 绘制当前活动的方块
        for y, row in enumerate(self.tetris.piece_shape):
            for x, val in enumerate(row):
                if val != 0 and val in self.block_color_map:
                    real_x = self.tetris.piece_x + x
                    real_y = self.tetris.piece_y + y
                    
                    rect_x = self.grid_offset_x + real_x * BLOCK_SIZE
                    rect_y = self.grid_offset_y + real_y * BLOCK_SIZE
                    
                    # 绘制方块主体
                    pygame.draw.rect(
                        self.screen, 
                        self.block_color_map[val], 
                        (rect_x, rect_y, BLOCK_SIZE, BLOCK_SIZE)
                    )
                    
                    # 绘制方块边框
                    pygame.draw.rect(
                        self.screen, 
                        (0, 0, 0),  # 黑色边框
                        (rect_x, rect_y, BLOCK_SIZE, BLOCK_SIZE), 
                        1
                    )
                    
                    # 绘制左上角的小三角形
                    offset = BLOCK_SIZE / 10
                    pygame.draw.polygon(
                        self.screen,
                        (200, 200, 200),  # 浅灰色三角形
                        [
                            (rect_x + offset, rect_y + offset),
                            (rect_x + 3 * offset, rect_y + offset),
                            (rect_x + offset, rect_y + 3 * offset)
                        ]
                    )
        
        # 游戏结束画面
        if self.tetris.dead:
            # 半透明黑色覆盖层
            overlay = pygame.Surface((GRID_COL_COUNT * BLOCK_SIZE, GRID_ROW_COUNT * BLOCK_SIZE), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))  # 黑色半透明
            self.screen.blit(overlay, (self.grid_offset_x, self.grid_offset_y))
            
            # 游戏结束文字
            font = pygame.font.SysFont(None, BLOCK_SIZE * 2)  # 大号字体
            text = font.render("GAME OVER", True, (255, 0, 0))  # 红色文字
            text_rect = text.get_rect(center=(
                self.grid_offset_x + GRID_COL_COUNT * BLOCK_SIZE // 2,
                self.grid_offset_y + GRID_ROW_COUNT * BLOCK_SIZE // 2
            ))
            self.screen.blit(text, text_rect)
        
        # 显示信息（在顶部信息区域）
        info_rect = pygame.Rect(0, 0, SCREEN_WIDTH, INFO_HEIGHT)
        pygame.draw.rect(self.screen, (30, 30, 30), info_rect)  # 深灰色信息栏
        
        texts = [
            f"Game: {self.game_count}",
            f"Score: {self.tetris.score}",
            f"Recording: {'ON' if self.recording else 'OFF'}"
        ]
        
        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, (255, 255, 255))  # 白色文字
            self.screen.blit(text_surface, (20, 10 + i * 30))
        
        pygame.display.flip()

    def run(self, num_games=10):
        """运行录制，生成指定数量的游戏数据"""
        for game_idx in range(1, num_games + 1):
            self.tetris.reset_game()
            self.init_recording(game_idx)
            
            # 游戏主循环
            clock = pygame.time.Clock()
            frame_idx = 0
            last_drop_time = pygame.time.get_ticks()
            auto_drop_interval = 1000  # 自动下落间隔（毫秒）
            action_delay = 200  # 每个操作后延迟100毫秒
            
            while not self.tetris.dead:
                current_time = pygame.time.get_ticks()
                frame_idx += 1

                # if frame_idx >= 20 * 5:
                #     print(f"📹 Reached max video duration, ending early.")
                #     break
                
                # 处理事件（以便可以退出）
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.stop_recording()
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        self.stop_recording()
                        pygame.quit()
                        sys.exit()
                
                # 获取Agent的动作
                action = self.agent.get_action(self.tetris)
                # 执行动作前添加延迟
                pygame.time.delay(action_delay)
                self.tetris.step(action)
                
                # 自动下落（如果时间到了）
                if current_time - last_drop_time > auto_drop_interval:
                    # 修改这里：ACTION.NONE -> ACTION.NOTHING
                    # self.tetris.step(ACTION.NOTHING)  # 无操作触发下落
                    self.tetris.drop_piece()
                    last_drop_time = current_time
                
                # 绘制和录制
                self.draw_game()
                self.record_frame(action, frame_idx)
                
                # # 控制游戏速度
                clock.tick(8)
            
            # 游戏结束，停止录制
            self.stop_recording()
            print(f"🎮 Game {game_idx} finished. Score: {self.tetris.score}")
        
        pygame.quit()
        print(f"✅ Successfully generated {num_games} games!")

def select_latest_agent():
    """选择最新保存的Agent"""
    if not os.path.exists(SAVE_DIR):
        print("❌ No saved agents found. Train an agent first.")
        return None
    
    agent_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".pkl")]
    if not agent_files:
        print("❌ No saved agents found. Train an agent first.")
        return None
    
    # 按修改时间排序
    agent_files.sort(key=lambda x: os.path.getmtime(os.path.join(SAVE_DIR, x)), reverse=True)
    return os.path.join(SAVE_DIR, agent_files[0])

if __name__ == "__main__":
    # 自动选择最新训练的Agent
    agent_path = select_latest_agent()
    if not agent_path:
        sys.exit(1)
    
    print(f"🤖 Using agent: {os.path.basename(agent_path)}")
    
    # 生成游戏数据
    recorder = AgentRecorder(agent_path, output_dir="agent_generated_data_0629")
    recorder.run(num_games=50000)  # 生成10个游戏的数据