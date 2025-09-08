import pygame
import random
from Pytorch import *
import torch
import cv2
import torch.mps

pygame.init()

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

paddle_width = 10
paddle_height = 100
ball_size = 20


class PongGame:

    def __init__(self, device, model, width=1470, height=820,):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Pong')

        self.paddle_location_main = self.height / 2 - paddle_height / 2
        self.paddle_location_2 = self.height / 2 - paddle_height / 2

        self.paddle_main = pygame.Rect(self.width - 40, self.paddle_location_main, paddle_width, paddle_height)
        self.paddle_ai = pygame.Rect(20, self.paddle_location_2, paddle_width, paddle_height)

        self.score_main = 0
        self.score_ai = 0
        self.paddle_ai_speed = 10

        self.initial_pace = 10
        self.pace = self.initial_pace
        self.pace_counter = 0

        self.model = model
        self.device = device
        self.cap = cv2.VideoCapture(0)

        self.clock = pygame.time.Clock()

        self.reset_ball()

    def reset_ball(self):

        ball_location_x = self.width / 2 - ball_size / 2
        ball_location_y = random.randint(100, self.height - 100)
        self.ball = pygame.Rect(ball_location_x, ball_location_y, ball_size, ball_size)
        self.ball_dx = random.choice([-self.pace, self.pace])
        self.ball_dy = random.choice([-self.pace, self.pace])

    def control_paddles(self):

        ret, frame = self.cap.read()
        if not ret:
            return
        box = get_prediction(self.model, self.device, frame)
        if box is not None:
            x_min, y_min, x_max, y_max = box
            box_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
            self.paddle_main.y = box_center[1] - paddle_height // 2

        # Clamp paddle inside the screen
        if self.paddle_main.top < 0:
            self.paddle_main.top = 0
        if self.paddle_main.bottom > self.height:
            self.paddle_main.bottom = self.height

        # AI paddle control - move only when the ball is on the AI paddle's side
        if self.ball.x < self.width / 2:
            # Introduce randomness to make the AI miss sometimes
            if random.random() < 0.2:
                self.paddle_ai.y += (self.paddle_ai_speed - 5)
            else:
                if self.paddle_ai.centery < self.ball.centery:
                    self.paddle_ai.y += self.paddle_ai_speed
                elif self.paddle_ai.centery > self.ball.centery:
                    self.paddle_ai.y -= self.paddle_ai_speed

        if self.paddle_ai.top < 0:
            self.paddle_ai.top = 0
        if self.paddle_ai.bottom > self.height:
            self.paddle_ai.bottom = self.height

    def move_ball(self):

        self.ball.x += self.ball_dx
        self.ball.y += self.ball_dy

        if (self.pace_counter == 1) and (self.pace <= 25):  # Increase pace up to a max of 20
            self.pace += 2
            self.pace_counter = 0

        if self.ball.top <= 0 or self.ball.bottom >= self.height:
            self.ball_dy = -self.ball_dy

        if self.ball.colliderect(self.paddle_main) or self.ball.colliderect(self.paddle_ai):
            self.ball_dx = -self.ball_dx
            self.pace_counter += 1

        if self.ball.left <= 0:
            self.score_main += 1
            self.reset_ball_with_pace()

        if self.ball.right >= self.width:
            self.score_ai += 1
            self.reset_ball_with_pace()

    def reset_ball_with_pace(self):

        self.pace = self.initial_pace + 2  # Reset pace to initial value
        self.pace_counter = 0  # Reset pace counter
        self.reset_ball()  # Reset ball position and direction

    def draw(self):

        pygame.mouse.set_visible(False)
        self.display.fill(BLACK)
        pygame.draw.rect(self.display, BLUE1, self.paddle_main)
        pygame.draw.rect(self.display, RED, self.paddle_ai)
        pygame.draw.rect(self.display, WHITE, self.ball)
        pygame.draw.aaline(self.display, WHITE, (self.width / 2, 0), (self.width / 2, self.height))

        font = pygame.font.Font(None, 74)
        score_text = font.render(f"{self.score_ai}  {self.score_main}", True, WHITE)
        self.display.blit(score_text, (self.width / 2 - score_text.get_width() / 2, 10))

        pygame.display.flip()

    def wait_for_space(self):

        waiting = True
        while waiting:
            self.display.fill(BLACK)
            font = pygame.font.Font(None, 74)
            text = font.render("Press SPACE to Start", True, WHITE)
            self.display.blit(text, (self.width / 2 - text.get_width() / 2, self.height / 2 - text.get_height() / 2))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False

    def run(self):

        self.wait_for_space()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            self.control_paddles()
            self.move_ball()
            self.draw()
            self.clock.tick(60)
        self.cap.release()
        pygame.quit()
        quit()


if __name__ == '__main__':

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = get_ssd_model(num_classes=2).to(device)
    model.load_state_dict(torch.load('Last_model_weights.pth'))
    game = PongGame(device=device, model=model)
    game.run()
