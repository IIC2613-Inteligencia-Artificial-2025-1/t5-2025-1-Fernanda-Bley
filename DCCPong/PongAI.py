# Implementado por Daniel Florea y Vicente Vega para IIC2613 - Inteligencia Artificial
# basado en la implementación de Tech with Tim - https://www.youtube.com/watch?v=vVGTZlnnX3U 

import pygame
import random

pygame.init()

WIDTH, HEIGHT = 700, 500
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED = 20, 100, 4
BALL_RADIUS, MAX_BALL_SPEED = 7, 5

font = pygame.font.SysFont('arial', 25)
SCORE_FONT = pygame.font.SysFont("comicsans", 50)

class Paddle:
    # Esta clase modela al pádel con el que interactúa el agente.

    COLOR = WHITE
    VEL = PADDLE_SPEED

    def __init__(self, x, y, width, height):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.width = width
        self.height = height

    def draw(self, win):
        # Muestra al pádel en pantalla
        pygame.draw.rect(
            win, self.COLOR, (self.x, self.y, self.width, self.height))

    def move(self, up = True):
        # Cambia la posición del pádel
        if up:
            self.y -= self.VEL
        else:
            self.y += self.VEL

    def reset(self):
        # Devuelve al pádel a su posición inicial
        self.x = self.original_x
        self.y = self.original_y


class Ball:
    # Esta clase modela a la pelota del juego.

    MAX_VEL = MAX_BALL_SPEED
    COLOR = WHITE

    def __init__(self, x, y, radius):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.radius = radius
        self.x_vel = self.MAX_VEL
        self.y_vel = 0
        self.bounces = 0

    def draw(self, win):
        # Muestra la pelota en pantalla
        pygame.draw.circle(win, self.COLOR, (self.x, self.y), self.radius)

    def move(self):
        # Cambia la posición de la pelota
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self):
        # Devuelve a la pelota a su posición inicial
        self.x = self.original_x
        self.y = self.original_y
        self.y_vel = 0
        self.x_vel = self.MAX_VEL
        self.bounces = 0

class PongAI:
    # Esta clase modela el gameplay del juego y su interacción con el agente.

    def __init__(self, w=WIDTH, h=HEIGHT, vis=True):
        self.w = w
        self.h = h
        self.vis = vis

        # En caso de querer cargar elementos visuales
        if self.vis:   
            self.WIN = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Pong")
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('PongAI')
            self.clock = pygame.time.Clock()

        self.score = 0
        self.hitpoint = 0
        self.reset()

    def reset(self):
        # Inicializamos el juego a en su estado original
        self.MAX_X = WIDTH
        self.left_paddle = Paddle(10, HEIGHT + 1, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.right_paddle = Paddle(WIDTH - 10 - PADDLE_WIDTH, HEIGHT //
                            2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.right_paddle.y = random.randint(0, HEIGHT - self.right_paddle.height)

        ball_x = self.left_paddle.x + self.left_paddle.width
        ball_y = random.randint(BALL_RADIUS, HEIGHT - BALL_RADIUS)
        self.ball = Ball(ball_x, ball_y, BALL_RADIUS)
        self.ball.y_vel = random.uniform(-5,5)
        self.ball.bounces = 0

        self.frame_iteration = 0

        # Calculamos el punto de término de la pelota en el mapa
        time_to_padel = (WIDTH - 10 - self.left_paddle.width - self.right_paddle.width)/self.ball.x_vel
        y_pos = self.ball.y + self.ball.y_vel * time_to_padel

        if self.ball.y_vel > 0:
            bounces = int(abs((y_pos + self.ball.radius) / HEIGHT))
        else:
            bounces = int(abs((HEIGHT - self.ball.y - self.ball.y_vel * time_to_padel + self.ball.radius) / HEIGHT))

        if bounces == 0:
            self.hitpoint = y_pos

        else:
            if self.ball.y_vel > 0:
                y_final_no_filter = y_pos - bounces * HEIGHT
            else:
                y_final_no_filter = y_pos + bounces * HEIGHT

            if bounces % 2:
                self.hitpoint = HEIGHT - y_final_no_filter
            else:
                self.hitpoint = y_final_no_filter

    def play_step(self, action):
        # Ejecuta una acción elegida por el agente previamente
        if self.vis:
            # Actualiza la interfaz en caso de haber visualización
            self.draw(self.WIN, [self.left_paddle, self.right_paddle], "", "")
            self.clock.tick(FPS)

        game_over = False 
        self.frame_iteration += 1

        if self.vis:
            # Revisa si el usuario ha cerrado la ventana
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        past_score = self.score

        # Mueve los componentes del juego en el espacio
        self._move(action)
        self.ball.move()

        # En caso de golpear a la pelota
        if self.handle_collision():
            self.score += 1
            self.reset()

        # En caso contrario, revisar si ya salió del área de juego
        elif self.ball.x > WIDTH:
            game_over = True
            self.score = 0  
            self.reset()

        # De no ser así, revisar si el punto medio del agente se encuentra a menos de un cuarto de pádel de donde llegará pelota
        if abs(self.hitpoint - (self.right_paddle.y + self.right_paddle.height/2)) <= self.right_paddle.height/4:
            # Recompensar al agente si se queda quieto, castigarlo si no
            if action == 1:
                reward = 1
            else:
                reward = -1

        # Si no, revisar si el punto de llegada de la pelota está encima del agente
        elif self.right_paddle.y > self.hitpoint:
            # Recompensar al agente si se mueve hacia arriba, castigarlo si no
            if action == 0:
                reward = 1
            else:
                reward = -1
        
        # Si la pelota está debajo del agente
        else:
            # Recompensar al agente si se mueve hacia abajo, castigarlo si no
            if action == 2:
                reward = 1
            else:
                reward = -1

        # Retornar información al agente
        return reward, game_over, past_score 

    def handle_collision(self):
        # Rebote con la parte de abajo de la pantalla
        if self.ball.y + self.ball.radius >= HEIGHT:
            self.ball.y = HEIGHT - self.ball.radius
            self.ball.y_vel *= -1
            self.ball.bounces += 1

        # Rebote con la parte de arriba de la pantalla
        elif self.ball.y - self.ball.radius <= 0:
            self.ball.y = 0 + self.ball.radius
            self.ball.y_vel *= -1
            self.ball.bounces += 1

        # Impacto con el pádel del jugador
        if self.ball.y >= self.right_paddle.y and self.ball.y <= self.right_paddle.y + self.right_paddle.height:
            if self.ball.x + self.ball.radius >= self.right_paddle.x:
                self.ball.bounces = 0
                return True
        return False
    
    def draw(self, win, paddles, left_score, right_score):
        # Encargada de mostrar los elementos gráficos del juego
        win.fill(BLACK)

        right_score_text = SCORE_FONT.render(f"{self.score}", 1, WHITE)
        win.blit(right_score_text, (WIDTH * (3/4) -
                                    right_score_text.get_width()//2, 20))

        for paddle in paddles:
            paddle.draw(win)

        for i in range(10, HEIGHT, HEIGHT//20):
            if i % 2 == 1:
                continue
            pygame.draw.rect(win, WHITE, (WIDTH//2 - 5, i, 10, HEIGHT//20))

        self.ball.draw(win)
        pygame.display.update()


    def _move(self, action):
        # No moverse simplemente no llama al método

        # Movimiento hacia arriba
        if action == 0 and self.right_paddle.y - self.right_paddle.VEL >= 0:
            self.right_paddle.move(up = True)

        # Movimiento hacia abajo
        if action == 2 and self.right_paddle.y + self.right_paddle.VEL + self.right_paddle.height <= HEIGHT:
            self.right_paddle.move(up = False)

# Si estás leyendo esto, queremos desearte mucho éxito en tu tarea <3
