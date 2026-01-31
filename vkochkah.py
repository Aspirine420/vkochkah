import pygame
import numpy as np

# --- 1. Энциклопедические знания (Обучающая выборка) ---
# [Ночь_Т, День_Т, Влажность, Осадки, Месяц, Широта]
X_train = np.array([
    [8, 13, 90, 40, 10, 55],  # ИДЕАЛ: Октябрь, дожди, прохладно -> 0.98
    [15, 22, 40, 5, 7, 55],   # ПЛОХО: Июль, жара, сухо -> 0.05
    [-2, 6, 85, 20, 11, 58],  # СРЕДНЕ: Ноябрь, заморозки -> 0.45
    [5, 12, 75, 15, 9, 56],   # ХОРОШО: Конец сентября -> 0.80
    [2, 8, 30, 0, 4, 54]      # ПЛОХО: Апрель, сухая весна -> 0.10
])
y_train = np.array([[0.98], [0.05], [0.45], [0.80], [0.10]])

# Границы для нормализации
X_min = np.array([-10, 0, 0, 0, 1, 40])
X_max = np.array([20, 35, 100, 100, 12, 70])
def scale(x): return (np.array(x) - X_min) / (X_max - X_min + 1e-5)

# --- 2. Инициализация и ОБУЧЕНИЕ ---
np.random.seed(42)
W1 = np.random.randn(6, 14) * 0.5
W2 = np.random.randn(14, 1) * 0.5
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

print("Нейросеть изучает данные из интернета...")
for epoch in range(15000): # 15к итераций обучения
    s_x = scale(X_train)
    # Forward pass
    h = sigmoid(np.dot(s_x, W1))
    out = sigmoid(np.dot(h, W2))
    # Backpropagation (корректировка нитей)
    error = y_train - out
    d_out = error * (out * (1 - out))
    d_h = d_out.dot(W2.T) * (h * (1 - h))
    W2 += h.T.dot(d_out) * 0.1
    W1 += s_x.T.dot(d_h) * 0.1
print("Обучение завершено. Веса настроены.")

# --- 3. Интерфейс ---
pygame.init()
W_WIN, H_WIN = 1200, 850
screen = pygame.display.set_mode((W_WIN, H_WIN))
font = pygame.font.SysFont("Consolas", 15, bold=True)
log_font = pygame.font.SysFont("Consolas", 14)

inputs = [8.0, 13.0, 90.0, 40.0, 10.0, 55.0] # Сразу ставим идеал
labels = ["T.НОЧЬ", "T.ДЕНЬ", "ВЛАЖН %", "ДОЖДЬ мм", "МЕСЯЦ", "ШИРОТА"]
active_idx, input_buffer, history = 0, "", [0.5]

def run():
    global active_idx, input_buffer, history
    clock = pygame.time.Clock()
    
    while True:
        screen.fill((5, 8, 15))
        # Сетка
        for x in range(0, W_WIN, 30): pygame.draw.line(screen, (15, 18, 30), (x, 0), (x, H_WIN))
        for y in range(0, H_WIN, 30): pygame.draw.line(screen, (15, 18, 30), (0, y), (W_WIN, y))
        
        # Мысли нейронки
        s_in = scale(inputs)
        h_layer = sigmoid(np.dot(s_in, W1))
        prob = sigmoid(np.dot(h_layer, W2)).item()

        # Визуализация нитей (Синапсы)
        xi, xh, xo = 380, 780, 1120
        yi, yh = np.linspace(150, 650, 6), np.linspace(100, 700, 14)
        for i in range(6):
            for j in range(14):
                # Показываем только сильные связи
                signal = s_in[i] * W1[i,j]
                alpha = int(np.clip(abs(signal) * 255, 10, 220))
                color = (0, 255, 200) if signal > 0 else (255, 20, 100)
                line_surf = pygame.Surface((W_WIN, H_WIN), pygame.SRCALPHA)
                pygame.draw.line(line_surf, (*color, alpha), (xi, yi[i]), (xh, yh[j]), 1)
                screen.blit(line_surf, (0,0))

        # Текстовый блок мыслей
        log_y = 700
        screen.blit(font.render("ЛОГ МЫСЛЕЙ НЕЙРОСЕТИ:", True, (0, 255, 150)), (50, log_y))
        msgs = [
            f"> Входной сигнал '{labels[active_idx]}' принят.",
            f"> Анализ весов: {'Позитивный' if prob > 0.5 else 'Негативный'} тренд.",
            f"> Резонанс мицелия: {'ВЫСОКИЙ' if prob > 0.8 else 'НИЗКИЙ'}.",
            f"> Вердикт: {'Грибы обнаружены' if prob > 0.8 else 'Условия не оптимальны'}."
        ]
        for i, m in enumerate(msgs):
            screen.blit(log_font.render(m, True, (150, 180, 255)), (50, log_y + 25 + i*20))

        # Отрисовка нейронов
        for i, y in enumerate(yi):
            act = (i == active_idx)
            pygame.draw.circle(screen, (0, 255, 255) if act else (100, 100, 100), (int(xi), int(y)), 8)
            txt = (input_buffer + "_") if act else f"{inputs[i]}"
            screen.blit(font.render(f"{labels[i]}: {txt}", True, (255,255,255)), (50, int(y)-8))

        for y in yh: pygame.draw.circle(screen, (100, 50, 200), (int(xh), int(y)), 6)

        # Результат
        res_c = (int(255*(1-prob)), int(255*prob), 50)
        pygame.draw.circle(screen, res_c, (xo, 400), 60, 2)
        pygame.draw.circle(screen, res_c, (xo, 400), int(prob*55))
        screen.blit(font.render(f"{prob:.1%}", True, res_c), (xo-35, 480))

        for event in pygame.event.get():
            if event.type == pygame.QUIT: return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    try: inputs[active_idx] = float(input_buffer)
                    except: pass
                    input_buffer, active_idx = "", (active_idx + 1) % 6
                    history.append(prob)
                elif event.key == pygame.K_BACKSPACE: input_buffer = input_buffer[:-1]
                elif event.unicode in "0123456789.": input_buffer += event.unicode
        
        pygame.display.flip()
        clock.tick(60)

run()
pygame.quit()
