import matplotlib.pyplot as plt
import pandas as pd

# Чтение данных
data = pd.read_csv("output1.csv")
length = data.shape[0]

# Получение данных для графиков
t = data["T"]
duration = data["Duration"].astype(float)
speedup = data["Speedup"].astype(float)

# Настройка размеров фигуры
plt.figure(figsize=(10, 6))

# Построение графиков на одном графике
fig, ax1 = plt.subplots(figsize=(10, 6))

# График времени выполнения
color = 'blue'
ax1.set_xlabel('Количество потоков')
ax1.set_ylabel('Время выполнения, мс', color=color)
ax1.plot(t, duration, label="Время выполнения", color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Создание второго графика с другой осью Y
ax2 = ax1.twinx()  # Дублирование оси X
color = 'green'
ax2.set_ylabel('Ускорение', color=color)
ax2.plot(t, speedup, label="Ускорение", color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Добавление легенды и заголовка
fig.suptitle("Графики времени выполнения и ускорения")
plt.tight_layout()
plt.show()
