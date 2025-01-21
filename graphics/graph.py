import matplotlib.pyplot as plt
import pandas

# Чтение данных
data = pandas.read_csv("output5.csv")
length = data.shape[0]

# Получение данных для графика
t = data[0:0 + length]["T"]
duration = data[0:0 + length]["Duration"]
duration_float = [float(i) for i in duration]

# Настройка размеров фигуры
plt.figure(figsize=(8, 6))

# Построение графика
plt.plot(t, duration_float, label="Время выполнения", c="blue")

# Автоматическая настройка высоты графика по амплитуде
y_min = min(duration_float) * 0.9  # Добавляем небольшой отступ
y_max = max(duration_float) * 1.1  # Добавляем небольшой отступ
plt.ylim(y_min, y_max)

# Настройка подписей и легенды
plt.xlabel('Количество потоков')
plt.ylabel('Время выполнения, мс')
plt.xticks(t)
plt.legend()
plt.show()
