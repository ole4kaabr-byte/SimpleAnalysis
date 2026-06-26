import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Генерация случайных данных
np.random.seed(0)  # для воспроизводимости
data = np.random.randint(-10000, 10001, size=1000)
series = pd.Series(data)

# Расчёт характеристик
min_value = series.min()
count_duplicates = series.duplicated().sum()
max_value = series.max()
sum_values = series.sum()
std_dev = series.std()

# Вывод результатов
print(f"Минимальное значение: {min_value}")
print(f"Количество повторяющихся значений: {count_duplicates}")
print(f"Максимальное значение: {max_value}")
print(f"Сумма чисел: {sum_values}")
print(f"Среднеквадратическое отклонение: {std_dev}")

# Визуализация данных
# Гистограмма с округлением до сотен
rounded_data = series.apply(lambda x: round(x / 100) * 100)
plt.figure()
plt.hist(rounded_data, bins=20, edgecolor='black')
plt.title('Гистограмма округленных данных')
plt.xlabel('Значения (округленные до сотен)')
plt.ylabel('Частота')
plt.savefig('histogram.png')
plt.close()

# Линейный график
plt.figure()
plt.plot(series.index, series, label='Исходные данные')
plt.title('Линейный график исходных данных')
plt.xlabel('Индексы')
plt.ylabel('Значения')
plt.legend()
plt.savefig('line_plot.png')
plt.close()

# Создание DataFrame с дополнительными столбцами
sorted_asc = series.sort_values()
sorted_desc = series.sort_values(ascending=False)
df = pd.DataFrame({
    'Исходные данные': series,
    'Отсортированные по возрастанию': sorted_asc,
    'Отсортированные по убыванию': sorted_desc
})

# Визуализация отсортированных данных
plt.figure()
plt.plot(sorted_asc.index, sorted_asc, label='По возрастанию')
plt.plot(sorted_desc.index, sorted_desc, label='По убыванию')
plt.title('Графики отсортированных данных')
plt.xlabel('Индексы')
plt.ylabel('Значения')
plt.legend()
plt.savefig('sorted_lines.png')
plt.close()

# Сохранение DataFrame
df.to_csv('dataframe.csv', index=False)
