import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ============================================
# Генерация блок-схемы алгоритма очистки данных
# ============================================
def create_cleaning_flowchart():
    """Создание блок-схемы для раздела 1.2.3"""
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('Блок-схема 1: Алгоритм очистки / верификации данных на входе', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Определение блоков
    blocks = [
        (5, 13, "НАЧАЛО\nПолучение сырого массива\ndata от генератора", 'ellipse'),
        (5, 11, "Проверка 1:\nЯвляется ли каждое\nзначение целым?\n(isinstance(value, int)\nили np.issubdtype())", 'decision'),
        (5, 9.0, "ДА: Продолжить\nНЕТ: Отбросить/округлить", 'process'),
        (5, 7.5, "Проверка 2:\nПопадание в диапазон\n[-10000, 10000]?\n(-10000 <= value <= 10000)", 'decision'),
        (5, 5.5, "ДА: Продолжить\nНЕТ: Отбросить или\nзаменить на граничное", 'process'),
        (5, 4.0, "Проверка 3:\nЯвляется ли значение\nNaN?\n(pd.isna(value))", 'decision'),
        (5, 2.0, "ДА: Отбросить\nНЕТ: Продолжить", 'process'),
        (5, 0.5, "КОНЕЦ\nФормирование\n'чистого' массива\ncleaned_data", 'ellipse')
    ]
    
    for x, y, text, btype in blocks:
        if btype == 'ellipse':
            circle = plt.Circle((x, y), 1.0, fill=True, color='lightblue', ec='black', lw=2)
            ax.add_patch(circle)
            ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
        elif btype == 'decision':
            diamond = plt.Polygon([(x, y+0.7), (x+1.2, y), (x, y-0.7), (x-1.2, y)], 
                                 fill=True, color='lightyellow', ec='black', lw=2)
            ax.add_patch(diamond)
            ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
        elif btype == 'process':
            rect = plt.Rectangle((x-1.2, y-0.4), 2.4, 0.8, fill=True, color='lightgreen', ec='black', lw=2)
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Добавление стрелок
    ax.annotate('', xy=(5, 12), xytext=(5, 12.3), 
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(5, 10.2), xytext=(5, 10.7), 
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # Добавление меток "ДА" и "НЕТ" для решений
    ax.text(6.5, 11.5, 'ДА', fontsize=10, fontweight='bold', color='green')
    ax.text(3.0, 11.5, 'НЕТ', fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('cleaning_flowchart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Сохранен файл: cleaning_flowchart.png")

# Генерация блок-схемы
create_cleaning_flowchart()

# ============================================
# Генерация случайных данных
# ============================================
np.random.seed(0)  # для воспроизводимости
data = np.random.randint(-10000, 10001, size=1000)
series = pd.Series(data, name='Значения')

# ============================================
# Функция очистки данных
# ============================================
def clean_data(data_series):
    """Верификация данных согласно блок-схеме"""
    cleaned = data_series.copy()
    
    # Проверка 1: Целое число
    if not np.issubdtype(cleaned.dtype, np.integer):
        print("Предупреждение: данные не являются целыми числами")
        cleaned = cleaned.round().astype(int)
    
    # Проверка 2: Попадание в диапазон
    mask_range = (cleaned >= -10000) & (cleaned <= 10000)
    if not mask_range.all():
        print(f"Найдено {len(cleaned) - mask_range.sum()} значений вне диапазона")
        cleaned = cleaned[mask_range]
    
    # Проверка 3: Пропуски/NaN
    mask_nan = cleaned.isna()
    if mask_nan.any():
        print(f"Найдено {mask_nan.sum()} пропущенных значений")
        cleaned = cleaned.dropna()
    
    return cleaned

# Применяем очистку
clean_series = clean_data(series)

# ============================================
# Расчёт характеристик
# ============================================
min_value = clean_series.min()
count_duplicates = clean_series.duplicated().sum()
max_value = clean_series.max()
sum_values = clean_series.sum()
std_dev = clean_series.std()

# ============================================
# Визуализация данных
# ============================================

# Гистограмма с округлением до сотен
rounded_data = clean_series.apply(lambda x: round(x / 100) * 100)
plt.figure(figsize=(10, 5))
plt.hist(rounded_data, bins=20, edgecolor='black', color='skyblue', alpha=0.7)
plt.title('Гистограмма распределения данных (округлённых до сотен)', fontsize=14)
plt.xlabel('Значения (округленные до сотен)', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('histogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nСохранен файл: histogram.png")

# Линейный график
plt.figure(figsize=(12, 5))
plt.plot(clean_series.index, clean_series, label='Исходные данные', color='blue', alpha=0.7, linewidth=0.8)
plt.title('Линейный график исходных данных', fontsize=14)
plt.xlabel('Индексы', fontsize=12)
plt.ylabel('Значения', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.savefig('line_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Сохранен файл: line_plot.png")

# ============================================
# Создание DataFrame
# ============================================
sorted_asc_values = clean_series.sort_values().values
sorted_desc_values = clean_series.sort_values(ascending=False).values

df = pd.DataFrame({
    'Исходные данные': clean_series,
    'Отсортированные по возрастанию': sorted_asc_values,
    'Отсортированные по убыванию': sorted_desc_values
})

# Визуализация отсортированных данных
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Отсортированные по возрастанию'], label='По возрастанию', color='green', linewidth=1.5)
plt.plot(df.index, df['Отсортированные по убыванию'], label='По убыванию', color='red', linewidth=1.5)
plt.title('Графики отсортированных данных', fontsize=14)
plt.xlabel('Индексы', fontsize=12)
plt.ylabel('Значения', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.savefig('sorted_lines.png', dpi=150, bbox_inches='tight')
plt.close()
print("Сохранен файл: sorted_lines.png")

# ============================================
# Сохранение DataFrame
# ============================================
output_filename = 'SimpleAnalysis_result.csv'
df.to_csv(output_filename, index=False)
print(f"\nСохранен файл: {output_filename}")

# ============================================
# Финальная проверка
# ============================================
print("\n" + "="*70)
print("ИТОГОВАЯ ПРОВЕРКА")
print("="*70)
print(f"✓ Блок-схема алгоритма очистки: cleaning_flowchart.png")
print(f"✓ Seed: 0 ")
print(f"✓ Min: {min_value} ")
print(f"✓ Max: {max_value} ")
print(f"✓ Sum: {sum_values} ")
print(f"✓ Std: {std_dev:.2f} ")
print(f"✓ Duplicates: {count_duplicates} ")
print(f"✓ Файл сохранения: {output_filename}")
print("="*70)
