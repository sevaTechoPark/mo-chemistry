import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distribution(y, target_name):
    y_log = np.log1p(y)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(y, bins=30, kde=True, color='blue', ax=axes[0])
    axes[0].set_title(f'Распределение {target_name}')
    axes[0].set_xlabel('Значение')
    axes[0].set_ylabel('Частота')
    axes[0].grid(True)
    
    sns.histplot(y_log, bins=30, kde=True, color='green', ax=axes[1])
    axes[1].set_title(f'Распределение логарифмированного {target_name}')
    axes[1].set_xlabel('Логарифмированное значение')
    axes[1].set_ylabel('Частота')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def remove_high_corr_columns(df, threshold=0.95):
    correlation_matrix = df.corr()

    to_remove = set()

    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            if abs(correlation_matrix.iloc[i, j]) >= threshold:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                
                if col2 not in to_remove:
                    to_remove.add(col2)

    reduced_df = df.drop(columns=to_remove)

    print(f"Удалено {len(to_remove)} признаков из-за высокой корреляции.")
    return reduced_df

def print_corr_columns(df, threshold=0.95):
    correlation_matrix = df.corr()

    high_corr_pairs = []
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            if abs(correlation_matrix.iloc[i, j]) >= threshold:
                high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

    for pair in high_corr_pairs:
        print(f"Признаки '{pair[0]}' и '{pair[1]}' коррелируют с коэффициентом {pair[2]:.2f}")

    if len(high_corr_pairs) == 0:
        print(f"Нет коррелирующих признаков >= {threshold}")

    return high_corr_pairs
