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
