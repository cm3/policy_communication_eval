import numpy as np
import pandas as pd
from itertools import combinations
import sys

def calculate_weighted_kappa(observed_matrix, weight_matrix):
    """重み付けカッパ係数を計算"""
    total_observations = np.sum(observed_matrix)

    # 観測一致度 Po
    Po = np.sum(weight_matrix * observed_matrix) / total_observations

    # 期待一致度 Pe
    row_totals = np.sum(observed_matrix, axis=1)
    col_totals = np.sum(observed_matrix, axis=0)
    expected_matrix = np.outer(row_totals, col_totals) / total_observations
    Pe = np.sum(weight_matrix * expected_matrix) / total_observations

    # カッパ係数
    kappa = (Po - Pe) / (1 - Pe) if (1 - Pe) != 0 else 0
    return Po, Pe, kappa

def create_contingency_table(data, categories):
    """各評価者ペアごとに観測頻度表を作成"""
    n_categories = len(categories)
    contingency_tables = {}

    for rater1, rater2 in combinations(data.columns, 2):
        table = np.zeros((n_categories, n_categories), dtype=int)
        for row in data.itertuples(index=False):
            i = categories.index(getattr(row, rater1))
            j = categories.index(getattr(row, rater2))
            table[i, j] += 1
        contingency_tables[(rater1, rater2)] = table

    return contingency_tables

def create_weight_matrix(categories):
    """二乗重み行列を作成"""
    n = len(categories)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            W[i, j] = 1 - ((i - j) ** 2) / ((n - 1) ** 2)
    return W

def main(csv_path, min_score, max_score):
    # CSV を読み込み（最初の2列は無視し、3列目以降のデータを取得）
    df = pd.read_csv(csv_path).iloc[:, 2:]

    # 指定されたスコア範囲を適用
    categories = list(range(min_score, max_score + 1))

    # 観測頻度表を作成
    contingency_tables = create_contingency_table(df, categories)

    # 重み行列を作成
    weight_matrix = create_weight_matrix(categories)

    # 各評価者ペアについてカッパ値を計算
    print("\n=== 重み付けカッパ係数 ===")
    for (rater1, rater2), observed_matrix in contingency_tables.items():
        Po, Pe, kappa = calculate_weighted_kappa(observed_matrix, weight_matrix)
        print(f"{rater1} vs {rater2}:")
        print(f"  - 観測一致度 (Po): {Po:.4f}")
        print(f"  - 期待一致度 (Pe): {Pe:.4f}")
        print(f"  - 重み付けカッパ係数: {kappa:.4f}\n")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("使用法: python script.py data.csv min_score max_score")
    else:
        csv_path = sys.argv[1]
        min_score = int(sys.argv[2])
        max_score = int(sys.argv[3])
        main(csv_path, min_score, max_score)
