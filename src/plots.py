"""Todas as visualizações (PNGs) dos Blocos 1-3 e do fechamento.

Extraído de ``dissertacao_mestrado.py`` (Fase 1 da modularização): ~30 funções
``plot_*``/``visualize_*`` que recebem listas de resultados/arrays e salvam
PNGs na pasta de execução. São consumidores puros — não chamam a API nem mutam
estado do runner. Única exceção de acoplamento: ``plot_phase_e_example_locations``
faz import tardio de ``select_examples_by_strategy`` do runner (documentado lá).
"""
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # backend não-interativo: salva em arquivo, não abre janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

from data_problems import (
    PROBLEM_A_CENTERS, PROBLEM_A_STD, PROBLEM_B_CENTERS, PROBLEM_B_STD,
    PROBLEM_C_CENTERS, PROBLEM_C_STD, PROBLEM_E_CENTERS,
)
from metrics import augment_features, compute_metric_confidence, predict_with_metric
from protocolo import EXAMPLE_STRATEGIES
from resultados import ResultadoExperimento, ResultadoPhaseEExperimento


def _save_panels_individually(panels, combined_filename, dpi=150):
    """Salva cada painel de uma figura composta como imagem individual.

    Args:
        panels: lista de (sufixo, draw_func, figsize) onde:
            - sufixo: string adicionada ao nome do arquivo (ex: "problema_a")
            - draw_func: callable(ax) que desenha em um único eixo
            - figsize: (largura, altura) da figura individual
        combined_filename: caminho completo do arquivo combinado (ex: "pasta/01_all.png")
        dpi: resolução das imagens individuais
    """
    if not combined_filename:
        return
    base, ext = os.path.splitext(combined_filename)
    for suffix, draw_func, figsize in panels:
        fig_ind, ax_ind = plt.subplots(1, 1, figsize=figsize)
        draw_func(ax_ind)
        fig_ind.tight_layout()
        fig_ind.savefig(f"{base}_{suffix}{ext}", dpi=dpi, bbox_inches='tight')
        plt.close(fig_ind)


def visualize_all_problems(
    X_a: np.ndarray, y_a: np.ndarray,
    X_b: np.ndarray, y_b: np.ndarray,
    X_c: np.ndarray, y_c: np.ndarray,
    filename: str = None
):
    """Visualiza os três problemas sintéticos lado a lado para inspeção visual da geometria."""
    problems = [
        (X_a, y_a, PROBLEM_A_CENTERS, PROBLEM_A_STD, "PROBLEMA A\n(Aprendizado da Métrica)"),
        (X_b, y_b, PROBLEM_B_CENTERS, PROBLEM_B_STD, "PROBLEMA B\n(Teste de Consistência 1)"),
        (X_c, y_c, PROBLEM_C_CENTERS, PROBLEM_C_STD, "PROBLEMA C\n(Teste de Consistência 2)")
    ]

    def _draw_problem(ax, X, y, centers, std, title):
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm",
                   alpha=0.7, edgecolor="k", s=60)
        ax.scatter(*centers[0], marker='*', s=300, c='blue',
                   edgecolor='black', linewidth=2, label='Centro 0', zorder=5)
        ax.scatter(*centers[1], marker='*', s=300, c='red',
                   edgecolor='black', linewidth=2, label='Centro 1', zorder=5)
        ax.set_title(f"{title}\nCentros: {centers}, σ={std}",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-5, 5)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (X, y, centers, std, title) in zip(axes, problems):
        _draw_problem(ax, X, y, centers, std, title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    suffixes = ["problema_a", "problema_b", "problema_c"]
    panels = [
        (suffixes[i], lambda ax, p=problems[i]: _draw_problem(ax, *p), (7, 5))
        for i in range(3)
    ]
    _save_panels_individually(panels, filename)


def visualize_problem_e_with_expert(
    X_e: np.ndarray, y_gt_e: np.ndarray,
    y_expert_e: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray,
    filename: str = None
):
    """
    Visualiza o Problema E (perito linear do Bloco 2) com a fronteira de decisão do perito.
    Compara os rótulos verdadeiros (ground truth) com os rótulos do perito lado a lado.
    """
    # Precomputa grid para reutilizar nos painéis
    x_min, x_max = -6, 6
    y_min, y_max = -5, 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_with_metric(grid_points, expert_centroids, expert_w)
    Z = Z.reshape(xx.shape)
    confidences, _ = compute_metric_confidence(grid_points, expert_centroids, expert_w)
    conf_grid = confidences.reshape(xx.shape)

    def _draw_ground_truth(ax):
        ax.scatter(X_e[:, 0], X_e[:, 1], c=y_gt_e, cmap="coolwarm",
                   alpha=0.7, edgecolor="k", s=60)
        ax.scatter(*PROBLEM_E_CENTERS[0], marker='*', s=300, c='blue',
                   edgecolor='black', linewidth=2, label='Centro Verdadeiro 0', zorder=5)
        ax.scatter(*PROBLEM_E_CENTERS[1], marker='*', s=300, c='red',
                   edgecolor='black', linewidth=2, label='Centro Verdadeiro 1', zorder=5)
        ax.set_title("Problema E: Rótulos Verdadeiros (Ground Truth)", fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    def _draw_expert_boundary(ax):
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0.5])
        ax.scatter(X_e[:, 0], X_e[:, 1], c=y_expert_e, cmap="coolwarm",
                   alpha=0.7, edgecolor="k", s=60)
        ax.scatter(*expert_centroids[0], marker='X', s=300, c='blue',
                   edgecolor='k', linewidth=2, label='Centróide do Perito 0', zorder=5)
        ax.scatter(*expert_centroids[1], marker='X', s=300, c='red',
                   edgecolor='k', linewidth=2, label='Centróide do Perito 1', zorder=5)
        ax.set_title(f"Rótulos do Perito (W=[{expert_w[0]:.1f}, {expert_w[1]:.1f}])",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    def _draw_margin_heatmap(ax):
        im = ax.contourf(xx, yy, conf_grid, levels=20, cmap="RdYlGn")
        ax.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0.5])
        plt.colorbar(im, ax=ax, label="Margem (confiança)")
        ax.scatter(X_e[:, 0], X_e[:, 1], c='black', alpha=0.3, s=20)
        ax.set_title("Margem do Perito (Distância à Fronteira)", fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _draw_ground_truth(axes[0])
    _draw_expert_boundary(axes[1])
    _draw_margin_heatmap(axes[2])
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("ground_truth", _draw_ground_truth, (7, 5)),
        ("fronteira_perito", _draw_expert_boundary, (7, 5)),
        ("margem_perito", _draw_margin_heatmap, (7, 5)),
    ]
    _save_panels_individually(panels, filename)


def plot_oracle_meialua(results: List[dict], filename: str = None):
    """Barras de fidelidade (métrica vs ground truth) da meia-lua por n_features.

    Mostra visualmente o ganho 2→3→4 features: a métrica diagonal sozinha (2)
    aproxima mal a meia-lua; com x1·x2 (3) e x1²/x2² (4) a aproximação melhora.
    """
    if not results:
        return
    df = pd.DataFrame(results)
    algos = sorted(df['algorithm'].unique())
    nfeats = sorted(df['n_features'].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(algos), 1)
    x = np.arange(len(nfeats))
    cmap = plt.get_cmap('tab10')
    for j, algo in enumerate(algos):
        means, stds = [], []
        for nf in nfeats:
            sub = df[(df['algorithm'] == algo) & (df['n_features'] == nf)]
            means.append(sub['fidelity_vs_true'].mean() if len(sub) else 0)
            stds.append(sub['fidelity_vs_true'].std() if len(sub) > 1 else 0)
        offset = (j - (len(algos) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               color=cmap(j), edgecolor='black', alpha=0.85, label=algo.upper())
    ax.set_xticks(x)
    ax.set_xticklabels([f'{nf} features' for nf in nfeats])
    ax.set_ylabel('Fidelidade da métrica vs ground truth')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    ax.set_title(
        'Oracle meia-lua — W que APROXIMA a fronteira não-linear\n'
        '(reunião 20/05: a fidelidade cresce ao aumentar features)',
        fontweight='bold', fontsize=11,
    )
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=140, bbox_inches='tight')
    plt.close()


def plot_consistency_comparison_extended(resultados: List[ResultadoExperimento], filename: str = None):
    """Gráfico comparativo estendido incluindo Problema C e métricas adicionais (Kappa, F1)."""
    df = pd.DataFrame([
        {
            'model': f"{r.provider}/{r.model_name} (t={r.temperature})",
            'n_shot': r.n_shot,
            'nomes': f"{r.nomes_classes[0]}/{r.nomes_classes[1]}",
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
            'kappa_b': r.kappa_problema_b,
            'kappa_c': r.kappa_problema_c,
            'f1_b': r.f1_problema_b,
            'f1_c': r.f1_problema_c,
            'fidelidade': r.fidelidade_problema_a,
        }
        for r in resultados
    ])

    df_cons = df.groupby('n_shot').agg({
        'consistencia_b': ['mean', 'std'], 'consistencia_c': ['mean', 'std'],
    }).reset_index()
    df_cons.columns = ['n_shot', 'b_mean', 'b_std', 'c_mean', 'c_std']
    x = np.arange(len(df_cons))
    width = 0.35

    df_kappa = df.groupby('n_shot').agg({'kappa_b': ['mean', 'std'], 'kappa_c': ['mean', 'std']}).reset_index()
    df_kappa.columns = ['n_shot', 'b_mean', 'b_std', 'c_mean', 'c_std']

    df_f1 = df.groupby('n_shot').agg({'f1_b': ['mean', 'std'], 'f1_c': ['mean', 'std']}).reset_index()
    df_f1.columns = ['n_shot', 'b_mean', 'b_std', 'c_mean', 'c_std']

    def _draw_consistency(ax):
        ax.bar(x - width/2, df_cons['b_mean'], width, yerr=df_cons['b_std'],
               label='Problema B', color='steelblue', capsize=3)
        ax.bar(x + width/2, df_cons['c_mean'], width, yerr=df_cons['c_std'],
               label='Problema C', color='coral', capsize=3)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(n)}-shot' for n in df_cons['n_shot']])
        ax.set_ylabel('Consistência')
        ax.set_title('Consistência: Problema B vs C', fontweight='bold')
        ax.legend(); ax.set_ylim(0, 1.1)

    def _draw_kappa(ax):
        ax.bar(x - width/2, df_kappa['b_mean'], width, yerr=df_kappa['b_std'],
               label='Problema B', color='steelblue', capsize=3)
        ax.bar(x + width/2, df_kappa['c_mean'], width, yerr=df_kappa['c_std'],
               label='Problema C', color='coral', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(n)}-shot' for n in df_kappa['n_shot']])
        ax.set_ylabel("Kappa de Cohen")
        ax.set_title("Kappa de Cohen: B vs C", fontweight='bold')
        ax.legend(); ax.set_ylim(-0.1, 1.1)

    def _draw_f1(ax):
        ax.bar(x - width/2, df_f1['b_mean'], width, yerr=df_f1['b_std'],
               label='Problema B', color='steelblue', capsize=3)
        ax.bar(x + width/2, df_f1['c_mean'], width, yerr=df_f1['c_std'],
               label='Problema C', color='coral', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(n)}-shot' for n in df_f1['n_shot']])
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score: B vs C', fontweight='bold')
        ax.legend(); ax.set_ylim(0, 1.1)

    def _draw_boxplot(ax):
        from matplotlib.patches import Patch
        metrics_data = []
        for col, label in [('consistencia_b', 'Acu B'), ('consistencia_c', 'Acu C'),
                           ('kappa_b', 'Kappa B'), ('kappa_c', 'Kappa C'),
                           ('f1_b', 'F1 B'), ('f1_c', 'F1 C')]:
            for val in df[col]:
                metrics_data.append({'Metric': label, 'Value': val})
        df_metrics = pd.DataFrame(metrics_data)
        colors_bp = ['steelblue', 'coral'] * 3
        positions = [0, 0.6, 1.5, 2.1, 3.0, 3.6]
        for i, (metric, color) in enumerate(zip(['Acu B', 'Acu C', 'Kappa B', 'Kappa C', 'F1 B', 'F1 C'], colors_bp)):
            data = df_metrics[df_metrics['Metric'] == metric]['Value']
            bp = ax.boxplot([data], positions=[positions[i]], widths=0.4, patch_artist=True)
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.7)
        ax.set_xticks([0.3, 1.8, 3.3])
        ax.set_xticklabels(['Acurácia', "Kappa de Cohen", 'F1-Score'])
        ax.set_ylabel('Pontuação')
        ax.set_title('Distribuição de Todas as Métricas', fontweight='bold')
        legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Problema B'),
                           Patch(facecolor='coral', alpha=0.7, label='Problema C')]
        ax.legend(handles=legend_elements, loc='lower right')

    # Figura combinada
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _draw_consistency(axes[0, 0])
    _draw_kappa(axes[0, 1])
    _draw_f1(axes[1, 0])
    _draw_boxplot(axes[1, 1])
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("consistencia", _draw_consistency, (7, 5)),
        ("kappa", _draw_kappa, (7, 5)),
        ("f1", _draw_f1, (7, 5)),
        ("distribuicao", _draw_boxplot, (7, 5)),
    ]
    _save_panels_individually(panels, filename)


def plot_class_names_effect(resultados: List[ResultadoExperimento], filename: str = None):
    """Analisa o efeito dos nomes das classes na consistência do LLM com a métrica."""
    df = pd.DataFrame([
        {
            'nomes': f"{r.nomes_classes[0]}/{r.nomes_classes[1]}",
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
        }
        for r in resultados
    ])

    df_names = df.groupby('nomes').agg({
        'consistencia_b': ['mean', 'std'],
        'consistencia_c': ['mean', 'std']
    }).reset_index()
    df_names.columns = ['nomes', 'b_mean', 'b_std', 'c_mean', 'c_std']

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df_names))
    width = 0.35

    ax.bar(x - width/2, df_names['b_mean'], width, yerr=df_names['b_std'],
           label='Problema B', color='steelblue', capsize=3, edgecolor='black')
    ax.bar(x + width/2, df_names['c_mean'], width, yerr=df_names['c_std'],
           label='Problema C', color='coral', capsize=3, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(df_names['nomes'], rotation=45, ha='right')
    ax.set_ylabel('Consistência Média')
    ax.set_title('Efeito dos Nomes das Classes na Consistência', fontsize=12, fontweight='bold')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(resultados: List[ResultadoExperimento], filename: str = None):
    """Compara modelos se múltiplos foram testados no experimento."""
    df = pd.DataFrame([
        {
            'model': f"{r.provider}/{r.model_name} (t={r.temperature})",
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
            'kappa_b': r.kappa_problema_b,
            'kappa_c': r.kappa_problema_c,
            'fidelidade': r.fidelidade_problema_a,
        }
        for r in resultados
    ])
    models = df['model'].unique()
    if len(models) < 2:
        print("  Pulando comparação entre modelos (apenas 1 modelo testado)")
        return

    df_models = df.groupby('model').agg({
        'consistencia_b': 'mean', 'consistencia_c': 'mean',
        'kappa_b': 'mean', 'kappa_c': 'mean', 'fidelidade': 'mean',
    }).reset_index().sort_values('consistencia_b', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    width = 0.35

    ax = axes[0]
    for i, (_, row) in enumerate(df_models.iterrows()):
        ax.bar(i - width/2, row['consistencia_b'], width, color=colors[i], edgecolor='black', alpha=0.8)
        ax.bar(i + width/2, row['consistencia_c'], width, color=colors[i], edgecolor='black', alpha=0.5, hatch='//')
    ax.set_xticks(range(len(df_models)))
    ax.set_xticklabels([m.split('/')[-1] for m in df_models['model']], rotation=45, ha='right')
    ax.set_ylabel('Consistência')
    ax.set_title('COMPARAÇÃO DE MODELOS: Consistência', fontweight='bold')
    ax.set_ylim(0, 1.1)

    ax = axes[1]
    metrics = ['consistencia_b', 'consistencia_c', 'kappa_b', 'kappa_c', 'fidelidade']
    labels = ['Consist. B', 'Consist. C', 'Kappa B', 'Kappa C', 'Fidelidade']
    x_metrics = np.arange(len(metrics))
    w = 0.8 / len(models)
    for i, (_, row) in enumerate(df_models.iterrows()):
        values = [row[m] for m in metrics]
        x_pos = x_metrics + i * w - (len(models) - 1) * w / 2
        ax.bar(x_pos, values, w * 0.9, color=colors[i], edgecolor='black', alpha=0.8,
               label=row['model'].split('/')[-1])
    ax.set_xticks(x_metrics)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Pontuação')
    ax.set_title('COMPARAÇÃO DE MODELOS: Todas as Métricas', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_seed_comparison(resultados: List[ResultadoExperimento], filename: str = None):
    """Compara os resultados entre diferentes sementes aleatórias para avaliar robustez."""
    df = pd.DataFrame([
        {
            'seed': r.random_seed,
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
        }
        for r in resultados
    ])
    seeds = sorted(df['seed'].unique())
    if len(seeds) < 2:
        print("  Pulando comparação entre sementes (apenas 1 semente testada)")
        return

    df_seeds = df.groupby('seed').agg({
        'consistencia_b': ['mean', 'std'],
        'consistencia_c': ['mean', 'std'],
    }).reset_index()
    df_seeds.columns = ['seed', 'b_mean', 'b_std', 'c_mean', 'c_std']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(seeds))
    width = 0.35
    ax.bar(x - width/2, df_seeds['b_mean'], width, yerr=df_seeds['b_std'],
           label='Problema B', color='steelblue', capsize=5, edgecolor='black')
    ax.bar(x + width/2, df_seeds['c_mean'], width, yerr=df_seeds['c_std'],
           label='Problema C', color='coral', capsize=5, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Semente {s}' for s in seeds])
    ax.set_ylabel('Consistência')
    ax.set_title('CONSISTÊNCIA POR SEMENTE ALEATÓRIA', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_phase_e_learning_curve(
    results_e: List[ResultadoPhaseEExperimento],
    filename: str = None
):
    """
    Curva de aprendizado mostrando como o LLM melhora com mais exemplos,
    discriminada por estratégia de seleção de exemplos.

    Esta é a VISUALIZAÇÃO PRINCIPAL da Fase E:
    - Eixo X: número de exemplos few-shot
    - Eixo Y: concordância LLM vs. Perito
    - Linhas: uma por estratégia (easy, hard, mixed, random)
    """
    df = pd.DataFrame([
        {
            'model': f"{r.provider}/{r.model_name}",
            'n_shot': r.n_shot,
            'strategy': r.example_strategy,
            'accuracy': r.accuracy_llm_vs_expert,
            'kappa': r.kappa_llm_vs_expert,
            'f1': r.f1_llm_vs_expert,
        }
        for r in results_e
    ])

    strategy_colors = {
        'easy': '#2ecc71',    # Green
        'hard': '#e74c3c',    # Red
        'mixed': '#3498db',   # Blue
        'random': '#95a5a6',  # Gray
    }
    strategy_markers = {
        'easy': 'o',
        'hard': 's',
        'mixed': 'D',
        'random': '^',
    }

    metrics_config = [
        ('accuracy', 'Concordância LLM vs. Perito (Acurácia)'),
        ('kappa', "Kappa de Cohen"),
        ('f1', 'F1-Score'),
    ]

    def _draw_metric(ax, metric_col, metric_name):
        for strategy in EXAMPLE_STRATEGIES:
            df_strat = df[df['strategy'] == strategy]
            if len(df_strat) == 0:
                continue
            df_grouped = df_strat.groupby('n_shot').agg({
                metric_col: ['mean', 'std']
            }).reset_index()
            df_grouped.columns = ['n_shot', 'mean', 'std']
            df_grouped = df_grouped.sort_values('n_shot')
            ax.errorbar(
                df_grouped['n_shot'], df_grouped['mean'],
                yerr=df_grouped['std'],
                marker=strategy_markers[strategy],
                color=strategy_colors[strategy],
                label=f'{strategy.capitalize()}',
                linewidth=2, markersize=8, capsize=4
            )
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance aleatória')
        ax.set_xlabel('Número de Exemplos Few-Shot', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'Fase E: {metric_name}\nvs Número de Exemplos', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (metric_col, metric_name) in enumerate(metrics_config):
        _draw_metric(axes[i], metric_col, metric_name)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    suffixes = ["accuracy", "kappa", "f1"]
    panels = [
        (suffixes[i], lambda ax, mc=metrics_config[i]: _draw_metric(ax, mc[0], mc[1]), (7, 5))
        for i in range(3)
    ]
    _save_panels_individually(panels, filename)


def plot_phase_e_strategy_comparison(
    results_e: List[ResultadoPhaseEExperimento],
    filename: str = None
):
    """
    Compara estratégias em cada nível de n_shot usando gráficos de barras agrupadas.
    Permite verificar qual estratégia de seleção de exemplos é mais eficaz para cada quantidade de shots.
    """
    df = pd.DataFrame([
        {
            'n_shot': r.n_shot,
            'strategy': r.example_strategy,
            'accuracy': r.accuracy_llm_vs_expert,
            'kappa': r.kappa_llm_vs_expert,
        }
        for r in results_e
    ])

    # Filtra apenas os resultados few-shot (exclui zero-shot, que é igual para todas as estratégias)
    df_fs = df[df['n_shot'] > 0]

    if len(df_fs) == 0:
        print("  Sem resultados few-shot para comparar estratégias.")
        return

    n_shots = sorted(df_fs['n_shot'].unique())

    strategy_colors = {
        'easy': '#2ecc71',
        'hard': '#e74c3c',
        'mixed': '#3498db',
        'random': '#95a5a6',
    }

    def _draw_n_shot(ax, n):
        df_n = df_fs[df_fs['n_shot'] == n]
        df_strat = df_n.groupby('strategy').agg({
            'accuracy': ['mean', 'std']
        }).reset_index()
        df_strat.columns = ['strategy', 'mean', 'std']
        bars = ax.bar(
            range(len(df_strat)), df_strat['mean'], yerr=df_strat['std'],
            color=[strategy_colors.get(s, 'gray') for s in df_strat['strategy']],
            edgecolor='black', capsize=5
        )
        ax.set_xticks(range(len(df_strat)))
        ax.set_xticklabels([s.capitalize() for s in df_strat['strategy']], fontsize=10)
        ax.set_ylabel('Concordância LLM vs. Perito')
        ax.set_title(f'{n} Exemplos Few-Shot', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        for bar, mean_val in zip(bars, df_strat['mean']):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{mean_val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Figura combinada
    fig, axes = plt.subplots(1, len(n_shots), figsize=(5 * len(n_shots), 5), squeeze=False)
    axes = axes[0]
    for ax_idx, n in enumerate(n_shots):
        _draw_n_shot(axes[ax_idx], n)
    plt.suptitle('Fase E: Comparação de Estratégias por Número de Exemplos',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        (f"{n}shot", lambda ax, n_val=n: _draw_n_shot(ax, n_val), (6, 5))
        for n in n_shots
    ]
    _save_panels_individually(panels, filename)


def plot_phase_e_example_locations(
    X_e: np.ndarray,
    y_expert: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray,
    n_examples: int = 10,
    random_state: int = 42,
    filename: str = None
):
    """
    Visualiza ONDE cada estratégia seleciona os exemplos.
    Exibe seleções easy/hard/mixed/random sobre a fronteira de decisão do perito.
    Fundamental para entender intuitivamente o que cada estratégia oferece ao LLM.
    """
    # Único ponto em que um plot reexecuta lógica de experimento: a seleção de
    # exemplos vive no runner (é núcleo do protocolo, não visualização). Import
    # tardio para não criar ciclo runner -> plots -> runner na carga dos módulos.
    from dissertacao_mestrado import select_examples_by_strategy

    strategies = ["easy", "hard", "mixed", "random"]

    x_min, x_max = -6, 6
    y_min, y_max = -5, 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_with_metric(grid_points, expert_centroids, expert_w)
    Z = Z.reshape(xx.shape)
    confidences_all, _ = compute_metric_confidence(X_e, expert_centroids, expert_w)

    def _draw_strategy(ax, strategy):
        ax.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, colors='k', linewidths=1.5, levels=[0.5])
        ax.scatter(X_e[:, 0], X_e[:, 1], c=y_expert, cmap="coolwarm",
                   alpha=0.2, edgecolor="gray", s=30)
        _, selected_indices = select_examples_by_strategy(
            X_e, y_expert, expert_w, expert_centroids,
            n_examples=n_examples, strategy=strategy,
            nome_classe_0="A", nome_classe_1="B",
            random_state=random_state, verbose=False
        )
        ax.scatter(X_e[selected_indices, 0], X_e[selected_indices, 1],
                   c=y_expert[selected_indices], cmap="coolwarm",
                   edgecolor="black", s=200, linewidth=2, marker='*',
                   zorder=5, label=f'Selecionados ({len(selected_indices)})')
        sel_margins = confidences_all[selected_indices]
        ax.set_title(f'Estratégia: {strategy.upper()}\n'
                     f'Margem média: {sel_margins.mean():.3f} '
                     f'[{sel_margins.min():.3f}, {sel_margins.max():.3f}]',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    # Figura combinada
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, strategy in zip(axes.flatten(), strategies):
        _draw_strategy(ax, strategy)
    plt.suptitle(f'Fase E: Estratégias de Seleção de Exemplos ({n_examples} exemplos)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        (s, lambda ax, strat=s: _draw_strategy(ax, strat), (7, 5))
        for s in strategies
    ]
    _save_panels_individually(panels, filename)


def plot_w_distribution(resultados: List[ResultadoExperimento], filename: str = None):
    """Distribuição do vetor W estimado entre sementes e repetições (pedido do orientador)."""
    data = []
    for r in resultados:
        if r.n_shot == 0:  # W é aprendido apenas no zero-shot (Fase A)
            data.append({
                'seed': r.random_seed,
                'w0': r.w_aprendido[0],
                'w1': r.w_aprendido[1],
                'ratio': r.w_aprendido[0] / r.w_aprendido[1] if r.w_aprendido[1] != 0 else float('inf'),
                'nomes': f"{r.nomes_classes[0]}/{r.nomes_classes[1]}",
                'rep': r.repeticao,
            })

    if not data:
        print("  Sem dados de W para plotar distribuição.")
        return

    df = pd.DataFrame(data)
    seeds = sorted(df['seed'].unique())
    colors_seed = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    seeds_str = [str(s) for s in seeds]
    w0_by_seed = [df[df['seed'] == s]['w0'].values for s in seeds]
    w1_by_seed = [df[df['seed'] == s]['w1'].values for s in seeds]
    x_seeds = np.arange(len(seeds))
    width = 0.35
    df_ratio = df.groupby('seed').agg({'ratio': ['mean', 'std']}).reset_index()
    df_ratio.columns = ['seed', 'mean', 'std']

    def _draw_w_scatter(ax):
        for i, seed in enumerate(seeds):
            subset = df[df['seed'] == seed]
            ax.scatter(subset['w0'], subset['w1'], c=[colors_seed[i]], s=100,
                       edgecolor='black', label=f'Semente {seed}', zorder=3)
        ax.set_xlabel('$w_1$ (peso da dimensão $x_1$)')
        ax.set_ylabel('$w_2$ (peso da dimensão $x_2$)')
        ax.set_title('Ŵ_LLM Estimado por Semente', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    def _draw_w_boxplot(ax):
        from matplotlib.patches import Patch
        bp1 = ax.boxplot(w0_by_seed, positions=x_seeds - width/2, widths=width*0.8, patch_artist=True)
        bp2 = ax.boxplot(w1_by_seed, positions=x_seeds + width/2, widths=width*0.8, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('steelblue'); patch.set_alpha(0.7)
        for patch in bp2['boxes']:
            patch.set_facecolor('coral'); patch.set_alpha(0.7)
        ax.set_xticks(x_seeds); ax.set_xticklabels(seeds_str)
        ax.set_xlabel('Semente Aleatória'); ax.set_ylabel('Valor do Peso')
        ax.set_title('Distribuição de $w_1$ e $w_2$ por Semente', fontweight='bold')
        ax.legend(handles=[Patch(facecolor='steelblue', alpha=0.7, label='$w_1$'),
                           Patch(facecolor='coral', alpha=0.7, label='$w_2$')])

    def _draw_w_ratio(ax):
        ax.bar(range(len(df_ratio)), df_ratio['mean'], yerr=df_ratio['std'],
               color='mediumpurple', edgecolor='black', capsize=5, alpha=0.8)
        ax.set_xticks(range(len(df_ratio)))
        ax.set_xticklabels([str(int(s)) for s in df_ratio['seed']])
        ax.set_xlabel('Semente Aleatória'); ax.set_ylabel('Razão $w_1/w_2$')
        ax.set_title('Razão $w_1/w_2$ por Semente', fontweight='bold')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Euclidiana (razão=1)')
        ax.legend()

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _draw_w_scatter(axes[0])
    _draw_w_boxplot(axes[1])
    _draw_w_ratio(axes[2])
    plt.suptitle('Distribuição do Vetor Ŵ_LLM Estimado entre Sementes e Repetições',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("scatter", _draw_w_scatter, (7, 5)),
        ("boxplot", _draw_w_boxplot, (7, 5)),
        ("ratio", _draw_w_ratio, (7, 5)),
    ]
    _save_panels_individually(panels, filename)


def plot_metric_errors_phase_a(
    X: np.ndarray, y_llm: np.ndarray, y_metric: np.ndarray,
    w: np.ndarray, centroids: np.ndarray, filename: str = None
):
    """Visualiza onde a métrica erra vs. a rotulação do LLM na Fase A (pedido do orientador)."""
    agreements = y_llm == y_metric
    disagreements = ~agreements
    fidelity = np.mean(agreements)

    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_with_metric(grid_points, centroids, w)
    Z = Z.reshape(xx.shape)

    confidences, _ = compute_metric_confidence(X, centroids, w)
    bins = np.linspace(0, confidences.max(), 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    error_rates = []
    for i in range(len(bins) - 1):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        error_rates.append(np.mean(disagreements[mask]) if np.sum(mask) > 0 else 0)

    def _draw_error_map(ax):
        ax.contourf(xx, yy, Z, alpha=0.15, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0.5])
        ax.scatter(X[agreements, 0], X[agreements, 1], c=y_llm[agreements],
                   cmap="coolwarm", alpha=0.5, edgecolor="gray", s=40, label=f'Concordam ({np.sum(agreements)})')
        ax.scatter(X[disagreements, 0], X[disagreements, 1],
                   c='yellow', edgecolor="red", s=150, linewidth=2, marker='X',
                   label=f'Discordam ({np.sum(disagreements)})', zorder=5)
        ax.scatter(*centroids[0], marker='D', s=200, c='blue', edgecolor='k', linewidth=2, zorder=6)
        ax.scatter(*centroids[1], marker='D', s=200, c='red', edgecolor='k', linewidth=2, zorder=6)
        ax.set_title(f'Fase A: Ŵ_LLM vs. Rotulação do LLM\nFidelidade: {fidelity:.1%}', fontweight='bold')
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

    def _draw_error_by_margin(ax):
        colors_bar = ['#e74c3c' if r > 0.3 else '#f39c12' if r > 0.1 else '#2ecc71' for r in error_rates]
        ax.bar(range(len(error_rates)), error_rates, color=colors_bar, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(error_rates)))
        ax.set_xticklabels([f'{b:.1f}' for b in bin_centers], rotation=45)
        ax.set_xlabel('Margem (distância à fronteira)')
        ax.set_ylabel('Taxa de Erro')
        ax.set_title('Taxa de Discordância por Faixa de Margem', fontweight='bold')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

    # Figura combinada
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _draw_error_map(axes[0])
    _draw_error_by_margin(axes[1])
    plt.suptitle('Análise de Erros da Métrica Ŵ_LLM na Fase A', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("mapa_erros", _draw_error_map, (7, 6)),
        ("erros_por_margem", _draw_error_by_margin, (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_class_order_bias(resultados: List[ResultadoExperimento], filename: str = None):
    """Compara resultados entre classes originais e invertidas para detectar viés de posição."""
    data = []
    for r in resultados:
        nome_pair = f"{r.nomes_classes[0]}/{r.nomes_classes[1]}"
        data.append({
            'pair': nome_pair,
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
            'fidelidade': r.fidelidade_problema_a,
        })

    if not data:
        return

    df = pd.DataFrame(data)
    df_grouped = df.groupby('pair').agg({
        'consistencia_b': ['mean', 'std'],
        'consistencia_c': ['mean', 'std'],
        'fidelidade': ['mean', 'std'],
    }).reset_index()
    df_grouped.columns = ['pair', 'b_mean', 'b_std', 'c_mean', 'c_std', 'fid_mean', 'fid_std']

    x = np.arange(len(df_grouped))
    width = 0.35

    def _draw_consistency_bias(ax):
        ax.bar(x - width/2, df_grouped['b_mean'], width, yerr=df_grouped['b_std'],
               label='Problema B', color='steelblue', capsize=3, edgecolor='black')
        ax.bar(x + width/2, df_grouped['c_mean'], width, yerr=df_grouped['c_std'],
               label='Problema C', color='coral', capsize=3, edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels(df_grouped['pair'], rotation=45, ha='right')
        ax.set_ylabel('Consistência')
        ax.set_title('Consistência por Par de Classes\n(inclui pares invertidos)', fontweight='bold')
        ax.legend(); ax.set_ylim(0, 1.1)

    def _draw_fidelity_bias(ax):
        ax.bar(x, df_grouped['fid_mean'], yerr=df_grouped['fid_std'],
               color='mediumpurple', capsize=3, edgecolor='black', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(df_grouped['pair'], rotation=45, ha='right')
        ax.set_ylabel('Fidelidade Fase A')
        ax.set_title('Fidelidade da Métrica por Par de Classes', fontweight='bold')
        ax.set_ylim(0, 1.1)

    # Figura combinada
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _draw_consistency_bias(axes[0])
    _draw_fidelity_bias(axes[1])
    plt.suptitle('Análise de Viés de Ordem/Posição das Classes', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("consistencia", _draw_consistency_bias, (7, 6)),
        ("fidelidade", _draw_fidelity_bias, (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_feature_names_effect(resultados: List[ResultadoExperimento], filename: str = None):
    """Compara métricas e pesos aprendidos entre diferentes nomes de features."""
    # Filtra resultados que têm feature_names não-padrão ou padrão para comparação
    feat_results = [r for r in resultados if r.feature_names != ("x1", "x2")
                    or (r.n_shot == 0 and r.nomes_classes == ("A", "B") and r.prompt_variant == "default")]

    if len(feat_results) < 2:
        return

    data = []
    for r in feat_results:
        label = f"{r.feature_names[0]}/{r.feature_names[1]}"
        data.append({
            'features': label,
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
            'fidelidade': r.fidelidade_problema_a,
            'w0': r.w_aprendido[0] if r.w_aprendido is not None else np.nan,
            'w1': r.w_aprendido[1] if r.w_aprendido is not None else np.nan,
        })

    if not data:
        return

    df = pd.DataFrame(data)
    df_grouped = df.groupby('features').agg({
        'consistencia_b': ['mean', 'std'],
        'consistencia_c': ['mean', 'std'],
        'fidelidade': ['mean', 'std'],
        'w0': ['mean', 'std'],
        'w1': ['mean', 'std'],
    }).reset_index()
    df_grouped.columns = ['features', 'b_mean', 'b_std', 'c_mean', 'c_std',
                          'fid_mean', 'fid_std', 'w0_mean', 'w0_std', 'w1_mean', 'w1_std']

    x = np.arange(len(df_grouped))
    width = 0.35

    def _draw_feat_consistency(ax):
        ax.bar(x - width/2, df_grouped['b_mean'], width, yerr=df_grouped['b_std'],
               label='Problema B', color='steelblue', capsize=3, edgecolor='black')
        ax.bar(x + width/2, df_grouped['c_mean'], width, yerr=df_grouped['c_std'],
               label='Problema C', color='coral', capsize=3, edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels(df_grouped['features'], rotation=30, ha='right')
        ax.set_ylabel('Consistência')
        ax.set_title('Consistência por Nome de Feature', fontweight='bold')
        ax.legend(); ax.set_ylim(0, 1.1)

    def _draw_feat_fidelity(ax):
        ax.bar(x, df_grouped['fid_mean'], yerr=df_grouped['fid_std'],
               color='mediumpurple', capsize=3, edgecolor='black', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(df_grouped['features'], rotation=30, ha='right')
        ax.set_ylabel('Fidelidade Fase A')
        ax.set_title('Fidelidade da Métrica por Nome de Feature', fontweight='bold')
        ax.set_ylim(0, 1.1)

    def _draw_feat_weights(ax):
        ax.bar(x - width/2, df_grouped['w0_mean'], width, yerr=df_grouped['w0_std'],
               label='w[0]', color='forestgreen', capsize=3, edgecolor='black')
        ax.bar(x + width/2, df_grouped['w1_mean'], width, yerr=df_grouped['w1_std'],
               label='w[1]', color='darkorange', capsize=3, edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels(df_grouped['features'], rotation=30, ha='right')
        ax.set_ylabel('Peso W')
        ax.set_title('Pesos Aprendidos por Nome de Feature', fontweight='bold')
        ax.legend()

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _draw_feat_consistency(axes[0])
    _draw_feat_fidelity(axes[1])
    _draw_feat_weights(axes[2])
    plt.suptitle('Efeito dos Nomes Semânticos de Features na Métrica Estimada',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("consistencia", _draw_feat_consistency, (7, 6)),
        ("fidelidade", _draw_feat_fidelity, (7, 6)),
        ("pesos", _draw_feat_weights, (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_classical_baselines_comparison(
    results_e: List[ResultadoPhaseEExperimento],
    results_baselines: List[Dict],
    filename: str = None
):
    """Compara LLM vs. baselines clássicos (k-NN, LR, SVM) na Fase E.

    Mostra curvas de aprendizado do LLM (mixed strategy) junto com os baselines treinados
    nos mesmos exemplos, respondendo: o LLM faz algo que um classificador trivial não faz?
    """
    if not results_e or not results_baselines:
        return

    # Filtra LLM: apenas estratégia mixed (a mais balanceada) para comparação justa
    df_llm = pd.DataFrame([{
        'n_shot': r.n_shot,
        'accuracy': r.accuracy_llm_vs_expert,
        'kappa': r.kappa_llm_vs_expert,
        'f1': r.f1_llm_vs_expert,
        'source': 'LLM (few-shot)',
    } for r in results_e if r.example_strategy == 'mixed'])

    df_bl = pd.DataFrame(results_baselines)
    df_bl = df_bl[df_bl['example_strategy'] == 'mixed']

    if df_llm.empty or df_bl.empty:
        return

    clf_colors = {
        'LLM (few-shot)': '#3498db',
        'k-NN': '#e74c3c',
        'Logistic Regression': '#2ecc71',
        'SVM (RBF)': '#9b59b6',
    }
    clf_markers = {
        'LLM (few-shot)': 'D',
        'k-NN': 'o',
        'Logistic Regression': 's',
        'SVM (RBF)': '^',
    }
    clf_names = df_bl['model'].unique()

    metrics_config = [
        ('accuracy', 'accuracy_vs_expert', 'Concordância vs. Perito'),
        ('kappa', 'kappa_vs_expert', 'Kappa de Cohen'),
        ('f1', 'f1_vs_expert', 'F1-Score'),
    ]

    def _draw_baseline_metric(ax, llm_col, bl_col, title):
        df_llm_grouped = df_llm.groupby('n_shot').agg({
            llm_col: ['mean', 'std']
        }).reset_index()
        df_llm_grouped.columns = ['n_shot', 'mean', 'std']
        df_llm_grouped = df_llm_grouped.sort_values('n_shot')
        ax.errorbar(
            df_llm_grouped['n_shot'], df_llm_grouped['mean'],
            yerr=df_llm_grouped['std'],
            marker=clf_markers['LLM (few-shot)'],
            color=clf_colors['LLM (few-shot)'],
            label='LLM (few-shot)',
            linewidth=2.5, markersize=9, capsize=4
        )
        for clf_name in clf_names:
            df_clf = df_bl[df_bl['model'] == clf_name]
            df_clf_grouped = df_clf.groupby('n_shot').agg({
                bl_col: ['mean', 'std']
            }).reset_index()
            df_clf_grouped.columns = ['n_shot', 'mean', 'std']
            df_clf_grouped = df_clf_grouped.sort_values('n_shot')
            color_key = clf_name
            for k in clf_colors:
                if k in clf_name or clf_name.startswith(k.split(' ')[0]):
                    color_key = k
                    break
            ax.errorbar(
                df_clf_grouped['n_shot'], df_clf_grouped['mean'],
                yerr=df_clf_grouped['std'],
                marker=clf_markers.get(color_key, 'x'),
                color=clf_colors.get(color_key, 'gray'),
                label=clf_name,
                linewidth=1.5, markersize=7, capsize=3,
                linestyle='--', alpha=0.8
            )
        ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.4, label='Chance')
        ax.set_xlabel('Número de Exemplos', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (llm_col, bl_col, title) in enumerate(metrics_config):
        _draw_baseline_metric(axes[i], llm_col, bl_col, title)
    plt.suptitle('LLM vs. Baselines Clássicos (mesmos exemplos few-shot, estratégia mixed)\n'
                 'O LLM faz algo diferente de um classificador trivial?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    suffixes = ["accuracy", "kappa", "f1"]
    panels = [
        (suffixes[i], lambda ax, mc=metrics_config[i]: _draw_baseline_metric(ax, *mc), (7, 6))
        for i in range(3)
    ]
    _save_panels_individually(panels, filename)


def plot_prompt_variant_comparison(resultados: List[ResultadoExperimento], filename: str = None):
    """Compara métricas e W aprendidos entre diferentes variantes de prompt.

    Testa se o prompt confunde a medição de consistência do LLM.
    """
    # Filtra resultados relevantes: classes A/B, n_shot=0, com variantes
    relevant = [r for r in resultados
                if r.nomes_classes == ("A", "B") and r.n_shot == 0]
    if not relevant:
        return

    df = pd.DataFrame([{
        'variant': r.prompt_variant,
        'fidelidade': r.fidelidade_problema_a,
        'consistencia_b': r.consistencia_problema_b,
        'consistencia_c': r.consistencia_problema_c,
        'kappa_b': r.kappa_problema_b,
        'kappa_c': r.kappa_problema_c,
        'w_0': r.w_aprendido[0],
        'w_1': r.w_aprendido[1],
    } for r in relevant])

    variants = sorted(df['variant'].unique())
    if len(variants) < 2:
        return

    colors = {
        'default': '#3498db',
        'geometric': '#2ecc71',
        'cot': '#e74c3c',
        'tabular': '#9b59b6',
    }
    labels_map = {
        'default': 'Default',
        'geometric': 'Geométrico',
        'cot': 'Chain-of-Thought',
        'tabular': 'Tabular',
    }

    x = np.arange(len(variants))
    width = 0.35
    df_grouped_pv = df.groupby('variant').agg({
        'consistencia_b': ['mean', 'std'], 'consistencia_c': ['mean', 'std'],
    }).reset_index()
    df_grouped_pv.columns = ['variant', 'b_mean', 'b_std', 'c_mean', 'c_std']
    df_grouped_pv = df_grouped_pv.set_index('variant').loc[variants].reset_index()

    df_fid = df.groupby('variant').agg({'fidelidade': ['mean', 'std']}).reset_index()
    df_fid.columns = ['variant', 'fid_mean', 'fid_std']
    df_fid = df_fid.set_index('variant').loc[variants].reset_index()
    bar_colors = [colors.get(v, 'gray') for v in variants]

    def _draw_pv_consistency(ax):
        ax.bar(x - width/2, df_grouped_pv['b_mean'], width, yerr=df_grouped_pv['b_std'],
               label='Problema B', color='steelblue', capsize=3, edgecolor='black')
        ax.bar(x + width/2, df_grouped_pv['c_mean'], width, yerr=df_grouped_pv['c_std'],
               label='Problema C', color='coral', capsize=3, edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels([labels_map.get(v, v) for v in variants], rotation=30, ha='right')
        ax.set_ylabel('Consistência')
        ax.set_title('Consistência por Variante de Prompt', fontweight='bold')
        ax.legend(); ax.set_ylim(0, 1.1)

    def _draw_pv_fidelity(ax):
        ax.bar(x, df_fid['fid_mean'], yerr=df_fid['fid_std'],
               color=bar_colors, capsize=3, edgecolor='black', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([labels_map.get(v, v) for v in variants], rotation=30, ha='right')
        ax.set_ylabel('Fidelidade Fase A')
        ax.set_title('Fidelidade da Métrica por Variante', fontweight='bold')
        ax.set_ylim(0, 1.1)

    def _draw_pv_w_scatter(ax):
        for variant in variants:
            df_v = df[df['variant'] == variant]
            ax.scatter(df_v['w_0'], df_v['w_1'], c=colors.get(variant, 'gray'),
                       label=labels_map.get(variant, variant),
                       s=80, edgecolors='black', linewidths=0.5, alpha=0.8)
        ax.set_xlabel('w[0] (peso x1)'); ax.set_ylabel('w[1] (peso x2)')
        ax.set_title('Métrica W Estimada por Variante\n(estabilidade entre prompts)', fontweight='bold')
        ax.legend(fontsize=8); ax.set_aspect('equal', adjustable='datalim'); ax.grid(True, alpha=0.3)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    _draw_pv_consistency(axes[0])
    _draw_pv_fidelity(axes[1])
    _draw_pv_w_scatter(axes[2])
    plt.suptitle('Sensibilidade ao Prompt: Mesmos dados, diferentes templates\n'
                 '(Se W muda, o prompt confunde a medição)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("consistencia", _draw_pv_consistency, (7, 6)),
        ("fidelidade", _draw_pv_fidelity, (7, 6)),
        ("w_scatter", _draw_pv_w_scatter, (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_example_order_bias(results_order: List[ResultadoPhaseEExperimento], filename: str = None):
    """Compara métricas entre diferentes ordenações dos exemplos few-shot.

    Detecta viés de recência: se a ordem dos exemplos afeta a performance do LLM.
    """
    if not results_order:
        return

    df = pd.DataFrame([{
        'n_shot': r.n_shot,
        'ordering': r.example_strategy.replace("mixed_order_", ""),
        'accuracy': r.accuracy_llm_vs_expert,
        'kappa': r.kappa_llm_vs_expert,
        'f1': r.f1_llm_vs_expert,
    } for r in results_order])

    n_shots = sorted(df['n_shot'].unique())
    orderings = df['ordering'].unique()

    order_colors = {
        'class0_first': '#3498db',
        'class1_first': '#e74c3c',
        'shuffled': '#2ecc71',
        'alternating': '#9b59b6',
    }
    order_labels = {
        'class0_first': 'Classe 0 primeiro',
        'class1_first': 'Classe 1 primeiro',
        'shuffled': 'Aleatório',
        'alternating': 'Alternado',
    }

    metrics_order = [
        ('accuracy', 'Concordância LLM vs. Perito'),
        ('kappa', 'Kappa de Cohen'),
        ('f1', 'F1-Score'),
    ]

    def _draw_order_metric(ax, metric, metric_name, show_legend=False):
        x = np.arange(len(n_shots))
        width = 0.18
        n_orderings = len(orderings)
        for i, ordering in enumerate(orderings):
            df_ord = df[df['ordering'] == ordering]
            means, stds = [], []
            for ns in n_shots:
                vals = df_ord[df_ord['n_shot'] == ns][metric]
                means.append(vals.mean()); stds.append(vals.std())
            offset = (i - n_orderings / 2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   label=order_labels.get(ordering, ordering),
                   color=order_colors.get(ordering, f'C{i}'),
                   capsize=3, edgecolor='black', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([f'{ns}-shot' for ns in n_shots])
        ax.set_ylabel(metric_name); ax.set_title(metric_name, fontweight='bold')
        if metric in ('accuracy', 'f1'):
            ax.set_ylim(0, 1.1)
        elif metric == 'kappa':
            ax.set_ylim(-0.1, 1.1)
        if show_legend:
            ax.legend(fontsize=8)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax_idx, (metric, metric_name) in enumerate(metrics_order):
        _draw_order_metric(axes[ax_idx], metric, metric_name, show_legend=(ax_idx == 0))
    plt.suptitle('Viés de Ordem dos Exemplos Few-Shot (Recency Bias)\n'
                 'Mesmos exemplos (mixed), diferentes ordenações',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    suffixes = ["accuracy", "kappa", "f1"]
    panels = [
        (suffixes[i], lambda ax, m=metrics_order[i]: _draw_order_metric(ax, m[0], m[1], show_legend=True), (7, 6))
        for i in range(3)
    ]
    _save_panels_individually(panels, filename)


def plot_dilution_experiment(results_dilution: List[ResultadoPhaseEExperimento], filename: str = None):
    """Gráfico do experimento de diluição: 3 hard fixos + N easy progressivos."""
    if not results_dilution:
        return

    df = pd.DataFrame([{
        'n_shot': r.n_shot,
        'accuracy': r.accuracy_llm_vs_expert,
        'kappa': r.kappa_llm_vs_expert,
        'strategy': r.example_strategy,
    } for r in results_dilution])

    df_acc = df.groupby('n_shot').agg({'accuracy': ['mean', 'std']}).reset_index()
    df_acc.columns = ['n_shot', 'mean', 'std']
    df_acc = df_acc.sort_values('n_shot')
    ref_3hard = df[df['n_shot'] == 3]

    df_kappa = df.groupby('n_shot').agg({'kappa': ['mean', 'std']}).reset_index()
    df_kappa.columns = ['n_shot', 'mean', 'std']
    df_kappa = df_kappa.sort_values('n_shot')

    def _draw_dilution_accuracy(ax):
        ax.errorbar(df_acc['n_shot'], df_acc['mean'], yerr=df_acc['std'],
                    marker='o', linewidth=2, markersize=8, capsize=4, color='#e74c3c')
        if len(ref_3hard) > 0:
            ax.axhline(y=ref_3hard['accuracy'].mean(), color='gray', linestyle='--',
                       alpha=0.7, label=f'3 hard puros ({ref_3hard["accuracy"].mean():.1%})')
        ax.set_xlabel('Total de Exemplos (3 hard + N easy)')
        ax.set_ylabel('Concordância LLM vs. Perito')
        ax.set_title('Experimento de Diluição: Acurácia', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

    def _draw_dilution_kappa(ax):
        ax.errorbar(df_kappa['n_shot'], df_kappa['mean'], yerr=df_kappa['std'],
                    marker='s', linewidth=2, markersize=8, capsize=4, color='#3498db')
        ax.set_xlabel('Total de Exemplos (3 hard + N easy)')
        ax.set_ylabel('Kappa de Cohen')
        ax.set_title('Experimento de Diluição: Kappa', fontweight='bold')
        ax.grid(True, alpha=0.3); ax.set_ylim(-0.1, 1.05)

    # Figura combinada
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_dilution_accuracy(axes[0])
    _draw_dilution_kappa(axes[1])
    plt.suptitle('Experimento de Diluição: Hard Fixos + Easy Progressivos', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("accuracy", _draw_dilution_accuracy, (7, 5)),
        ("kappa", _draw_dilution_kappa, (7, 5)),
    ]
    _save_panels_individually(panels, filename)


def plot_r3_comparison(results_r2: List, results_r3: List, filename: str = None):
    """Compara fidelidade com 2 pesos vs 3 pesos (projeção R3) usando as mesmas
    classificações do LLM (que viu apenas x1, x2). Se a fidelidade R3 > R2,
    há evidência de não-linearidade implícita no processo decisório do LLM."""
    if not results_r2 or not results_r3:
        return

    r2_means = np.mean([r['accuracy'] for r in results_r2])
    r2_stds = np.std([r['accuracy'] for r in results_r2])
    r3_means = np.mean([r['accuracy'] for r in results_r3])
    r3_stds = np.std([r['accuracy'] for r in results_r3])
    has_nnls = any('accuracy_nnls' in r for r in results_r3)

    algo_data = [('Perceptron', [r['accuracy'] for r in results_r3], 'steelblue')]
    if has_nnls:
        algo_data.append(('NNLS', [r.get('accuracy_nnls', np.nan) for r in results_r3], 'coral'))

    def _draw_r3_comparison(ax):
        labels_r3 = ['2 pesos\n($w_1$, $w_2$)', '3 pesos\n($w_1$, $w_2$, $w_3$)\n$x_3 = x_1 \\cdot x_2$']
        means = [r2_means, r3_means]
        stds = [r2_stds, r3_stds]
        bars = ax.bar(labels_r3, means, yerr=stds, color=['steelblue', 'coral'],
                      edgecolor='black', capsize=5, alpha=0.8)
        ax.set_ylabel('Fidelidade (métrica vs. LLM)')
        ax.set_title('Perceptron: Métrica Linear vs. Quadrática', fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{mean:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        delta = means[1] - means[0]
        ax.text(0.5, 0.05, f'$\\Delta$ = {delta:+.1%}', ha='center', va='bottom',
                transform=ax.transAxes, fontsize=10, style='italic',
                color='green' if delta > 0 else 'gray')

    def _draw_r3_algo(ax):
        x_pos = np.arange(len(algo_data))
        for i, (name, accs, color) in enumerate(algo_data):
            accs_clean = [a for a in accs if not np.isnan(a)]
            m = np.mean(accs_clean) if accs_clean else 0
            s = np.std(accs_clean) if len(accs_clean) > 1 else 0
            ax.bar(i, m, yerr=s, color=color, edgecolor='black', capsize=5, alpha=0.8)
            ax.text(i, m + 0.02, f'{m:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos); ax.set_xticklabels([d[0] for d in algo_data])
        ax.set_ylabel('Fidelidade R3 (3 pesos)')
        ax.set_title('R3: Fidelidade por Algoritmo', fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

    # Figura combinada
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_r3_comparison(axes[0])
    _draw_r3_algo(axes[1])
    plt.suptitle('Não-linearidade Implícita: Fidelidade com 2 vs 3 pesos\n'
                 '(LLM viu apenas $x_1$, $x_2$ — $x_3 = x_1 \\cdot x_2$ adicionado nos bastidores)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("linear_vs_quadratica", _draw_r3_comparison, (7, 5)),
        ("algoritmos_r3", _draw_r3_algo, (7, 5)),
    ]
    _save_panels_individually(panels, filename)


def plot_algorithm_comparison(results_perceptron: List, results_alternative: List,
                              filename: str = None):
    """Compara W aprendido por Perceptron vs NNLS em todas as fases (A, B, C)."""
    if not results_perceptron or not results_alternative:
        return

    bar_width = 0.25
    w_perc = np.array([[r.w_aprendido[0], r.w_aprendido[1]] for r in results_perceptron])
    w_alt = np.array([[r.w_aprendido[0], r.w_aprendido[1]] for r in results_alternative])
    fid_perc = [r.fidelidade_problema_a for r in results_perceptron]
    fid_alt = [r.fidelidade_problema_a for r in results_alternative]

    algo_list = [
        ('Perceptron', results_perceptron, 'steelblue'),
        ('NNLS', results_alternative, 'coral'),
    ]

    def _draw_algo_w_scatter(ax):
        ax.scatter(w_perc[:, 0], w_perc[:, 1], c='steelblue', s=100, edgecolor='k', label='Perceptron', zorder=3)
        ax.scatter(w_alt[:, 0], w_alt[:, 1], c='coral', s=100, edgecolor='k', label='NNLS', zorder=3)
        ax.set_xlabel('$w_1$'); ax.set_ylabel('$w_2$')
        ax.set_title('Ŵ_LLM: Perceptron vs. NNLS', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

    def _draw_algo_fidelity(ax):
        box_data = [fid_perc, fid_alt]
        box_labels = ['Perceptron', 'NNLS']
        box_colors = ['steelblue', 'coral']
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        for i, color in enumerate(box_colors):
            bp['boxes'][i].set_facecolor(color); bp['boxes'][i].set_alpha(0.7)
        ax.set_ylabel('Fidelidade')
        ax.set_title('Fase A: Fidelidade (Métrica vs. LLM)', fontweight='bold')
        ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3, axis='y')

    def _draw_algo_phase(ax, phase_letter, cons_attr, kappa_attr):
        x_pos = np.array([0, 1])
        n_algos_local = len(algo_list)
        for j, (name, res, color) in enumerate(algo_list):
            cons_vals = [getattr(r, cons_attr) for r in res]
            kappa_vals = [getattr(r, kappa_attr) for r in res]
            means = [np.mean(cons_vals), np.mean(kappa_vals)]
            stds = [np.std(cons_vals) if len(cons_vals) > 1 else 0,
                    np.std(kappa_vals) if len(kappa_vals) > 1 else 0]
            offset = (j - (n_algos_local - 1) / 2) * bar_width
            ax.bar(x_pos + offset, means, bar_width, yerr=stds,
                   color=color, alpha=0.7, edgecolor='k', capsize=4, label=name)
        ax.set_xticks(x_pos); ax.set_xticklabels(['Consistência', 'Kappa'])
        ax.set_title(f'Fase {phase_letter}: Consistência no Problema {phase_letter}', fontweight='bold')
        ax.set_ylim(-0.1, 1.1); ax.grid(True, alpha=0.3, axis='y'); ax.legend(fontsize=8)

    # Figura combinada
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _draw_algo_w_scatter(axes[0, 0])
    _draw_algo_fidelity(axes[0, 1])
    _draw_algo_phase(axes[1, 0], 'B', 'consistencia_problema_b', 'kappa_problema_b')
    _draw_algo_phase(axes[1, 1], 'C', 'consistencia_problema_c', 'kappa_problema_c')
    plt.suptitle('Comparação de Algoritmos de Otimização Inversa\n(Fases A, B e C)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("w_scatter", _draw_algo_w_scatter, (7, 6)),
        ("fidelidade", _draw_algo_fidelity, (7, 6)),
        ("fase_b", lambda ax: _draw_algo_phase(ax, 'B', 'consistencia_problema_b', 'kappa_problema_b'), (7, 6)),
        ("fase_c", lambda ax: _draw_algo_phase(ax, 'C', 'consistencia_problema_c', 'kappa_problema_c'), (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_gamma_convergence(diagnostics: List[dict], filename: str = None):
    """Diagnóstico da busca binária em γ no Perceptron Estruturado.

    Item b da reunião 30/04/2026 (~520s): orientador pediu para verificar se
    γ converge crescentemente entre execuções (não fica estagnado). Cada entrada
    de `diagnostics` é {"label": str, "gamma_final": float, "history": [dicts]}.
    Cada dict do history: {iter, gamma, gamma_lo, gamma_hi, violations, viable}.

    Painel 1: evolução de γ por iteração (curva por execução).
    Painel 2: γ_lo (limite inferior viável) por iteração — deve ser monotônico.
    """
    if not diagnostics:
        print("  ⚠ Sem histórico de γ para plotar (nenhuma chamada com return_history=True).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("tab10")

    for idx, entry in enumerate(diagnostics):
        hist = entry.get("history") or []
        if not hist:
            continue
        iters = [h["iter"] for h in hist]
        gammas = [h["gamma"] for h in hist]
        gamma_los = [h["gamma_lo"] for h in hist]
        viable = [h["viable"] for h in hist]

        color = cmap(idx % 10)
        label = f"{entry.get('label', f'exec_{idx}')} (γ*={entry.get('gamma_final', float('nan')):.3f})"

        # Painel 1: γ testado por iteração — marcadores diferentes para viable/inviable
        axes[0].plot(iters, gammas, "-", color=color, alpha=0.5, linewidth=1.0)
        v_iters = [i for i, v in zip(iters, viable) if v]
        v_gammas = [g for g, v in zip(gammas, viable) if v]
        nv_iters = [i for i, v in zip(iters, viable) if not v]
        nv_gammas = [g for g, v in zip(gammas, viable) if not v]
        axes[0].scatter(v_iters, v_gammas, s=40, marker="o", color=color, label=label,
                        edgecolor="black", linewidth=0.4)
        axes[0].scatter(nv_iters, nv_gammas, s=40, marker="x", color=color)

        # Painel 2: γ_lo (melhor margem viável até o momento) — DEVE ser monotônico crescente
        axes[1].plot(iters, gamma_los, "-o", color=color, alpha=0.8, markersize=4,
                     label=label)

    axes[0].set_xlabel("Iteração da busca binária")
    axes[0].set_ylabel("γ testado")
    axes[0].set_title("γ candidato por iteração\n(○ = viável, × = inviável)", fontsize=10, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")

    axes[1].set_xlabel("Iteração da busca binária")
    axes[1].set_ylabel("γ_lo (melhor margem viável)")
    axes[1].set_title("γ_lo monotônico (orientador, item b ~520s)", fontsize=10, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")

    plt.suptitle("Convergência da busca binária em γ — Perceptron Estruturado (CILAMCE 2017, Eq. 31)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  Imagem salva: {os.path.basename(filename)}")
    plt.close(fig)

    # Verificação textual: γ_lo deve ser monotônico não-decrescente em cada execução
    print("\n  Diagnóstico da busca binária em γ:")
    for entry in diagnostics:
        hist = entry.get("history") or []
        if not hist:
            continue
        gamma_los = [h["gamma_lo"] for h in hist]
        monotone = all(gamma_los[i] <= gamma_los[i + 1] for i in range(len(gamma_los) - 1))
        status = "✓ monotônico" if monotone else "⚠ NÃO monotônico"
        print(f"    {entry.get('label', '?'):40s}  iters={len(hist):3d}  γ*={entry.get('gamma_final', float('nan')):.4f}  {status}")


def plot_oracle_w_recovery(oracle_results: List[dict], filename: str = None):
    """Visualiza a recuperação de W conhecido pelos algoritmos (validação do oráculo)."""
    if not oracle_results:
        return

    from matplotlib.lines import Line2D

    colors_algo = {'perceptron': 'steelblue', 'nnls': 'coral'}
    labels = list(dict.fromkeys(f"{r['problem']}\n{r['expert_name']}" for r in oracle_results))
    label_results = {}
    for r in oracle_results:
        key = f"{r['problem']}\n{r['expert_name']}"
        algo = r['algorithm']
        label_results.setdefault(key, {})[algo] = r
    n_labels = len(labels)
    x_base = np.arange(n_labels)

    def _draw_w_scatter(ax):
        for r in oracle_results:
            cl = colors_algo[r['algorithm']]
            true_norm = np.array([r['true_w_0'], r['true_w_1']])
            true_norm = true_norm / np.sum(true_norm)
            ax.scatter(true_norm[0], r['recovered_w_norm_0'], c=cl, s=80,
                       edgecolor='k', linewidth=0.5, zorder=3, alpha=0.7)
            ax.scatter(true_norm[1], r['recovered_w_norm_1'], c=cl, s=80,
                       edgecolor='k', linewidth=0.5, zorder=3, alpha=0.7)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Recuperação perfeita')
        ax.set_xlabel('$w$ verdadeiro (normalizado)')
        ax.set_ylabel('$w$ recuperado (normalizado)')
        ax.set_title('W Normalizado: Verdadeiro vs Recuperado', fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='PERCEPTRON'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='coral', markersize=8, label='NNLS'),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

    def _draw_ratio(ax):
        bar_width = 0.2
        max_ratio_for_plot = 10.0
        for i, label in enumerate(labels):
            algos = label_results[label]
            first = list(algos.values())[0]
            true_ratio = min(first['true_w_ratio'], max_ratio_for_plot)
            ax.bar(x_base[i] - bar_width, true_ratio, bar_width, color='gray',
                   edgecolor='k', alpha=0.7, label='Verdadeiro' if i == 0 else '')
            for j, algo in enumerate(['perceptron', 'nnls']):
                if algo in algos:
                    ratio = algos[algo]['recovered_w_ratio']
                    if ratio != float('inf'):
                        ratio_plot = min(ratio, max_ratio_for_plot)
                        ax.bar(x_base[i] + bar_width * j, ratio_plot, bar_width,
                               color=colors_algo[algo], edgecolor='k', alpha=0.7,
                               label=algo.upper() if i == 0 else '')
        ax.set_xticks(x_base)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('Razão $w_1 / w_2$')
        ax.set_title('Razão $w_1/w_2$: Verdadeira vs Recuperada', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    def _draw_cosine(ax):
        bar_width = 0.3
        for i, label in enumerate(labels):
            algos = label_results[label]
            for j, algo in enumerate(['perceptron', 'nnls']):
                if algo in algos:
                    ax.bar(x_base[i] + bar_width * (j - 0.5), algos[algo]['cosine_similarity'],
                           bar_width, color=colors_algo[algo], edgecolor='k', alpha=0.7,
                           label=algo.upper() if i == 0 else '')
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfeito (1.0)')
        ax.set_xticks(x_base)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('Similaridade de Cosseno')
        ax.set_title('Cosseno entre W Verdadeiro e Recuperado', fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    def _draw_fidelity(ax):
        bar_width = 0.3
        for i, label in enumerate(labels):
            algos = label_results[label]
            for j, algo in enumerate(['perceptron', 'nnls']):
                if algo in algos:
                    ax.bar(x_base[i] + bar_width * (j - 0.5), algos[algo]['fidelity'] * 100,
                           bar_width, color=colors_algo[algo], edgecolor='k', alpha=0.7,
                           label=algo.upper() if i == 0 else '')
        ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='Meta 95%')
        ax.set_xticks(x_base)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('Fidelidade (%)')
        ax.set_title('Fidelidade: Rótulos Verdadeiros vs Métrica Recuperada', fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Figura combinada
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _draw_w_scatter(axes[0, 0])
    _draw_ratio(axes[0, 1])
    _draw_cosine(axes[1, 0])
    _draw_fidelity(axes[1, 1])
    plt.suptitle('Validação do Oráculo: Recuperação de W Conhecido\n(Centroides verdadeiros, sem LLM)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("w_scatter", _draw_w_scatter, (7, 7)),
        ("ratio", _draw_ratio, (8, 6)),
        ("cosseno", _draw_cosine, (8, 6)),
        ("fidelidade", _draw_fidelity, (8, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_oracle_transfer(oracle_results: List[dict], filename: str = None):
    """Visualiza a fidelidade por problema e algoritmo (todos os problemas lado a lado)."""
    if not oracle_results:
        return

    colors_algo = {'perceptron': 'steelblue', 'nnls': 'coral'}
    expert_names = list(dict.fromkeys(r['expert_name'] for r in oracle_results))

    def _draw_expert(ax, ename):
        subset = [r for r in oracle_results if r['expert_name'] == ename]
        probs_for_expert = list(dict.fromkeys(r['problem'] for r in subset))
        true_w_str = f"[{subset[0]['true_w_0']}, {subset[0]['true_w_1']}]"
        bar_width = 0.25
        x_pos = np.arange(len(probs_for_expert))
        for j, algo in enumerate(['perceptron', 'nnls']):
            values = []
            for prob in probs_for_expert:
                r_match = [r for r in subset if r['algorithm'] == algo and r['problem'] == prob]
                values.append(r_match[0]['fidelity'] * 100 if r_match else 0)
            bars = ax.bar(x_pos + bar_width * (j - 1), values, bar_width,
                          color=colors_algo[algo], edgecolor='k', alpha=0.7,
                          label=algo.upper())
            for k, v in enumerate(values):
                ax.text(x_pos[k] + bar_width * (j - 1), v + 1, f'{v:.1f}%',
                        ha='center', va='bottom', fontsize=7)
        ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90%')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.replace('Problema ', '') for p in probs_for_expert], fontsize=9)
        ax.set_xlabel('Problema')
        ax.set_ylabel('Fidelidade (%)')
        ax.set_title(f'{ename}\nW = {true_w_str}', fontweight='bold')
        ax.set_ylim(0, 110)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Figura combinada
    fig, axes = plt.subplots(1, len(expert_names), figsize=(5 * len(expert_names), 5), squeeze=False)
    for idx, ename in enumerate(expert_names):
        _draw_expert(axes[0, idx], ename)
    plt.suptitle('Validação do Oráculo: Fidelidade por Problema\n(Centroides verdadeiros, sem LLM)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais por expert
    if filename:
        base, ext = os.path.splitext(filename)
        for ename in expert_names:
            fig_ind, ax_ind = plt.subplots(1, 1, figsize=(6, 5))
            _draw_expert(ax_ind, ename)
            fig_ind.tight_layout()
            fig_ind.savefig(f"{base}_{ename}{ext}", dpi=150, bbox_inches='tight')
            plt.close(fig_ind)


def plot_dataset_overview(data: dict, seed: int, filename: str = None):
    """Visão completa dos 4 datasets: ground truth vs classificação LLM."""
    colors_gt = {0: '#3498db', 1: '#e74c3c'}
    colors_llm = {0: '#2980b9', 1: '#c0392b'}

    problems = [
        ('A', data['X_a'], data['y_gt_a'], data.get('y_llm_a')),
        ('B', data['X_b'], data['y_gt_b'], data.get('y_llm_b')),
        ('C', data['X_c'], data['y_gt_c'], data.get('y_llm_c')),
        ('D', data['X_e'], data['y_gt_e'], None),
    ]

    def _draw_problem(axes_pair, name, X, y_gt, y_llm):
        ax_gt, ax_llm = axes_pair
        for c in [0, 1]:
            mask = y_gt == c
            ax_gt.scatter(X[mask, 0], X[mask, 1], c=colors_gt[c], s=15, alpha=0.6, label=f'Classe {c}')
        centroid_0 = X[y_gt == 0].mean(axis=0)
        centroid_1 = X[y_gt == 1].mean(axis=0)
        ax_gt.scatter(*centroid_0, c='black', marker='X', s=120, zorder=5, edgecolors='white', linewidths=1)
        ax_gt.scatter(*centroid_1, c='black', marker='X', s=120, zorder=5, edgecolors='white', linewidths=1)
        ax_gt.set_title(f'Problema {name} — Ground Truth (n={len(X)})', fontweight='bold', fontsize=10)
        ax_gt.legend(fontsize=8)
        ax_gt.grid(True, alpha=0.3)

        if y_llm is not None:
            for c in [0, 1]:
                mask = y_llm == c
                ax_llm.scatter(X[mask, 0], X[mask, 1], c=colors_llm[c], s=15, alpha=0.6, label=f'Classe {c}')
            c0_llm = X[y_llm == 0].mean(axis=0) if np.any(y_llm == 0) else centroid_0
            c1_llm = X[y_llm == 1].mean(axis=0) if np.any(y_llm == 1) else centroid_1
            ax_llm.scatter(*c0_llm, c='black', marker='X', s=120, zorder=5, edgecolors='white', linewidths=1)
            ax_llm.scatter(*c1_llm, c='black', marker='X', s=120, zorder=5, edgecolors='white', linewidths=1)
            n0, n1 = np.sum(y_llm == 0), np.sum(y_llm == 1)
            ax_llm.set_title(f'Problema {name} — LLM (C0={n0}, C1={n1})', fontweight='bold', fontsize=10)
            ax_llm.legend(fontsize=8)
        else:
            ax_llm.text(0.5, 0.5, 'Sem dados LLM', ha='center', va='center', transform=ax_llm.transAxes, fontsize=12, color='gray')
            ax_llm.set_title(f'Problema {name} — LLM', fontweight='bold', fontsize=10)
        ax_llm.grid(True, alpha=0.3)

    # Figura combinada
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for col, (name, X, y_gt, y_llm) in enumerate(problems):
        _draw_problem((axes[0, col], axes[1, col]), name, X, y_gt, y_llm)
    fig.suptitle(f'Visão Geral dos Datasets — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais por problema (GT + LLM)
    if filename:
        base, ext = os.path.splitext(filename)
        for name, X, y_gt, y_llm in problems:
            fig_ind, axes_ind = plt.subplots(2, 1, figsize=(6, 10))
            _draw_problem((axes_ind[0], axes_ind[1]), name, X, y_gt, y_llm)
            fig_ind.suptitle(f'Problema {name} — Seed {seed}', fontsize=13, fontweight='bold')
            fig_ind.tight_layout()
            fig_ind.savefig(f"{base}_problema_{name.lower()}{ext}", dpi=150, bbox_inches='tight')
            plt.close(fig_ind)


def plot_hits_and_errors(data: dict, seed: int, filename: str = None):
    """Acertos e erros da métrica vs LLM para Problemas A, B, C."""
    problems = [
        ('A', data['X_a'], data.get('y_llm_a'), data.get('y_metric_a')),
        ('B', data['X_b'], data.get('y_llm_b'), data.get('y_metric_b')),
        ('C', data['X_c'], data.get('y_llm_c'), data.get('y_metric_c')),
    ]
    learned_metric = data.get('learned_metric')

    def _draw_problem_row(axes_row, name, X, y_llm, y_metric):
        if y_llm is None or y_metric is None:
            for col in range(3):
                axes_row[col].text(0.5, 0.5, 'Sem dados', ha='center', va='center',
                                   transform=axes_row[col].transAxes, fontsize=12, color='gray')
                axes_row[col].set_title(f'Problema {name}')
            return

        n_min = min(len(y_llm), len(y_metric), len(X))
        X_plot = X[:n_min]
        y_llm_plot = y_llm[:n_min]
        y_metric_plot = y_metric[:n_min]
        hits = y_llm_plot == y_metric_plot
        errors = ~hits

        ax1 = axes_row[0]
        ax1.scatter(X_plot[hits, 0], X_plot[hits, 1], c='#27ae60', s=15, alpha=0.5, label=f'Acerto ({hits.sum()})')
        ax1.scatter(X_plot[errors, 0], X_plot[errors, 1], c='#e74c3c', s=30, alpha=0.8, marker='x', label=f'Erro ({errors.sum()})')
        acc = hits.sum() / len(hits) * 100
        ax1.set_title(f'Problema {name}: Acertos/Erros ({acc:.1f}%)', fontweight='bold', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = axes_row[1]
        if learned_metric is not None:
            x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
            y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            Z = predict_with_metric(grid_points, learned_metric.centroids, learned_metric.w)
            Z = Z.reshape(xx.shape)
            ax2.contourf(xx, yy, Z, alpha=0.2, cmap='RdBu')
            ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        for c in [0, 1]:
            mask = y_llm_plot == c
            ax2.scatter(X_plot[mask, 0], X_plot[mask, 1], s=15, alpha=0.5, label=f'LLM Classe {c}')
        ax2.set_title(f'Problema {name}: Fronteira W_A', fontweight='bold', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax3 = axes_row[2]
        if learned_metric is not None:
            confidences, _ = compute_metric_confidence(X_plot, learned_metric.centroids, learned_metric.w)
            sc = ax3.scatter(X_plot[:, 0], X_plot[:, 1], c=confidences, cmap='viridis', s=15, alpha=0.6)
            ax3.scatter(X_plot[errors, 0], X_plot[errors, 1], c='red', s=50, marker='x', linewidths=2, label=f'Erros ({errors.sum()})')
            plt.colorbar(sc, ax=ax3, label='Margem')
        ax3.set_title(f'Problema {name}: Confiança + Erros', fontweight='bold', fontsize=10)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    # Figura combinada
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    for row, (name, X, y_llm, y_metric) in enumerate(problems):
        _draw_problem_row(axes[row], name, X, y_llm, y_metric)
    fig.suptitle(f'Acertos e Erros: Métrica vs LLM — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais por problema (1×3: acertos, fronteira, confiança)
    if filename:
        base, ext = os.path.splitext(filename)
        for name, X, y_llm, y_metric in problems:
            fig_ind, axes_ind = plt.subplots(1, 3, figsize=(18, 5))
            _draw_problem_row(axes_ind, name, X, y_llm, y_metric)
            fig_ind.suptitle(f'Acertos e Erros: Problema {name} — Seed {seed}', fontsize=13, fontweight='bold')
            fig_ind.tight_layout()
            fig_ind.savefig(f"{base}_problema_{name.lower()}{ext}", dpi=150, bbox_inches='tight')
            plt.close(fig_ind)


def plot_w_comparison_algorithms(data: dict, seed: int, filename: str = None):
    """Comparação visual dos W aprendidos: Perceptron vs NNLS."""
    learned_metric = data.get('learned_metric')
    w_nnls = data.get('w_nnls')
    X = data['X_a']
    y_llm = data.get('y_llm_a')

    if learned_metric is None or y_llm is None:
        return

    w_perc = learned_metric.w
    centroids_perc = learned_metric.centroids
    centroids_nnls = data.get('centroids_nnls', centroids_perc)

    has_nnls = w_nnls is not None

    # Helper para plotar fronteira
    def plot_boundary(ax, X, y, w, centroids, title):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = predict_with_metric(grid_points, centroids, w)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.15, cmap='RdBu')
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        for c in [0, 1]:
            mask = y == c
            ax.scatter(X[mask, 0], X[mask, 1], s=15, alpha=0.5, label=f'Classe {c}')
        ax.scatter(*centroids[0], c='black', marker='X', s=150, zorder=5, edgecolors='white', linewidths=1.5)
        ax.scatter(*centroids[1], c='black', marker='X', s=150, zorder=5, edgecolors='white', linewidths=1.5)
        w_norm = w / np.sum(w) if np.sum(w) > 0 else w
        ax.set_title(f'{title}\nW=[{w[0]:.3f}, {w[1]:.3f}] norm=[{w_norm[0]:.2f}, {w_norm[1]:.2f}]',
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _draw_bar_chart(ax_bar):
        x_pos = np.arange(2)
        algos = [('Perceptron', w_perc, 'steelblue')]
        if has_nnls:
            algos.append(('NNLS', w_nnls, 'coral'))
        n_algos = len(algos)
        width = 0.8 / n_algos
        for j, (name, w, color) in enumerate(algos):
            offset = (j - (n_algos - 1) / 2) * width
            bars = ax_bar.bar(x_pos + offset, w, width, label=name, color=color, alpha=0.8, edgecolor='k')
            for bar in bars:
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(['w1 (x1)', 'w2 (x2)'])
        ax_bar.set_ylabel('Peso')
        ratio_strs = []
        for name, w, _ in algos:
            ratio = w[0] / w[1] if w[1] > 0 else float('inf')
            ratio_strs.append(f'{name}={ratio:.2f}')
        ax_bar.set_title(f'Comparação de Pesos\nRatio w1/w2: {", ".join(ratio_strs)}',
                         fontweight='bold', fontsize=10)
        ax_bar.legend(fontsize=8)
        ax_bar.grid(True, alpha=0.3, axis='y')

    # Figura combinada: boundary plots + bar chart
    idx = 0
    n_boundary = 1 + int(has_nnls)
    n_cols = n_boundary + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    plot_boundary(axes[idx], X, y_llm, w_perc, centroids_perc, 'Perceptron Estruturado')
    idx += 1
    if has_nnls:
        plot_boundary(axes[idx], X, y_llm, w_nnls, centroids_nnls, 'NNLS (Mín. Quadrados)')
        idx += 1
    _draw_bar_chart(axes[idx])

    fig.suptitle(f'Comparação de Algoritmos de Otimização Inversa — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    if filename:
        base, ext = os.path.splitext(filename)
        # Fronteira Perceptron
        fig_ind, ax_ind = plt.subplots(1, 1, figsize=(7, 6))
        plot_boundary(ax_ind, X, y_llm, w_perc, centroids_perc, 'Perceptron Estruturado')
        fig_ind.tight_layout()
        fig_ind.savefig(f"{base}_perceptron{ext}", dpi=150, bbox_inches='tight')
        plt.close(fig_ind)
        if has_nnls:
            fig_ind, ax_ind = plt.subplots(1, 1, figsize=(7, 6))
            plot_boundary(ax_ind, X, y_llm, w_nnls, centroids_nnls, 'NNLS (Mín. Quadrados)')
            fig_ind.tight_layout()
            fig_ind.savefig(f"{base}_nnls{ext}", dpi=150, bbox_inches='tight')
            plt.close(fig_ind)
        # Bar chart
        fig_ind, ax_ind = plt.subplots(1, 1, figsize=(7, 6))
        _draw_bar_chart(ax_ind)
        fig_ind.tight_layout()
        fig_ind.savefig(f"{base}_barras{ext}", dpi=150, bbox_inches='tight')
        plt.close(fig_ind)


def plot_confusion_matrices_detailed(data: dict, seed: int, filename: str = None):
    """Matrizes de confusão: LLM vs Métrica e LLM vs Ground Truth para cada problema."""
    problems = [
        ('A', data.get('y_llm_a'), data.get('y_metric_a'), data['y_gt_a']),
        ('B', data.get('y_llm_b'), data.get('y_metric_b'), data['y_gt_b']),
        ('C', data.get('y_llm_c'), data.get('y_metric_c'), data['y_gt_c']),
    ]

    def _draw_problem_col(axes_col, name, y_llm, y_metric, y_gt):
        ax1, ax2 = axes_col
        if y_llm is not None and y_metric is not None:
            n_min = min(len(y_llm), len(y_metric))
            cm = np.zeros((2, 2), dtype=int)
            for i in range(n_min):
                cm[int(y_metric[i]), int(y_llm[i])] += 1
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                        xticklabels=['C0', 'C1'], yticklabels=['C0', 'C1'])
            ax1.set_xlabel('LLM')
            ax1.set_ylabel('Métrica')
            acc = np.trace(cm) / cm.sum() * 100
            ax1.set_title(f'Problema {name}: LLM vs Métrica\n({acc:.1f}% concordância)', fontweight='bold', fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'Sem dados', ha='center', va='center', transform=ax1.transAxes, color='gray')
            ax1.set_title(f'Problema {name}: LLM vs Métrica')

        if y_llm is not None:
            n_min = min(len(y_llm), len(y_gt))
            cm2 = np.zeros((2, 2), dtype=int)
            for i in range(n_min):
                cm2[int(y_gt[i]), int(y_llm[i])] += 1
            sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=ax2,
                        xticklabels=['C0', 'C1'], yticklabels=['C0', 'C1'])
            ax2.set_xlabel('LLM')
            ax2.set_ylabel('Ground Truth')
            acc2 = np.trace(cm2) / cm2.sum() * 100
            ax2.set_title(f'Problema {name}: LLM vs GT\n({acc2:.1f}% acurácia)', fontweight='bold', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'Sem dados', ha='center', va='center', transform=ax2.transAxes, color='gray')
            ax2.set_title(f'Problema {name}: LLM vs GT')

    # Figura combinada
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, (name, y_llm, y_metric, y_gt) in enumerate(problems):
        _draw_problem_col((axes[0, col], axes[1, col]), name, y_llm, y_metric, y_gt)
    fig.suptitle(f'Matrizes de Confusão — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais por problema (2×1: LLM vs Métrica + LLM vs GT)
    if filename:
        base, ext = os.path.splitext(filename)
        for name, y_llm, y_metric, y_gt in problems:
            fig_ind, axes_ind = plt.subplots(2, 1, figsize=(6, 10))
            _draw_problem_col((axes_ind[0], axes_ind[1]), name, y_llm, y_metric, y_gt)
            fig_ind.suptitle(f'Matrizes de Confusão: Problema {name} — Seed {seed}', fontsize=13, fontweight='bold')
            fig_ind.tight_layout()
            fig_ind.savefig(f"{base}_problema_{name.lower()}{ext}", dpi=150, bbox_inches='tight')
            plt.close(fig_ind)


def plot_margin_analysis_detailed(data: dict, seed: int, filename: str = None):
    """Análise detalhada de margens/confiança da métrica estimada."""
    learned_metric = data.get('learned_metric')
    if learned_metric is None:
        return

    X_a = data['X_a']
    y_llm_a = data.get('y_llm_a')
    y_metric_a = data.get('y_metric_a')
    conf_a, _ = compute_metric_confidence(X_a, learned_metric.centroids, learned_metric.w)

    def _draw_histograma(ax):
        ax.hist(conf_a, bins=30, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(np.median(conf_a), color='red', linestyle='--', label=f'Mediana: {np.median(conf_a):.2f}')
        ax.axvline(np.mean(conf_a), color='orange', linestyle='--', label=f'Média: {np.mean(conf_a):.2f}')
        ax.set_xlabel('Margem')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição de Margens — Problema A', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _draw_taxa_erro(ax):
        if y_llm_a is not None and y_metric_a is not None:
            n_min = min(len(y_llm_a), len(y_metric_a), len(conf_a))
            errors = y_llm_a[:n_min] != y_metric_a[:n_min]
            conf_plot = conf_a[:n_min]
            bins = np.linspace(0, conf_plot.max(), 8)
            bin_indices = np.digitize(conf_plot, bins)
            error_rates = []
            bin_centers = []
            bin_counts = []
            for b in range(1, len(bins)):
                mask = bin_indices == b
                if mask.sum() > 0:
                    error_rates.append(errors[mask].mean() * 100)
                    bin_centers.append((bins[b-1] + bins[b]) / 2)
                    bin_counts.append(mask.sum())
            bars = ax.bar(bin_centers, error_rates, width=(bins[1]-bins[0])*0.8, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.5)
            for bar, count in zip(bars, bin_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'n={count}', ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('Margem')
        ax.set_ylabel('Taxa de Erro (%)')
        ax.set_title('Taxa de Erro por Faixa de Margem', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _draw_mapa_confianca(ax):
        sc = ax.scatter(X_a[:, 0], X_a[:, 1], c=conf_a, cmap='viridis', s=20, alpha=0.7)
        ax.scatter(*learned_metric.centroids[0], c='red', marker='X', s=200, zorder=5, edgecolors='white', linewidths=2)
        ax.scatter(*learned_metric.centroids[1], c='red', marker='X', s=200, zorder=5, edgecolors='white', linewidths=2)
        plt.colorbar(sc, ax=ax, label='Margem')
        ax.set_title('Mapa de Confiança — Problema A', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _draw_violin(ax):
        all_margins = [conf_a]
        labels = ['A']
        for pname, X_p in [('B', data['X_b']), ('C', data['X_c'])]:
            conf_p, _ = compute_metric_confidence(X_p, learned_metric.centroids, learned_metric.w)
            all_margins.append(conf_p)
            labels.append(pname)
        parts = ax.violinplot(all_margins, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#3498db')
            pc.set_alpha(0.5)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([f'Problema {l}' for l in labels])
        ax.set_ylabel('Margem')
        ax.set_title('Distribuição de Margens por Problema', fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Figura combinada
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    _draw_histograma(axes[0, 0])
    _draw_taxa_erro(axes[0, 1])
    _draw_mapa_confianca(axes[1, 0])
    _draw_violin(axes[1, 1])
    fig.suptitle(f'Análise de Margens — Seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Figuras individuais
    panels = [
        ("histograma", _draw_histograma, (7, 6)),
        ("taxa_erro", _draw_taxa_erro, (7, 6)),
        ("mapa_confianca", _draw_mapa_confianca, (7, 6)),
        ("violin", _draw_violin, (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_experiment_summary_dashboard(data: dict, seed: int,
                                      results_abc: list, results_e: list,
                                      filename: str = None):
    """Dashboard completo do experimento para uma seed."""
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

    learned_metric = data.get('learned_metric')
    colors = {0: '#3498db', 1: '#e74c3c'}

    # ─── BLOCO 1 (topo): 4 datasets com ground truth ───
    for col, (name, X, y_gt) in enumerate([
        ('A', data['X_a'], data['y_gt_a']),
        ('B', data['X_b'], data['y_gt_b']),
        ('C', data['X_c'], data['y_gt_c']),
        ('D', data['X_e'], data['y_gt_e']),
    ]):
        ax = fig.add_subplot(gs[0, col])
        for c in [0, 1]:
            mask = y_gt == c
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[c], s=10, alpha=0.5)
        ax.set_title(f'Prob. {name} (n={len(X)})', fontweight='bold', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    # ─── BLOCO 2: Fase A — acertos/erros + fronteira ───
    y_llm_a = data.get('y_llm_a')
    y_metric_a = data.get('y_metric_a')
    X_a = data['X_a']

    ax_a1 = fig.add_subplot(gs[1, 0])
    if y_llm_a is not None and y_metric_a is not None:
        hits = y_llm_a == y_metric_a
        ax_a1.scatter(X_a[hits, 0], X_a[hits, 1], c='#27ae60', s=10, alpha=0.5, label=f'OK ({hits.sum()})')
        ax_a1.scatter(X_a[~hits, 0], X_a[~hits, 1], c='#e74c3c', s=25, marker='x', label=f'Erro ({(~hits).sum()})')
        ax_a1.set_title(f'Fase A: Fidelidade {hits.mean():.1%}', fontweight='bold', fontsize=9)
        ax_a1.legend(fontsize=7)
    ax_a1.grid(True, alpha=0.2)

    ax_a2 = fig.add_subplot(gs[1, 1])
    if learned_metric is not None:
        x_min, x_max = X_a[:, 0].min() - 1, X_a[:, 0].max() + 1
        y_min, y_max = X_a[:, 1].min() - 1, X_a[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
        Z = predict_with_metric(np.c_[xx.ravel(), yy.ravel()], learned_metric.centroids, learned_metric.w).reshape(xx.shape)
        ax_a2.contourf(xx, yy, Z, alpha=0.15, cmap='RdBu')
        ax_a2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5)
    if y_llm_a is not None:
        for c in [0, 1]:
            ax_a2.scatter(X_a[y_llm_a == c, 0], X_a[y_llm_a == c, 1], c=colors[c], s=10, alpha=0.4)
    w_str = f'W=[{learned_metric.w[0]:.3f}, {learned_metric.w[1]:.3f}]' if learned_metric else ''
    ax_a2.set_title(f'Fase A: Fronteira {w_str}', fontweight='bold', fontsize=9)
    ax_a2.grid(True, alpha=0.2)

    # Margens Fase A
    ax_a3 = fig.add_subplot(gs[1, 2])
    if learned_metric is not None:
        conf_a, _ = compute_metric_confidence(X_a, learned_metric.centroids, learned_metric.w)
        ax_a3.hist(conf_a, bins=25, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.3)
        ax_a3.axvline(np.median(conf_a), color='red', linestyle='--', linewidth=1, label=f'Med={np.median(conf_a):.2f}')
        ax_a3.set_title('Margens Problema A', fontweight='bold', fontsize=9)
        ax_a3.legend(fontsize=7)
    ax_a3.grid(True, alpha=0.2)

    # W dos algoritmos
    ax_w = fig.add_subplot(gs[1, 3])
    if learned_metric is not None:
        algs = ['Perceptron']
        w_vals = [learned_metric.w]
        if data.get('w_nnls') is not None:
            algs.append('NNLS')
            w_vals.append(data['w_nnls'])
        x_pos = np.arange(2)
        width = 0.8 / len(algs)
        for i, (alg, w) in enumerate(zip(algs, w_vals)):
            offset = (i - (len(algs)-1)/2) * width
            ax_w.bar(x_pos + offset, w, width * 0.9, label=f'{alg}', alpha=0.8)
        ax_w.set_xticks(x_pos)
        ax_w.set_xticklabels(['w₁', 'w₂'])
        ax_w.set_title('Pesos W', fontweight='bold', fontsize=9)
        ax_w.legend(fontsize=7)
    ax_w.grid(True, alpha=0.2, axis='y')

    # ─── BLOCO 3: Fases B/C ───
    for col_offset, (name, X, y_llm, y_metric) in enumerate([
        ('B', data['X_b'], data.get('y_llm_b'), data.get('y_metric_b')),
        ('C', data['X_c'], data.get('y_llm_c'), data.get('y_metric_c')),
    ]):
        ax = fig.add_subplot(gs[2, col_offset])
        if y_llm is not None and y_metric is not None:
            n_min = min(len(y_llm), len(y_metric), len(X))
            hits = y_llm[:n_min] == y_metric[:n_min]
            ax.scatter(X[:n_min][hits, 0], X[:n_min][hits, 1], c='#27ae60', s=10, alpha=0.5)
            ax.scatter(X[:n_min][~hits, 0], X[:n_min][~hits, 1], c='#e74c3c', s=25, marker='x')
            ax.set_title(f'Fase {name}: Consistência {hits.mean():.1%}', fontweight='bold', fontsize=9)
        else:
            ax.set_title(f'Fase {name}: Sem dados', fontsize=9)
        ax.grid(True, alpha=0.2)

    # Barras de métricas B/C
    ax_bars = fig.add_subplot(gs[2, 2])
    seed_results = [r for r in results_abc if r.random_seed == seed and r.n_shot == 0
                    and r.nomes_classes == ("A", "B") and r.repeticao == 0]
    if seed_results:
        r = seed_results[0]
        metrics = ['Consist. B', 'Kappa B', 'F1 B', 'Consist. C', 'Kappa C', 'F1 C']
        values = [r.consistencia_problema_b, r.kappa_problema_b, r.f1_problema_b,
                  r.consistencia_problema_c, r.kappa_problema_c, r.f1_problema_c]
        bar_colors = ['#3498db']*3 + ['#e67e22']*3
        ax_bars.barh(metrics, values, color=bar_colors, alpha=0.8)
        for i, v in enumerate(values):
            ax_bars.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8)
        ax_bars.set_xlim(0, 1.15)
        ax_bars.set_title('Métricas Zero-Shot', fontweight='bold', fontsize=9)
    ax_bars.grid(True, alpha=0.2, axis='x')

    # Fidelidade por n_shot
    ax_fid = fig.add_subplot(gs[2, 3])
    seed_abc = [r for r in results_abc if r.random_seed == seed and r.nomes_classes == ("A", "B")]
    if seed_abc:
        for metric_name, getter, color in [
            ('Consist. B', lambda r: r.consistencia_problema_b, '#3498db'),
            ('Consist. C', lambda r: r.consistencia_problema_c, '#e67e22'),
        ]:
            by_nshot = {}
            for r in seed_abc:
                by_nshot.setdefault(r.n_shot, []).append(getter(r))
            nshots = sorted(by_nshot.keys())
            means = [np.mean(by_nshot[n]) for n in nshots]
            ax_fid.plot(nshots, means, 'o-', label=metric_name, color=color)
        ax_fid.set_xlabel('n_shot')
        ax_fid.set_title('Consistência vs n_shot', fontweight='bold', fontsize=9)
        ax_fid.legend(fontsize=7)
    ax_fid.grid(True, alpha=0.2)

    # ─── BLOCO 4: Fase E ───
    seed_d = [r for r in results_e if r.random_seed == seed]
    ax_d1 = fig.add_subplot(gs[3, 0:2])
    if seed_d:
        for strategy in ['easy', 'hard', 'mixed', 'random']:
            strat_results = [r for r in seed_d if r.example_strategy == strategy]
            if strat_results:
                by_nshot = {}
                for r in strat_results:
                    by_nshot.setdefault(r.n_shot, []).append(r.accuracy_llm_vs_expert)
                nshots = sorted(by_nshot.keys())
                means = [np.mean(by_nshot[n]) for n in nshots]
                ax_d1.plot(nshots, means, 'o-', label=strategy)
        ax_d1.set_xlabel('n_shot')
        ax_d1.set_ylabel('Acurácia LLM vs Expert')
        ax_d1.set_title('Fase E: Learning Curve por Estratégia', fontweight='bold', fontsize=9)
        ax_d1.legend(fontsize=8)
    else:
        ax_d1.text(0.5, 0.5, 'Fase E não executada', ha='center', va='center', transform=ax_d1.transAxes, color='gray')
    ax_d1.grid(True, alpha=0.2)

    # Confusion matrix A (LLM vs Métrica)
    ax_cm = fig.add_subplot(gs[3, 2])
    if y_llm_a is not None and y_metric_a is not None:
        cm = np.zeros((2, 2), dtype=int)
        for i in range(len(y_llm_a)):
            cm[int(y_metric_a[i]), int(y_llm_a[i])] += 1
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['C0', 'C1'], yticklabels=['C0', 'C1'])
        ax_cm.set_xlabel('LLM')
        ax_cm.set_ylabel('Métrica')
        ax_cm.set_title('Conf. Matrix A: LLM vs Métrica', fontweight='bold', fontsize=9)

    # Info textual
    ax_info = fig.add_subplot(gs[3, 3])
    ax_info.axis('off')
    info_lines = [f'Seed: {seed}']
    if learned_metric:
        w = learned_metric.w
        w_norm = w / np.sum(w) if np.sum(w) > 0 else w
        info_lines.extend([
            f'W Perceptron: [{w[0]:.4f}, {w[1]:.4f}]',
            f'W norm: [{w_norm[0]:.3f}, {w_norm[1]:.3f}]',
            f'Gamma: {learned_metric.gamma:.4f}',
        ])
    if data.get('w_nnls') is not None:
        wn = data['w_nnls']
        info_lines.append(f'W NNLS: [{wn[0]:.4f}, {wn[1]:.4f}]')
    if seed_results:
        r = seed_results[0]
        info_lines.extend([
            f'',
            f'Fidelidade A: {r.fidelidade_problema_a:.1%}',
            f'Consistência B: {r.consistencia_problema_b:.1%}',
            f'Consistência C: {r.consistencia_problema_c:.1%}',
            f'Kappa B: {r.kappa_problema_b:.3f}',
            f'Kappa C: {r.kappa_problema_c:.3f}',
        ])
    ax_info.text(0.05, 0.95, '\n'.join(info_lines), transform=ax_info.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_info.set_title('Resumo', fontweight='bold', fontsize=9)

    fig.suptitle(f'Dashboard Completo do Experimento — Seed {seed}', fontsize=16, fontweight='bold')
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_meialua_svm_vs_llm(
    X_ml: np.ndarray,
    y_true: np.ndarray,
    y_llm: np.ndarray,
    metric: Optional[dict] = None,
    seed: int = 0,
    filename: str = None,
):
    """Compara a superfície de separação na meia-lua (item 7, reunião 20/05 ~1285s).

    O orientador pediu para sobrepor a superfície de separação de um SVM gaussiano
    (RBF) à fronteira induzida pelo LLM. Dois painéis usam o MESMO SVM RBF como
    "régua" de fronteira não-linear, mudando apenas os rótulos que ele ajusta:
      (1) SVM RBF treinado no GROUND TRUTH — a fronteira não-linear de referência.
      (2) SVM RBF treinado nos RÓTULOS DO LLM — a fronteira efetiva do LLM; por
          cima, em tracejado verde, a fronteira da métrica diagonal aprendida.

    Se o LLM colapsou em classe única (zero-shot), o painel 2 mostra os pontos
    com a classe única e uma anotação explícita — o colapso é um achado, não um
    motivo para suprimir a figura. Retorna True se o arquivo foi salvo.
    """
    try:
        from sklearn.svm import SVC
    except Exception:
        return False
    if len(np.unique(y_true)) < 2:
        return False
    llm_collapsed = len(np.unique(y_llm)) < 2

    pad = 0.5
    xs = np.linspace(X_ml[:, 0].min() - pad, X_ml[:, 0].max() + pad, 200)
    ys = np.linspace(X_ml[:, 1].min() - pad, X_ml[:, 1].max() + pad, 200)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = [
        ("SVM RBF treinado no ground truth", y_true),
        ("SVM RBF treinado nos rótulos do LLM", y_llm),
    ]
    for panel_idx, (ax, (title, labels)) in enumerate(zip(axes, panels)):
        if panel_idx == 1 and llm_collapsed:
            unica = int(np.unique(y_llm)[0])
            ax.scatter(X_ml[:, 0], X_ml[:, 1],
                       c='tab:blue' if unica == 0 else 'tab:red',
                       s=22, edgecolor='black', linewidth=0.3, alpha=0.85)
            ax.set_title('Rótulos do LLM (zero-shot): CLASSE ÚNICA — sem fronteira',
                         fontweight='bold', fontsize=11)
            ax.text(0.5, 0.04,
                    f'100% dos pontos rotulados como classe {unica};\n'
                    'SVM e métrica não ajustáveis (colapso zero-shot)',
                    transform=ax.transAxes, ha='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.grid(True, alpha=0.3)
            continue
        clf = SVC(kernel='rbf', gamma='scale', C=1.0)
        clf.fit(X_ml, labels)
        zz = clf.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                    colors=['tab:blue', 'tab:red'], alpha=0.15)
        ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=1.6)
        for c in [0, 1]:
            mask = labels == c
            ax.scatter(X_ml[mask, 0], X_ml[mask, 1],
                       c='tab:blue' if c == 0 else 'tab:red',
                       s=22, edgecolor='black', linewidth=0.3, alpha=0.85)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.grid(True, alpha=0.3)

    # Sobrepõe a fronteira da métrica diagonal aprendida (tracejada) no painel 2
    if metric is not None:
        try:
            n_feat = metric['n_features']
            grid_aug = augment_features(grid, n_feat)
            zz_m = predict_with_metric(grid_aug, metric['centroids'], metric['w_perc']).reshape(xx.shape)
            axes[1].contour(xx, yy, zz_m, levels=[0.5], colors='green',
                            linewidths=1.8, linestyles='dashed')
            axes[1].plot([], [], color='green', linestyle='dashed',
                         label=f"métrica diagonal ({n_feat} feat)")
            axes[1].legend(loc='upper right', fontsize=9)
        except Exception:
            pass

    fig.suptitle(
        f'Meia-lua (seed={seed}) — superfície SVM gaussiano vs LLM\n'
        '(reunião 20/05: comparar a fronteira do SVM RBF com a do LLM)',
        fontweight='bold', fontsize=12,
    )
    plt.tight_layout()
    saved = False
    if filename:
        plt.savefig(filename, dpi=140, bbox_inches='tight')
        saved = True
    plt.close()
    return saved


def plot_problem_overview(
    X: np.ndarray,
    y_true: np.ndarray,
    title: str,
    feature_names: Tuple[str, str] = ("x1", "x2"),
    optimal_boundary_fn: Optional[callable] = None,
    filename: Optional[str] = None,
) -> None:
    """Visualiza um problema (scatter por classe + fronteira ótima opcional).

    Usado para os problemas novos (meia-lua, peso × altura) que nao aparecem
    nos gráficos `01_all_three_problems.png` (limitados a A/B/C).

    Args:
        optimal_boundary_fn: função f(x1, x2) cuja curva f=0 será sobreposta
            (ex.: classificador ótimo bayesiano do peso×altura é a elipse
            x2² - x2 + x1² - x1 + cte).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap_class = {0: "tab:blue", 1: "tab:red"}
    for c in [0, 1]:
        mask = y_true == c
        ax.scatter(
            X[mask, 0], X[mask, 1], c=cmap_class[c], s=50,
            edgecolor="black", linewidth=0.5, alpha=0.85,
            label=f"Classe {c}",
        )

    if optimal_boundary_fn is not None:
        pad = 0.5
        xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 200)
        ys = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 200)
        xx, yy = np.meshgrid(xs, ys)
        try:
            zz = optimal_boundary_fn(xx, yy)
            ax.contour(xx, yy, zz, levels=[0.0], colors="black",
                       linewidths=1.8, linestyles="--")
            ax.plot([], [], color="black", linestyle="--", label="Fronteira ótima")
        except Exception:
            pass

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=140, bbox_inches="tight")
    plt.close()


def plot_llm_labels_per_problem(
    X: np.ndarray,
    y_llm: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    title: str = "Rotulação do LLM",
    feature_names: Tuple[str, str] = ("x1", "x2"),
    filename: Optional[str] = None,
) -> None:
    """Visualiza scatter ponto-a-ponto das rotulações do LLM.

    Item G do plano (e-mail orientador 22:06):
        "Mostrar graficamente todas as rotulações feitas pela LLM."

    Se y_true for fornecido, gera 3 painéis: LLM, real, erros (LLM ≠ real).
    """
    n_panels = 3 if y_true is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    cmap_class = {0: "tab:blue", 1: "tab:red"}

    # Painel 1: rotulação do LLM
    for c in [0, 1]:
        mask = y_llm == c
        axes[0].scatter(X[mask, 0], X[mask, 1], c=cmap_class[c], s=40,
                        edgecolor="black", linewidth=0.4, alpha=0.85,
                        label=f"LLM = {c}")
    axes[0].set_xlabel(feature_names[0])
    axes[0].set_ylabel(feature_names[1])
    axes[0].set_title(f"Rotulação do LLM\n(n={len(X)})")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    if y_true is not None:
        # Painel 2: rótulo real (ground truth)
        for c in [0, 1]:
            mask = y_true == c
            axes[1].scatter(X[mask, 0], X[mask, 1], c=cmap_class[c], s=40,
                            edgecolor="black", linewidth=0.4, alpha=0.85,
                            label=f"Real = {c}")
        axes[1].set_xlabel(feature_names[0])
        axes[1].set_ylabel(feature_names[1])
        acc = accuracy_score(y_true, y_llm)
        axes[1].set_title(f"Rótulo Real\n(acurácia do LLM = {acc:.1%})")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # Painel 3: erros (LLM ≠ real)
        errors = y_llm != y_true
        axes[2].scatter(X[~errors, 0], X[~errors, 1], c="lightgray", s=25,
                        edgecolor="none", alpha=0.6, label="LLM acertou")
        axes[2].scatter(X[errors, 0], X[errors, 1], c="black", s=60,
                        marker="x", linewidths=2, label="LLM errou")
        axes[2].set_xlabel(feature_names[0])
        axes[2].set_ylabel(feature_names[1])
        axes[2].set_title(f"Erros do LLM\n(n_erros = {int(errors.sum())} / {len(X)})")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=140, bbox_inches="tight")
    plt.close()


def plot_external_learning_curve(
    phase_e_results: List[dict],
    filename: Optional[str] = None,
) -> None:
    """Curva de aprendizado (acurácia × n_shot) por (problema, variante)."""
    if not phase_e_results:
        return
    df = pd.DataFrame(phase_e_results)
    df['feature_names_str'] = df['feature_names'].apply(
        lambda t: '/'.join(t) if isinstance(t, tuple) else t
    )
    df['group'] = df['problem_name'].astype(str) + ' | ' + df['feature_names_str']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap('tab10')

    for idx, metric_col in enumerate(['accuracy_llm_vs_metric', 'accuracy_llm_vs_true']):
        ax = axes[idx]
        for i, group in enumerate(sorted(df['group'].unique())):
            sub = df[df['group'] == group].sort_values('n_shot')
            stats = sub.groupby('n_shot')[metric_col].agg(['mean', 'std']).reset_index()
            ax.errorbar(
                stats['n_shot'], stats['mean'], yerr=stats['std'],
                marker='o', label=group, color=cmap(i % 10), capsize=3,
            )
        ax.set_xlabel('n_shot')
        ax.set_ylabel(metric_col.replace('_', ' '))
        ax.set_title('LLM vs métrica' if idx == 0 else 'LLM vs rótulo real', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

    fig.suptitle('Fase E (problemas externos) — curva de aprendizado',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=140, bbox_inches='tight')
    plt.close()


def plot_external_features_comparison(
    phase_a_results: List[dict],
    filename: Optional[str] = None,
) -> None:
    """Bar chart comparativo 2/3/4 features por (problema, variante).

    Responde diretamente ao "Verificar se houve ganho?" do e-mail 22:04 ponto 3.
    """
    if not phase_a_results:
        return
    df = pd.DataFrame(phase_a_results)
    df['feature_names_str'] = df['feature_names'].apply(
        lambda t: '/'.join(t) if isinstance(t, tuple) else t
    )
    df['group'] = df['problem_name'].astype(str) + ' | ' + df['feature_names_str']
    groups = sorted(df['group'].unique())
    n_features_list = sorted(df['n_features'].unique())

    metrics = [
        ('fidelity_perc_vs_llm', 'Fidelidade Perc (vs LLM)'),
        ('accuracy_perc_vs_true', 'Acurácia Perc (vs real)'),
        ('fidelity_nnls_vs_llm', 'Fidelidade NNLS (vs LLM)'),
        ('accuracy_nnls_vs_true', 'Acurácia NNLS (vs real)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    cmap = plt.get_cmap('tab10')

    for ax, (col, title) in zip(axes, metrics):
        bar_width = 0.8 / len(n_features_list)
        x = np.arange(len(groups))
        for j, nf in enumerate(n_features_list):
            means, stds = [], []
            for g in groups:
                sub = df[(df['group'] == g) & (df['n_features'] == nf)]
                means.append(sub[col].mean() if len(sub) else 0)
                stds.append(sub[col].std() if len(sub) > 1 else 0)
            offset = (j - (len(n_features_list) - 1) / 2) * bar_width
            ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                   color=cmap(j), edgecolor='black', alpha=0.8,
                   label=f'{nf} features')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8)

    fig.suptitle('Fase A externos — comparação 2 / 3 / 4 features\n(orientador 22:04 ponto 3: "Verificar se houve ganho")',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=140, bbox_inches='tight')
    plt.close()


def plot_external_decision_boundary(
    phase_a_results: List[dict],
    filename: Optional[str] = None,
) -> None:
    """Fronteira de decisão projetada em R2 para cada (problema, variante, n_feat).

    Para cada combinação (problema, variante), mostra um painel por n_features
    com os pontos e a fronteira d_W(x, c_0) = d_W(x, c_1) reconstruída no espaço
    de 2/3/4 features mas projetada de volta em R2 via grid em (x1, x2).
    """
    if not phase_a_results:
        return
    df = pd.DataFrame(phase_a_results)
    df['feature_names_str'] = df['feature_names'].apply(
        lambda t: '/'.join(t) if isinstance(t, tuple) else t
    )

    # Agrega: para cada (problema, variante) usa a primeira seed disponível
    groups = df.groupby(['problem_name', 'feature_names_str'])
    n_panels = len(groups)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(n_panels, 3, figsize=(15, 5 * n_panels), squeeze=False)
    for row_idx, ((problem_name, fnames), sub) in enumerate(groups):
        seed = sub['seed'].iloc[0]
        sub_seed = sub[sub['seed'] == seed].sort_values('n_features')
        for col_idx, n_feat in enumerate([2, 3, 4]):
            ax = axes[row_idx, col_idx]
            row = sub_seed[sub_seed['n_features'] == n_feat]
            if len(row) == 0:
                ax.set_title(f'{problem_name} | {fnames} | n_feat={n_feat}\n(não rodado)',
                             fontsize=9)
                ax.axis('off')
                continue
            r = row.iloc[0]
            X_aug = r['X_aug']
            y_llm = r['y_llm']
            centroids = r['centroids']
            w = r['w_perc']

            X2 = X_aug[:, :2]
            pad = 0.5
            xs = np.linspace(X2[:, 0].min() - pad, X2[:, 0].max() + pad, 120)
            ys = np.linspace(X2[:, 1].min() - pad, X2[:, 1].max() + pad, 120)
            xx, yy = np.meshgrid(xs, ys)
            grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
            grid_aug = augment_features(grid_xy, n_feat)
            zz = predict_with_metric(grid_aug, centroids, w).reshape(xx.shape)

            ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                        colors=['tab:blue', 'tab:red'], alpha=0.18)
            ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=1.4)
            for c in [0, 1]:
                mask = y_llm == c
                ax.scatter(X2[mask, 0], X2[mask, 1],
                           c='tab:blue' if c == 0 else 'tab:red',
                           s=18, edgecolor='black', linewidth=0.3, alpha=0.85)
            ax.set_title(
                f'{problem_name} | {fnames}\n'
                f'n_feat={n_feat} | seed={seed} | acc_real={r["accuracy_perc_vs_true"]:.1%}',
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)

    fig.suptitle('Fronteira de decisão (Perceptron) — projetada em R2',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=140, bbox_inches='tight')
    plt.close()


def plot_phase_e_llm_vs_perceptron(
    phase_e_results: List[dict],
    filename: Optional[str] = None,
) -> None:
    """Bar chart comparativo: LLM vs Perceptron baseline na Fase E externa.

    Responde à instrução da reunião (~2110s): treinar Perceptron sobre os mesmos
    exemplos do expert e comparar com o LLM in-context.
    """
    if not phase_e_results:
        return
    df = pd.DataFrame(phase_e_results)
    df['feature_names_str'] = df['feature_names'].apply(
        lambda t: '/'.join(t) if isinstance(t, tuple) else t
    )
    df['group'] = df['problem_name'].astype(str) + ' | ' + df['feature_names_str']
    groups = sorted(df['group'].unique())
    n_shots = sorted(df['n_shot'].unique())

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5), squeeze=False)
    for col_idx, group in enumerate(groups):
        ax = axes[0, col_idx]
        sub = df[df['group'] == group]
        means_llm, means_perc = [], []
        stds_llm, stds_perc = [], []
        for ns in n_shots:
            ss = sub[sub['n_shot'] == ns]
            means_llm.append(ss['accuracy_llm_vs_true'].mean())
            stds_llm.append(ss['accuracy_llm_vs_true'].std() if len(ss) > 1 else 0)
            pb = ss['accuracy_perceptron_baseline_vs_true'].dropna()
            means_perc.append(pb.mean() if len(pb) else np.nan)
            stds_perc.append(pb.std() if len(pb) > 1 else 0)
        x = np.arange(len(n_shots))
        w = 0.4
        ax.bar(x - w/2, means_llm, w, yerr=stds_llm, capsize=3,
               color='tab:orange', edgecolor='black', alpha=0.85, label='LLM')
        ax.bar(x + w/2, means_perc, w, yerr=stds_perc, capsize=3,
               color='tab:blue', edgecolor='black', alpha=0.85, label='Perceptron baseline')
        ax.set_xticks(x)
        ax.set_xticklabels([str(ns) for ns in n_shots])
        ax.set_xlabel('n_shot')
        ax.set_ylabel('Acurácia (vs rótulo real)')
        ax.set_ylim(0, 1.05)
        ax.set_title(group, fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8)

    fig.suptitle('Fase E externa — LLM in-context vs Perceptron treinado nos mesmos exemplos',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=140, bbox_inches='tight')
    plt.close()
