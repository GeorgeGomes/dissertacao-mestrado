"""Todas as visualizações (PNGs) dos Blocos 1-3 e do fechamento.

Extraído de ``dissertacao_mestrado.py`` (etapa 1 da modularização, 2026): as funções
``plot_<problema>_passoN_*`` / ``plot_final_*`` / ``plot_fg_*`` que recebem listas de resultados
(dataclasses ou dicts dos CSVs) e arrays e salvam PNGs na pasta de execução. São consumidores puros: não chamam a API nem mutam
estado do runner. Única exceção de acoplamento: ``plot_e_passo3_exemplos``
faz import tardio de ``e_passo3_selecionar_exemplos`` do runner (documentado lá).
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

from execucao_io import asset_variant, com_cor
from estilo_cinza import garantir_legendas, para_cinza
from data_problems import (
    PROBLEM_A_CENTERS, PROBLEM_A_STD, PROBLEM_B_CENTERS, PROBLEM_B_STD,
    PROBLEM_C_CENTERS, PROBLEM_C_STD, PROBLEM_E_CENTERS,
)
from metrics import augment_features, compute_centroids, compute_metric_confidence, predict_with_metric
from protocolo import EXAMPLE_STRATEGIES, colapsou
from resultados import ResultadoABC, ResultadoAprendiz

# Nomes em português nas figuras (14/10/2026): as tabelas já usavam Fácil/Difícil/Mista/Aleatória e
# "Regressão Logística"; as figuras rotulavam em inglês. As siglas easy/hard/mixed/random ficam nos dados.
ESTRATEGIA_PT = {'easy': 'Fácil', 'hard': 'Difícil', 'mixed': 'Mista', 'random': 'Aleatória'}
CLASSIFICADOR_PT = {'Logistic Regression': 'Regressão Logística', 'k-NN': 'k-NN', 'SVM (RBF)': 'SVM (RBF)'}
ALGORITMO_PT = {'perceptron': 'Perceptron', 'nnls': 'NNLS'}
# Nome visível da dimensão da métrica (15/10/2026): "R²/R³/R⁴" em todo texto visível, nunca "R2/R3/R4".
DIM_PT = {2: 'R²', 3: 'R³', 4: 'R⁴'}


def _dim(n) -> str:
    """``2 → 'R²'``; fora de {2, 3, 4} cai em ``R<n>``."""
    try:
        return DIM_PT.get(int(n), f'R{int(n)}')
    except (TypeError, ValueError):
        return f'R{n}'


def _ordem_fg(grupo: str):
    """Chave de ordenação dos grupos "<rótulo> | <atributos>" das figuras de F/G: F primeiro; nos G,
    Mulher/Homem antes de A/B (ordem do CLAUDE.md), depois os atributos."""
    g = str(grupo)
    return (0 if g.startswith("F") else 1, g[:2], "Mulher/Homem" not in g, g)


def _rotulo_fg(problem_base: str) -> str:
    """Nome legível de um problema de F/G nas figuras: ``F_meia_lua`` → ``F (meia-lua)``,
    ``G1_homem_mulher`` → ``G1 Mulher/Homem``, ``G2_homem_mulher_classesAB`` → ``G2 A/B``."""
    pb = str(problem_base)
    if pb.startswith("F_"):
        return "F (meia-lua)"
    if pb.startswith("G"):
        return f"{pb[:2]} {'A/B' if pb.endswith('classesAB') else 'Mulher/Homem'}"
    return pb


def salvar_figura(fig, filename, dpi=150, ja_cinza=False, cor=None):
    """Salva a figura em DOIS arquivos: ``<stem>_color.png`` (como desenhada) e
    ``<stem>_gray.png`` (preto e branco com símbolos/hachuras, via `estilo_cinza.para_cinza`).
    ``cor="color"`` ou ``cor="gray"`` salva só aquela versão (figuras desenhadas duas vezes,
    uma por estilo, como as de símbolos); ``ja_cinza=True`` equivale a ``cor="gray"``.
    A figura é MODIFICADA no lugar pela conversão; chame por último, antes de fechá-la."""
    if not filename:
        return
    pendentes = garantir_legendas(fig)          # toda figura com séries tem legenda
    if pendentes:
        print(f"  ⚠️ sem legenda em {os.path.basename(filename)}: {pendentes}")
    if ja_cinza:
        cor = "gray"
    if cor in ("color", "gray"):
        fig.savefig(com_cor(filename, cor), dpi=dpi, bbox_inches='tight')
        return
    fig.savefig(com_cor(filename, "color"), dpi=dpi, bbox_inches='tight')
    para_cinza(fig)
    fig.savefig(com_cor(filename, "gray"), dpi=dpi, bbox_inches='tight')


def _save_panels_individually(panels, combined_filename, dpi=150, ja_cinza=False, cor=None):
    """Salva cada painel de uma figura composta como imagem individual.

    Args:
        panels: lista de (sufixo, draw_func, figsize) onde:
            - sufixo: string adicionada ao nome do arquivo (ex: "problema_a")
            - draw_func: callable(ax) que desenha em um único eixo
            - figsize: (largura, altura) da figura individual
        combined_filename: caminho completo do arquivo combinado
            (ex: "pasta/bloco1_graphic_05_A_passo4_w_distribuicao__gpt4mini.png"); o painel vira
            "pasta/bloco1_graphic_05_A_passo4_w_distribuicao_<sufixo>__gpt4mini.png" (asset_variant)
            e cada um sai em ``_color`` e ``_gray`` (salvar_figura)
        dpi: resolução das imagens individuais
    """
    if not combined_filename:
        return
    for suffix, draw_func, figsize in panels:
        fig_ind, ax_ind = plt.subplots(1, 1, figsize=figsize)
        draw_func(ax_ind)
        fig_ind.tight_layout()
        salvar_figura(fig_ind, asset_variant(combined_filename, suffix), dpi=dpi, ja_cinza=ja_cinza, cor=cor)
        plt.close(fig_ind)


def _config_base(resultados, nomes=None, k_comum=False):
    """Filtra os resultados dos Problemas A, B e C para a configuração BASE do grid.

    Mantém só prompt ``default`` e atributos ``x1/x2`` (as variantes de prompt e de
    nome de atributo rodam em outra grade de k e contaminariam as médias por k);
    ``nomes`` restringe ainda aos nomes de classe dados (ex.: ``("A", "B")``);
    ``k_comum`` mantém só os k presentes em TODOS os pares de nomes de classe
    (os nomes alternativos rodam em K_VIESES, o par A/B na grade completa).
    """
    base = [r for r in resultados
            if getattr(r, 'prompt_variant', 'default') == 'default'
            and tuple(getattr(r, 'feature_names', ('x1', 'x2'))) == ('x1', 'x2')]
    if nomes is not None:
        base = [r for r in base if tuple(r.nomes_classes) == tuple(nomes)]
    if k_comum and base:
        por_nome = {}
        for r in base:
            por_nome.setdefault(tuple(r.nomes_classes), set()).add(r.k_passo4_bc)
        comuns = set.intersection(*por_nome.values())
        base = [r for r in base if r.k_passo4_bc in comuns]
    return base


def _centroides_locais(X, y_llm, learned_metric):
    """Centróides dos rótulos do LLM no próprio problema (B/C), como em
    bc_passos_2_a_4; com o LLM colapsado (regra única `colapsou`: classe única ou
    minoritária < MIN_PONTOS_CLASSE_MINORITARIA) usa os centróides do A."""
    if y_llm is not None and not colapsou(y_llm):
        return compute_centroids(X, y_llm)
    return learned_metric.centroids


def _grouped_bar_offsets(n_series: int, bar_width: float) -> np.ndarray:
    """Deslocamentos centrados no tick para ``n_series`` barras agrupadas."""
    return (np.arange(n_series) - (n_series - 1) / 2.0) * bar_width

def _main_expert(results_e):
    """Filtra os resultados do Problema E (Passo 3) para UM especialista (o primeiro que aparece, que é o
    principal ``E1`` no protocolo; antigo ``aniso_x2``). Com RUN_PROBLEMA_E2_E3 há 3 especialistas e
    misturá-los numa curva única diluiria o sinal. Retorna (lista, nome do especialista)."""
    if not results_e:
        return results_e, None
    name = getattr(results_e[0], 'expert_name', None)
    if name is None:
        return results_e, None
    return [r for r in results_e if getattr(r, 'expert_name', name) == name], name



def plot_abc_passo1(
    X_a: np.ndarray, y_a: np.ndarray,
    X_b: np.ndarray, y_b: np.ndarray,
    X_c: np.ndarray, y_c: np.ndarray,
    filename: str = None
):
    """Visualiza os três problemas sintéticos lado a lado para inspeção visual da geometria."""
    problems = [
        (X_a, y_a, PROBLEM_A_CENTERS, PROBLEM_A_STD, "PROBLEMA A, Passo 1\n(estimação da métrica nos Passos 2 a 6)"),
        (X_b, y_b, PROBLEM_B_CENTERS, PROBLEM_B_STD, "PROBLEMA B, Passo 1\n(transferência nos Passos 2 a 4)"),
        (X_c, y_c, PROBLEM_C_CENTERS, PROBLEM_C_STD, "PROBLEMA C, Passo 1\n(transferência nos Passos 2 a 4)")
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
    fig.suptitle('Problemas A, B e C, Passo 1: os pontos gerados (rótulo real)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    suffixes = ["problema_a", "problema_b", "problema_c"]
    panels = [
        (suffixes[i], lambda ax, p=problems[i]: _draw_problem(ax, *p), (7, 5))
        for i in range(3)
    ]
    _save_panels_individually(panels, filename)


def plot_e_passo1(
    X_e: np.ndarray, y_gt_e: np.ndarray,
    y_expert_e: np.ndarray,
    expert_w: np.ndarray,
    expert_centroids: np.ndarray,
    filename: str = None
):
    """
    Visualiza o Problema E (especialista linear do Bloco 2) com a fronteira de decisão do especialista.
    Compara os rótulos verdadeiros (ground truth) com os rótulos do especialista lado a lado.
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

    def _draw_rotulo_real(ax):
        ax.scatter(X_e[:, 0], X_e[:, 1], c=y_gt_e, cmap="coolwarm",
                   alpha=0.7, edgecolor="k", s=60)
        ax.scatter(*PROBLEM_E_CENTERS[0], marker='*', s=300, c='blue',
                   edgecolor='black', linewidth=2, label='Centro Verdadeiro 0', zorder=5)
        ax.scatter(*PROBLEM_E_CENTERS[1], marker='*', s=300, c='red',
                   edgecolor='black', linewidth=2, label='Centro Verdadeiro 1', zorder=5)
        ax.set_title("Problema E, Passo 1: rótulo real", fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    def _draw_expert_boundary(ax):
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0.5])
        ax.scatter(X_e[:, 0], X_e[:, 1], c=y_expert_e, cmap="coolwarm",
                   alpha=0.7, edgecolor="k", s=60)
        ax.scatter(*expert_centroids[0], marker='X', s=300, c='blue',
                   edgecolor='k', linewidth=2, label='Centróide do Especialista 0', zorder=5)
        ax.scatter(*expert_centroids[1], marker='X', s=300, c='red',
                   edgecolor='k', linewidth=2, label='Centróide do Especialista 1', zorder=5)
        ax.set_title(f"Problema E, Passo 1: rótulos do especialista\n(W=[{expert_w[0]:.1f}, {expert_w[1]:.1f}])",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    def _draw_margin_heatmap(ax):
        im = ax.contourf(xx, yy, conf_grid, levels=20, cmap="RdYlGn")
        ax.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0.5])
        plt.colorbar(im, ax=ax, label="Margem (confiança)")
        ax.scatter(X_e[:, 0], X_e[:, 1], c='black', alpha=0.3, s=20, label='amostras do Problema E')
        ax.set_title("Problema E, Passo 1: margem do especialista\n(distância à fronteira)", fontsize=11, fontweight='bold')
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _draw_rotulo_real(axes[0])
    _draw_expert_boundary(axes[1])
    _draw_margin_heatmap(axes[2])
    fig.suptitle(f'Problema E, Passo 1: o especialista (W=[{expert_w[0]:.1f}, {expert_w[1]:.1f}]) rotula os 150 pontos',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    panels = [
        ("rotulo_real", _draw_rotulo_real, (7, 5)),
        ("fronteira_perito", _draw_expert_boundary, (7, 5)),
        ("margem_perito", _draw_margin_heatmap, (7, 5)),
    ]
    _save_panels_individually(panels, filename)


def plot_d_passo5_teto(results: List[dict], filename: str = None):
    """Teto do Problema D: acurácia da métrica ajustada ao rótulo real da meia-lua,
    por número de atributos (2, 3, 4). Mostra até onde a métrica diagonal em espaço
    aumentado aproxima a fronteira não-linear.
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
            means.append(sub['accuracy_vs_true'].mean() if len(sub) else 0)
            stds.append(sub['accuracy_vs_true'].std() if len(sub) > 1 else 0)
        offset = (j - (len(algos) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               color=cmap(j), edgecolor='black', alpha=0.85, label=ALGORITMO_PT.get(algo, algo))
    ax.set_xticks(x)
    ax.set_xticklabels([f'{nf} atributos' for nf in nfeats])
    ax.set_ylabel('Acurácia da métrica vs rótulo real')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    ax.set_title(
        'Problema D, Passo 5 (teto): W ajustado ao rótulo real da meia-lua em R², R³ e R⁴\n'
        '(a métrica diagonal só aproxima a fronteira não-linear)',
        fontweight='bold', fontsize=11,
    )
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=140)
    plt.close()


def plot_bc_passo3_consistencia(resultados: List[ResultadoABC], filename: str = None):
    """Consistência, Cohen's κ e F1-Macro nos Problemas B e C (bloco1_graphic_10), por k."""
    resultados = _config_base(resultados, nomes=("A", "B"))   # grade completa só no par A/B
    if not resultados:
        return
    df = pd.DataFrame([
        {
            'model': f"{r.provider}/{r.model_name} (t={r.temperature})",
            'k': r.k_passo4_bc,
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

    df_cons = df.groupby('k').agg({
        'consistencia_b': ['mean', 'std'], 'consistencia_c': ['mean', 'std'],
    }).reset_index()
    df_cons.columns = ['k', 'b_mean', 'b_std', 'c_mean', 'c_std']
    x = np.arange(len(df_cons))
    width = 0.35

    df_kappa = df.groupby('k').agg({'kappa_b': ['mean', 'std'], 'kappa_c': ['mean', 'std']}).reset_index()
    df_kappa.columns = ['k', 'b_mean', 'b_std', 'c_mean', 'c_std']

    df_f1 = df.groupby('k').agg({'f1_b': ['mean', 'std'], 'f1_c': ['mean', 'std']}).reset_index()
    df_f1.columns = ['k', 'b_mean', 'b_std', 'c_mean', 'c_std']

    def _draw_consistency(ax):
        ax.bar(x - width/2, df_cons['b_mean'], width, yerr=df_cons['b_std'],
               label='Problema B', color='steelblue', capsize=3)
        ax.bar(x + width/2, df_cons['c_mean'], width, yerr=df_cons['c_std'],
               label='Problema C', color='coral', capsize=3)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='perfeito (1,0)')
        ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.7, label='acaso (0,5)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(n)}-shot' for n in df_cons['k']])
        ax.set_ylabel('Consistência')
        ax.set_title('Consistência (Passo 3): Problema B vs C', fontweight='bold')
        ax.set_ylim(0, 1.35); ax.legend(loc='upper center', ncol=4, fontsize=8)   # acima das barras

    def _draw_kappa(ax):
        ax.bar(x - width/2, df_kappa['b_mean'], width, yerr=df_kappa['b_std'],
               label='Problema B', color='steelblue', capsize=3)
        ax.bar(x + width/2, df_kappa['c_mean'], width, yerr=df_kappa['c_std'],
               label='Problema C', color='coral', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(n)}-shot' for n in df_kappa['k']])
        ax.set_ylabel("Cohen's κ")
        ax.set_title("Cohen's κ (Passo 3): B vs C", fontweight='bold')
        ax.set_ylim(-0.1, 1.3); ax.legend(loc='upper center', ncol=2, fontsize=8)

    def _draw_f1(ax):
        ax.bar(x - width/2, df_f1['b_mean'], width, yerr=df_f1['b_std'],
               label='Problema B', color='steelblue', capsize=3)
        ax.bar(x + width/2, df_f1['c_mean'], width, yerr=df_f1['c_std'],
               label='Problema C', color='coral', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(n)}-shot' for n in df_f1['k']])
        ax.set_ylabel('F1-Macro')
        ax.set_title('F1-Macro (Passo 3): B vs C', fontweight='bold')
        ax.set_ylim(0, 1.3); ax.legend(loc='upper center', ncol=2, fontsize=8)

    def _draw_boxplot(ax):
        from matplotlib.patches import Patch
        metrics_data = []
        nomes_metricas = ['Consist. B', 'Consist. C', "Cohen's κ B", "Cohen's κ C", 'F1-Macro B', 'F1-Macro C']
        for col, label in zip(['consistencia_b', 'consistencia_c', 'kappa_b', 'kappa_c', 'f1_b', 'f1_c'],
                              nomes_metricas):
            for val in df[col]:
                metrics_data.append({'Metric': label, 'Value': val})
        df_metrics = pd.DataFrame(metrics_data)
        colors_bp = ['steelblue', 'coral'] * 3
        positions = [0, 0.6, 1.5, 2.1, 3.0, 3.6]
        for i, (metric, color) in enumerate(zip(nomes_metricas, colors_bp)):
            data = df_metrics[df_metrics['Metric'] == metric]['Value']
            bp = ax.boxplot([data], positions=[positions[i]], widths=0.4, patch_artist=True)
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.7)
        ax.set_xticks([0.3, 1.8, 3.3])
        ax.set_xticklabels(['Consistência', "Cohen's κ", 'F1-Macro'])
        ax.set_ylabel('Pontuação')
        ax.set_title('Distribuição de todas as métricas (Passo 3; Passo 4 onde k > 0)', fontweight='bold')
        legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Problema B'),
                           Patch(facecolor='coral', alpha=0.7, label='Problema C')]
        ax.legend(handles=legend_elements, loc='lower right')

    # Figura combinada
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _draw_consistency(axes[0, 0])
    _draw_kappa(axes[0, 1])
    _draw_f1(axes[1, 0])
    _draw_boxplot(axes[1, 1])
    fig.suptitle('Problemas B e C, Passo 3 (e Passo 4 onde há few-shot): LLM contra a métrica do A',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    panels = [
        ("consistencia", _draw_consistency, (7, 5)),
        ("kappa", _draw_kappa, (7, 5)),
        ("f1", _draw_f1, (7, 5)),
        ("distribuicao", _draw_boxplot, (7, 5)),
    ]
    _save_panels_individually(panels, filename)


# Pares de nomes de classe do protocolo (mesma lista de NOMES_CLASSES no runner; o
# plots.py não importa o runner para não criar ciclo). Os invertidos são a tupla ao
# contrário (NOMES_CLASSES_INVERTIDAS).
PARES_NOMES_BASE: List[Tuple[str, str]] = [("A", "B"), ("0", "1"), ("Positivo", "Negativo"), ("Azul", "Vermelho")]


def _par_base(nomes) -> Optional[Tuple[str, str]]:
    """Par base a que ``nomes`` pertence (ele mesmo ou o invertido); None se não for do protocolo."""
    t = tuple(nomes)
    for par in PARES_NOMES_BASE:
        if t == par or t == par[::-1]:
            return par
    return None


def plot_vies_nomes_classe(resultados: List[ResultadoABC], filename: str = None):
    """Efeito do NOME das classes na consistência (bloco1_graphic_16): só os 4 pares base
    (A/B, 0/1, Positivo/Negativo, Azul/Vermelho), sem os invertidos (esses ficam no
    bloco1_graphic_15), um painel por regime (zero-shot e 10-shot), sem misturar k na média."""
    resultados = _config_base(resultados, k_comum=True)   # mesmos k para todos os nomes
    resultados = [r for r in resultados if tuple(r.nomes_classes) in PARES_NOMES_BASE]
    if not resultados:
        return
    df = pd.DataFrame([
        {
            'nomes': f"{r.nomes_classes[0]}/{r.nomes_classes[1]}",
            'k': r.k_passo4_bc,
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
        }
        for r in resultados
    ])
    ordem = [f"{a}/{b}" for a, b in PARES_NOMES_BASE if f"{a}/{b}" in set(df['nomes'])]
    regimes = sorted(df['k'].unique())

    def _draw_regime(ax, k):
        d = df[df['k'] == k]
        g = d.groupby('nomes').agg({'consistencia_b': ['mean', 'std'], 'consistencia_c': ['mean', 'std']})
        g.columns = ['b_mean', 'b_std', 'c_mean', 'c_std']
        g = g.reindex(ordem).fillna({'b_std': 0.0, 'c_std': 0.0})
        x = np.arange(len(ordem))
        width = 0.35
        ax.bar(x - width/2, g['b_mean'], width, yerr=g['b_std'],
               label='Problema B', color='steelblue', capsize=3, edgecolor='black')
        ax.bar(x + width/2, g['c_mean'], width, yerr=g['c_std'],
               label='Problema C', color='coral', capsize=3, edgecolor='black')
        for i_n, nome in enumerate(ordem):      # slot vazio = colapso (do A, ou de B/C) em todas as sementes
            for dx, col in ((-width/2, 'b_mean'), (width/2, 'c_mean')):
                if np.isnan(g.loc[nome, col]):
                    ax.text(x[i_n] + dx, 0.02, 'colapso', rotation=90, fontsize=7, ha='center', va='bottom')
        ax.set_xticks(x)
        ax.set_xticklabels(ordem, rotation=30, ha='right')
        ax.set_ylabel('Consistência (média ± desvio entre sementes)')
        ax.set_title(f'{"zero-shot" if k == 0 else f"{k}-shot"}', fontweight='bold')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='0,5 (acaso)')
        ax.set_ylim(0, 1.3)                                   # espaço para a legenda acima das barras
        ax.legend(fontsize=8, loc='upper center', ncol=3)
        ax.grid(True, alpha=0.3, axis='y')

    fig, axes = plt.subplots(1, len(regimes), figsize=(6.5 * len(regimes), 5), squeeze=False)
    for ax, k in zip(axes[0], regimes):
        _draw_regime(ax, k)
    fig.suptitle('Viés de nomes de classe (Problema A, Passo 2, e B/C, Passos 3 e 4): consistência por par base e regime',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()
    _save_panels_individually([(f"{k}shot", (lambda ax, kk=k: _draw_regime(ax, kk)), (7, 5)) for k in regimes],
                              filename)


def plot_final_comparacao_modelos(resultados: List[ResultadoABC], filename: str = None):
    """Compara modelos se múltiplos foram testados (final_graphic_08): configuração base A/B, prompt
    default, x1/x2 e SÓ zero-shot (``k_passo4_bc == 0``: Passo 4 do A sobre o Passo 2; Passo 3 de
    B/C), como o título promete; até 15/10/2026 as médias misturavam o Passo 4 de B/C (k > 0)."""
    resultados = [r for r in _config_base(resultados, nomes=("A", "B")) if r.k_passo4_bc == 0]
    if not resultados:
        return
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
        ax.bar(i - width/2, row['consistencia_b'], width, color=colors[i], edgecolor='black', alpha=0.8,
               label='Problema B' if i == 0 else None)
        ax.bar(i + width/2, row['consistencia_c'], width, color=colors[i], edgecolor='black', alpha=0.5,
               hatch='//', label='Problema C' if i == 0 else None)
    ax.set_xticks(range(len(df_models)))
    ax.set_xticklabels([m.split('/')[-1] for m in df_models['model']], rotation=45, ha='right')
    ax.set_ylabel('Consistência')
    ax.set_title('Consistência em B e C (Passo 3, zero-shot; média entre sementes)', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', fontsize=9)

    ax = axes[1]
    metrics = ['consistencia_b', 'consistencia_c', 'kappa_b', 'kappa_c', 'fidelidade']
    labels = ['Consist. B', 'Consist. C', "Cohen's κ B", "Cohen's κ C", 'Fidelidade A']
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
    ax.set_title('Todas as métricas (configuração base, zero-shot; média entre sementes)', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle('Comparação de modelos (classes A/B, prompt padrão, x1/x2): A, Passo 4; B e C, Passo 3',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()


def plot_bc_passo3_sementes(resultados: List[ResultadoABC], filename: str = None):
    """Consistência em B e C por semente (bloco1_graphic_07): ZERO-SHOT da configuração base
    (uma coleta por seed, logo sem barra de erro). Se não houver zero-shot, usa todos os k e
    diz "média sobre k" no título."""
    resultados = _config_base(resultados, nomes=("A", "B"))
    if not resultados:
        return
    zero = [r for r in resultados if r.k_passo4_bc == 0]
    regime = 'zero-shot' if zero else f"média sobre k ∈ {sorted({r.k_passo4_bc for r in resultados})}"
    if zero:
        resultados = zero
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
    df_seeds = df_seeds.fillna({'b_std': 0.0, 'c_std': 0.0})   # 1 coleta por seed em zero-shot

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
    ax.set_title(f'Problemas B e C, Passo 3: consistência por semente ({regime}, classes A/B, prompt padrão)',
                 fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()


def plot_e_passo3_curva(
    results_e: List[ResultadoAprendiz],
    filename: str = None
):
    """
    Curva de aprendizado mostrando como o LLM melhora com mais exemplos,
    discriminada por estratégia de seleção de exemplos.

    Esta é a VISUALIZAÇÃO PRINCIPAL do Passo 3 do Problema E:
    - Eixo X: número de exemplos few-shot
    - Eixo Y: acurácia (LLM vs especialista)
    - Linhas: uma por estratégia (easy, hard, mixed, random)
    """
    results_e, expert_label = _main_expert(results_e)
    df = pd.DataFrame([
        {
            'model': f"{r.provider}/{r.model_name}",
            'k': r.k,
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
        ('accuracy', 'Acurácia (LLM vs especialista)'),
        ('kappa', "Cohen's κ"),
        ('f1', 'F1-Macro'),
    ]

    def _draw_metric(ax, metric_col, metric_name):
        for strategy in EXAMPLE_STRATEGIES:
            df_strat = df[df['strategy'] == strategy]
            if len(df_strat) == 0:
                continue
            df_grouped = df_strat.groupby('k').agg({
                metric_col: ['mean', 'std']
            }).reset_index()
            df_grouped.columns = ['k', 'mean', 'std']
            df_grouped = df_grouped.sort_values('k')
            ax.errorbar(
                df_grouped['k'], df_grouped['mean'],
                yerr=df_grouped['std'],
                marker=strategy_markers[strategy],
                color=strategy_colors[strategy],
                label=ESTRATEGIA_PT.get(strategy, strategy),
                linewidth=2, markersize=8, capsize=4
            )
        if metric_col == 'kappa':      # para Cohen's κ o acaso é 0, não 0,5 (15/10/2026)
            ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='acaso (κ = 0)')
        else:
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='acaso (0,5)')
        ax.set_xlabel('k exemplos', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'Problema {expert_label or "E"}, Passo 3: {metric_name}\nvs número de exemplos', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (metric_col, metric_name) in enumerate(metrics_config):
        _draw_metric(axes[i], metric_col, metric_name)
    fig.suptitle(f'Problema {expert_label or "E"}, Passo 3: curva de aprendizado por estratégia (especialista {expert_label})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    suffixes = ["accuracy", "kappa", "f1"]
    panels = [
        (suffixes[i], lambda ax, mc=metrics_config[i]: _draw_metric(ax, mc[0], mc[1]), (7, 5))
        for i in range(3)
    ]
    _save_panels_individually(panels, filename)


def plot_e_passo3_estrategias(
    results_e: List[ResultadoAprendiz],
    filename: str = None
):
    """
    Compara estratégias em cada nível de k_exemplos usando gráficos de barras agrupadas.
    Permite verificar qual estratégia de seleção de exemplos é mais eficaz para cada quantidade de shots.
    """
    results_e, expert_label = _main_expert(results_e)
    df = pd.DataFrame([
        {
            'k': r.k,
            'strategy': r.example_strategy,
            'accuracy': r.accuracy_llm_vs_expert,
            'kappa': r.kappa_llm_vs_expert,
        }
        for r in results_e
    ])

    # Filtra apenas os resultados few-shot (exclui zero-shot, que é igual para todas as estratégias)
    df_fs = df[df['k'] > 0]

    if len(df_fs) == 0:
        print("  Sem resultados few-shot para comparar estratégias.")
        return

    valores_k = sorted(df_fs['k'].unique())

    strategy_colors = {
        'easy': '#2ecc71',
        'hard': '#e74c3c',
        'mixed': '#3498db',
        'random': '#95a5a6',
    }

    def _draw_k(ax, n):
        df_n = df_fs[df_fs['k'] == n]
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
        ax.set_xticklabels([ESTRATEGIA_PT.get(s, s) for s in df_strat['strategy']], fontsize=10)
        ax.set_ylabel('Acurácia (LLM vs especialista)')
        ax.set_title(f'Problema {expert_label or "E"}, Passo 3: k = {n} exemplos', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        for bar, mean_val in zip(bars, df_strat['mean']):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{mean_val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Figura combinada
    fig, axes = plt.subplots(1, len(valores_k), figsize=(5 * len(valores_k), 5), squeeze=False)
    axes = axes[0]
    for ax_idx, n in enumerate(valores_k):
        _draw_k(axes[ax_idx], n)
    plt.suptitle(f'Problema {expert_label or "E"}, Passo 3: comparação de estratégias por número de exemplos (especialista {expert_label})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    panels = [
        (f"{n}shot", lambda ax, n_val=n: _draw_k(ax, n_val), (6, 5))
        for n in valores_k
    ]
    _save_panels_individually(panels, filename)


def plot_e_passo3_exemplos(
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
    Exibe seleções easy/hard/mixed/random sobre a fronteira de decisão do especialista.
    Fundamental para entender intuitivamente o que cada estratégia oferece ao LLM.
    """
    # Único ponto em que um plot reexecuta lógica de experimento: a seleção de
    # exemplos vive no runner (é núcleo do protocolo, não visualização). Import
    # tardio para não criar ciclo runner -> plots -> runner na carga dos módulos.
    from dissertacao_mestrado import e_passo3_selecionar_exemplos

    strategies = list(EXAMPLE_STRATEGIES)

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
        _, selected_indices = e_passo3_selecionar_exemplos(
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
        ax.set_title(f'Estratégia: {ESTRATEGIA_PT.get(strategy, strategy)}\n'
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
    plt.suptitle(f'Problema E, Passo 3: onde cada estratégia escolhe os exemplos ({n_examples} exemplos)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    panels = [
        (s, lambda ax, strat=s: _draw_strategy(ax, strat), (7, 5))
        for s in strategies
    ]
    _save_panels_individually(panels, filename)


def _w_unitario(r) -> np.ndarray:
    """Direção de Ŵ como vetor unitário (norma 2): ``w_direction`` do resultado quando
    existe, senão ``w_aprendido / ||w_aprendido||``. A magnitude do W do Perceptron
    depende de γ (varia de 1e-4 a 0,5 entre seeds), então comparar W bruto entre
    configurações mostra escala, não direção (revisão de 09/10/2026, F4)."""
    wd = getattr(r, 'w_direction', None)
    if wd is not None and np.all(np.isfinite(np.asarray(wd, float))) and np.linalg.norm(wd) > 0:
        wd = np.asarray(wd, float)[:2]
        return wd / np.linalg.norm(wd)
    w = np.asarray(getattr(r, 'w_aprendido', None) if getattr(r, 'w_aprendido', None) is not None
                   else [np.nan, np.nan], float)[:2]
    n = np.linalg.norm(w)
    return w / n if n > 0 else w


def _eixos_w_unitario(ax):
    """Eixos do quarto de círculo unitário para scatters de direção de Ŵ."""
    t = np.linspace(0, np.pi / 2, 50)
    arco, = ax.plot(np.cos(t), np.sin(t), color='gray', linestyle=':', linewidth=1, label='||w|| = 1')
    arco._referencia = True       # referência: a versão P&B não lhe dá marcador (estilo_cinza)
    ax.set_xlabel('$w_1 / ||w||$ (peso de $x_1$)'); ax.set_ylabel('$w_2 / ||w||$ (peso de $x_2$)')
    ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.05, 1.1)
    ax.set_aspect('equal', adjustable='box')


def plot_a_passo4_w_distribuicao(resultados: List[ResultadoABC], filename: str = None):
    """Distribuição de Ŵ_LLM entre sementes e nomes de classe (pedido do orientador).

    Só a configuração base: zero-shot (W é estimado só no Passo 4, sobre os rótulos do Passo 2), prompt
    `default` e atributos x1/x2; as variantes de prompt e de nomes de atributo
    têm gráficos próprios (bloco1_graphic_18 e bloco1_graphic_17) e contaminariam a
    dispersão. Em zero-shot há UMA coleta por (seed, par de nomes de classe): os pontos
    por semente são os pares de nomes (bloco1_graphic_05)."""
    data = []
    for r in resultados:
        if (r.k_passo4_bc == 0 and r.prompt_variant == "default"
                and tuple(r.feature_names) == ("x1", "x2")):
            data.append({
                'seed': r.random_seed,
                'w0': r.w_aprendido[0],
                'w1': r.w_aprendido[1],
                'ratio': r.w_aprendido[0] / r.w_aprendido[1] if r.w_aprendido[1] != 0 else float('inf'),
                'nomes': f"{r.nomes_classes[0]}/{r.nomes_classes[1]}",
            })

    if not data:
        print("  Sem dados de W para plotar distribuição.")
        return

    df = pd.DataFrame(data)
    seeds = sorted(df['seed'].unique())
    seeds_str = [str(s) for s in seeds]
    # Boxplot com pesos NORMALIZADOS por ΣW (14/10/2026): a escala bruta do Ŵ do Perceptron é
    # arbitrária (depende de γ), só a direção é comparável entre sementes e pares de nomes.
    _soma = (df['w0'].abs() + df['w1'].abs()).replace(0, np.nan)
    w0_by_seed = [(df[df['seed'] == s]['w0'] / _soma[df['seed'] == s]).dropna().values for s in seeds]
    w1_by_seed = [(df[df['seed'] == s]['w1'] / _soma[df['seed'] == s]).dropna().values for s in seeds]
    x_seeds = np.arange(len(seeds))
    width = 0.35
    # razão ≥ 0: barras mín–máx entre pares de nomes (± desvio descia abaixo de 0; 15/10/2026)
    df_ratio = df.groupby('seed').agg({'ratio': ['mean', 'min', 'max']}).reset_index()
    df_ratio.columns = ['seed', 'mean', 'min', 'max']

    nomes_pares = list(dict.fromkeys(df['nomes']))
    marcadores = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    marcador_por_nome = {n: marcadores[i % len(marcadores)] for i, n in enumerate(nomes_pares)}

    def _draw_w_scatter(ax):
        # 14/10/2026: UMA série por par de nomes (marcador = par; a legenda nasce dos rótulos e a versão
        # P&B troca marcador e legenda juntos); a semente vai como texto ao lado do ponto; eixos em
        # vetor UNITÁRIO (a escala do Ŵ do Perceptron é arbitrária, só a direção compara).
        _norma = np.sqrt(df['w0'] ** 2 + df['w1'] ** 2).replace(0, np.nan)
        cores_par = plt.cm.tab10(np.linspace(0, 1, max(len(nomes_pares), 2)))
        for i_n, nome in enumerate(nomes_pares):
            sel = df['nomes'] == nome
            # uma cor por série (par de nomes): a conversão P&B mantém uma entrada de legenda por par
            ax.scatter(df.loc[sel, 'w0'] / _norma[sel], df.loc[sel, 'w1'] / _norma[sel], color=cores_par[i_n], s=90,
                       marker=marcador_por_nome[nome], edgecolor='black', label=f'classes {nome}', zorder=3)
        for _, r in df.iterrows():
            nr = np.sqrt(r['w0'] ** 2 + r['w1'] ** 2)
            if nr > 0:
                ax.annotate(str(int(r['seed'])), (r['w0'] / nr, r['w1'] / nr), textcoords='offset points',
                            xytext=(4, 3), fontsize=6)
        ax.set_xlabel('$w_1/‖W‖$ (direção, dimensão $x_1$)')
        ax.set_ylabel('$w_2/‖W‖$ (direção, dimensão $x_2$)')
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_title('Direção de Ŵ_LLM por par de nomes de classe (marcador); número = semente', fontweight='bold', fontsize=10)
        ax.legend(fontsize=7, ncol=2, loc='lower left'); ax.grid(True, alpha=0.3)

    def _draw_w_boxplot(ax):
        from matplotlib.patches import Patch
        bp1 = ax.boxplot(w0_by_seed, positions=x_seeds - width/2, widths=width*0.8, patch_artist=True)
        bp2 = ax.boxplot(w1_by_seed, positions=x_seeds + width/2, widths=width*0.8, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('steelblue'); patch.set_alpha(0.7)
        for patch in bp2['boxes']:
            patch.set_facecolor('coral'); patch.set_alpha(0.7)
        ax.set_xticks(x_seeds); ax.set_xticklabels(seeds_str)
        ax.set_xlabel('Semente aleatória'); ax.set_ylabel('Peso normalizado por ΣW')
        ax.set_title('Distribuição de $w_1$ e $w_2$ (normalizados por ΣW) por semente\n(pontos = pares de nomes de classe)', fontweight='bold', fontsize=10)
        ax.legend(handles=[Patch(facecolor='steelblue', alpha=0.7, label='$w_1$'),
                           Patch(facecolor='coral', alpha=0.7, label='$w_2$')])

    def _draw_w_ratio(ax):
        yerr = [(df_ratio['mean'] - df_ratio['min']).clip(lower=0), (df_ratio['max'] - df_ratio['mean']).clip(lower=0)]
        ax.bar(range(len(df_ratio)), df_ratio['mean'], yerr=yerr,
               color='mediumpurple', edgecolor='black', capsize=5, alpha=0.8, label='razão $w_1/w_2$ (média; mín–máx)')
        ax.set_xticks(range(len(df_ratio)))
        ax.set_xticklabels([str(int(s)) for s in df_ratio['seed']])
        ax.set_xlabel('Semente aleatória'); ax.set_ylabel('Razão $w_1/w_2$')
        ax.set_title('Razão $w_1/w_2$ por semente (média; mín–máx entre pares de nomes)', fontweight='bold', fontsize=10)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Euclidiana (razão=1)')
        ax.legend()

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _draw_w_scatter(axes[0])
    _draw_w_boxplot(axes[1])
    _draw_w_ratio(axes[2])
    plt.suptitle('Problema A, Passo 4: distribuição do vetor Ŵ_LLM estimado entre sementes e nomes de classe (rótulos zero-shot do Passo 2)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    panels = [
        ("scatter", _draw_w_scatter, (7, 5)),
        ("boxplot", _draw_w_boxplot, (7, 5)),
        ("ratio", _draw_w_ratio, (7, 5)),
    ]
    _save_panels_individually(panels, filename)


def plot_a_passo4_erros(
    X: np.ndarray, y_llm: np.ndarray, y_metric: np.ndarray,
    w: np.ndarray, centroids: np.ndarray, filename: str = None
):
    """Onde a métrica estimada (Problema A, Passo 4) discorda dos rótulos zero-shot do LLM (Passo 2)
    (bloco1_graphic_03; pedido do orientador).

    Painel 1: mapa com os pontos em que métrica = LLM e métrica ≠ LLM, sobre a fronteira.
    Painel 2: 1 − fidelidade por faixa de margem. A margem é calculada com W NORMALIZADO
    (W / ΣW): a magnitude de W do Perceptron depende de γ (chega a 1e-3), o que deixava
    todas as faixas do eixo x em "0.0"; a normalização não muda a fronteira nem a ordem dos pontos."""
    y_llm = np.asarray(y_llm); y_metric = np.asarray(y_metric)
    w = np.asarray(w, float)
    w_plot = w / np.sum(w) if np.sum(w) > 0 else w
    agreements = y_llm == y_metric
    disagreements = ~agreements
    fidelity = np.mean(agreements)

    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_with_metric(grid_points, centroids, w_plot)
    Z = Z.reshape(xx.shape)

    confidences, _ = compute_metric_confidence(X, centroids, w_plot)
    bins = np.linspace(0, confidences.max(), 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    error_rates, bin_counts = [], []
    for i in range(len(bins) - 1):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if i == len(bins) - 2:
            mask |= confidences == bins[-1]        # inclui o máximo na última faixa
        error_rates.append(np.mean(disagreements[mask]) if np.sum(mask) > 0 else 0)
        bin_counts.append(int(np.sum(mask)))

    def _draw_error_map(ax):
        ax.contourf(xx, yy, Z, alpha=0.15, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0.5])
        for cls, cor in ((0, 'tab:blue'), (1, 'tab:red')):
            m = agreements & (y_llm == cls)
            if m.any():
                ax.scatter(X[m, 0], X[m, 1], c=cor, alpha=0.5, edgecolor="gray", s=40,
                           label=f'métrica = LLM, classe {cls} ({int(m.sum())})')
        ax.scatter(X[disagreements, 0], X[disagreements, 1],
                   c='yellow', edgecolor="red", s=150, linewidth=2, marker='X',
                   label=f'métrica ≠ LLM ({np.sum(disagreements)})', zorder=5)
        ax.scatter(*centroids[0], marker='D', s=200, c='blue', edgecolor='k', linewidth=2, zorder=6, label='centróide 0')
        ax.scatter(*centroids[1], marker='D', s=200, c='red', edgecolor='k', linewidth=2, zorder=6, label='centróide 1')
        ax.set_title(f'Problema A, Passo 4: métrica Ŵ_LLM vs rótulos zero-shot do LLM (Passo 2)\nFidelidade (métrica vs LLM): {fidelity:.1%}', fontweight='bold')
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

    def _draw_error_by_margin(ax):
        colors_bar = ['#e74c3c' if r > 0.3 else '#f39c12' if r > 0.1 else '#2ecc71' for r in error_rates]
        bars = ax.bar(range(len(error_rates)), error_rates, color=colors_bar, edgecolor='black', alpha=0.8,
                      label='1 − fidelidade na faixa')
        for bar, n in zip(bars, bin_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'n={n}',
                    ha='center', va='bottom', fontsize=7)
        ax.set_xticks(range(len(error_rates)))
        ax.set_xticklabels([f'{b:.2g}' for b in bin_centers], rotation=45)
        ax.set_xlabel('Margem (distância à fronteira, W normalizado por ΣW)')
        ax.set_ylabel('1 − fidelidade (fração métrica ≠ LLM)')
        ax.set_title('1 − fidelidade por faixa de margem', fontweight='bold')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='0,5 (acaso)')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Figura combinada
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _draw_error_map(axes[0])
    _draw_error_by_margin(axes[1])
    plt.suptitle('Problema A, Passo 4: discordâncias entre a métrica Ŵ_LLM e os rótulos do LLM (Passo 2)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    panels = [
        ("mapa_erros", _draw_error_map, (7, 6)),
        ("erros_por_margem", _draw_error_by_margin, (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_vies_ordem_classes(resultados: List[ResultadoABC], filename: str = None):
    """Viés de ordem/posição das classes (bloco1_graphic_15): barras PAREADAS normal × invertido
    para cada par base. Painéis: consistência em B e em C por regime (k ∈ K_VIESES: zero-shot =
    Passo 3 de B/C, k-shot = Passo 4), sem misturar k na média, e UM painel de fidelidade no A
    (Passo 4, sobre os rótulos zero-shot do Passo 2): a métrica do A é estimada uma vez e não
    depende de k, logo o antigo painel "Fidelidade A, 10-shot" era cópia do zero-shot e foi
    removido (15/10/2026). Painéis individuais: ``_consistencia_{b,c}_<k>shot`` e ``_fidelidade_0shot``."""
    resultados = _config_base(resultados, k_comum=True)   # pares normais × invertidos nos mesmos k
    rows = []
    for r in resultados:
        par = _par_base(r.nomes_classes)
        if par is None:
            continue
        rows.append({
            'par': f"{par[0]}/{par[1]}",
            'ordem': 'normal' if tuple(r.nomes_classes) == par else 'invertido',
            'k': r.k_passo4_bc,
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
            'fidelidade': r.fidelidade_problema_a,
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    if df['ordem'].nunique() < 2:
        print("  Pulando viés de ordem das classes (não há pares invertidos nos resultados)")
        return
    pares = [f"{a}/{b}" for a, b in PARES_NOMES_BASE if f"{a}/{b}" in set(df['par'])]
    regimes = sorted(df['k'].unique())
    cores = {'normal': 'steelblue', 'invertido': 'coral'}
    hatch = {'normal': None, 'invertido': '//'}

    k_fid = 0 if 0 in regimes else regimes[0]      # a fidelidade do A não depende de k: uma linha por (seed, par, ordem)

    def _draw(ax, col, k, ylabel):
        d = df[df['k'] == k]
        x = np.arange(len(pares))
        width = 0.38
        for j, ordem in enumerate(('normal', 'invertido')):
            g = d[d['ordem'] == ordem].groupby('par')[col].agg(['mean', 'std']).reindex(pares)
            g['std'] = g['std'].fillna(0.0)
            rot = [f"{a}/{b}" if ordem == 'normal' else f"{b}/{a}" for a, b in (p.split('/') for p in pares)]
            ax.bar(x + (j - 0.5) * width, g['mean'], width, yerr=g['std'], capsize=3,
                   color=cores[ordem], hatch=hatch[ordem], edgecolor='black',
                   label=f'{ordem} ({", ".join(rot)})')
            for i_p, par in enumerate(pares):   # slot vazio = colapso (do A, ou de B/C) em todas as sementes
                if np.isnan(g.loc[par, 'mean']):
                    ax.text(x[i_p] + (j - 0.5) * width, 0.02, 'colapso', rotation=90, fontsize=7, ha='center', va='bottom')
        ax.set_xticks(x)
        ax.set_xticklabels(pares, rotation=30, ha='right')
        ax.set_ylabel(f'{ylabel} (média ± desvio entre sementes)')
        if col == 'fidelidade':
            titulo = f'{ylabel} (Passo 4, sobre o zero-shot do Passo 2; não depende de k)'
        else:
            titulo = f'{ylabel}, {"zero-shot (Passo 3)" if k == 0 else f"{k}-shot (Passo 4)"}'
        ax.set_title(titulo, fontweight='bold', fontsize=10)
        ax.set_ylim(0, 1.45)                                  # espaço para a legenda (2 linhas) acima das barras
        ax.legend(fontsize=7, loc='upper center', ncol=1)
        ax.grid(True, alpha=0.3, axis='y')

    paineis_bc = [('consistencia_b', 'Consistência B'), ('consistencia_c', 'Consistência C')]
    n_reg = len(regimes)
    fig = plt.figure(figsize=(18, 5 * n_reg))
    gs = fig.add_gridspec(n_reg, 3)
    for i, k in enumerate(regimes):
        for j, (col, ylabel) in enumerate(paineis_bc):
            _draw(fig.add_subplot(gs[i, j]), col, k, ylabel)
    _draw(fig.add_subplot(gs[:, 2]), 'fidelidade', k_fid, 'Fidelidade A')   # um painel só, ocupa a 3ª coluna
    fig.suptitle('Viés de ordem das classes: par normal × par invertido. Consistência em B/C por regime '
                 '(Passo 3 zero-shot; Passo 4 com k exemplos) e fidelidade no A (Passo 4, zero-shot)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    panels = []
    for k in regimes:
        for col, ylabel in paineis_bc:
            panels.append((f"{col}_{k}shot", (lambda ax, c=col, kk=k, yl=ylabel: _draw(ax, c, kk, yl)), (7, 5.5)))
    panels.append(("fidelidade_0shot", (lambda ax: _draw(ax, 'fidelidade', k_fid, 'Fidelidade A')), (7, 5.5)))
    _save_panels_individually(panels, filename)


def plot_vies_nomes_atributo(resultados: List[ResultadoABC], filename: str = None):
    """Compara métricas e Ŵ_LLM estimada entre diferentes nomes de atributos."""
    # Filtra resultados que têm feature_names não-padrão ou padrão para comparação
    feat_results = [r for r in resultados if r.feature_names != ("x1", "x2")
                    or (r.k_passo4_bc == 0 and r.nomes_classes == ("A", "B") and r.prompt_variant == "default")]

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
            'w0': _w_unitario(r)[0],
            'w1': _w_unitario(r)[1],
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
        ax.set_title('Consistência por nome de atributo', fontweight='bold')
        ax.legend(); ax.set_ylim(0, 1.1)

    def _draw_feat_fidelity(ax):
        ax.bar(x, df_grouped['fid_mean'], yerr=df_grouped['fid_std'],
               color='mediumpurple', capsize=3, edgecolor='black', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(df_grouped['features'], rotation=30, ha='right')
        ax.set_ylabel('Fidelidade no Problema A (Passo 4)')
        ax.set_title('Fidelidade da métrica por nome de atributo', fontweight='bold')
        ax.set_ylim(0, 1.1)

    def _draw_feat_weights(ax):
        ax.bar(x - width/2, df_grouped['w0_mean'], width, yerr=df_grouped['w0_std'],
               label='$w_1/||w||$ (1º atributo)', color='forestgreen', capsize=3, edgecolor='black')
        ax.bar(x + width/2, df_grouped['w1_mean'], width, yerr=df_grouped['w1_std'],
               label='$w_2/||w||$ (2º atributo)', color='darkorange', capsize=3, edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels(df_grouped['features'], rotation=30, ha='right')
        ax.set_ylabel('Componente do vetor unitário')
        ax.set_ylim(0, 1.15)
        ax.set_title('Direção de Ŵ_LLM (vetor unitário) por nome de atributo', fontweight='bold')
        ax.legend(fontsize=8)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _draw_feat_consistency(axes[0])
    _draw_feat_fidelity(axes[1])
    _draw_feat_weights(axes[2])
    plt.suptitle('Viés de nomes de atributo (Problema A, Passo 2, e B/C, Passo 3): efeito dos nomes semânticos na métrica estimada',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    panels = [
        ("consistencia", _draw_feat_consistency, (7, 6)),
        ("fidelidade", _draw_feat_fidelity, (7, 6)),
        ("pesos", _draw_feat_weights, (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_e_passo6_baselines(
    results_e: List[ResultadoAprendiz],
    results_baselines: List[Dict],
    filename: str = None
):
    """Compara LLM (Problema E, Passo 3) vs. baselines clássicos (k-NN, LR, SVM; Passo 6).

    Mostra curvas de aprendizado do LLM (mixed strategy) junto com os baselines treinados
    nos mesmos exemplos, respondendo: o LLM faz algo que um classificador trivial não faz?
    """
    if not results_e or not results_baselines:
        return

    # Um especialista só (o principal), tanto no LLM quanto nos baselines
    results_e, expert_label = _main_expert(results_e)
    results_baselines = [b for b in results_baselines
                         if b.get('expert_name', expert_label) == expert_label]

    # Filtra LLM: apenas estratégia mixed (a mais balanceada) para comparação justa e só k > 0:
    # "LLM (com exemplos)" não inclui o zero-shot (k = 0), que não tem baseline treinado (15/10/2026)
    df_llm = pd.DataFrame([{
        'k': r.k,
        'accuracy': r.accuracy_llm_vs_expert,
        'kappa': r.kappa_llm_vs_expert,
        'f1': r.f1_llm_vs_expert,
        'source': 'LLM (com exemplos)',
    } for r in results_e if r.example_strategy == 'mixed' and r.k > 0])

    df_bl = pd.DataFrame(results_baselines)
    df_bl = df_bl[df_bl['example_strategy'] == 'mixed'].copy()

    if df_llm.empty or df_bl.empty:
        return

    # k-NN leva o k no nome ("k-NN (k=3)" com 4 exemplos, "k-NN (k=5)" com 10+): agrupa por
    # FAMÍLIA (nome antes de " (k=") para ser uma série só; os k usados vão para o rodapé.
    df_bl['family'] = df_bl['model'].astype(str).str.replace(r'\s*\(k=\d+\)\s*$', '', regex=True)
    ks_por_familia = {}
    for fam, sub in df_bl.groupby('family'):
        ks = sorted({int(m) for m in sub['model'].astype(str).str.extract(r'k=(\d+)')[0].dropna()})
        if ks:
            ks_por_familia[fam] = ks

    clf_colors = {
        'LLM (com exemplos)': '#3498db',
        'k-NN': '#e74c3c',
        'Logistic Regression': '#2ecc71',
        'SVM (RBF)': '#9b59b6',
    }
    clf_markers = {
        'LLM (com exemplos)': 'D',
        'k-NN': 'o',
        'Logistic Regression': 's',
        'SVM (RBF)': '^',
    }
    clf_names = list(dict.fromkeys(df_bl['family']))

    metrics_config = [
        ('accuracy', 'accuracy_vs_expert', 'Acurácia (vs especialista)'),
        ('kappa', 'kappa_vs_expert', "Cohen's κ"),
        ('f1', 'f1_vs_expert', 'F1-Macro'),
    ]

    def _draw_baseline_metric(ax, llm_col, bl_col, title):
        df_llm_grouped = df_llm.groupby('k').agg({
            llm_col: ['mean', 'std']
        }).reset_index()
        df_llm_grouped.columns = ['k', 'mean', 'std']
        df_llm_grouped = df_llm_grouped.sort_values('k')
        ax.errorbar(
            df_llm_grouped['k'], df_llm_grouped['mean'],
            yerr=df_llm_grouped['std'],
            marker=clf_markers['LLM (com exemplos)'],
            color=clf_colors['LLM (com exemplos)'],
            label='LLM (com exemplos)',
            linewidth=2.5, markersize=9, capsize=4
        )
        for clf_name in clf_names:
            df_clf = df_bl[df_bl['family'] == clf_name]
            df_clf_grouped = df_clf.groupby('k').agg({
                bl_col: ['mean', 'std']
            }).reset_index()
            df_clf_grouped.columns = ['k', 'mean', 'std']
            df_clf_grouped = df_clf_grouped.sort_values('k')
            color_key = clf_name
            for k in clf_colors:
                if k in clf_name or clf_name.startswith(k.split(' ')[0]):
                    color_key = k
                    break
            ax.errorbar(
                df_clf_grouped['k'], df_clf_grouped['mean'],
                yerr=df_clf_grouped['std'],
                marker=clf_markers.get(color_key, 'x'),
                color=clf_colors.get(color_key, 'gray'),
                label=CLASSIFICADOR_PT.get(clf_name, clf_name),
                linewidth=1.5, markersize=7, capsize=3,
                linestyle='--', alpha=0.8
            )
        if llm_col == 'kappa':         # para Cohen's κ o acaso é 0, não 0,5 (15/10/2026)
            ax.axhline(y=0.0, color='red', linestyle=':', alpha=0.4, label='acaso (κ = 0)')
        else:
            ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.4, label='acaso (0,5)')
        ax.set_xlabel('k exemplos', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (llm_col, bl_col, title) in enumerate(metrics_config):
        _draw_baseline_metric(axes[i], llm_col, bl_col, title)
    rodape_k = '; '.join(f"{fam}: k = {', '.join(map(str, ks))} (k cresce com o número de exemplos)"
                         for fam, ks in ks_por_familia.items())
    plt.suptitle(f'Problema {expert_label or "E"}, Passos 3 e 6: LLM vs baselines clássicos (mesmos exemplos, estratégia mista)\n'
                 'O LLM faz algo diferente de um classificador trivial?',
                 fontsize=13, fontweight='bold')
    if rodape_k:
        fig.text(0.5, 0.005, rodape_k, ha='center', va='bottom', fontsize=8, style='italic')
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    suffixes = ["accuracy", "kappa", "f1"]
    panels = [
        (suffixes[i], lambda ax, mc=metrics_config[i]: _draw_baseline_metric(ax, *mc), (7, 6))
        for i in range(3)
    ]
    _save_panels_individually(panels, filename)


def plot_vies_variantes_prompt(resultados: List[ResultadoABC], filename: str = None):
    """Compara métricas e Ŵ_LLM estimada entre diferentes variantes de prompt.

    Testa se o prompt confunde a medição de consistência do LLM.
    """
    # Filtra resultados relevantes: classes A/B, k_exemplos=0, com variantes
    relevant = [r for r in resultados
                if r.nomes_classes == ("A", "B") and r.k_passo4_bc == 0
                and tuple(getattr(r, 'feature_names', ('x1', 'x2'))) == ('x1', 'x2')]
    if not relevant:
        return

    df = pd.DataFrame([{
        'variant': r.prompt_variant,
        'fidelidade': r.fidelidade_problema_a,
        'consistencia_b': r.consistencia_problema_b,
        'consistencia_c': r.consistencia_problema_c,
        'kappa_b': r.kappa_problema_b,
        'kappa_c': r.kappa_problema_c,
        'w_0': _w_unitario(r)[0],
        'w_1': _w_unitario(r)[1],
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
        ax.set_ylabel('Fidelidade no Problema A (Passo 4)')
        ax.set_title('Fidelidade da Métrica por Variante', fontweight='bold')
        ax.set_ylim(0, 1.1)

    def _draw_pv_w_scatter(ax):
        for variant in variants:
            df_v = df[df['variant'] == variant]
            ax.scatter(df_v['w_0'], df_v['w_1'], c=colors.get(variant, 'gray'),
                       label=labels_map.get(variant, variant),
                       s=80, edgecolors='black', linewidths=0.5, alpha=0.8)
        _eixos_w_unitario(ax)
        ax.set_title('Direção de Ŵ_LLM (vetor unitário) por variante de prompt\n(estabilidade entre prompts)', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Figura combinada
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    _draw_pv_consistency(axes[0])
    _draw_pv_fidelity(axes[1])
    _draw_pv_w_scatter(axes[2])
    plt.suptitle('Viés de variantes de prompt (Problema A, Passo 2, e B/C, Passo 3): mesmos dados, diferentes templates\n'
                 '(Se W muda, o prompt confunde a medição)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    panels = [
        ("consistencia", _draw_pv_consistency, (7, 6)),
        ("fidelidade", _draw_pv_fidelity, (7, 6)),
        ("w_scatter", _draw_pv_w_scatter, (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_e_passo5_ordem(results_order: List[ResultadoAprendiz], filename: str = None):
    """Compara métricas entre diferentes ordenações dos exemplos few-shot.

    Detecta viés de recência: se a ordem dos exemplos afeta a performance do LLM.
    """
    if not results_order:
        return

    df = pd.DataFrame([{
        'k': r.k,
        'ordering': r.example_strategy.replace("mixed_order_", ""),
        'accuracy': r.accuracy_llm_vs_expert,
        'kappa': r.kappa_llm_vs_expert,
        'f1': r.f1_llm_vs_expert,
    } for r in results_order])

    valores_k = sorted(df['k'].unique())
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
        ('accuracy', 'Acurácia (LLM vs especialista)'),
        ('kappa', "Cohen's κ"),
        ('f1', 'F1-Macro'),
    ]

    def _draw_order_metric(ax, metric, metric_name, show_legend=False):
        x = np.arange(len(valores_k))
        width = 0.18
        n_orderings = len(orderings)
        for i, ordering in enumerate(orderings):
            df_ord = df[df['ordering'] == ordering]
            means, stds = [], []
            for ns in valores_k:
                vals = df_ord[df_ord['k'] == ns][metric]
                means.append(vals.mean()); stds.append(vals.std())
            offset = (i - n_orderings / 2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   label=order_labels.get(ordering, ordering),
                   color=order_colors.get(ordering, f'C{i}'),
                   capsize=3, edgecolor='black', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([f'k = {ns}' for ns in valores_k]); ax.set_xlabel('k exemplos')
        ax.set_ylabel(metric_name); ax.set_title(f'Problema E1, Passo 5: {metric_name}', fontweight='bold')
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
    plt.suptitle('Problema E1, Passo 5: viés de ordem dos exemplos (viés de recência)\n'
                 'Mesmos exemplos (estratégia mista), diferentes ordenações',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    suffixes = ["accuracy", "kappa", "f1"]
    panels = [
        (suffixes[i], lambda ax, m=metrics_order[i]: _draw_order_metric(ax, m[0], m[1], show_legend=True), (7, 6))
        for i in range(3)
    ]
    _save_panels_individually(panels, filename)


def _dilution_reference(df: pd.DataFrame):
    """(n_hard, linhas de referência "hard puros") a partir de ``example_strategy``.

    O nome da estratégia é ``dilution_<N>hard_<M>easy`` (runner); a referência são as
    coletas com M = 0. Fallback (nomes fora do padrão): N = menor ``k_exemplos`` observado.
    """
    m = df['strategy'].astype(str).str.extract(r'dilution_(\d+)hard_(\d+)easy').astype(float)
    n_hard_col, n_easy_col = m[0], m[1]
    if n_hard_col.notna().any():
        n_hard = int(n_hard_col.dropna().iloc[0])
        ref = df[n_easy_col == 0]
    else:
        n_hard = int(df['k'].min())
        ref = df[df['k'] == n_hard]
    return n_hard, ref


def plot_e_passo4_diluicao(results_dilution: List[ResultadoAprendiz], filename: str = None):
    """Gráfico do experimento de diluição: N hard fixos + M easy progressivos."""
    if not results_dilution:
        return

    df = pd.DataFrame([{
        'k': r.k,
        'accuracy': r.accuracy_llm_vs_expert,
        'kappa': r.kappa_llm_vs_expert,
        'strategy': r.example_strategy,
    } for r in results_dilution])

    df_acc = df.groupby('k').agg({'accuracy': ['mean', 'std']}).reset_index()
    df_acc.columns = ['k', 'mean', 'std']
    df_acc = df_acc.sort_values('k')
    n_hard, ref_hard = _dilution_reference(df)
    xlabel = f'k exemplos ({n_hard} difíceis fixos + N fáceis)'

    df_kappa = df.groupby('k').agg({'kappa': ['mean', 'std']}).reset_index()
    df_kappa.columns = ['k', 'mean', 'std']
    df_kappa = df_kappa.sort_values('k')

    def _draw_dilution_accuracy(ax):
        ax.errorbar(df_acc['k'], df_acc['mean'], yerr=df_acc['std'],
                    marker='o', linewidth=2, markersize=8, capsize=4, color='#e74c3c',
                    label='acurácia (LLM vs especialista)')
        if len(ref_hard) > 0:
            ax.axhline(y=ref_hard['accuracy'].mean(), color='gray', linestyle='--',
                       alpha=0.7, label=f'só os {n_hard} difíceis ({ref_hard["accuracy"].mean():.1%})')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Acurácia (LLM vs especialista)')
        ax.set_title('Problema E1, Passo 4 (diluição): Acurácia (LLM vs especialista)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

    def _draw_dilution_kappa(ax):
        ax.errorbar(df_kappa['k'], df_kappa['mean'], yerr=df_kappa['std'],
                    marker='s', linewidth=2, markersize=8, capsize=4, color='#3498db',
                    label="Cohen's κ (LLM vs especialista)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cohen's κ")
        ax.set_title("Problema E1, Passo 4 (diluição): Cohen's κ", fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(-0.1, 1.05)

    # Figura combinada
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_dilution_accuracy(axes[0])
    _draw_dilution_kappa(axes[1])
    plt.suptitle('Problema E1, Passo 4: experimento de diluição (difíceis fixos + fáceis progressivos)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    panels = [
        ("accuracy", _draw_dilution_accuracy, (7, 5)),
        ("kappa", _draw_dilution_kappa, (7, 5)),
    ]
    _save_panels_individually(panels, filename)


def plot_a_passo4_r2_r3_r4(results_r2: List, results_r3: List, results_r4: Optional[List] = None,
                       filename: str = None):
    """Fidelidade da métrica com 2, 3 e 4 pesos (R2 → R3 → R4) sobre as MESMAS
    classificações zero-shot do LLM no Problema A (que viu apenas x1, x2).

    R3 acrescenta x3 = x1·x2 (hipérbole); R4 acrescenta x1², x2² (elipse). Se a
    fidelidade cresce com a dimensão, há evidência de não-linearidade implícita no
    critério do LLM; em um problema linear espera-se ganho ~nulo. ``results_r4`` é
    opcional (execuções antigas só têm R3)."""
    if not results_r2 or not results_r3:
        return

    spaces = [('2 pesos\n($w_1$, $w_2$)', results_r2),
              ('3 pesos\n$x_3 = x_1 \\cdot x_2$', results_r3)]
    if results_r4:
        spaces.append(('4 pesos\n$x_3 = x_1^2$, $x_4 = x_2^2$', results_r4))
    space_short = ['R²', 'R³', 'R⁴'][:len(spaces)]
    space_colors = ['steelblue', 'coral', 'seagreen'][:len(spaces)]

    _LEGADO = {'fidelity': 'accuracy', 'fidelity_nnls': 'accuracy_nnls'}   # chaves até 08/10/2026

    def _val(r, key):
        v = r.get(key)
        if v is None:
            v = r.get(_LEGADO.get(key, key))
        return v

    def _stats(rs, key):
        vals = [_val(r, key) for r in rs]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)

    means = [_stats(rs, 'fidelity')[0] for _, rs in spaces]
    stds = [_stats(rs, 'fidelity')[1] for _, rs in spaces]
    algo_list = [('Perceptron', 'fidelity', 'steelblue'), ('NNLS', 'fidelity_nnls', 'coral')]

    def _draw_r3_comparison(ax):
        bars = ax.bar([lbl for lbl, _ in spaces], means, yerr=stds, color=space_colors,
                      edgecolor='black', capsize=5, alpha=0.8, label='Fidelidade (média ± desvio entre seeds)')
        ax.set_ylabel('Fidelidade (métrica vs LLM)')
        ax.set_title('Perceptron: fidelidade por dimensão da métrica (R² → R³ → R⁴)',
                     fontweight='bold')
        ax.set_ylim(0, 1.32)                                  # espaço para valores, Δ e legenda acima das barras
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='0,5 (acaso)')
        # valor ACIMA da tampa da barra de erro (antes colidia com ela); Δ entre barras numa
        # linha própria, acima de todas as tampas (15/10/2026)
        for bar, mean, sd in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + (0.0 if np.isnan(sd) else sd) + 0.015,
                    f'{mean:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        y_delta = 1.04
        for i in range(1, len(means)):
            delta = means[i] - means[i - 1]
            ax.annotate(f'$\\Delta$ = {delta:+.1%}', xy=(i - 0.5, y_delta), ha='center', va='bottom',
                        fontsize=9, style='italic', color='green' if delta > 0 else 'gray')
        ax.legend(fontsize=8, loc='upper center', ncol=2)

    def _draw_r3r4_algo(ax):
        n_alg = len(algo_list)
        bar_width = 0.8 / n_alg
        x_pos = np.arange(len(spaces))
        offsets = _grouped_bar_offsets(n_alg, bar_width)
        for j, (name, key, color) in enumerate(algo_list):
            m = [_stats(rs, key)[0] for _, rs in spaces]
            sd = [_stats(rs, key)[1] for _, rs in spaces]
            ax.bar(x_pos + offsets[j], m, bar_width, yerr=sd, color=color,
                   edgecolor='black', capsize=4, alpha=0.8, label=name)
            for k, v in enumerate(m):
                if not np.isnan(v):     # valor acima da tampa da barra de erro (15/10/2026)
                    ax.text(x_pos[k] + offsets[j], v + (sd[k] if not np.isnan(sd[k]) else 0.0) + 0.015,
                            f'{v:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(x_pos); ax.set_xticklabels(space_short)
        ax.set_ylabel('Fidelidade (métrica vs LLM)')
        ax.set_title('Fidelidade por algoritmo em R² / R³ / R⁴', fontweight='bold')
        ax.set_ylim(0, 1.3)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='0,5 (acaso)')
        ax.legend(loc='upper center', ncol=3, fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # Figura combinada
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_r3_comparison(axes[0])
    _draw_r3r4_algo(axes[1])
    plt.suptitle('Problema A, Passo 4: fidelidade com 2, 3 e 4 pesos (R², R³, R⁴) sobre os rótulos zero-shot do Passo 2\n'
                 '(LLM viu apenas $x_1$, $x_2$; atributos extras adicionados nos bastidores)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    panels = [
        ("linear_vs_quadratica", _draw_r3_comparison, (7, 5)),
        ("algoritmos_r3r4", _draw_r3r4_algo, (7, 5)),
    ]
    _save_panels_individually(panels, filename)




def plot_a_passo4_perceptron_vs_nnls(results_perceptron: List, results_alternative: List,
                              filename: str = None):
    """Compara Ŵ_LLM estimada por Perceptron vs NNLS: fidelidade no A (Passo 4) e consistência em B e C (Passo 3)."""
    if not results_perceptron or not results_alternative:
        return

    bar_width = 0.25
    w_perc = np.array([_w_unitario(r) for r in results_perceptron])
    w_alt = np.array([_w_unitario(r) for r in results_alternative])
    fid_perc = [r.fidelidade_problema_a for r in results_perceptron]
    fid_alt = [r.fidelidade_problema_a for r in results_alternative]

    algo_list = [
        ('Perceptron', results_perceptron, 'steelblue'),
        ('NNLS', results_alternative, 'coral'),
    ]

    def _draw_algo_w_scatter(ax):
        ax.scatter(w_perc[:, 0], w_perc[:, 1], c='steelblue', s=100, edgecolor='k', label='Perceptron', zorder=3)
        ax.scatter(w_alt[:, 0], w_alt[:, 1], c='coral', s=100, edgecolor='k', label='NNLS', zorder=3)
        _eixos_w_unitario(ax)
        ax.set_title('Direção de Ŵ_LLM (vetor unitário): Perceptron vs NNLS', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

    def _draw_algo_fidelity(ax):
        box_data = [fid_perc, fid_alt]
        box_labels = ['Perceptron', 'NNLS']
        box_colors = ['steelblue', 'coral']
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        for i, color in enumerate(box_colors):
            bp['boxes'][i].set_facecolor(color); bp['boxes'][i].set_alpha(0.7)
        ax.set_ylabel('Fidelidade')
        ax.set_title('Problema A, Passo 4: fidelidade (métrica vs LLM)', fontweight='bold')
        ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3, axis='y')

    def _draw_algo_bc(ax, problema, cons_attr, kappa_attr):
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
        ax.set_xticks(x_pos); ax.set_xticklabels(['Consistência', "Cohen's κ"])
        ax.set_title(f'Problema {problema}, Passo 3: consistência', fontweight='bold')
        ax.set_ylim(-0.1, 1.1); ax.grid(True, alpha=0.3, axis='y'); ax.legend(fontsize=8)

    # Figura combinada
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _draw_algo_w_scatter(axes[0, 0])
    _draw_algo_fidelity(axes[0, 1])
    _draw_algo_bc(axes[1, 0], 'B', 'consistencia_problema_b', 'kappa_problema_b')
    _draw_algo_bc(axes[1, 1], 'C', 'consistencia_problema_c', 'kappa_problema_c')
    plt.suptitle('Comparação de algoritmos de otimização inversa\n(Problema A, Passo 4; Problemas B e C, Passo 3)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    panels = [
        ("w_scatter", _draw_algo_w_scatter, (7, 6)),
        ("fidelidade", _draw_algo_fidelity, (7, 6)),
        ("bc_passo3_b", lambda ax: _draw_algo_bc(ax, 'B', 'consistencia_problema_b', 'kappa_problema_b'), (7, 6)),
        ("bc_passo3_c", lambda ax: _draw_algo_bc(ax, 'C', 'consistencia_problema_c', 'kappa_problema_c'), (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_a_passo4_gamma(diagnostics: List[dict], filename: str = None):
    """Diagnóstico da busca binária em γ no Perceptron Estruturado (bloco1_graphic_09).

    Verifica se γ converge crescentemente entre execuções (não fica estagnado). Cada
    entrada de `diagnostics` é {"label": str, "gamma_final": float, "history": [dicts]}.
    Cada dict do history: {iter, gamma, gamma_lo, gamma_hi, violations, viable}.

    Painel 1: evolução de γ por iteração (curva por execução).
    Painel 2: γ_lo (limite inferior viável) por iteração, que deve ser monotônico.
    """
    if not diagnostics:
        print("  ⚠ Sem histórico de γ para plotar (nenhuma chamada com return_history=True).")
        return

    # 14/10/2026: com 36 configurações (seed × nomes × variante) a legenda dentro dos eixos cobria as
    # curvas; agora fica abaixo da figura, em colunas, e só no painel 2 (as cores são as mesmas).
    n_leg = max(1, len(diagnostics))
    ncol_leg = 4
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5 + 0.18 * int(np.ceil(n_leg / ncol_leg))))
    cmap = plt.get_cmap("tab20" if n_leg > 10 else "tab10")

    for idx, entry in enumerate(diagnostics):
        hist = entry.get("history") or []
        if not hist:
            continue
        iters = [h["iter"] for h in hist]
        gammas = [h["gamma"] for h in hist]
        gamma_los = [h["gamma_lo"] for h in hist]
        viable = [h["viable"] for h in hist]

        color = cmap(idx % (20 if n_leg > 10 else 10))
        label = f"{entry.get('label', f'exec_{idx}')} (γ*={entry.get('gamma_final', float('nan')):.3f})"

        # Painel 1: γ testado por iteração, marcadores diferentes para viável/inviável.
        # O rótulo fica na LINHA (sempre desenhada): o scatter "viável" fica vazio quando γ*=0
        # e, rotulado, deixava um handle sem correspondência na versão em cinza.
        axes[0].plot(iters, gammas, "-", color=color, alpha=0.5, linewidth=1.0, label=label)
        v_iters = [i for i, v in zip(iters, viable) if v]
        v_gammas = [g for g, v in zip(gammas, viable) if v]
        nv_iters = [i for i, v in zip(iters, viable) if not v]
        nv_gammas = [g for g, v in zip(gammas, viable) if not v]
        axes[0].scatter(v_iters, v_gammas, s=40, marker="o", color=color, label="_nolegend_",
                        edgecolor="black", linewidth=0.4)
        axes[0].scatter(nv_iters, nv_gammas, s=40, marker="x", color=color, label="_nolegend_")

        # Painel 2: γ_lo (melhor margem viável até o momento), que DEVE ser monotônico crescente
        axes[1].plot(iters, gamma_los, "-o", color=color, alpha=0.8, markersize=4,
                     label=label)

    axes[0].set_xlabel("Iteração da busca binária")
    axes[0].set_ylabel("γ testado")
    axes[0].set_title("γ candidato por iteração\n(○ = viável, × = inviável)", fontsize=10, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Iteração da busca binária")
    axes[1].set_ylabel("γ_lo (melhor margem viável)")
    axes[1].set_title("γ_lo (melhor margem viável) por iteração", fontsize=10, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    handles, labels = axes[1].get_legend_handles_labels()
    n_lin = int(np.ceil(len(labels) / ncol_leg))
    fig.legend(handles, labels, loc="lower center", ncol=ncol_leg, fontsize=6.5, frameon=False,
               title="seed | nomes de classe | variante de prompt (γ* final)", title_fontsize=7)

    plt.suptitle("Problema A, Passo 4: convergência da busca binária em γ (Perceptron Estruturado)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, min(0.45, 0.04 + 0.022 * n_lin), 1, 0.95))

    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
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


def _oraculo_accuracy(r: dict) -> float:
    """Acurácia da métrica recuperada vs rótulos do oráculo: coluna ``accuracy``
    (09/10/2026); execuções antigas gravaram a mesma grandeza como ``fidelity``."""
    v = r.get('accuracy')
    if v is None:
        v = r.get('fidelity')
    return float('nan') if v is None else float(v)


_ORACULO_ESPECIALISTA_LEGADO = {'euclidiana': 'B'}
# Problemas 1/2/3: E3 (euclidiano; 'B' nas execuções até 13/10/2026), E1 e E2.
_ORACULO_PROBLEMAS = ('Oráculo sobre E3', 'Oráculo sobre B', 'Oráculo sobre E1', 'Oráculo sobre E2')


def _oraculo_rotulo(r: dict) -> str:
    """Rótulo do eixo para uma linha do oráculo: "Oráculo sobre E3" (até 13/10/2026 "sobre B"),
    "Oráculo sobre E1", "Oráculo sobre E2". Aceita o nome antigo ``euclidiana`` do Oráculo sobre B."""
    prob = str(r.get('problem', ''))
    exp = _ORACULO_ESPECIALISTA_LEGADO.get(str(r.get('expert_name', '')), str(r.get('expert_name', '')))
    for antigo, novo in _ORACULO_ESPECIALISTA_LEGADO.items():
        prob = prob.replace(antigo, novo)
    if not exp or exp in prob:
        return prob
    return f"{prob}\n{exp}"


def _agregar_oraculo(resultados_oraculo: List[dict]):
    """Agrega as linhas do oráculo por (rótulo do problema, algoritmo) sobre as seeds:
    média, mínimo e máximo de acurácia, cosseno, razão w1/w2 e w normalizado, mais o
    W verdadeiro (igual em todas as seeds). Devolve (rótulos na ordem de chegada, dict)."""
    # Só B/E1/E2 (os Problemas 1/2/3); linhas de configurações extras de execuções
    # anteriores a 13/10/2026 (gaussianas anisotrópicas) são ignoradas.
    resultados_oraculo = [r for r in resultados_oraculo if _oraculo_rotulo(r) in _ORACULO_PROBLEMAS]
    labels = list(dict.fromkeys(_oraculo_rotulo(r) for r in resultados_oraculo))
    grupos: Dict[tuple, List[dict]] = {}
    for r in resultados_oraculo:
        grupos.setdefault((_oraculo_rotulo(r), r['algorithm']), []).append(r)
    agg = {}
    for (label, algo), rs in grupos.items():
        def _stat(vals):
            v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
            if v.size == 0:
                return (np.nan, np.nan, np.nan, 0)
            return (float(v.mean()), float(v.min()), float(v.max()), int(v.size))
        ratios = [r.get('recovered_w_ratio') for r in rs]
        agg[(label, algo)] = {
            'n_seeds': len({r.get('random_seed') for r in rs}),
            'n': len(rs),
            'true_w': np.array([rs[0]['true_w_0'], rs[0]['true_w_1']], float),
            'true_w_ratio': float(rs[0]['true_w_ratio']),
            'accuracy': _stat([_oraculo_accuracy(r) for r in rs]),
            'cosine': _stat([r.get('cosine_similarity') for r in rs]),
            'ratio': _stat(ratios),
            'n_ratio_inf': int(sum(1 for x in ratios if x is not None and np.isinf(x))),
            'w_norm_0': _stat([r.get('recovered_w_norm_0') for r in rs]),
            'w_norm_1': _stat([r.get('recovered_w_norm_1') for r in rs]),
        }
    return labels, agg


def plot_oraculo_recuperacao_w(resultados_oraculo: List[dict], filename: str = None):
    """Recuperação de W conhecido pelos dois algoritmos (validação do oráculo,
    bloco1_graphic_19): Oráculo sobre E3 (W = [1, 1]; sobre B até 13/10/2026) e sobre E1/E2, os Problemas
    1/2/3 do texto do orientador. Cada barra é a MÉDIA entre as seeds, com barra de erro mín–máx
    (antes de 09/10/2026 a figura mostrava só a última seed). Lê ``accuracy``
    (fallback ``fidelity`` nas execuções antigas)."""
    if not resultados_oraculo:
        return

    from matplotlib.lines import Line2D

    colors_algo = {'perceptron': 'steelblue', 'nnls': 'coral'}
    nomes_algo = {'perceptron': 'Perceptron', 'nnls': 'NNLS'}
    labels, agg = _agregar_oraculo(resultados_oraculo)
    n_labels = len(labels)
    x_base = np.arange(n_labels)
    n_seeds = max((v['n_seeds'] for v in agg.values()), default=1)
    faixa = 'mín–máx entre seeds' if n_seeds > 1 else '1 seed'

    def _yerr(stat, escala=1.0):
        """Barra mín–máx em torno da média (nunca negativa: arredondamento de ponto flutuante)."""
        m, lo, hi, _ = stat
        if np.isnan(m):
            return None
        return [[max(0.0, (m - lo) * escala)], [max(0.0, (hi - m) * escala)]]

    marcadores_algo = {'perceptron': 'o', 'nnls': 's'}     # marcador por algoritmo (distinção sem cor)

    def _draw_w_scatter(ax):
        for (label, algo), v in agg.items():
            cl = colors_algo.get(algo, 'gray')
            mk = marcadores_algo.get(algo, 'o')
            true_norm = v['true_w'] / np.sum(v['true_w'])
            for comp, stat in ((0, v['w_norm_0']), (1, v['w_norm_1'])):
                m, lo, hi, _ = stat
                if np.isnan(m):
                    continue
                ax.errorbar([true_norm[comp]], [m], yerr=_yerr(stat), fmt=mk, color=cl,
                            markersize=7, markeredgecolor='k', markeredgewidth=0.5, capsize=3,
                            alpha=0.75, zorder=3)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='recuperação perfeita')
        ax.set_xlabel('$w$ verdadeiro (normalizado por ΣW)')
        ax.set_ylabel('$w$ recuperado (normalizado por ΣW)')
        ax.set_title(f'W normalizado: verdadeiro vs recuperado\n(média; barras = {faixa})', fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        legend_elements = [
            Line2D([0], [0], marker=marcadores_algo['perceptron'], color='w', markerfacecolor='steelblue',
                   markeredgecolor='k', markersize=8, label='Perceptron'),
            Line2D([0], [0], marker=marcadores_algo['nnls'], color='w', markerfacecolor='coral',
                   markeredgecolor='k', markersize=8, label='NNLS'),
            Line2D([0], [0], color='k', linestyle='--', alpha=0.4, label='recuperação perfeita'),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

    def _draw_ratio(ax):
        bar_width = 0.25
        # Escala log: as razões verdadeiras são 0,2, 1 e 5 (B, E1, E2); em escala linear 0,2 sumia
        valores = [agg[(l, a)]['true_w_ratio'] for l in labels for a in colors_algo if (l, a) in agg]
        valores += [agg[k]['ratio'][2] for k in agg if not np.isnan(agg[k]['ratio'][2])]
        valores = [v for v in valores if np.isfinite(v) and v > 0]
        y_max = max(valores + [1.0]) * 3
        y_min = min(valores + [1.0]) / 3
        for i, label in enumerate(labels):
            first = next((agg[(label, a)] for a in colors_algo if (label, a) in agg), None)
            if first is None:
                continue
            tr = first['true_w_ratio']
            ax.bar(x_base[i] - bar_width, tr if np.isfinite(tr) else y_max, bar_width, color='gray',
                   edgecolor='k', alpha=0.7, label='verdadeira' if i == 0 else None)
            if not np.isfinite(tr):
                ax.annotate('∞', (x_base[i] - bar_width, y_max), ha='center', va='bottom', fontsize=9)
            for j, algo in enumerate(colors_algo):
                if (label, algo) not in agg:
                    continue
                v = agg[(label, algo)]
                m, lo, hi, n_fin = v['ratio']
                xpos = x_base[i] + bar_width * j
                if n_fin > 0:
                    ax.bar(xpos, m, bar_width, yerr=_yerr(v['ratio']), capsize=2,
                           color=colors_algo[algo], edgecolor='k', alpha=0.7,
                           label=nomes_algo[algo] if i == 0 else None)
                if v['n_ratio_inf']:
                    y_txt = m if n_fin > 0 else y_max / 3
                    ax.annotate(f"∞ em {v['n_ratio_inf']}/{v['n']}", (xpos, y_txt), textcoords='offset points',
                                xytext=(0, 3), ha='center', va='bottom', fontsize=6)
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max * 1.8)
        ax.axhline(1.0, color='k', linestyle=':', alpha=0.5, label='razão 1 (euclidiana)')
        ax.set_xticks(x_base)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('Razão $w_1 / w_2$ (escala log)')
        ax.set_title(f'Razão $w_1/w_2$: verdadeira vs recuperada\n(média; barras = {faixa}; ∞ = $w_2$ = 0)', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y', which='both')

    def _draw_barras(ax, chave, ylabel, titulo, ref=None, ref_label=None, escala=1.0, ylim=(0, 1.15)):
        bar_width = 0.3
        for i, label in enumerate(labels):
            for j, algo in enumerate(colors_algo):
                if (label, algo) not in agg:
                    continue
                m, lo, hi, _ = agg[(label, algo)][chave]
                if np.isnan(m):
                    continue
                ax.bar(x_base[i] + bar_width * (j - 0.5), m * escala, bar_width,
                       yerr=_yerr(agg[(label, algo)][chave], escala), capsize=2,
                       color=colors_algo[algo], edgecolor='k', alpha=0.7,
                       label=nomes_algo[algo] if i == 0 else None)
        if ref is not None:
            ax.axhline(y=ref, color='green', linestyle='--', alpha=0.5, label=ref_label)
        ax.set_xticks(x_base)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{titulo}\n(média; barras = {faixa})', fontweight='bold')
        ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    def _draw_cosine(ax):
        _draw_barras(ax, 'cosine', 'Similaridade de cosseno', 'Cosseno entre W verdadeiro e recuperado',
                     ref=1.0, ref_label='perfeito (1,0)')

    def _draw_acuracia_oraculo(ax):
        _draw_barras(ax, 'accuracy', 'Acurácia (vs rótulos do oráculo), %',
                     'Acurácia (vs rótulos do oráculo): métrica recuperada',
                     ref=95, ref_label='meta 95%', escala=100.0, ylim=(0, 108))

    # Figura combinada
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _draw_w_scatter(axes[0, 0])
    _draw_ratio(axes[0, 1])
    _draw_cosine(axes[1, 0])
    _draw_acuracia_oraculo(axes[1, 1])
    _exps = [str(l).replace('Oráculo sobre ', '').split('\n')[0] for l in labels]
    _sobre = (', '.join(_exps[:-1]) + ' e ' + _exps[-1]) if len(_exps) > 1 else (_exps[0] if _exps else '')
    plt.suptitle(f'Oráculo sobre {_sobre}, Passos 1 a 3: recuperação de W conhecido\n'
                 f'(centróides verdadeiros, sem LLM; {n_seeds} seed(s))',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    panels = [
        ("w_scatter", _draw_w_scatter, (7, 7)),
        ("ratio", _draw_ratio, (8, 6)),
        ("cosseno", _draw_cosine, (8, 6)),
        ("acuracia", _draw_acuracia_oraculo, (8, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_final_dados(data: dict, seed: int, filename: str = None):
    """Visão completa dos 4 datasets sintéticos (A, B, C lineares + E, especialista linear),
    final_graphic_01: rótulo real em cima; embaixo, rótulo do LLM zero-shot (A, B, C) e, no E,
    rótulo do especialista (``y_expert_e``) ou do LLM zero-shot (``y_llm_e``) se existirem no
    dict; sem nenhum dos dois, o E sai só com o painel de cima."""
    colors_gt = {0: '#3498db', 1: '#e74c3c'}
    colors_llm = {0: '#2980b9', 1: '#c0392b'}

    y_e_inf = data.get('y_expert_e')
    rotulador_e = 'especialista'
    if y_e_inf is None:
        y_e_inf = data.get('y_llm_e')
        rotulador_e = 'LLM'
    problems = [
        ('A', data['X_a'], data['y_gt_a'], data.get('y_llm_a'), 'LLM'),
        ('B', data['X_b'], data['y_gt_b'], data.get('y_llm_b'), 'LLM'),
        ('C', data['X_c'], data['y_gt_c'], data.get('y_llm_c'), 'LLM'),
        ('E', data['X_e'], data['y_gt_e'], y_e_inf, rotulador_e),  # Problema E = especialista linear (Bloco 2)
    ]

    def _draw_problem(axes_pair, name, X, y_gt, y_llm, rotulador='LLM'):
        ax_gt, ax_llm = axes_pair
        for c in [0, 1]:
            mask = y_gt == c
            ax_gt.scatter(X[mask, 0], X[mask, 1], c=colors_gt[c], s=15, alpha=0.6, label=f'Classe {c}')
        centroid_0 = X[y_gt == 0].mean(axis=0)
        centroid_1 = X[y_gt == 1].mean(axis=0)
        ax_gt.scatter(*centroid_0, c='black', marker='X', s=120, zorder=5, edgecolors='white', linewidths=1)
        ax_gt.scatter(*centroid_1, c='black', marker='X', s=120, zorder=5, edgecolors='white', linewidths=1)
        ax_gt.set_title(f'Problema {name}, rótulo real (n={len(X)})', fontweight='bold', fontsize=10)
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
            ax_llm.set_title(f'Problema {name}, rótulo do {rotulador} (C0={n0}, C1={n1})', fontweight='bold', fontsize=10)
            ax_llm.legend(fontsize=8)
        elif ax_llm is not None:
            ax_llm.text(0.5, 0.5, f'Sem rótulo do {rotulador}', ha='center', va='center',
                        transform=ax_llm.transAxes, fontsize=12, color='gray')
            ax_llm.set_title(f'Problema {name}, rótulo do {rotulador}', fontweight='bold', fontsize=10)
        if ax_llm is not None:
            ax_llm.grid(True, alpha=0.3)

    # Figura combinada
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for col, (name, X, y_gt, y_llm, rot) in enumerate(problems):
        _draw_problem((axes[0, col], axes[1, col]), name, X, y_gt, y_llm, rot)
        if y_llm is None and name == 'E':
            axes[1, col].axis('off')   # E sem rótulo do especialista/LLM: só o painel de cima
    painel_e = ('no E, rótulo do ESPECIALISTA (Passo 1)' if rotulador_e == 'especialista' and y_e_inf is not None
                else 'no E, rótulo do LLM zero-shot (Passo 2)' if y_e_inf is not None else 'E só com o rótulo real')
    fig.suptitle(f'Visão geral dos conjuntos de dados, seed {seed}: rótulo real (Passo 1); '
                 f'em A, B e C, rótulo do LLM zero-shot (Passo 2); {painel_e}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais por problema (rótulo real + rótulo do LLM/especialista)
    if filename:
        for name, X, y_gt, y_llm, rot in problems:
            if y_llm is None and name == 'E':
                fig_ind, ax_unico = plt.subplots(1, 1, figsize=(6, 5))
                _draw_problem((ax_unico, None), name, X, y_gt, None, rot)
            else:
                fig_ind, axes_ind = plt.subplots(2, 1, figsize=(6, 10))
                _draw_problem((axes_ind[0], axes_ind[1]), name, X, y_gt, y_llm, rot)
            passos = 'Passo 1 (rótulo real e do especialista)' if rot == 'especialista' else 'Passos 1 e 2'
            fig_ind.suptitle(f'Problema {name}, seed {seed}: {passos}', fontsize=13, fontweight='bold')
            fig_ind.tight_layout()
            salvar_figura(fig_ind, asset_variant(filename, f"problema_{name.lower()}"), dpi=150)
            plt.close(fig_ind)


def plot_final_acertos_erros(data: dict, seed: int, filename: str = None):
    """Métrica vs LLM ponto a ponto nos Problemas A (fidelidade) e B/C (consistência): final_graphic_03."""
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

        metrica_nome = 'Fidelidade' if name == 'A' else 'Consistência'
        ax1 = axes_row[0]
        ax1.scatter(X_plot[hits, 0], X_plot[hits, 1], c='#27ae60', s=15, alpha=0.5, label=f'métrica = LLM ({hits.sum()})')
        ax1.scatter(X_plot[errors, 0], X_plot[errors, 1], c='#e74c3c', s=30, alpha=0.8, marker='x', label=f'métrica ≠ LLM ({errors.sum()})')
        acc = hits.sum() / len(hits) * 100
        ax1.set_title(f'Problema {name}: {metrica_nome} {acc:.1f}%', fontweight='bold', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = axes_row[1]
        if learned_metric is not None:
            x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
            y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            # B/C: W_A com centróides LOCAIS (rótulos do LLM no próprio problema),
            # a mesma regra que gerou y_metric; só em A valem os centróides do A.
            cents = learned_metric.centroids if name == 'A' else _centroides_locais(X_plot, y_llm_plot, learned_metric)
            Z = predict_with_metric(grid_points, cents, learned_metric.w)
            Z = Z.reshape(xx.shape)
            ax2.contourf(xx, yy, Z, alpha=0.2, cmap='RdBu')
            ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        for c in [0, 1]:
            mask = y_llm_plot == c
            ax2.scatter(X_plot[mask, 0], X_plot[mask, 1], s=15, alpha=0.5, label=f'LLM Classe {c}')
        ax2.set_title(f'Problema {name}: Fronteira W_A' + ('' if name == 'A' else ' (centróides locais)'),
                      fontweight='bold', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax3 = axes_row[2]
        if learned_metric is not None:
            cents = learned_metric.centroids if name == 'A' else _centroides_locais(X_plot, y_llm_plot, learned_metric)
            confidences, _ = compute_metric_confidence(X_plot, cents, learned_metric.w)
            sc = ax3.scatter(X_plot[:, 0], X_plot[:, 1], c=confidences, cmap='viridis', s=15, alpha=0.6)
            ax3.scatter(X_plot[errors, 0], X_plot[errors, 1], c='red', s=50, marker='x', linewidths=2, label=f'métrica ≠ LLM ({errors.sum()})')
            plt.colorbar(sc, ax=ax3, label='Margem')
        ax3.set_title(f'Problema {name}: margem da métrica e pontos com métrica ≠ LLM', fontweight='bold', fontsize=10)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    # Figura combinada
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    for row, (name, X, y_llm, y_metric) in enumerate(problems):
        _draw_problem_row(axes[row], name, X, y_llm, y_metric)
    fig.suptitle(f'Métrica vs LLM, seed {seed}: Problema A, Passo 4 (fidelidade); Problemas B e C, Passo 3 (consistência)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais por problema (1×3: acertos, fronteira, confiança)
    if filename:
        for name, X, y_llm, y_metric in problems:
            fig_ind, axes_ind = plt.subplots(1, 3, figsize=(18, 5))
            _draw_problem_row(axes_ind, name, X, y_llm, y_metric)
            fig_ind.suptitle(f'Métrica vs LLM: Problema {name}, Passo {4 if name == "A" else 3}, seed {seed}', fontsize=13, fontweight='bold')
            fig_ind.tight_layout()
            salvar_figura(fig_ind, asset_variant(filename, f"problema_{name.lower()}"), dpi=150)
            plt.close(fig_ind)


def plot_final_w_algoritmos(data: dict, seed: int, filename: str = None):
    """Comparação visual das Ŵ_LLM estimadas: Perceptron vs NNLS."""
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
        # pesos NORMALIZADOS por ΣW (15/10/2026): a escala do Ŵ do Perceptron depende de γ e a do
        # NNLS é outra; só a direção compara (mesma convenção de bloco1_graphic_05)
        x_pos = np.arange(2)
        algos = [('Perceptron', np.asarray(w_perc, float), 'steelblue')]
        if has_nnls:
            algos.append(('NNLS', np.asarray(w_nnls, float), 'coral'))
        n_algos = len(algos)
        width = 0.8 / n_algos
        for j, (name, w, color) in enumerate(algos):
            wn = w / np.sum(w) if np.sum(w) > 0 else w
            offset = (j - (n_algos - 1) / 2) * width
            bars = ax_bar.bar(x_pos + offset, wn, width, label=name, color=color, alpha=0.8, edgecolor='k')
            for bar in bars:
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(['$w_1/ΣW$ ($x_1$)', '$w_2/ΣW$ ($x_2$)'])
        ax_bar.set_ylabel('Peso normalizado por ΣW')
        ax_bar.set_ylim(0, 1.15)
        ratio_strs = []
        for name, w, _ in algos:
            ratio = w[0] / w[1] if w[1] > 0 else float('inf')
            ratio_strs.append(f'{name}={ratio:.2f}')
        ax_bar.set_title(f'Comparação dos pesos (W normalizado por ΣW)\nRazão $w_1/w_2$: {", ".join(ratio_strs)}',
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

    fig.suptitle(f'Problema A, Passo 4: comparação de algoritmos de otimização inversa, seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    if filename:
        # Fronteira Perceptron
        fig_ind, ax_ind = plt.subplots(1, 1, figsize=(7, 6))
        plot_boundary(ax_ind, X, y_llm, w_perc, centroids_perc, 'Perceptron Estruturado')
        fig_ind.tight_layout()
        salvar_figura(fig_ind, asset_variant(filename, "perceptron"), dpi=150)
        plt.close(fig_ind)
        if has_nnls:
            fig_ind, ax_ind = plt.subplots(1, 1, figsize=(7, 6))
            plot_boundary(ax_ind, X, y_llm, w_nnls, centroids_nnls, 'NNLS (Mín. Quadrados)')
            fig_ind.tight_layout()
            salvar_figura(fig_ind, asset_variant(filename, "nnls"), dpi=150)
            plt.close(fig_ind)
        # Bar chart
        fig_ind, ax_ind = plt.subplots(1, 1, figsize=(7, 6))
        _draw_bar_chart(ax_ind)
        fig_ind.tight_layout()
        salvar_figura(fig_ind, asset_variant(filename, "barras"), dpi=150)
        plt.close(fig_ind)


def plot_final_matrizes_confusao(data: dict, seed: int, filename: str = None):
    """Matrizes de confusão: LLM vs Métrica e LLM vs rótulo real para cada problema."""
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
            metrica_nome = 'Fidelidade' if name == 'A' else 'Consistência'
            ax1.set_title(f'Problema {name}: LLM vs Métrica\n({metrica_nome} {acc:.1f}%)', fontweight='bold', fontsize=10)
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
            ax2.set_ylabel('Rótulo real')
            acc2 = np.trace(cm2) / cm2.sum() * 100
            ax2.set_title(f'Problema {name}: LLM vs rótulo real\n(Acurácia (vs rótulo real) {acc2:.1f}%)', fontweight='bold', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'Sem dados', ha='center', va='center', transform=ax2.transAxes, color='gray')
            ax2.set_title(f'Problema {name}: LLM vs rótulo real')

    # Figura combinada
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, (name, y_llm, y_metric, y_gt) in enumerate(problems):
        _draw_problem_col((axes[0, col], axes[1, col]), name, y_llm, y_metric, y_gt)
    fig.suptitle(f'Matrizes de confusão, seed {seed}: Problema A, Passos 2 e 4; Problemas B e C, Passos 2 e 3', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais por problema (2×1: LLM vs Métrica + LLM vs GT)
    if filename:
        for name, y_llm, y_metric, y_gt in problems:
            fig_ind, axes_ind = plt.subplots(2, 1, figsize=(6, 10))
            _draw_problem_col((axes_ind[0], axes_ind[1]), name, y_llm, y_metric, y_gt)
            fig_ind.suptitle(f'Matrizes de confusão: Problema {name}, Passos 2 e {4 if name == "A" else 3}, seed {seed}', fontsize=13, fontweight='bold')
            fig_ind.tight_layout()
            salvar_figura(fig_ind, asset_variant(filename, f"problema_{name.lower()}"), dpi=150)
            plt.close(fig_ind)


def plot_final_margem(data: dict, seed: int, filename: str = None):
    """Análise detalhada de margens/confiança da métrica estimada."""
    learned_metric = data.get('learned_metric')
    if learned_metric is None:
        return

    X_a = data['X_a']
    y_llm_a = data.get('y_llm_a')
    y_metric_a = data.get('y_metric_a')
    # margens com W NORMALIZADO por ΣW (15/10/2026, como bloco1_graphic_03): a magnitude do Ŵ do
    # Perceptron depende de γ e deixava o eixo da margem em "0.0"; a normalização não muda a
    # fronteira nem a ordem dos pontos
    w_norm = np.asarray(learned_metric.w, float)
    w_norm = w_norm / np.sum(w_norm) if np.sum(w_norm) > 0 else w_norm
    conf_a, _ = compute_metric_confidence(X_a, learned_metric.centroids, w_norm)

    def _draw_histograma(ax):
        ax.hist(conf_a, bins=30, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5,
                label='margem (W/ΣW)')
        ax.axvline(np.median(conf_a), color='red', linestyle='--', label=f'Mediana: {np.median(conf_a):.2f}')
        ax.axvline(np.mean(conf_a), color='orange', linestyle=':', label=f'Média: {np.mean(conf_a):.2f}')
        ax.set_xlabel('Margem (W normalizado por ΣW)')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição de Margens, Problema A', fontweight='bold')
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
            bars = ax.bar(bin_centers, error_rates, width=(bins[1]-bins[0])*0.8, color='#e74c3c', alpha=0.7,
                          edgecolor='black', linewidth=0.5, label='1 − fidelidade na faixa')
            for bar, count in zip(bars, bin_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'n={count}', ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('Margem (W normalizado por ΣW)')
        ax.set_ylabel('1 − fidelidade (%)')
        ax.set_title('1 − fidelidade por faixa de margem (Problema A)', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _draw_mapa_confianca(ax):
        sc = ax.scatter(X_a[:, 0], X_a[:, 1], c=conf_a, cmap='viridis', s=20, alpha=0.7)
        ax.scatter(*learned_metric.centroids[0], c='red', marker='X', s=200, zorder=5, edgecolors='white', linewidths=2, label='centróide 0')
        ax.scatter(*learned_metric.centroids[1], c='red', marker='X', s=200, zorder=5, edgecolors='white', linewidths=2, label='centróide 1')
        plt.colorbar(sc, ax=ax, label='Margem')
        ax.set_title('Mapa de Confiança, Problema A', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _draw_violin(ax):
        all_margins = [conf_a]
        labels = ['A']
        for pname, X_p, y_llm_p in [('B', data['X_b'], data.get('y_llm_b')), ('C', data['X_c'], data.get('y_llm_c'))]:
            # Margens de B/C com W_A e centróides locais (como na consistência)
            if y_llm_p is None or colapsou(y_llm_p):   # colapso em B/C: sem centróides locais, sem margens
                all_margins.append(np.array([0.0]))
                labels.append(f'{pname} (colapso)')
                continue
            cents_p = _centroides_locais(X_p, y_llm_p, learned_metric)
            conf_p, _ = compute_metric_confidence(X_p, cents_p, w_norm)
            all_margins.append(conf_p)
            labels.append(pname)
        parts = ax.violinplot(all_margins, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#3498db')
            pc.set_alpha(0.5)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([f'Problema {l}' for l in labels])
        ax.set_ylabel('Margem (W normalizado por ΣW)')
        ax.set_title('Distribuição de Margens por Problema', fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Figura combinada
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    _draw_histograma(axes[0, 0])
    _draw_taxa_erro(axes[0, 1])
    _draw_mapa_confianca(axes[1, 0])
    _draw_violin(axes[1, 1])
    fig.suptitle(f'Análise de margens da métrica: Problema A (Passo 4) e Problemas B e C (Passo 3), seed {seed}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()

    # Figuras individuais
    panels = [
        ("histograma", _draw_histograma, (7, 6)),
        ("taxa_erro", _draw_taxa_erro, (7, 6)),
        ("mapa_confianca", _draw_mapa_confianca, (7, 6)),
        ("violin", _draw_violin, (7, 6)),
    ]
    _save_panels_individually(panels, filename)


def plot_final_painel_resumo(data: dict, seed: int,
                                      results_abc: list, results_e: list,
                                      filename: str = None):
    """Dashboard completo do experimento para uma seed (final_graphic_06). Desde 15/10/2026 os
    pesos e as margens usam W NORMALIZADO por ΣW (a escala do Ŵ do Perceptron depende de γ, só a
    direção compara; mesma convenção de bloco1_graphic_05), as estratégias do E saem em
    português (`ESTRATEGIA_PT`) e a grade começa logo abaixo do supertítulo (sem faixa em branco)."""
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3, left=0.04, right=0.98, bottom=0.03, top=0.95)

    learned_metric = data.get('learned_metric')
    colors = {0: '#3498db', 1: '#e74c3c'}

    def _norm(w):
        w = np.asarray(w, float)
        return w / np.sum(w) if np.sum(w) > 0 else w

    # ─── LINHA 1 (topo): 4 datasets com ground truth ───
    for col, (name, X, y_gt) in enumerate([
        ('A', data['X_a'], data['y_gt_a']),
        ('B', data['X_b'], data['y_gt_b']),
        ('C', data['X_c'], data['y_gt_c']),
        ('E', data['X_e'], data['y_gt_e']),  # Problema E = especialista linear (Bloco 2)
    ]):
        ax = fig.add_subplot(gs[0, col])
        for c in [0, 1]:
            mask = y_gt == c
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[c], s=10, alpha=0.5, label=f'rótulo real, classe {c}')
        ax.set_title(f'Prob. {name} (n={len(X)})', fontweight='bold', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    # ─── LINHA 2: Problema A, Passo 4: métrica = LLM / métrica ≠ LLM + fronteira ───
    y_llm_a = data.get('y_llm_a')
    y_metric_a = data.get('y_metric_a')
    X_a = data['X_a']

    ax_a1 = fig.add_subplot(gs[1, 0])
    if y_llm_a is not None and y_metric_a is not None:
        hits = y_llm_a == y_metric_a
        ax_a1.scatter(X_a[hits, 0], X_a[hits, 1], c='#27ae60', s=10, alpha=0.5, label=f'métrica = LLM ({hits.sum()})')
        ax_a1.scatter(X_a[~hits, 0], X_a[~hits, 1], c='#e74c3c', s=25, marker='x', label=f'métrica ≠ LLM ({(~hits).sum()})')
        ax_a1.set_title(f'A, Passo 4: fidelidade {hits.mean():.1%}', fontweight='bold', fontsize=9)
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
            ax_a2.scatter(X_a[y_llm_a == c, 0], X_a[y_llm_a == c, 1], c=colors[c], s=10, alpha=0.4,
                          label=f'LLM, classe {c}')
        ax_a2.legend(fontsize=7)
    if learned_metric:
        _wn = _norm(learned_metric.w)
        w_str = f'W/ΣW=[{_wn[0]:.3f}, {_wn[1]:.3f}]'
    else:
        w_str = ''
    ax_a2.set_title(f'A, Passo 4: fronteira {w_str}', fontweight='bold', fontsize=9)
    ax_a2.grid(True, alpha=0.2)

    # Margens da métrica do A (W normalizado por ΣW: a fronteira e a ordem dos pontos não mudam)
    ax_a3 = fig.add_subplot(gs[1, 2])
    if learned_metric is not None:
        conf_a, _ = compute_metric_confidence(X_a, learned_metric.centroids, _norm(learned_metric.w))
        ax_a3.hist(conf_a, bins=25, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.3,
                   label='margem (W/ΣW)')
        ax_a3.axvline(np.median(conf_a), color='red', linestyle='--', linewidth=1, label=f'Med={np.median(conf_a):.2f}')
        ax_a3.set_title('Margens Problema A (W normalizado por ΣW)', fontweight='bold', fontsize=9)
        ax_a3.legend(fontsize=7)
    ax_a3.grid(True, alpha=0.2)

    # W dos algoritmos, NORMALIZADOS por ΣW (escala do Perceptron é arbitrária; só a direção compara)
    ax_w = fig.add_subplot(gs[1, 3])
    if learned_metric is not None:
        algs = ['Perceptron']
        w_vals = [_norm(learned_metric.w)]
        if data.get('w_nnls') is not None:
            algs.append('NNLS')
            w_vals.append(_norm(data['w_nnls']))
        x_pos = np.arange(2)
        width = 0.8 / len(algs)
        for i, (alg, w) in enumerate(zip(algs, w_vals)):
            offset = (i - (len(algs)-1)/2) * width
            ax_w.bar(x_pos + offset, w, width * 0.9, label=f'{alg}', alpha=0.8)
        ax_w.set_xticks(x_pos)
        ax_w.set_xticklabels(['w₁/ΣW', 'w₂/ΣW'])
        ax_w.set_ylim(0, 1.05)
        ax_w.set_title('Pesos W normalizados por ΣW', fontweight='bold', fontsize=9)
        ax_w.legend(fontsize=7)
    ax_w.grid(True, alpha=0.2, axis='y')

    # ─── LINHA 3: Problemas B e C, Passo 3 ───
    for col_offset, (name, X, y_llm, y_metric) in enumerate([
        ('B', data['X_b'], data.get('y_llm_b'), data.get('y_metric_b')),
        ('C', data['X_c'], data.get('y_llm_c'), data.get('y_metric_c')),
    ]):
        ax = fig.add_subplot(gs[2, col_offset])
        if y_llm is not None and y_metric is not None:
            n_min = min(len(y_llm), len(y_metric), len(X))
            hits = y_llm[:n_min] == y_metric[:n_min]
            ax.scatter(X[:n_min][hits, 0], X[:n_min][hits, 1], c='#27ae60', s=10, alpha=0.5,
                       label=f'métrica = LLM ({hits.sum()})')
            ax.scatter(X[:n_min][~hits, 0], X[:n_min][~hits, 1], c='#e74c3c', s=25, marker='x',
                       label=f'métrica ≠ LLM ({(~hits).sum()})')
            ax.set_title(f'{name}, Passo 3: consistência {hits.mean():.1%}', fontweight='bold', fontsize=9)
            ax.legend(fontsize=7)
        else:
            ax.set_title(f'{name}, Passo 3: sem dados', fontsize=9)
        ax.grid(True, alpha=0.2)

    # Barras de métricas B/C
    ax_bars = fig.add_subplot(gs[2, 2])
    seed_results = [r for r in _config_base(results_abc, nomes=("A", "B"))
                    if r.random_seed == seed and r.k_passo4_bc == 0 and r.repeticao == 0]
    if seed_results:
        r = seed_results[0]
        metrics = ['Consist. B', "Cohen's κ B", 'F1-Macro B', 'Consist. C', "Cohen's κ C", 'F1-Macro C']
        values = [r.consistencia_problema_b, r.kappa_problema_b, r.f1_problema_b,
                  r.consistencia_problema_c, r.kappa_problema_c, r.f1_problema_c]
        bar_colors = ['#3498db']*3 + ['#e67e22']*3
        ax_bars.barh(metrics, values, color=bar_colors, alpha=0.8)
        for i, v in enumerate(values):
            ax_bars.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8)
        ax_bars.set_xlim(0, 1.15)
        ax_bars.set_title('B e C, Passo 3 (zero-shot): consistência e κ', fontweight='bold', fontsize=9)
    ax_bars.grid(True, alpha=0.2, axis='x')

    # Fidelidade por k_exemplos
    ax_fid = fig.add_subplot(gs[2, 3])
    seed_abc = [r for r in _config_base(results_abc, nomes=("A", "B")) if r.random_seed == seed]
    if seed_abc:
        for metric_name, getter, color in [
            ('Consist. B', lambda r: r.consistencia_problema_b, '#3498db'),
            ('Consist. C', lambda r: r.consistencia_problema_c, '#e67e22'),
        ]:
            by_nshot = {}
            for r in seed_abc:
                by_nshot.setdefault(r.k_passo4_bc, []).append(getter(r))
            nshots = sorted(by_nshot.keys())
            means = [np.mean(by_nshot[n]) for n in nshots]
            ax_fid.plot(nshots, means, 'o-', label=metric_name, color=color)
        ax_fid.set_xlabel('k')
        ax_fid.set_title('B e C, Passos 3 e 4: consistência vs k', fontweight='bold', fontsize=9)
        ax_fid.legend(fontsize=7)
    ax_fid.grid(True, alpha=0.2)

    # ─── LINHA 4: Problema E, Passo 3 (especialista principal) ───
    results_e_main, expert_label = _main_expert(results_e)
    seed_e = [r for r in results_e_main if r.random_seed == seed]
    ax_e1 = fig.add_subplot(gs[3, 0:2])
    if seed_e:
        for strategy in EXAMPLE_STRATEGIES:
            strat_results = [r for r in seed_e if r.example_strategy == strategy]
            if strat_results:
                by_nshot = {}
                for r in strat_results:
                    by_nshot.setdefault(r.k, []).append(r.accuracy_llm_vs_expert)
                nshots = sorted(by_nshot.keys())
                means = [np.mean(by_nshot[n]) for n in nshots]
                ax_e1.plot(nshots, means, 'o-', label=ESTRATEGIA_PT.get(strategy, strategy))
        ax_e1.set_xlabel('k exemplos')
        ax_e1.set_ylabel('Acurácia (LLM vs especialista)')
        ax_e1.set_title(f'Problema {expert_label or "E"}, Passo 3: curva de aprendizado por estratégia', fontweight='bold', fontsize=9)
        ax_e1.legend(fontsize=8)
    else:
        ax_e1.text(0.5, 0.5, 'Problema E, Passo 3 não executado', ha='center', va='center', transform=ax_e1.transAxes, color='gray')
    ax_e1.grid(True, alpha=0.2)

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
        ax_cm.set_title('Matriz de confusão A: LLM vs Métrica', fontweight='bold', fontsize=9)

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
            '',
            f'Fidelidade A: {r.fidelidade_problema_a:.1%}',
            f'Consistência B: {r.consistencia_problema_b:.1%}',
            f'Consistência C: {r.consistencia_problema_c:.1%}',
            f"Cohen's κ B: {r.kappa_problema_b:.3f}",
            f"Cohen's κ C: {r.kappa_problema_c:.3f}",
        ])
    ax_info.text(0.05, 0.95, '\n'.join(info_lines), transform=ax_info.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_info.set_title('Resumo', fontweight='bold', fontsize=9)

    fig.suptitle(f'Painel resumo do experimento, seed {seed}: A, Passos 2 a 4; B e C, Passo 3; E, Passo 3',
                 fontsize=16, fontweight='bold', y=0.985)
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()


def plot_d_passo6_svm(
    X_ml: np.ndarray,
    y_true: np.ndarray,
    y_llm: np.ndarray,
    metric: Optional[dict] = None,
    seed: int = 0,
    filename: str = None,
):
    """Compara a superfície de separação na meia-lua.

    O orientador pediu para sobrepor a superfície de separação de um SVM gaussiano
    (RBF) à fronteira induzida pelo LLM. Dois painéis usam o MESMO SVM RBF como
    "régua" de fronteira não-linear, mudando apenas os rótulos que ele ajusta:
      (1) SVM RBF treinado no GROUND TRUTH: a fronteira não-linear de referência.
      (2) SVM RBF treinado nos RÓTULOS DO LLM: a fronteira efetiva do LLM; por
          cima, em tracejado verde, a fronteira da métrica diagonal estimada.

    Em AMBOS os painéis os pontos são coloridos pelo GROUND TRUTH : com as classes corretas no painel 2, os erros do LLM
    ficam visíveis contra a superfície ajustada aos rótulos dele, evidenciando
    que o LLM erra muito mais que o SVM RBF de referência.

    Se o LLM colapsou no zero-shot (regra única `colapsou`: classe única ou classe
    minoritária com menos de MIN_PONTOS_CLASSE_MINORITARIA pontos), o painel 2 mostra
    os pontos nas classes corretas e uma anotação explícita com a partição n0/n1,
    pois o colapso é um achado, não um motivo para suprimir a figura. Retorna True
    se o arquivo foi salvo.
    """
    try:
        from sklearn.svm import SVC
    except Exception:
        return False
    if len(np.unique(y_true)) < 2:
        return False
    llm_collapsed = colapsou(y_llm)   # regra única: classe única ou minoritária < 5

    pad = 0.5
    xs = np.linspace(X_ml[:, 0].min() - pad, X_ml[:, 0].max() + pad, 200)
    ys = np.linspace(X_ml[:, 1].min() - pad, X_ml[:, 1].max() + pad, 200)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = [
        ("SVM RBF treinado no rótulo real", y_true),
        ("SVM RBF treinado nos rótulos do LLM", y_llm),
    ]
    for panel_idx, (ax, (title, labels)) in enumerate(zip(axes, panels)):
        if panel_idx == 1 and llm_collapsed:
            n0, n1 = int(np.sum(np.asarray(y_llm) == 0)), int(np.sum(np.asarray(y_llm) == 1))
            for c in [0, 1]:
                mask = y_true == c
                ax.scatter(X_ml[mask, 0], X_ml[mask, 1],
                           c='tab:blue' if c == 0 else 'tab:red',
                           s=22, edgecolor='black', linewidth=0.3, alpha=0.85,
                           label=f'rótulo real, classe {c}')
            ax.set_title('Rótulos do LLM (Passo 2, zero-shot): colapso, sem fronteira',
                         fontweight='bold', fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.text(0.5, 0.04,
                    f'LLM rotulou {n0}/{n1} (classe 0/classe 1): colapso ou quase-colapso\n'
                    '(classe minoritária < 5 pontos); SVM e métrica não ajustáveis',
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
        # Pontos SEMPRE nas classes corretas (rótulo real):
        # no painel 2 isso expõe visualmente os erros do LLM contra a superfície.
        for c in [0, 1]:
            mask = y_true == c
            ax.scatter(X_ml[mask, 0], X_ml[mask, 1],
                       c='tab:blue' if c == 0 else 'tab:red',
                       s=22, edgecolor='black', linewidth=0.3, alpha=0.85,
                       label=f'rótulo real, classe {c}')
        ax.plot([], [], color='black', linewidth=1.6, label='fronteira do SVM RBF')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    # Sobrepõe a fronteira da métrica diagonal estimada (tracejada) no painel 2
    if metric is not None:
        try:
            n_feat = metric['n_features']
            grid_aug = augment_features(grid, n_feat)
            zz_m = predict_with_metric(grid_aug, metric['centroids'], metric['w_perc']).reshape(xx.shape)
            axes[1].contour(xx, yy, zz_m, levels=[0.5], colors='green',
                            linewidths=1.8, linestyles='dashed')
            axes[1].plot([], [], color='green', linestyle='dashed',
                         label=f"métrica diagonal ({n_feat} atributos)")
            axes[1].legend(loc='upper right', fontsize=9)
        except Exception:
            pass

    fig.suptitle(
        f'Problema D, Passo 6 (seed={seed}): superfície do SVM gaussiano vs rótulos zero-shot do LLM (Passo 2)',

        fontweight='bold', fontsize=12,
    )
    plt.tight_layout()
    saved = False
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=140)
        saved = True
    plt.close()
    return saved


def _nomes_classes(class_names) -> Tuple[str, str]:
    """Nomes das classes para legendas: ``class_names`` (ex.: ("Mulher", "Homem")) ou "classe 0/1"."""
    if class_names and len(class_names) >= 2:
        return (str(class_names[0]), str(class_names[1]))
    return ("classe 0", "classe 1")


def plot_problema_passo1(
    X: np.ndarray,
    y_true: np.ndarray,
    title: str,
    feature_names: Tuple[str, str] = ("x1", "x2"),
    optimal_boundary_fn: Optional[callable] = None,
    filename: Optional[str] = None,
    class_names: Optional[Tuple[str, str]] = None,
) -> None:
    """Visualiza um problema (scatter por classe + fronteira ótima opcional):
    bloco1_graphic_12 (meia-lua), bloco3_graphic_01/02 (peso × altura).

    Args:
        optimal_boundary_fn: função f(x1, x2) cuja curva f=0 será sobreposta
            (ex.: classificador ótimo bayesiano do peso×altura é a elipse
            x2² - x2 + x1² - x1 + cte).
        class_names: nomes das classes na legenda (ex.: ("Mulher", "Homem"));
            sem ele, "classe 0/1".
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap_class = {0: "tab:blue", 1: "tab:red"}
    nomes = _nomes_classes(class_names)
    for c in [0, 1]:
        mask = y_true == c
        ax.scatter(
            X[mask, 0], X[mask, 1], c=cmap_class[c], s=50,
            edgecolor="black", linewidth=0.5, alpha=0.85,
            label=f"{nomes[c]} (n={int(mask.sum())})",
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
        salvar_figura(plt.gcf(), filename, dpi=140)
    plt.close()


def plot_fg_rotulos_llm(
    X: np.ndarray,
    y_llm: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    title: str = "Rotulação do LLM",
    feature_names: Tuple[str, str] = ("x1", "x2"),
    filename: Optional[str] = None,
    y_metric: Optional[np.ndarray] = None,
    class_names: Optional[Tuple[str, str]] = None,
) -> None:
    """Visualiza scatter ponto-a-ponto das rotulações do LLM (final_graphic_07).

    Pedido do orientador: "mostrar graficamente todas as rotulações feitas pela LLM."

    Painéis: rótulo do LLM; rótulo real (se ``y_true``), com a Acurácia (vs rótulo real);
    rótulo do especialista que rotulou os exemplos (se ``y_metric``), com a Acurácia (LLM vs
    especialista); e os pontos em que LLM ≠ rótulo real. ``class_names`` (ex.:
    ("Mulher", "Homem")) entra nas legendas; sem ele, "classe 0/1".
    """
    y_llm = np.asarray(y_llm)
    nomes = _nomes_classes(class_names)
    paineis = ['llm']
    if y_true is not None:
        paineis.append('real')
    if y_metric is not None:
        paineis.append('especialista')
    if y_true is not None:
        paineis.append('erros')
    n_panels = len(paineis)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    axes = list(np.atleast_1d(axes))
    cmap_class = {0: "tab:blue", 1: "tab:red"}

    def _scatter_classes(ax, y, prefixo):
        for c in [0, 1]:
            mask = np.asarray(y) == c
            ax.scatter(X[mask, 0], X[mask, 1], c=cmap_class[c], s=40,
                       edgecolor="black", linewidth=0.4, alpha=0.85,
                       label=f"{prefixo} = {nomes[c]} ({int(mask.sum())})")
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax, painel in zip(axes, paineis):
        if painel == 'llm':
            _scatter_classes(ax, y_llm, 'LLM')
            ax.set_title(f"Rótulo do LLM\n(n={len(X)})")
        elif painel == 'real':
            _scatter_classes(ax, y_true, 'real')
            acc = accuracy_score(y_true, y_llm)
            ax.set_title(f"Rótulo real\n(Acurácia (vs rótulo real) do LLM = {acc:.1%})")
        elif painel == 'especialista':
            _scatter_classes(ax, y_metric, 'especialista')
            acc_m = accuracy_score(np.asarray(y_metric), y_llm)
            ax.set_title(f"Rótulo do especialista (quem rotulou os exemplos)\n"
                         f"(Acurácia (LLM vs especialista) = {acc_m:.1%})")
        else:
            errors = y_llm != np.asarray(y_true)
            ax.scatter(X[~errors, 0], X[~errors, 1], c="lightgray", s=25,
                       edgecolor="none", alpha=0.6, label="LLM = rótulo real")
            ax.scatter(X[errors, 0], X[errors, 1], c="black", s=60,
                       marker="x", linewidths=2, label="LLM ≠ rótulo real")
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.set_title(f"LLM ≠ rótulo real\n({int(errors.sum())} de {len(X)} pontos)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=140)
    plt.close()


def plot_fg_aprendiz_curva(
    aprendiz: List[dict],
    filename: Optional[str] = None,
) -> None:
    """Curva de aprendizado em contexto do Problema F (Passo 3) e do Problema G (Passo 6)
    (bloco23_graphic_01): um painel por (problema, variante de atributo), acurácia × k.
    Em cada painel, linha cheia = Acurácia (LLM vs rótulo real) e linha tracejada =
    Acurácia (LLM vs especialista), em que o especialista é quem rotulou os exemplos
    (coluna ``accuracy_llm_vs_expert``; execuções antigas: ``accuracy_llm_vs_metric``).
    Desde 14/10/2026 ``n_features`` é o nº de atributos MOSTRADOS ao LLM (fator do Passo 6 do
    G): uma cor/marcador por dimensão (R², R³, R⁴) dentro do painel. Legenda única da figura
    (até 14/10/2026 eram 2 painéis com uma série por grupo, ilegível com 25 séries)."""
    if not aprendiz:
        return
    df = pd.DataFrame(aprendiz)
    if 'accuracy_llm_vs_expert' not in df.columns and 'accuracy_llm_vs_metric' in df.columns:
        df['accuracy_llm_vs_expert'] = df['accuracy_llm_vs_metric']
    if 'n_features' not in df.columns:
        df['n_features'] = 2
    df['feature_names_str'] = df['feature_names'].apply(
        lambda t: '/'.join(t) if isinstance(t, tuple) else t
    )
    # F_meia_lua_seed{N} é um problema por semente: agrega os seeds no mesmo grupo
    df['problem_base'] = df['problem_name'].astype(str).str.replace(r'_seed\d+$', '', regex=True)
    df['group'] = df['problem_base'].map(_rotulo_fg) + ' | ' + df['feature_names_str']
    groups = sorted(df['group'].unique(), key=_ordem_fg)   # F, depois G1/G2 com Mulher/Homem antes de A/B
    dims = sorted(int(d) for d in df['n_features'].dropna().unique())
    estilo_dim = {2: ('tab:blue', 'o'), 3: ('tab:orange', 's'), 4: ('tab:green', '^')}
    metricas = [('accuracy_llm_vs_true', 'Acurácia (LLM vs rótulo real)', '-'),
                ('accuracy_llm_vs_expert', 'Acurácia (LLM vs especialista)', '--')]
    metricas = [m for m in metricas if m[0] in df.columns]

    ncols = min(3, len(groups)); nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows + 0.6), squeeze=False)
    for ax in axes.ravel()[len(groups):]:
        ax.set_visible(False)
    for idx, group in enumerate(groups):
        ax = axes.ravel()[idx]
        sub = df[df['group'] == group]
        for d in dims:
            sd = sub[sub['n_features'] == d]
            if sd.empty:
                continue
            cor, marcador = estilo_dim.get(d, ('tab:gray', 'D'))
            for j, (col, nome, ls) in enumerate(metricas):
                stats = sd.groupby('k')[col].agg(['mean', 'std']).reset_index().sort_values('k')
                # a legenda do eixo nomeia a dimensão UMA vez (série cheia) e as métricas por proxies pretos
                ax.errorbar(stats['k'], stats['mean'], yerr=stats['std'].fillna(0), marker=marcador,
                            linestyle=ls, color=cor, capsize=3, markersize=5, linewidth=1.6,
                            label=(f'{_dim(d)} no prompt' + (' (x1, x2)' if d == 2 else ' (+ x1·x2)' if d == 3 else ' (+ x1², x2²)' if d == 4 else ''))
                            if j == 0 else '_nolegend_')
        ax.set_title(group, fontweight='bold', fontsize=10)
        ax.set_xlabel('k exemplos')
        ax.set_ylabel('Acurácia')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        from matplotlib.lines import Line2D
        handles, labels = ax.get_legend_handles_labels()
        handles += [Line2D([], [], color='black', linestyle=ls, label=nome) for _, nome, ls in metricas]
        labels += [nome for _, nome, _ in metricas]
        ax.legend(handles, labels, fontsize=6.5, loc='lower right', framealpha=0.85)

    fig.suptitle(_suptitle_fg(groups, 'Problema F, Passo 3', 'Problema G, Passo 6') + ': curva de aprendizado em contexto; '
                 'média ± desvio entre seeds × reps' + (' (uma série por nº de atributos no prompt)' if len(dims) > 1 else ''),
                 fontweight='bold', fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=140)
    plt.close()


# Variantes de nome de atributo do protocolo de F/G (mesmas de G1_NOMES_ATRIBUTO/G2_NOMES_ATRIBUTO no
# runner, que plots.py não importa): unidas às vistas nos dados para anotar o colapso de uma variante
# que não deixou linha em NENHUM subproblema do modelo (ex.: x1/x2 do G2 no Gemini, 15/10/2026).
VARIANTES_ATRIBUTO_FG = {'F': ('x1/x2',), 'G1': ('x1/x2', 'peso/altura'), 'G2': ('x1/x2', 'peso_kg/altura_m')}


def _configuracoes_fg_esperadas(df: pd.DataFrame):
    """Configurações (problema-base, variante de atributo) ESPERADAS nas figuras de F/G, a partir
    das presentes em ``df`` (colunas ``problem_base`` e ``feature_names_str``): por subproblema
    (F, G1, G2), o produto das variantes de nome de classe vistas (no G, também a outra variante,
    Mulher/Homem ↔ A/B, pois o runner roda as duas) pelas variantes de atributo vistas. Devolve
    ``(esperadas, faltantes)``, listas de pares na ordem de `_ordem_fg`; uma configuração faltante é
    uma em que o LLM colapsou no zero-shot (o runner não grava linha) e a figura a anota em vez
    de omiti-la (15/10/2026)."""
    vistas = set(zip(df['problem_base'].astype(str), df['feature_names_str'].astype(str)))

    def _prefixo(pb: str) -> str:
        return 'F' if pb.startswith('F') else pb[:2]

    bases: Dict[str, set] = {}
    feats: Dict[str, set] = {}
    for pb, fn in vistas:
        px = _prefixo(pb)
        bases.setdefault(px, set()).add(pb)
        feats.setdefault(px, set()).add(fn)
        feats[px] |= set(VARIANTES_ATRIBUTO_FG.get(px, ()))
        if px.startswith('G'):      # as duas variantes de nomes de classe do G
            bases[px].add(pb[:-len('_classesAB')] if pb.endswith('_classesAB') else pb + '_classesAB')
    esperadas = sorted({(pb, fn) for px in bases for pb in bases[px] for fn in feats[px]},
                       key=lambda t: _ordem_fg(f"{_rotulo_fg(t[0])} | {t[1]}"))
    faltantes = [c for c in esperadas if c not in vistas]
    return esperadas, faltantes


def _suptitle_fg(groups, texto_f: str, texto_g: str) -> str:
    """Prefixo do supertítulo das figuras de F/G: só cita o Problema F quando há painel do F."""
    tem_f = any(str(g).startswith('F') for g in groups)
    tem_g = any(str(g).startswith('G') for g in groups)
    partes = ([texto_f] if tem_f else []) + ([texto_g] if tem_g else [])
    return ', e '.join(partes) if len(partes) == 2 else (partes[0] if partes else 'Problemas F e G')


def plot_fg_passos_2_e_4_r2_r3_r4(
    passo2: List[dict],
    filename: Optional[str] = None,
) -> None:
    """Barras comparativas da métrica estimada com 2/3/4 atributos por (problema, variante de
    atributo) sobre os rótulos zero-shot (Problema F, Passo 2; Problema G, Passos 2 e 4;
    bloco23_graphic_02): fidelidade (métrica vs LLM) e acurácia DA MÉTRICA (vs rótulo real),
    Perceptron e NNLS. Mostra se houve ganho ao acrescentar atributos (2 → 3 → 4). Uma
    configuração esperada sem linha (colapso do LLM no zero-shot) aparece no eixo com a anotação
    "colapso no zero-shot" em vez de sumir (15/10/2026)."""
    if not passo2:
        return
    df = pd.DataFrame(passo2)
    df['feature_names_str'] = df['feature_names'].apply(
        lambda t: '/'.join(t) if isinstance(t, tuple) else t
    )
    # F_meia_lua_seed{N} é um problema por semente: agrega os seeds no mesmo grupo
    df['problem_base'] = df['problem_name'].astype(str).str.replace(r'_seed\d+$', '', regex=True)
    df['group'] = df['problem_base'].map(_rotulo_fg) + ' | ' + df['feature_names_str']
    esperadas, faltantes = _configuracoes_fg_esperadas(df)
    groups = [f"{_rotulo_fg(pb)} | {fn}" for pb, fn in esperadas]   # F, depois G1/G2 com Mulher/Homem antes de A/B
    grupos_colapso = {f"{_rotulo_fg(pb)} | {fn}" for pb, fn in faltantes}
    n_features_list = sorted(df['n_features'].unique())

    metrics = [
        ('fidelity_perc_vs_llm', 'Fidelidade da métrica, Perceptron (vs LLM)'),
        ('accuracy_perc_vs_true', 'Acurácia da métrica, Perceptron (vs rótulo real)'),
        ('fidelity_nnls_vs_llm', 'Fidelidade da métrica, NNLS (vs LLM)'),
        ('accuracy_nnls_vs_true', 'Acurácia da métrica, NNLS (vs rótulo real)'),
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
                means.append(sub[col].mean() if len(sub) else np.nan)
                stds.append(sub[col].std() if len(sub) > 1 else 0)
            offset = (j - (len(n_features_list) - 1) / 2) * bar_width
            ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                   color=cmap(j), edgecolor='black', alpha=0.8,
                   label=f'{nf} atributos')
        for i_g, g in enumerate(groups):
            if g in grupos_colapso:
                ax.text(x[i_g], 0.03, 'colapso no zero-shot', rotation=90, fontsize=8, ha='center', va='bottom')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_ylim(0, 1.3)                                   # legenda acima das barras
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8, loc='upper center', ncol=len(n_features_list))

    fig.suptitle(_suptitle_fg(groups, 'Problema F, Passo 2', 'Problema G, Passos 2 e 4')
                 + ': métrica estimada com 2, 3 e 4 atributos (R², R³, R⁴) sobre os rótulos zero-shot',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=140)
    plt.close()


def plot_fg_fronteiras(
    passo2: List[dict],
    filename: Optional[str] = None,
) -> None:
    """Fronteira de decisão da métrica (Perceptron) projetada em R² para cada (problema, variante
    de atributo, nº de atributos), bloco23_graphic_03: Problema F, Passo 2; Problema G, Passo 4.

    Para cada configuração (problema, variante), um painel por n_features com os pontos
    (rótulo do LLM) e a fronteira d_W(x, c_0) = d_W(x, c_1) reconstruída no espaço de 2/3/4
    atributos e projetada de volta em R² via grade em (x1, x2). Só a PRIMEIRA semente de cada
    configuração é mostrada (dito no supertítulo). Uma configuração esperada sem linha (colapso do
    LLM no zero-shot) ganha uma linha de painéis com a anotação "colapso no zero-shot" (15/10/2026)."""
    if not passo2:
        return
    df = pd.DataFrame(passo2)
    df['feature_names_str'] = df['feature_names'].apply(
        lambda t: '/'.join(t) if isinstance(t, tuple) else t
    )

    # Agrega por problema-base (F_meia_lua_seed{N} → F_meia_lua) e variante; usa a 1ª seed
    df['problem_base'] = df['problem_name'].astype(str).str.replace(r'_seed\d+$', '', regex=True)
    esperadas, faltantes = _configuracoes_fg_esperadas(df)
    n_panels = len(esperadas)
    if n_panels == 0:
        return
    seeds_mostradas = []

    fig, axes = plt.subplots(n_panels, 3, figsize=(15, 5 * n_panels), squeeze=False)
    for row_idx, (problem_name, fnames) in enumerate(esperadas):   # problem_name = problema-base
        rotulo = _rotulo_fg(problem_name)
        if (problem_name, fnames) in faltantes:
            for col_idx in range(3):
                ax = axes[row_idx, col_idx]
                ax.set_title(f'{rotulo} | {fnames}', fontsize=9, fontweight='bold')
                ax.text(0.5, 0.5, 'colapso no zero-shot\n(sem métrica estimada)', ha='center', va='center',
                        transform=ax.transAxes, fontsize=11, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
            continue
        sub = df[(df['problem_base'] == problem_name) & (df['feature_names_str'] == fnames)]
        seed = sub['seed'].iloc[0]
        seeds_mostradas.append(int(seed) if float(seed).is_integer() else seed)
        sub_seed = sub[sub['seed'] == seed].sort_values('n_features')
        for col_idx, n_feat in enumerate([2, 3, 4]):
            ax = axes[row_idx, col_idx]
            row = sub_seed[sub_seed['n_features'] == n_feat]
            if len(row) == 0:
                ax.set_title(f'{rotulo} | {fnames} | {n_feat} atributos\n(não rodado)',
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
                if mask.any():
                    ax.scatter(X2[mask, 0], X2[mask, 1],
                               c='tab:blue' if c == 0 else 'tab:red',
                               s=18, edgecolor='black', linewidth=0.3, alpha=0.85,
                               label=f'classe {c} (rótulo do LLM)')
            ax.set_title(
                f'{rotulo} | {fnames}\n'
                f'{n_feat} atributos ({_dim(n_feat)}) | seed={seed} | Acurácia da métrica (vs rótulo real) = {r["accuracy_perc_vs_true"]:.1%}',
                fontsize=9,
            )
            # a métrica é SEMPRE estimada no espaço normalizado [-1, 1]; no G2 o prompt
            # mostra kg e m, mas os eixos aqui continuam normalizados
            norm = ' (normalizado)' if str(problem_name).startswith('G') else ''
            ax.set_xlabel(f'x1{norm}'); ax.set_ylabel(f'x2{norm}')
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

    groups = [f"{_rotulo_fg(pb)} | {fn}" for pb, fn in esperadas]
    seeds_txt = ', '.join(str(s_) for s_ in dict.fromkeys(seeds_mostradas)) or '—'
    fig.suptitle(_suptitle_fg(groups, 'Problema F, Passo 2', 'Problema G, Passo 4')
                 + ': fronteira de decisão da métrica (Perceptron) projetada em R²\n'
                 f'(só a 1ª semente de cada configuração é mostrada: seed {seeds_txt}; rótulos zero-shot do LLM)',
                 fontweight='bold', fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.9 / (5 * n_panels)))
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=140)
    plt.close()


def plot_fg_llm_vs_perceptron(
    aprendiz: List[dict],
    filename: Optional[str] = None,
) -> None:
    """Barras comparativas, por k: LLM em contexto vs Perceptron treinado nos MESMOS exemplos
    rotulados pelo especialista (Problema F, Passo 4; Problema G, Passo 7; bloco23_graphic_04).
    Um painel por (problema, variante de atributo[, nº de atributos no prompt]); legenda única
    da figura, para não cobrir as barras (15/10/2026)."""
    if not aprendiz:
        return
    df = pd.DataFrame(aprendiz)
    df['feature_names_str'] = df['feature_names'].apply(
        lambda t: '/'.join(t) if isinstance(t, tuple) else t
    )
    # F_meia_lua_seed{N} é um problema por semente: agrega os seeds no mesmo grupo
    df['problem_base'] = df['problem_name'].astype(str).str.replace(r'_seed\d+$', '', regex=True)
    df['group'] = df['problem_base'].map(_rotulo_fg) + ' | ' + df['feature_names_str']
    # 14/10/2026: `n_features` = atributos MOSTRADOS ao LLM (fator do Passo 6 do G); uma série/painel por dimensão
    if 'n_features' in df.columns and df['n_features'].nunique() > 1:
        df['group'] = df['group'] + ' | ' + df['n_features'].map(_dim) + ' no prompt'
    groups = sorted(df['group'].unique(), key=_ordem_fg)   # F, depois G1/G2 com Mulher/Homem antes de A/B
    valores_k = sorted(df['k'].unique())

    # Grade com até 3 colunas (13/10/2026): na completa são 9 painéis (F + 8 combinações do G),
    # ilegíveis numa única linha.
    ncols = min(4 if len(groups) > 12 else 3, len(groups)); nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)
    for ax in axes.ravel()[len(groups):]:
        ax.set_visible(False)
    for col_idx, group in enumerate(groups):
        ax = axes.ravel()[col_idx]
        sub = df[df['group'] == group]
        means_llm, means_perc = [], []
        stds_llm, stds_perc = [], []
        for ns in valores_k:
            ss = sub[sub['k'] == ns]
            means_llm.append(ss['accuracy_llm_vs_true'].mean())
            stds_llm.append(ss['accuracy_llm_vs_true'].std() if len(ss) > 1 else 0)
            pb = ss['accuracy_perceptron_baseline_vs_true'].dropna()
            means_perc.append(pb.mean() if len(pb) else np.nan)
            stds_perc.append(pb.std() if len(pb) > 1 else 0)
        x = np.arange(len(valores_k))
        w = 0.4
        b_llm = ax.bar(x - w/2, means_llm, w, yerr=stds_llm, capsize=3,
                       color='tab:orange', edgecolor='black', alpha=0.85, label='LLM (em contexto)')
        b_perc = ax.bar(x + w/2, means_perc, w, yerr=stds_perc, capsize=3,
                        color='tab:blue', edgecolor='black', alpha=0.85, label='Perceptron (Passo 4 F / Passo 7 G)')
        ax.set_xticks(x)
        ax.set_xticklabels([str(ns) for ns in valores_k])
        ax.set_xlabel('k exemplos')
        ax.set_ylabel('Acurácia (vs rótulo real)')
        ax.set_ylim(0, 1.05)
        ax.set_title(group, fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    tem_f = any(g.startswith('F') for g in groups)
    fig.suptitle(('Problema F, Passo 4, e Problema G, Passo 7' if tem_f else 'Problema G, Passo 7')
                 + ': LLM em contexto vs Perceptron treinado nos mesmos exemplos',
                 fontweight='bold', fontsize=12)
    # legenda ÚNICA da figura, abaixo dos painéis: dentro de cada painel cobria as barras (15/10/2026)
    fig.legend([b_llm, b_perc], ['LLM (em contexto)', 'Perceptron (Passo 4 do F / Passo 7 do G) nos mesmos exemplos'],
               loc='lower center', ncol=2, fontsize=9, frameon=True)
    # rect deixa o supertítulo fora da 1ª linha de painéis (25 painéis desde 14/10/2026) e a legenda embaixo
    plt.tight_layout(rect=(0, 0.035 if nrows > 2 else 0.07, 1, 0.985 if nrows > 2 else 0.95))
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=140)
    plt.close()


# =============================================================================
# PASSOS 2 A 5 DOS PROBLEMAS A E D (orientador, 01/10/2026): figuras em preto e branco
# com SÍMBOLOS por classe e curva fidelidade/acurácia × k
# =============================================================================

def _draw_symbols_scheme(ax, X: np.ndarray, scheme: dict,
                         feature_names: Tuple[str, str] = ("x1", "x2"),
                         reference_centroids: Optional[dict] = None, colorido: bool = False) -> None:
    """Desenha um esquema de rotulação com símbolos: classe 0 = '+', classe 1 = '−' (menos);
    centróide = 'o' grande (classe 0 vazado, classe 1 cheio): quando o esquema tem
    ``boundary``, são os ``boundary['centroids']`` que traçaram a fronteira (os da
    estimação), senão os centróides dos próprios rótulos; exemplos do prompt circundados;
    fronteira da métrica tracejada (na legenda); centróides de referência (origem → destino,
    p/ B e C) como quadrados e setas. ``scheme['rep']`` (int, opcional) entra no subtítulo
    como "rep N" quando o título não o cita. ``colorido=False`` = tudo em preto (versão
    ``_gray``); ``colorido=True`` = classe 0 azul, classe 1 vermelha, fronteira roxa e
    exemplos verdes (versão ``_color``)."""
    y = np.asarray(scheme['y']).astype(int)
    b = scheme.get('boundary')
    tem_fronteira = bool(b and b.get('w') is not None and b.get('centroids') is not None)
    cor_cls = {0: 'tab:blue', 1: 'tab:red'} if colorido else {0: 'k', 1: 'k'}
    cor_fronteira = 'tab:purple' if colorido else 'k'
    cor_exemplo = 'tab:green' if colorido else '0.45'
    # '+' (mais) para a classe 0 e '_' (traço horizontal = sinal de menos) para a classe 1
    for cls, marker, nome in ((0, '+', '+'), (1, '_', '−')):
        m = y == cls
        if m.any():
            ax.scatter(X[m, 0], X[m, 1], marker=marker, s=55, linewidths=1.4, color=cor_cls[cls],
                       label=f'classe {cls} ({nome})')
    # centróides: os da fronteira (estimação) quando há boundary; senão os dos rótulos
    cents = np.asarray(b['centroids'], float) if tem_fronteira else None
    origem_c = 'da estimação' if tem_fronteira else 'dos rótulos'
    for cls in (0, 1):
        m = y == cls
        if cents is not None and len(cents) > cls:
            c = cents[cls][:2]
        elif m.any():
            c = X[m].mean(axis=0)
        else:
            continue
        ax.scatter([c[0]], [c[1]], marker='o', s=220,
                   facecolors=('none' if cls == 0 else cor_cls[cls]), edgecolors=cor_cls[cls],
                   linewidths=2, zorder=5, label=f'centróide c{cls} ({origem_c})')
        ax.annotate(f'c{cls}', (c[0], c[1]), textcoords='offset points', xytext=(9, 9), fontsize=9)
    ex = scheme.get('example_idx')
    if ex is not None and len(ex):
        ex = np.asarray(ex, dtype=int)
        ax.scatter(X[ex, 0], X[ex, 1], marker='o', s=150, facecolors='none', edgecolors=cor_exemplo,
                   linewidths=1.2, zorder=4, label='exemplo no prompt (rótulo real)')
    if tem_fronteira:
        pad = 0.6
        xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 220)
        ys = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 220)
        xx, yy = np.meshgrid(xs, ys)
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        n_feat = int(b.get('n_features', 2))
        zz = predict_with_metric(augment_features(grid, n_feat), cents,
                                 np.asarray(b['w'], float)).reshape(xx.shape)
        if zz.min() != zz.max():
            ax.contour(xx, yy, zz, levels=[0.5], colors=cor_fronteira, linestyles='--', linewidths=1.4)
            ax.plot([], [], color=cor_fronteira, linestyle='--', linewidth=1.4,
                    label=f'fronteira da métrica ({_dim(n_feat)})')
    if reference_centroids:
        src = np.asarray(reference_centroids.get('from', []), float)
        dst = np.asarray(reference_centroids.get('to', []), float)
        for i, p in enumerate(src):
            ax.scatter([p[0]], [p[1]], marker='s', s=120, facecolors='none', edgecolors='k',
                       linewidths=1.5, zorder=6, label='centróide gerador do A' if i == 0 else None)
        for i, (p, q) in enumerate(zip(src, dst)):
            ax.annotate('', xy=(q[0], q[1]), xytext=(p[0], p[1]),
                        arrowprops=dict(arrowstyle='->', color='k', lw=1.2))
            ax.scatter([q[0]], [q[1]], marker='D', s=70, facecolors='k', edgecolors='k',
                       zorder=6, label='centróide gerador deste problema' if i == 0 else None)
    title = scheme.get('title', '')
    rep_val = scheme.get('rep')
    if rep_val is not None and 'rep ' not in title:
        title = f"{title} (rep {int(rep_val)})"
    if scheme.get('subtitle'):
        title += f"\n{scheme['subtitle']}"
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(feature_names[0]); ax.set_ylabel(feature_names[1])
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.9)


def plot_rotulos_simbolos(X: np.ndarray, schemes: List[dict], filename: str,
                        feature_names: Tuple[str, str] = ("x1", "x2"), suptitle: str = "",
                        reference_centroids: Optional[dict] = None) -> None:
    """Painéis 1×N (um por esquema de rotulação: ground truth, LLM, métrica…) com símbolos
    + (classe 0) e − (classe 1); salvos em ``_color`` (azul/vermelho) e ``_gray`` (preto).

    `schemes`: lista de dicts {'suffix', 'title', 'y', 'subtitle'?, 'example_idx'?,
    'boundary'?: {'w', 'centroids', 'n_features'}}. Salva a figura combinada e um PNG por
    painel (`asset_variant` com o `suffix`)."""
    if X is None or not schemes:
        return
    n = len(schemes)
    if reference_centroids and reference_centroids.get('label'):
        # texto da rotação (B/C) uma vez só, no título da figura, não por painel
        suptitle = (suptitle + "\n" if suptitle else "") + reference_centroids['label']
    # desenhada DUAS vezes (símbolos coloridos por classe → _color; tudo em preto → _gray),
    # em vez da conversão genérica, para os círculos dos exemplos e os centróides manterem a forma
    for colorido, cor in ((True, "color"), (False, "gray")):
        fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.2))
        axes = np.atleast_1d(axes)
        for ax, sc in zip(axes, schemes):
            _draw_symbols_scheme(ax, X, sc, feature_names, reference_centroids, colorido=colorido)
        if suptitle:
            fig.suptitle(suptitle, fontsize=11, fontweight='bold')
        plt.tight_layout()
        if filename:
            salvar_figura(fig, filename, dpi=150, cor=cor)
        plt.close(fig)
        panels = [(sc['suffix'], (lambda ax, sc=sc, c=colorido: _draw_symbols_scheme(
            ax, X, sc, feature_names, reference_centroids, colorido=c)), (5.6, 5.4)) for sc in schemes]
        _save_panels_individually(panels, filename, cor=cor)


def _slug_painel(texto: str) -> str:
    """Sufixo de painel para `asset_variant`: minúsculas, sem acento, só [a-z0-9_]."""
    import re
    import unicodedata
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", t)).strip("_")


def plot_passos_2_a_5_curva(rows: List[dict], filename: str, algorithm: str = "perceptron",
                              panels: Optional[List[Tuple[str, str, Dict[str, object]]]] = None,
                              ncols: Optional[int] = None, suptitle: Optional[str] = None) -> None:
    """Curva dos Passos 2 a 5 dos Problemas A e D (bloco1_graphic_04): fidelidade (métrica vs LLM, nos n − k
    pontos avaliados), acurácia da métrica vs rótulo real (nos n pontos) e acurácia do LLM
    vs rótulo real (nos n − k avaliados) em função de k (0 = zero-shot), média com barras
    mín–máx entre seeds × reps, e o teto (W ajustado ao rótulo real) tracejado. O teto vem
    de ``accuracy_teto_vs_true`` no regime ``passo5_teto`` (desde 09/10/2026, em que
    ``fidelity_vs_llm`` é NaN nesse regime); execuções antigas guardavam o teto em
    ``fidelity_vs_llm``. Painéis padrão: Problema A (R²) e Problema D em R², R³, R⁴.

    ``panels`` (opcional) substitui os painéis padrão por uma lista de
    ``(sufixo, título, filtro)``: o painel mostra as linhas em que cada coluna do
    ``filtro`` tem o valor dado (ex.: ``{"problem_name": "G1_homem_mulher",
    "feature_names": "peso/altura", "n_features": 2}``); ``sufixo`` nomeia o painel
    individual (`asset_variant`). Painéis sem linhas são omitidos. ``ncols`` é o número de
    colunas da grade (padrão: todos os painéis numa linha); ``suptitle`` troca o título geral.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df[df['algorithm'] == algorithm]
    if df.empty:
        return
    if 'collapsed' not in df.columns:
        df = df.assign(collapsed=0)
    if panels is None:
        panels = [(f"problema_{p.lower()}" + ("" if p == "A" else f"_r{nf}"), f"Problema {p}, {_dim(nf)}",
                   {"problem": p, "n_features": nf})
                  for p, nf in (("A", 2), ("D", 2), ("D", 3), ("D", 4))]

    def _mask(filtro):
        m = pd.Series(True, index=df.index)
        for col, val in filtro.items():
            m &= (df[col] == val) if col in df.columns else False
        return m

    panels_spec = [(sfx, tit, flt) for sfx, tit, flt in panels if _mask(flt).any()]
    if not panels_spec:
        return
    series = [('fidelity_vs_llm', '-', 'o', 'Fidelidade (métrica vs LLM), n − k avaliados'),
              ('accuracy_metric_vs_true_full', '--', 's', 'Acurácia da métrica (vs rótulo real), n pontos'),
              ('llm_accuracy_vs_true', ':', '^', 'Acurácia do LLM (vs rótulo real), n − k avaliados')]

    def _teto(gt: pd.DataFrame) -> float:
        for col in ('accuracy_teto_vs_true', 'accuracy_metric_vs_true_full', 'fidelity_vs_llm'):
            if col in gt.columns and not np.isnan(gt[col].mean()):
                return float(gt[col].mean())
        return float('nan')

    def _draw(ax, titulo, filtro):
        sub = df[_mask(filtro)]
        llm = sub[sub['regime'] != 'passo5_teto']
        llm = llm[llm['collapsed'].fillna(0).astype(int) == 0]   # colapsos ficam fora das médias
        gt = sub[sub['regime'] == 'passo5_teto']
        n_grupos = 0
        for col, ls, mk, lab in series:
            if col not in llm.columns:
                continue
            g = llm.groupby('k')[col].agg(['mean', 'min', 'max', 'count']).reset_index().sort_values('k')
            g = g.dropna(subset=['mean'])
            if g.empty:
                continue
            n_grupos = max(n_grupos, int(g['count'].max()))
            ax.errorbar(g['k'], g['mean'],
                        yerr=[(g['mean'] - g['min']).clip(lower=0), (g['max'] - g['mean']).clip(lower=0)],
                        fmt=mk + ls, color='k', capsize=3, markersize=6, linewidth=1.3,
                        markerfacecolor=('k' if mk == 'o' else 'white'), label=lab)
        n_col = sub[sub['regime'] != 'passo5_teto'].groupby('k')['collapsed'].sum()
        for k_val, n_c in n_col.items():
            if n_c > 0:
                ax.annotate(f'{int(n_c)} colapso(s) excluído(s)', (k_val, 0.06), textcoords='offset points',
                            xytext=(4, 0), fontsize=7, ha='left')
        teto = _teto(gt) if not gt.empty else float('nan')
        if not np.isnan(teto):
            ax.axhline(teto, color='k', linestyle='-.', linewidth=1.0,
                       label=f'teto: W ajustado ao rótulo real ({teto:.1%}; melhor esforço se γ = 0)')
        ks = sorted(sub[sub['regime'] != 'passo5_teto']['k'].unique())
        ax.set_xticks(ks)
        ax.set_xlabel('k exemplos (0 = zero-shot; rótulo real)')
        ax.set_ylabel('fração (média; mín–máx)')
        ax.set_ylim(0, 1.05)
        ax.set_title(f'{titulo}\n({ALGORITMO_PT.get(algorithm, algorithm)}; até {n_grupos} coleta{"s" if n_grupos != 1 else ""} por k)',
                     fontweight='bold', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='lower right', framealpha=0.85)

    n_pan = len(panels_spec)
    # Grade: 4 painéis (A R² + D R²/R³/R⁴) em 2×2 (13/10/2026; numa linha só a figura saía
    # larga e baixa, com eixos ilegíveis); demais casos numa linha, salvo `ncols` explícito.
    ncols = min(int(ncols), n_pan) if ncols else (2 if n_pan == 4 else n_pan)
    nrows = int(np.ceil(n_pan / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, (_sfx, tit, flt) in zip(axes, panels_spec):
        _draw(ax, tit, flt)
    for ax in axes[n_pan:]:
        ax.set_visible(False)
    fig.suptitle(suptitle or ('Problemas A e D, Passos 2 a 5: zero-shot (Passo 2) → k exemplos do rótulo real (Passo 3) → métrica (Passo 4) → teto (Passo 5)\n'
                              'fidelidade e acurácia do LLM sobre os n − k pontos avaliados; acurácia da métrica sobre os n pontos'),
                 fontsize=11, fontweight='bold')
    # Reserva espaço para o supertítulo (duas linhas) acima da primeira linha de painéis.
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.75 / (4.8 * nrows)))
    if filename:
        salvar_figura(plt.gcf(), filename, dpi=150)
    plt.close()
    _save_panels_individually([(sfx, (lambda ax, t=tit, f=flt: _draw(ax, t, f)), (6, 4.8))
                               for sfx, tit, flt in panels_spec], filename)


def plot_g_passos_2_a_5_curva(rows: List[dict], filename: str, algorithm: str = "perceptron",
                                n_features: int = 2) -> None:
    """Curva dos Passos 2 a 5 do Problema G (bloco3_graphic_03): um painel por
    (subproblema, nomes de classe, nomes de atributo) em R² (8 painéis na completa, grade
    4 × 2), a partir das linhas de `bloco3_g_passos_2_a_5` (colunas `subproblem`,
    `class_names`, `feature_names`). Painéis individuais ``_g1_mulher_homem_peso_altura``
    etc. (`_slug_painel`, antes do alias). Mesmas séries e teto de `plot_passos_2_a_5_curva`."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    chaves = ("subproblem", "class_names", "feature_names")
    if any(c not in df.columns for c in chaves):
        return
    combos = sorted(set(zip(df["subproblem"], df["class_names"], df["feature_names"])),
                    key=lambda t: (t[0], str(t[1]).replace(" ", "") != "Mulher/Homem", t[2]))   # Mulher/Homem antes de A/B
    panels = [(_slug_painel(f"{sp}_{cn}_{fn}"), f"{sp} | classes {cn} | atributos {fn} | {_dim(n_features)}",
               {"subproblem": sp, "class_names": cn, "feature_names": fn, "n_features": n_features})
              for sp, cn, fn in combos]
    plot_passos_2_a_5_curva(
        rows, filename, algorithm=algorithm, panels=panels, ncols=2,
        suptitle=(f'Problema G (peso × altura), Passos 2 a 5 nos 70 pontos de treino, {_dim(n_features)}: '
                  'zero-shot (Passo 2) → k exemplos do rótulo real (Passo 3) → métrica e teto (Passo 4) → base da eleição (Passo 5)\n'
                  'fidelidade e acurácia do LLM sobre os n − k pontos avaliados; acurácia da métrica sobre os n pontos; '
                  'média entre sementes e repetições (acurácia literal: com classes A/B a correspondência nome↔classe é arbitrária)'))
