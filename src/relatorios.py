"""Relatórios de texto e estatística descritiva dos experimentos.

Extraído de ``dissertacao_mestrado.py`` (etapa 1 da modularização, 2026): funções
``print_*`` que compõem o log_execucao.txt, ``resumo_entre_sementes`` e a síntese
cruzada linear x não-linear. São consumidores puros das listas de resultados —
não chamam a API nem mutam estado do runner.
"""
import os
import textwrap
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from execucao_io import decorrido, llm_asset
from metrics import compute_metric_confidence, predict_with_metric
from protocolo import EXAMPLE_STRATEGIES, EXPERT_W, PERCEPTRON_PARAMS
from relaxed_perceptron import train_relaxed_perceptron
from resultados import ResultadoABC, ResultadoAprendiz
from tabelas import faixa_kappa

# Limiar de H1 (consistência): κ > 0,4, piso "moderada" de Landis e Koch (1977).
LIMIAR_H1_KAPPA = 0.4


def print_section(title: str, char: str = "="):
    """Imprime um cabeçalho de seção formatado."""
    line = char * 70
    marca = decorrido()          # 15/10/2026: tempo decorrido no cabeçalho (vazio fora do runner)
    print(f"\n{line}")
    print(f" {title}" + (f"   {marca}" if marca else ""))
    print(f"{line}\n")


def print_box(text: str, largura: int = 90):
    """Imprime texto em uma caixa formatada.

    Linhas mais longas que ``largura`` são QUEBRADAS (``textwrap.wrap``) em várias
    linhas dentro da caixa, nunca truncadas com "...": o veredito de H4 e os demais
    resumos saem inteiros no log (09/10/2026).
    """
    lines = []
    for line in text.strip().split('\n'):
        if len(line) <= largura:
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, width=largura, break_long_words=True,
                                       break_on_hyphens=False) or [""])
    max_len = max(len(line) for line in lines)
    print("┌" + "─" * (max_len + 2) + "┐")
    for line in lines:
        print(f"│ {line:<{max_len}} │")
    print("└" + "─" * (max_len + 2) + "┘")


def print_a_passo4_erros_por_margem(dados_a_passo2: dict):
    """Análise quantitativa de erros por região (distância à fronteira).

    Responde: onde a métrica discorda do LLM? Perto da fronteira (margem baixa) ou
    longe? Complementa bloco1_graphic_03_A_passo4_erros_seed* com números no log; a
    coluna "1 − fidelidade" é a fração de pontos discordantes por faixa de margem.
    """
    print_section("ANÁLISE DE ERROS POR REGIÃO (DISTÂNCIA À FRONTEIRA)", "═")

    for seed_key, data in sorted(dados_a_passo2.items()):
        X = data['X']
        y_llm = data['y_llm']
        y_metric = data['y_metric']
        w = data['w']
        centroids = data['centroids']

        agreements = y_llm == y_metric
        disagreements = ~agreements
        n_total = len(X)
        n_errors = np.sum(disagreements)

        if n_errors == 0:
            print(f"\n  Seed {seed_key}: Fidelidade perfeita (0 discordâncias)")
            continue

        confidences, _ = compute_metric_confidence(X, centroids, w)

        # Dividir em 3 regiões: baixa margem (< p33), média, alta (> p66)
        p33 = np.percentile(confidences, 33)
        p66 = np.percentile(confidences, 66)

        regions = {
            'Baixa margem (fronteira)': confidences < p33,
            'Margem média': (confidences >= p33) & (confidences < p66),
            'Alta margem (longe)': confidences >= p66,
        }

        print(f"\n  Seed {seed_key}: {n_errors}/{n_total} discordâncias ({n_errors/n_total:.1%})")
        print(f"  Margem: min={confidences.min():.3f}, mediana={np.median(confidences):.3f}, "
              f"max={confidences.max():.3f}")
        print(f"\n  {'Região':<28} {'N pontos':>10} {'Discord.':>8} {'1 − fidelidade':>14} {'% das discord.':>14}")
        print(f"  {'─'*28} {'─'*10} {'─'*8} {'─'*14} {'─'*14}")

        for region_name, mask in regions.items():
            n_region = np.sum(mask)
            n_err_region = np.sum(disagreements[mask])
            rate = n_err_region / n_region if n_region > 0 else 0
            pct_errors = n_err_region / n_errors if n_errors > 0 else 0
            print(f"  {region_name:<28} {n_region:>10} {n_err_region:>8} {rate:>14.1%} {pct_errors:>14.1%}")

        # Correlação margem × acerto
        from scipy import stats as sp_stats
        corr, p_val = sp_stats.pointbiserialr(agreements.astype(int), confidences)
        print(f"\n  Correlação ponto-bisserial (concordância × margem): r={corr:.3f}, p={p_val:.4f} "
              f"(n = {n_total} pontos de uma só semente; p-valor só indicativo)")
        if corr > 0 and p_val < 0.05:
            print("  → Confirmado: pontos com maior margem concordam mais (esperado)")
        elif p_val >= 0.05:
            print("  → Correlação NÃO significativa: discordâncias não se concentram na fronteira")
        else:
            print("  → Correlação NEGATIVA significativa: discordâncias concentradas LONGE da fronteira (inesperado)")

        # Margem média dos pontos concordantes vs discordantes
        margin_correct = confidences[agreements]
        margin_errors = confidences[disagreements]
        print(f"\n  Margem média dos pontos concordantes:  {np.mean(margin_correct):.3f} (±{np.std(margin_correct):.3f})")
        print(f"  Margem média dos pontos discordantes:  {np.mean(margin_errors):.3f} (±{np.std(margin_errors):.3f})")

        if len(margin_errors) >= 2 and len(margin_correct) >= 2:
            stat, p_mw = sp_stats.mannwhitneyu(margin_correct, margin_errors, alternative='greater')
            print(f"  Mann-Whitney U (concordantes > discordantes): U={stat:.0f}, p={p_mw:.4f} "
                  f"(n = {len(margin_correct)} concordantes vs {len(margin_errors)} discordantes de uma só semente; "
                  "p-valor só indicativo)")
            if p_mw < 0.05:
                print("  → Significativo: discordâncias têm margem menor que concordâncias")
            else:
                print("  → NÃO significativo: discordâncias não se concentram em margens baixas")

    print()


def print_a_passo4_sensibilidade_perceptron(dados_a_passo2: dict):
    """Análise de sensibilidade dos hiperparâmetros do Perceptron Estruturado.

    Re-executa o Perceptron com diferentes combinações de (eta, C) sobre os
    mesmos rótulos zero-shot do Problema A (Passo 2; sem chamadas adicionais à API). Reporta se a
    direção de W é estável, o que indicaria robustez à escolha de hiperparâmetros.

    Custo computacional: ~O(n_configs × n_seeds × custo_perceptron), sem API calls.
    """
    print_section("SENSIBILIDADE DOS HIPERPARÂMETROS DO PERCEPTRON", "═")
    print("\n  Re-execução do Perceptron com diferentes (eta, C) sobre os mesmos")
    print("  rótulos zero-shot do Problema A (Passo 2). Nenhuma chamada adicional à API é feita.")
    print("  Se a direção de W (cosseno) é estável, o resultado é robusto.\n")

    # Configurações a testar (inclui a configuração padrão para referência)
    # Faixa de C inclui [0.1, 1.0] conforme Coelho et al. CILAMCE 2017, p. 16
    ETA_VALUES = [0.0001, 0.001, 0.01, 0.1]
    C_VALUES = [0.1, 1.0, 10.0]
    DELTA_GAMMA_VALUES = [0.01, 0.05, 0.1]

    # Configuração padrão do protocolo (PERCEPTRON_PARAMS, protocolo.py) — lida de lá
    # para que este relatório nunca divirja do que o runner realmente usa.
    DEFAULT_ETA = PERCEPTRON_PARAMS["eta"]
    DEFAULT_C = PERCEPTRON_PARAMS["C"]
    DEFAULT_DELTA = PERCEPTRON_PARAMS["delta_gamma"]
    MAX_EPOCHS = PERCEPTRON_PARAMS["max_epochs"]
    TOL = PERCEPTRON_PARAMS["tol"]

    for seed_key, data in sorted(dados_a_passo2.items()):
        X = data['X']
        y_llm = data['y_llm']
        centroids = data['centroids']
        w_default = data['w']
        w_default_norm = np.linalg.norm(w_default)
        w_default_dir = w_default / w_default_norm if w_default_norm > 0 else w_default

        print(f"  Seed {seed_key}: W padrão (eta={DEFAULT_ETA}, C={DEFAULT_C}) = "
              f"[{w_default[0]:.4f}, {w_default[1]:.4f}]")

        # --- Sensibilidade a eta (C fixo) ---
        print(f"\n  Variando eta (C={DEFAULT_C}, delta_gamma={DEFAULT_DELTA}):")
        print(f"    {'eta':>6} {'W bruto':>22} {'W unitário':>22} {'cos(default)':>14} {'Fidelidade':>12}")
        print(f"    {'─'*6} {'─'*22} {'─'*22} {'─'*14} {'─'*12}")

        cos_sims_eta = []
        for eta in ETA_VALUES:
            w_test, _ = train_relaxed_perceptron(
                X, y_llm, centroids,
                eta=eta, C=DEFAULT_C, delta_gamma=DEFAULT_DELTA,
                max_epochs=MAX_EPOCHS, tol=TOL, verbose=False, use_best_effort=True
            )
            w_norm = np.linalg.norm(w_test)
            w_dir = w_test / w_norm if w_norm > 0 else w_test
            cos_sim = float(np.dot(w_default_dir, w_dir)) if w_norm > 0 else 0.0
            cos_sims_eta.append(cos_sim)
            y_pred = predict_with_metric(X, centroids, w_test)
            fid = np.mean(y_pred == y_llm)
            marker = " ← padrão" if eta == DEFAULT_ETA else ""
            print(f"    {eta:>6g} [{w_test[0]:>8.4f}, {w_test[1]:>8.4f}] "
                  f"[{w_dir[0]:>8.4f}, {w_dir[1]:>8.4f}] "
                  f"{cos_sim:>14.4f} {fid:>12.1%}{marker}")

        # --- Sensibilidade a C (eta fixo) ---
        print(f"\n  Variando C (eta={DEFAULT_ETA}, delta_gamma={DEFAULT_DELTA}):")
        print(f"    {'C':>6} {'W bruto':>22} {'W unitário':>22} {'cos(default)':>14} {'Fidelidade':>12}")
        print(f"    {'─'*6} {'─'*22} {'─'*22} {'─'*14} {'─'*12}")

        cos_sims_c = []
        for C in C_VALUES:
            w_test, _ = train_relaxed_perceptron(
                X, y_llm, centroids,
                eta=DEFAULT_ETA, C=C, delta_gamma=DEFAULT_DELTA,
                max_epochs=MAX_EPOCHS, tol=TOL, verbose=False, use_best_effort=True
            )
            w_norm = np.linalg.norm(w_test)
            w_dir = w_test / w_norm if w_norm > 0 else w_test
            cos_sim = float(np.dot(w_default_dir, w_dir)) if w_norm > 0 else 0.0
            cos_sims_c.append(cos_sim)
            y_pred = predict_with_metric(X, centroids, w_test)
            fid = np.mean(y_pred == y_llm)
            marker = " ← padrão" if C == DEFAULT_C else ""
            print(f"    {C:>6.1f} [{w_test[0]:>8.4f}, {w_test[1]:>8.4f}] "
                  f"[{w_dir[0]:>8.4f}, {w_dir[1]:>8.4f}] "
                  f"{cos_sim:>14.4f} {fid:>12.1%}{marker}")

        # --- Sensibilidade a delta_gamma (eta, C fixos) ---
        print(f"\n  Variando delta_gamma (eta={DEFAULT_ETA}, C={DEFAULT_C}):")
        print(f"    {'δγ':>6} {'W bruto':>22} {'W unitário':>22} {'cos(default)':>14} {'Fidelidade':>12}")
        print(f"    {'─'*6} {'─'*22} {'─'*22} {'─'*14} {'─'*12}")

        cos_sims_dg = []
        for dg in DELTA_GAMMA_VALUES:
            w_test, _ = train_relaxed_perceptron(
                X, y_llm, centroids,
                eta=DEFAULT_ETA, C=DEFAULT_C, delta_gamma=dg,
                max_epochs=MAX_EPOCHS, tol=TOL, verbose=False, use_best_effort=True
            )
            w_norm = np.linalg.norm(w_test)
            w_dir = w_test / w_norm if w_norm > 0 else w_test
            cos_sim = float(np.dot(w_default_dir, w_dir)) if w_norm > 0 else 0.0
            cos_sims_dg.append(cos_sim)
            y_pred = predict_with_metric(X, centroids, w_test)
            fid = np.mean(y_pred == y_llm)
            marker = " ← padrão" if dg == DEFAULT_DELTA else ""
            print(f"    {dg:>6.2f} [{w_test[0]:>8.4f}, {w_test[1]:>8.4f}] "
                  f"[{w_dir[0]:>8.4f}, {w_dir[1]:>8.4f}] "
                  f"{cos_sim:>14.4f} {fid:>12.1%}{marker}")

        # --- Veredicto ---
        all_cos = cos_sims_eta + cos_sims_c + cos_sims_dg
        min_cos = min(all_cos) if all_cos else 0
        mean_cos = np.mean(all_cos) if all_cos else 0
        n_configs = len(all_cos)

        print(f"\n  Resumo seed {seed_key}: {n_configs} configurações testadas")
        print(f"    Cosseno mínimo com padrão: {min_cos:.4f}")
        print(f"    Cosseno médio com padrão:  {mean_cos:.4f}")

        if min_cos > 0.95:
            print("    → W ROBUSTO aos hiperparâmetros (cos mín > 0.95)")
        elif min_cos > 0.80:
            print("    → W MODERADAMENTE sensível (0.80 < cos mín < 0.95)")
        else:
            print("    → W SENSÍVEL aos hiperparâmetros (cos mín < 0.80): cautela nas conclusões")

        print()

    print("  Nota: apenas a DIREÇÃO de W importa (escala é arbitrária).")
    print("  Cosseno > 0.95 entre configs = hiperparâmetros não afetam a fronteira.\n")


def print_e_passo5_ordem(results_order: List[ResultadoAprendiz]):
    """Análise quantitativa do viés de ordem dos exemplos few-shot (recency bias).

    Testa se a ordenação dos exemplos afeta significativamente a performance do LLM.
    Recency bias: LLMs tendem a dar mais peso aos últimos exemplos do prompt.
    """
    from scipy import stats as sp_stats

    print_section("PROBLEMA E1, PASSO 5: VIÉS DE ORDEM DOS EXEMPLOS (RECENCY BIAS)", "═")

    df = pd.DataFrame([{
        'seed': r.random_seed, 'rep': r.repeticao,
        'k': r.k,
        'ordering': r.example_strategy.replace("mixed_order_", ""),
        'accuracy': r.accuracy_llm_vs_expert,
        'kappa': r.kappa_llm_vs_expert,
        'f1': r.f1_llm_vs_expert,
    } for r in results_order])

    orderings = sorted(df['ordering'].unique())
    valores_k = sorted(df['k'].unique())

    # Tabela resumo (acurácia = LLM vs especialista)
    print(f"\n  {'Ordenação':<18} {'k':>6} {'Acurácia (LLM vs especialista)':>32} {'±std':>8} {'Cohen\'s κ':>10} {'n':>4}")
    print(f"  {'─'*18} {'─'*6} {'─'*10} {'─'*8} {'─'*8} {'─'*4}")

    for ns in valores_k:
        for ordering in orderings:
            subset = df[(df['ordering'] == ordering) & (df['k'] == ns)]
            if len(subset) > 0:
                print(f"  {ordering:<18} {ns:>6} {subset['accuracy'].mean():>32.1%} "
                      f"{subset['accuracy'].std():>8.1%} {subset['kappa'].mean():>10.3f} "
                      f"{len(subset):>4}")
        print()

    # Teste estatístico por k: Kruskal-Wallis (não-paramétrico, >2 grupos)
    print("  TESTES DE SIGNIFICÂNCIA:")
    print(f"  {'─'*60}")

    for ns in valores_k:
        groups = []
        group_names = []
        for ordering in orderings:
            vals = df[(df['ordering'] == ordering) & (df['k'] == ns)]['accuracy'].values
            if len(vals) >= 2:
                groups.append(vals)
                group_names.append(ordering)

        if len(groups) >= 2:
            # Kruskal-Wallis: H0 = todas as ordenações têm mesma distribuição
            stat, p_val = sp_stats.kruskal(*groups)
            n_grp = sorted({len(g) for g in groups})
            n_txt = str(n_grp[0]) if len(n_grp) == 1 else f"{n_grp[0]} a {n_grp[-1]}"
            print(f"\n  {ns}-shot: Kruskal-Wallis H={stat:.3f}, p={p_val:.4f} "
                  f"(n = {n_txt} sementes por ordenação; p-valor só indicativo)")

            if p_val < 0.05:
                print("  → SIGNIFICATIVO: a ordem dos exemplos AFETA a performance")
                # Identificar qual ordenação é melhor/pior
                means = {name: np.mean(g) for name, g in zip(group_names, groups)}
                best = max(means, key=means.get)
                worst = min(means, key=means.get)
                diff = means[best] - means[worst]
                print(f"    Melhor: {best} ({means[best]:.1%}), Pior: {worst} ({means[worst]:.1%}), Δ={diff:+.1%}")

                # Teste pareado: class0_first vs class1_first (recency bias direto)
                if 'class0_first' in group_names and 'class1_first' in group_names:
                    c0 = df[(df['ordering'] == 'class0_first') & (df['k'] == ns)]
                    c1 = df[(df['ordering'] == 'class1_first') & (df['k'] == ns)]
                    # Parear por seed
                    c0_by_seed = c0.groupby('seed')['accuracy'].mean()
                    c1_by_seed = c1.groupby('seed')['accuracy'].mean()
                    common = c0_by_seed.index.intersection(c1_by_seed.index)
                    if len(common) >= 3:
                        diff_paired = c0_by_seed.loc[common].values - c1_by_seed.loc[common].values
                        mean_d, lo_d, hi_d = resumo_entre_sementes(diff_paired)
                        print(f"    Classe0_first - Classe1_first: Δ={mean_d:+.3f} "
                              f"mín–máx entre sementes=[{lo_d:+.3f}, {hi_d:+.3f}]")
                        if lo_d > 0:
                            print("    → Recency bias: última classe vista (classe 1) é FAVORECIDA")
                        elif hi_d < 0:
                            print("    → Recency bias: última classe vista (classe 0) é FAVORECIDA")
            else:
                print("  → NÃO significativo: a ordem dos exemplos NÃO afeta a performance")

    # Variabilidade geral por ordering (colapsando os k)
    overall = df.groupby('ordering')['accuracy'].agg(['mean', 'std'])
    max_range = overall['mean'].max() - overall['mean'].min()
    print(f"\n  Variação total entre ordenações: {max_range:.1%}")
    if max_range < 0.03:
        print("  → Efeito de ordem NEGLIGÍVEL (< 3 p.p.)")
    elif max_range < 0.10:
        print("  → Efeito de ordem MODERADO (3-10 p.p.): reportar como limitação")
    else:
        print("  → Efeito de ordem GRANDE (> 10 p.p.): recency bias significativo")

    print()


def malformadas_e(df: pd.DataFrame) -> Tuple[int, int]:
    """(com exemplos, zero-shot) respostas malformadas do Problema E (Passos 3 e 2) SEM contar o zero-shot em dobro.

    O zero-shot do Problema E (Passo 2) é coletado UMA vez por (modelo, seed) e a mesma linha é
    replicada para cada estratégia × especialista (4 × 3 = 12 cópias com o mesmo
    `n_malformed`). Somar a coluna inteira contava essas cópias 12 vezes (revisão de
    09/10/2026). Aqui o few-shot soma normalmente (cada linha é uma coleta própria) e o
    zero-shot conta uma vez por (model, seed, rep). Colunas exigidas: `k`,
    `n_malformed`, `model`, `seed`, `rep`.
    """
    few = int(df.loc[df['k'] > 0, 'n_malformed'].sum())
    zero_df = df[df['k'] == 0].drop_duplicates(subset=['model', 'seed', 'rep'])
    return few, int(zero_df['n_malformed'].sum())


def print_e_passos_2_3(results_e: List[ResultadoAprendiz]):
    """Imprime análise detalhada dos resultados do Problema E (Passos 2 e 3; antes de 13/10/2026: Fase E)."""
    print_section("ANÁLISE DO PROBLEMA E, PASSOS 2 E 3: LLM COMO APRENDIZ", "═")

    df = pd.DataFrame([
        {
            'model': f"{r.provider}/{r.model_name} (temp={r.temperature})",
            'seed': r.random_seed,
            'rep': r.repeticao,
            'k': r.k,
            'strategy': r.example_strategy,
            'accuracy': r.accuracy_llm_vs_expert,
            'kappa': r.kappa_llm_vs_expert,
            'f1': r.f1_llm_vs_expert,
            'acc_vs_gt': r.accuracy_llm_vs_gt,
            'expert_vs_gt': r.accuracy_expert_vs_gt,
            'n_disagreements': r.n_disagreements,
            'n_malformed': r.n_malformed_responses,
            'expert': r.expert_name,
            'expert_w': tuple(float(v) for v in np.asarray(r.expert_w).ravel()),
        }
        for r in results_e
    ])

    models = df['model'].unique()

    # Uma análise por (modelo, especialista): com RUN_PROBLEMA_E2_E3 há 3 especialistas
    # (E1, E2, E3; antigos aniso_x2, aniso_x1, euclidean) e misturá-los diluiria as curvas.
    for model in models:
      for expert in df[df['model'] == model]['expert'].unique():
        model_df = df[(df['model'] == model) & (df['expert'] == expert)]
        expert_w = model_df['expert_w'].iloc[0]
        print(f"\n{'═' * 70}")
        print(f" MODELO: {model} | ESPECIALISTA: {expert}")
        print(f"{'═' * 70}")

        print(f"\n  Métrica do Especialista W = [{expert_w[0]:.2f}, {expert_w[1]:.2f}]")
        print(f"  Acurácia (vs rótulo real) do especialista, sobre o teste [contexto]: {model_df['expert_vs_gt'].mean():.1%}")

        # 1. Curva de aprendizado por estratégia
        print("\n  1. CURVA DE APRENDIZADO (Acurácia (LLM vs especialista)):")
        print("  " + "-" * 65)
        header = f"  {'k':>6}"
        for strat in EXAMPLE_STRATEGIES:
            header += f" | {strat:>12}"
        print(header)
        print("  " + "-" * 65)

        for n in sorted(model_df['k'].unique()):
            line = f"  {n:>6}"
            for strat in EXAMPLE_STRATEGIES:
                subset = model_df[(model_df['k'] == n) & (model_df['strategy'] == strat)]
                if len(subset) > 0:
                    mean = subset['accuracy'].mean()
                    std = subset['accuracy'].std()
                    line += f" | {mean:.1%}±{std:.1%}"
                else:
                    line += f" | {'N/D':>12}"
            print(line)

        # 2. Melhor estratégia por k
        print("\n  2. MELHOR ESTRATÉGIA POR k:")
        print("  " + "-" * 50)
        for n in sorted(model_df['k'].unique()):
            if n == 0:
                continue
            subset = model_df[model_df['k'] == n]
            best = subset.groupby('strategy')['accuracy'].mean().idxmax()
            best_val = subset.groupby('strategy')['accuracy'].mean().max()
            print(f"     {n:>3}-shot: {best.upper():>8} ({best_val:.1%})")

        # 3. Melhoria do zero-shot para o melhor few-shot
        print("\n  3. MELHORIA DO ZERO-SHOT PARA O MELHOR FEW-SHOT:")
        print("  " + "-" * 50)
        zero_shot_acc = model_df[model_df['k'] == 0]['accuracy'].mean()
        print(f"     Linha de base zero-shot: {zero_shot_acc:.1%}")

        for strat in EXAMPLE_STRATEGIES:
            strat_df = model_df[model_df['strategy'] == strat]
            if len(strat_df) == 0:
                continue
            best_n = strat_df.groupby('k')['accuracy'].mean().idxmax()
            best_acc = strat_df.groupby('k')['accuracy'].mean().max()
            improvement = best_acc - zero_shot_acc
            print(f"     {strat.upper():>8}: melhor={best_acc:.1%} em {best_n}-shot "
                  f"(Δ={improvement:+.1%})")

        # 4. Respostas malformadas deste especialista (só few-shot: o zero-shot é
        # compartilhado entre especialistas e entra uma vez no total do modelo, abaixo)
        few_malformed_expert, _ = malformadas_e(model_df)
        if few_malformed_expert > 0:
            print(f"\n  ⚠️ Respostas malformadas (few-shot deste especialista): {few_malformed_expert}")

      # Total por MODELO: few-shot de todos os especialistas + zero-shot UMA vez por seed
      few_m, zero_m = malformadas_e(df[df['model'] == model])
      print(f"\n  Respostas malformadas no Problema E ({model}): few-shot (Passo 3) {few_m} + zero-shot (Passo 2) {zero_m} "
            f"(contado uma vez por seed) = {few_m + zero_m}")

    # Conclusão geral do experimento — especialista PRINCIPAL (E1, EXPERT_W = [0,3; 1,5])
    main_expert = df['expert'].iloc[0]
    df_main = df[df['expert'] == main_expert]
    overall_zero = df_main[df_main['k'] == 0]['accuracy'].mean()
    overall_best = df_main.groupby(['k', 'strategy'])['accuracy'].mean().max()
    best_config = df_main.groupby(['k', 'strategy'])['accuracy'].mean().idxmax()

    print_box(f"""
CONCLUSÃO DO PROBLEMA E, PASSOS 2 E 3: LLM COMO APRENDIZ (especialista principal: {main_expert})

Métrica do especialista: W = [{EXPERT_W[0]:.2f}, {EXPERT_W[1]:.2f}]
(Pondera a dimensão x2 {EXPERT_W[1]/EXPERT_W[0]:.1f}x mais do que x1)

Linha de base zero-shot: {overall_zero:.1%}
Melhor resultado few-shot: {overall_best:.1%} (estratégia {best_config[1]}, {best_config[0]}-shot)
Melhoria geral: {overall_best - overall_zero:+.1%}

{'✓ O LLM CONSEGUE aprender com exemplos do especialista — o desempenho melhora com mais exemplos.' if overall_best - overall_zero > 0.1 else
 '~ O LLM mostra aprendizado MODERADO com exemplos do especialista.' if overall_best - overall_zero > 0.05 else
 '✗ O LLM NÃO melhora significativamente com os exemplos do especialista.'}
"""
    )

    # --- Avaliação de H4 (easy vs hard) calculada dos dados ---
    # Só o especialista principal: misturar E1/E2/E3 confundiria a comparação.
    df_exemplos = df_main[df_main['k'] > 0]
    if len(df_exemplos) > 0:
        print_box(texto_h4(df_exemplos))


def texto_h4(df_exemplos: pd.DataFrame) -> str:
    """Veredicto de H4 ("hard > easy") CALCULADO dos dados (09/10/2026; antes era um
    texto fixo "easy supera hard consistentemente").

    `df_exemplos`: linhas few-shot (k > 0) de UM especialista, com colunas
    `strategy`, `k` e `accuracy` (acurácia LLM vs especialista). Reporta a média
    por estratégia (média das médias por k, para random com 3 reps não pesar 3×),
    quem venceu em média, por quanto, e se houve cruzamento em algum k (em
    particular no maior, 40-shot na grade completa).
    """
    por_n = df_exemplos.groupby(['strategy', 'k'])['accuracy'].mean()
    strat_means = por_n.groupby(level='strategy').mean()
    linhas = [f'  {s.upper():>8}: {v:.1%}' for s, v in sorted(strat_means.items(), key=lambda x: -x[1])]
    easy_mean = strat_means.get('easy', None)
    hard_mean = strat_means.get('hard', None)
    random_mean = strat_means.get('random', None)
    if easy_mean is None or hard_mean is None:
        veredicto = "RESULTADO: dados insuficientes para avaliar H4 (faltam easy ou hard)."
        detalhe = ""
    else:
        delta = hard_mean - easy_mean
        if abs(delta) < 0.005:
            veredicto = "RESULTADO: EMPATE entre easy e hard na média (|Δ| < 0,5 p.p.); H4 não é sustentada nem refutada."
        elif delta > 0:
            veredicto = f"RESULTADO: H4 sustentável nesta execução: hard ({hard_mean:.1%}) supera easy ({easy_mean:.1%}) em média (Δ = {delta:+.1%})."
        else:
            veredicto = f"RESULTADO: H4 REFUTADA nesta execução: easy ({easy_mean:.1%}) supera hard ({hard_mean:.1%}) em média (Δ = {delta:+.1%})."
        # cruzamentos por k
        cruz = []
        for n in sorted(df_exemplos['k'].unique()):
            e = por_n.get(('easy', n)); h = por_n.get(('hard', n))
            if e is None or h is None:
                continue
            vence = 'hard' if h > e else ('easy' if e > h else 'empate')
            cruz.append(f"{int(n)}-shot: {vence} (easy {e:.1%} | hard {h:.1%})")
        n_max = int(df_exemplos['k'].max())
        e_max, h_max = por_n.get(('easy', n_max)), por_n.get(('hard', n_max))
        vencedor_medio = 'hard' if delta > 0 else 'easy'
        if e_max is not None and h_max is not None and abs(delta) >= 0.005:
            venc_max = 'hard' if h_max > e_max else ('easy' if e_max > h_max else 'empate')
            cruzou = (f"Em {n_max}-shot o vencedor é {venc_max}: "
                      + ("CRUZAMENTO em relação à média." if venc_max not in (vencedor_medio, 'empate')
                         else "sem cruzamento em relação à média."))
        else:
            cruzou = ""
        detalhe = "Por k: " + "; ".join(cruz) + ("\n" + cruzou if cruzou else "")
        if random_mean is not None:
            detalhe += f"\nRandom (linha de base): {random_mean:.1%}" + (
                "; hard fica abaixo de random." if hard_mean < random_mean else
                "; easy fica abaixo de random." if easy_mean < random_mean else "; easy e hard superam random.")
    return f"""
AVALIAÇÃO DA HIPÓTESE H4 (Problema E, Passo 3): "Exemplos hard são mais informativos que easy"

Acurácia (LLM vs especialista) média por estratégia (média das médias por k > 0):
{chr(10).join(linhas)}

{veredicto}
{detalhe}

Nota: a hipótese foi formulada a priori; o veredicto acima é calculado dos dados
desta execução e reportado como resultado genuíno, não como falha do método.
"""


def print_abc_resumo(resultados: List[ResultadoABC]):
    """Imprime a análise final dos Problemas A, B e C (antes de 13/10/2026: Fases A, B e C)."""
    print_section("ANÁLISE FINAL: PROBLEMAS A, B E C", "═")

    df = pd.DataFrame([
        {
            'model': f"{r.provider}/{r.model_name} (temp={r.temperature})",
            'seed': r.random_seed,
            'k': r.k_passo4_bc,
            'nomes': f"{r.nomes_classes[0]}/{r.nomes_classes[1]}",
            'prompt_variant': getattr(r, 'prompt_variant', 'default'),
            'features': '/'.join(getattr(r, 'feature_names', ('x1', 'x2'))),
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
            'kappa_b': r.kappa_problema_b,
            'kappa_c': r.kappa_problema_c,
            'f1_b': r.f1_problema_b,
            'f1_c': r.f1_problema_c,
            'fidelidade': r.fidelidade_problema_a,
            'n_disagreements_b': r.n_disagreements_b,
            'n_disagreements_c': r.n_disagreements_c,
            'n_malformed': r.n_malformed_responses,
        }
        for r in resultados
    ])
    # Configuração base (prompt default, atributos x1/x2): as variantes de viés
    # rodam em outra grade de k e contaminariam as médias por k e o resumo.
    df = df[(df['prompt_variant'] == 'default') & (df['features'] == 'x1/x2')]
    if df.empty:
        print("  (sem resultados na configuração base)")
        return

    models = df['model'].unique()

    for model in models:
        model_df = df[df['model'] == model]
        print(f"\n{'═' * 70}")
        print(f" MODELO: {model}")
        print(f"{'═' * 70}")

        print("\n  1. CONSISTÊNCIA POR NÚMERO DE EXEMPLOS (classes A/B; k = 0: B e C, Passo 3, zero-shot; "
              "k > 0: Passo 4, exemplos rotulados pela métrica do A):")
        print(f"     {'k':>6} | {'Consist B':>10} | {'Consist C':>10} | {'Cohen\'s κ B':>12} | {'Cohen\'s κ C':>12}")
        print("     " + "-" * 55)
        model_ab = model_df[model_df['nomes'] == 'A/B']
        for n in sorted(model_ab['k'].unique()):
            subset = model_ab[model_ab['k'] == n]
            c_b = f"{subset['consistencia_b'].mean():.1%}±{subset['consistencia_b'].std():.1%}"
            c_c = f"{subset['consistencia_c'].mean():.1%}±{subset['consistencia_c'].std():.1%}"
            k_b = f"{subset['kappa_b'].mean():.3f}"
            k_c = f"{subset['kappa_c'].mean():.3f}"
            print(f"     {n:>6} | {c_b:>10} | {c_c:>10} | {k_b:>12} | {k_c:>12}")

        print("\n  2. CONSISTÊNCIA POR NOMES DE CLASSE (k presentes em todos os pares):")
        k_comuns = set.intersection(*[set(model_df[model_df['nomes'] == n]['k']) for n in model_df['nomes'].unique()])
        for nome in model_df['nomes'].unique():
            subset = model_df[(model_df['nomes'] == nome) & (model_df['k'].isin(k_comuns))]
            print(f"     {nome:20s}: B={subset['consistencia_b'].mean():.1%}, C={subset['consistencia_c'].mean():.1%}")

    df_ab = df[df['nomes'] == 'A/B'] if (df['nomes'] == 'A/B').any() else df
    # O resumo é o Passo 3 de B e C (k = 0, zero-shot contra a métrica do A). As linhas com k > 0 são o
    # Passo 4 (exemplos rotulados pela métrica, H2) e entram só numa linha à parte, identificada.
    df_p3 = df_ab[df_ab['k'] == 0] if (df_ab['k'] == 0).any() else df_ab
    df_p4 = df_ab[df_ab['k'] > 0]
    mean_b = df_p3['consistencia_b'].mean()
    mean_c = df_p3['consistencia_c'].mean()
    linha_p4 = ""
    if not df_p4.empty:
        ks_p4 = ", ".join(str(int(k)) for k in sorted(df_p4['k'].unique()))
        linha_p4 = (f"Passo 4 (k = {ks_p4}, exemplos rotulados pela métrica do A; H2), média sobre k: "
                    f"B = {df_p4['consistencia_b'].mean():.1%}, C = {df_p4['consistencia_c'].mean():.1%}\n")
    if min(mean_b, mean_c) > 0.85:
        leitura = "✓ Consistência ALTA (> 85%) em B e C no Passo 3."
    elif min(mean_b, mean_c) > 0.7:
        leitura = "~ Consistência MODERADA (70% a 85%) em B e/ou C no Passo 3."
    else:
        leitura = "✗ Consistência BAIXA (até 70%) em B e/ou C no Passo 3."

    print_box(f"""
RESUMO DOS PROBLEMAS A, B E C (A, Passos 2 e 4; B e C, Passo 3, zero-shot; classes A/B, prompt default, atributos x1/x2)

Consistência Média Problema B, Passo 3 (zero-shot): {mean_b:.1%} ± {df_p3['consistencia_b'].std():.1%}
Consistência Média Problema C, Passo 3 (zero-shot): {mean_c:.1%} ± {df_p3['consistencia_c'].std():.1%}
{linha_p4}
{leitura}
Leitura descritiva (limiares de 85% e 70% da consistência). O critério oficial de H1 é Cohen's κ > 0,4
(Landis e Koch 1977), calculado por modelo no bloco "H1 (CONSISTÊNCIA): VEREDITO POR MODELO" do sumário estatístico.
""")


def resumo_entre_sementes(data) -> Tuple[float, float, float]:
    """Resumo descritivo de uma métrica entre sementes (e repetições): média, mínimo e máximo.

    Substitui o intervalo de confiança por bootstrap, removido em 13/10/2026 a pedido do
    orientador: com 3 sementes o IC bootstrap não acrescenta informação além da faixa
    observada. Para uma diferença pareada, a faixa [mín, máx] não conter zero significa
    que TODAS as sementes apontam no mesmo sentido (critério usado como "consistente").
    Devolve (média, mínimo, máximo); vetor vazio devolve NaN nos três.
    """
    v = np.asarray([x for x in np.atleast_1d(data) if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(v.mean()), float(v.min()), float(v.max())


def diferenca_consistente(lo: float, hi: float) -> bool:
    """True quando a faixa [mín, máx] de uma diferença pareada não contém zero."""
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def texto_h1(resultados: List[ResultadoABC], limiar: float = LIMIAR_H1_KAPPA) -> str:
    """Veredito de H1 (consistência) por modelo, para o log (09/10/2026).

    H1: a métrica estimada no A prevê o LLM em B e C com κ > ``limiar`` (0,4, piso
    "moderada" de Landis e Koch, 1977). Para cada modelo usa a configuração base
    (classes A/B, prompt default, atributos x1/x2), ZERO-SHOT, Perceptron (as linhas de
    `ResultadoABC` são do Perceptron; o NNLS vive em bloco1_a_passo4_perceptron_vs_nnls)
    e sem colapso (`collapsed`/`collapsed_b`/`collapsed_c` = 0; κ NaN fora). O κ é
    tirado por seed (média das repetições) e o veredito usa o κ MÉDIO entre seeds; a
    linha traz também quantas seeds passam do limiar.
    """
    rows = [{
        'model': f"{r.provider}/{r.model_name} (temp={r.temperature})",
        'seed': r.random_seed, 'k': r.k_passo4_bc,
        'nomes': f"{r.nomes_classes[0]}/{r.nomes_classes[1]}",
        'prompt_variant': getattr(r, 'prompt_variant', 'default'),
        'features': '/'.join(getattr(r, 'feature_names', ('x1', 'x2'))),
        'collapsed': int(getattr(r, 'collapsed', 0) or 0),
        'collapsed_b': int(getattr(r, 'collapsed_b', 0) or 0),
        'collapsed_c': int(getattr(r, 'collapsed_c', 0) or 0),
        'kappa_b': r.kappa_problema_b, 'kappa_c': r.kappa_problema_c,
    } for r in resultados]
    cab = (f"  H1 (κ > {limiar:.1f}, piso 'moderada' de Landis e Koch): Problemas B e C, Passo 3 "
           "(métrica do A, Passo 4, sobre os rótulos zero-shot do Passo 2); configuração base "
           "(classes A/B, prompt default, atributos x1/x2), zero-shot, Perceptron, sem colapso")
    if not rows:
        return cab + "\n    (sem resultados dos Problemas A, B e C)"
    df = pd.DataFrame(rows)
    base = df[(df['nomes'] == 'A/B') & (df['prompt_variant'] == 'default')
              & (df['features'] == 'x1/x2') & (df['k'] == 0) & (df['collapsed'] == 0)]
    if base.empty:
        return cab + "\n    (sem linhas zero-shot na configuração base sem colapso)"
    linhas = [cab]
    for model in base['model'].unique():
        m = base[base['model'] == model]
        linhas.append(f"  MODELO: {model}")
        vereditos = []
        for prob, col, col_colapso in (("B", 'kappa_b', 'collapsed_b'), ("C", 'kappa_c', 'collapsed_c')):
            sub = m[m[col_colapso] == 0].dropna(subset=[col])
            if sub.empty:
                linhas.append(f"    Problema {prob}: sem κ válido (colapso em todas as seeds)")
                vereditos.append(f"NÃO AVALIÁVEL em {prob}")
                continue
            por_seed = sub.groupby('seed')[col].mean()
            k_medio = float(por_seed.mean())
            n_seeds = int(len(por_seed))
            n_passam = int((por_seed > limiar).sum())
            linhas.append(f"    Problema {prob}: κ médio = {k_medio:.3f} ({faixa_kappa(k_medio)}) | "
                          f"seeds com κ > {limiar:.1f}: {n_passam}/{n_seeds} | "
                          f"κ por seed: {', '.join(f'{s}={k:.3f}' for s, k in por_seed.items())}")
            veredito = "SUSTENTADA" if k_medio > limiar else "NÃO SUSTENTADA"
            vereditos.append(f"{veredito} em {prob} (κ médio {k_medio:.3f}, {n_passam}/{n_seeds} seeds passam)")
        linhas.append(f"    H1 (κ > {limiar:.1f}): " + " | ".join(vereditos))
    return "\n".join(linhas)


def print_sumario_estatistico(
    resultados_abc: List[ResultadoABC],
    resultados_e: List[ResultadoAprendiz],
):
    """Imprime o sumário estatístico: média e faixa entre sementes, diferenças pareadas e Cohen's d.

    Sem bootstrap (removido em 13/10/2026 a pedido do orientador) nem Wilcoxon (removido em 13/10/2026 por decisão do autor):
    - média, mínimo e máximo entre sementes (× repetições) para as métricas principais
    - diferenças pareadas por semente: Δ médio e faixa [mín, máx]; "consistente" quando a faixa
      não contém zero (todas as sementes no mesmo sentido)
    - tamanho de efeito (Cohen's d) quando aplicável
    """
    print_section("SUMÁRIO ESTATÍSTICO (média e faixa entre sementes, diferenças pareadas, Cohen's d)", "═")

    # --- Problemas A, B e C ---
    if resultados_abc:
        # Veredito de H1 (κ > 0,4) por modelo, antes das faixas entre sementes.
        print("\n  ════════════════════════════════════════════════")
        print("  H1 (CONSISTÊNCIA): VEREDITO POR MODELO")
        print("  ════════════════════════════════════════════════")
        print(texto_h1(resultados_abc))

        print("\n  ════════════════════════════════════════════════")
        print("  PROBLEMAS A, B e C (A, Passo 4; B e C, Passos 3 e 4): MÉDIA E FAIXA ENTRE SEMENTES")
        print("  ════════════════════════════════════════════════")

        # Filtrar apenas classes A/B, prompt default para métricas limpas
        df = pd.DataFrame([{
            'seed': r.random_seed, 'k': r.k_passo4_bc,
            'nomes': f"{r.nomes_classes[0]}/{r.nomes_classes[1]}",
            'prompt_variant': r.prompt_variant,
            'features': f"{r.feature_names[0]}/{r.feature_names[1]}",
            'fidelidade': r.fidelidade_problema_a,
            'consistencia_b': r.consistencia_problema_b,
            'consistencia_c': r.consistencia_problema_c,
            'kappa_b': r.kappa_problema_b,
            'kappa_c': r.kappa_problema_c,
            'f1_b': r.f1_problema_b,
            'f1_c': r.f1_problema_c,
        } for r in resultados_abc])

        # Métricas por k (classes A/B, prompt default, atributos x1/x2).
        # O filtro de atributos evita a contaminação da faixa da config base pelas
        # linhas do experimento de nomes de atributos (que também usam A/B+default).
        df_default = df[(df['nomes'] == 'A/B') & (df['prompt_variant'] == 'default')
                        & (df['features'] == 'x1/x2')]
        if len(df_default) > 0:
            print("\n  Configuração base: classes A/B, prompt default, atributos x1/x2")
            print(f"  n = {len(df_default)} observações ({len(df_default['seed'].unique())} seeds)")
            ks = sorted(df_default['k'].unique())
            # A métrica do A é estimada UMA vez (Passo 4, sobre os rótulos zero-shot do Passo 2) e é a
            # mesma em todos os k de B/C: a fidelidade sai uma única vez, fora do laço por k.
            k_ref = 0 if 0 in ks else ks[0]
            fid = df_default[df_default['k'] == k_ref]['fidelidade'].values
            mean_f, lo_f, hi_f = resumo_entre_sementes(fid)
            print(f"\n  Fidelidade A (Perceptron, Passo 4; a mesma métrica em todos os k de B/C): "
                  f"média={mean_f:.3f} mín–máx=[{lo_f:.3f}, {hi_f:.3f}] n={len(fid)}")
            print(f"\n  {'Métrica':<25} {'k':>6} {'Média':>8} {'mín–máx':>20} {'n':>4}")
            print(f"  {'─'*25} {'─'*6} {'─'*8} {'─'*20} {'─'*4}")

            for k_exemplos in ks:
                subset = df_default[df_default['k'] == k_exemplos]
                rotulo = ("Passo 3 (zero-shot)" if k_exemplos == 0
                          else f"Passo 4 ({int(k_exemplos)} exemplos rotulados pela métrica do A)")
                print(f"  --- B e C, {rotulo}; Perceptron ---")
                for col, label in [
                    ('consistencia_b', 'Consistência B (Perc)'),
                    ('consistencia_c', 'Consistência C (Perc)'),
                    ('kappa_b', "Cohen's κ B (Perc)"),
                    ('kappa_c', "Cohen's κ C (Perc)"),
                ]:
                    values = subset[col].values
                    mean, lo, hi = resumo_entre_sementes(values)
                    faixa_str = f"[{lo:.3f}, {hi:.3f}]"
                    print(f"  {label:<25} {k_exemplos:>6} {mean:>8.3f} {faixa_str:>20} {len(values):>4}")
                print()

        # Diferença pareada: Passo 3 (zero-shot) vs. Passo 4 (10 exemplos rotulados pela métrica do A), H2
        df_0 = df_default[df_default['k'] == 0].groupby('seed')[['consistencia_b', 'consistencia_c']].mean()
        df_10 = df_default[df_default['k'] == 10].groupby('seed')[['consistencia_b', 'consistencia_c']].mean() if 10 in df_default['k'].values else None

        if df_10 is not None:
            common_seeds = df_0.index.intersection(df_10.index)
            if len(common_seeds) >= 3:
                print("\n  DIFERENÇA PAREADA POR SEMENTE: B e C, Passo 3 (zero-shot) vs. Passo 4 "
                      "(10 exemplos rotulados pela métrica do A), hipótese H2")
                print(f"  {'─'*60}")
                for col, label in [('consistencia_b', 'Consistência B'), ('consistencia_c', 'Consistência C')]:
                    vals_0 = df_0.loc[common_seeds, col].values
                    vals_10 = df_10.loc[common_seeds, col].values
                    diff = vals_10 - vals_0

                    mean_diff, lo_diff, hi_diff = resumo_entre_sementes(diff)

                    # Cohen's d com desvios POPULACIONAIS (np.std, ddof = 0; com 3 sementes o d sai maior
                    # do que com ddof = 1). Mantido assim para não alterar os valores já reportados.
                    pooled_std = np.sqrt((np.std(vals_0)**2 + np.std(vals_10)**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

                    consist = "consistente entre sementes" if diferenca_consistente(lo_diff, hi_diff) else "sinal varia entre sementes"
                    print(f"  {label}: Δ={mean_diff:+.3f} mín–máx=[{lo_diff:+.3f}, {hi_diff:+.3f}] ({consist})  "
                          f"d={cohens_d:.2f}")

                print("\n  Interpretação Cohen's d: |d|<0.2 negligível, 0.2-0.5 pequeno, 0.5-0.8 médio, >0.8 grande "
                      "(d calculado com np.std, ddof = 0)")

        # (O teste "métrica estimada vs. Euclidiana" e a diagonal_limitation_flag foram
        # removidos em 09/10/2026, M15: não são métricas da seção IV.3.)

        # --- Estabilidade de W entre seeds (H5) ---
        # A magnitude de W é arbitrária (depende da escala da margem-alvo).
        # O que importa é a DIREÇÃO: w_ratio = w1/w2 ou o ângulo do vetor unitário.
        # A métrica do A é estimada UMA vez (Passo 4, rótulos zero-shot do Passo 2) e repetida nas linhas
        # de todos os k de B/C: o bloco sai uma única vez (k de referência), não por k.
        print("\n  ════════════════════════════════════════════════")
        print("  ESTABILIDADE DE W ENTRE SEEDS (H5)")
        print("  ════════════════════════════════════════════════")
        print("\n  IMPORTANTE: Valores absolutos de W são irrelevantes para a fronteira")
        print("  de decisão: apenas a DIREÇÃO (razão w1/w2) determina a geometria.")
        print("  Multiplicar W por k>0 não altera classificações.\n")

        ks_h5 = sorted(df_default['k'].unique())
        for k_val in ([0] if 0 in ks_h5 else ks_h5[:1]):   # uma só passagem: o W é o mesmo em todos os k
            subset_results = [r for r in resultados_abc
                              if r.nomes_classes == ("A", "B")
                              and r.prompt_variant == "default"
                              and tuple(r.feature_names) == ("x1", "x2")
                              and r.k_passo4_bc == k_val
                              and r.w_aprendido is not None
                              and np.linalg.norm(r.w_aprendido) > 0]

            if len(subset_results) < 2:
                continue

            # Coletar direções (vetores unitários) e razões
            directions = []
            ratios = []
            w_raw = []
            for r in subset_results:
                w = r.w_aprendido
                w_raw.append(w.copy())
                norm = np.linalg.norm(w)
                if norm > 0:
                    directions.append(w / norm)
                ratio = w[0] / w[1] if w[1] != 0 else float('inf')
                ratios.append(ratio)

            ratios_finite = [r for r in ratios if np.isfinite(r)]

            print(f"  Métrica do A, Passo 4 (estimada uma vez sobre os rótulos zero-shot do Passo 2; a mesma em "
                  f"todos os k de B/C; linhas com k={k_val}): {len(subset_results)} observações, "
                  f"{len(set(r.random_seed for r in subset_results))} seeds:")

            # Tabela: seed | W bruto | W normalizado | w1/w2 | cos(NNLS)
            print(f"    {'Seed':>6} {'Rep':>4} {'W bruto':>20} {'W unitário':>20} {'w1/w2':>8} {'cos(NNLS)':>10}")
            print(f"    {'─'*6} {'─'*4} {'─'*20} {'─'*20} {'─'*8} {'─'*10}")
            for r in subset_results:
                w = r.w_aprendido
                norm = np.linalg.norm(w)
                w_unit = w / norm if norm > 0 else w
                ratio = w[0] / w[1] if w[1] != 0 else float('inf')
                cos_nnls = r.w_cosine_sim_nnls
                print(f"    {r.random_seed:>6} {r.repeticao:>4} "
                      f"[{w[0]:>7.3f}, {w[1]:>7.3f}] "
                      f"[{w_unit[0]:>7.3f}, {w_unit[1]:>7.3f}] "
                      f"{ratio:>8.3f} "
                      f"{cos_nnls:>10.4f}")

            # Similaridade cosseno entre todos os pares de W (entre seeds)
            if len(directions) >= 2:
                cos_sims = []
                for i in range(len(directions)):
                    for j in range(i + 1, len(directions)):
                        cos_sims.append(float(np.dot(directions[i], directions[j])))
                mean_cos, lo_cos, hi_cos = resumo_entre_sementes(cos_sims)
                print("\n    Similaridade cosseno entre seeds (pares): "
                      f"média={mean_cos:.4f}, range=[{lo_cos:.4f}, {hi_cos:.4f}]")
                if mean_cos > 0.95:
                    print("    → W ESTÁVEL entre seeds (cos > 0.95): H5 sustentável")
                elif mean_cos > 0.80:
                    print("    → W MODERADAMENTE estável (0.80 < cos < 0.95)")
                else:
                    print("    → W INSTÁVEL entre seeds (cos < 0.80): H5 em risco")

            # Faixa da razão w1/w2 entre sementes
            if len(ratios_finite) >= 2:
                mean_r, lo_r, hi_r = resumo_entre_sementes(ratios_finite)
                print(f"    Razão w1/w2: média={mean_r:.4f} mín–máx=[{lo_r:.4f}, {hi_r:.4f}]")
                cv = np.std(ratios_finite) / np.mean(ratios_finite) if np.mean(ratios_finite) != 0 else float('inf')
                print(f"    Coeficiente de variação (CV): {cv:.2%}")
                if cv < 0.15:
                    print("    → Razão w1/w2 ESTÁVEL (CV < 15%)")
                elif cv < 0.30:
                    print("    → Razão w1/w2 MODERADAMENTE estável (15% < CV < 30%)")
                else:
                    print("    → Razão w1/w2 INSTÁVEL (CV > 30%): métrica pode não ser captável")

            # Similaridade cosseno média entre algoritmos (robustez ao método)
            cos_nnls_vals = [r.w_cosine_sim_nnls for r in subset_results if r.w_cosine_sim_nnls > 0]
            if cos_nnls_vals:
                print(f"    Cosseno Perceptron-NNLS: média={np.mean(cos_nnls_vals):.4f} "
                      f"(min={np.min(cos_nnls_vals):.4f}, max={np.max(cos_nnls_vals):.4f})")

            print()

    # --- Problema E ---
    if resultados_e:
        print("\n  ════════════════════════════════════════════════")
        print("  PROBLEMA E (Passos 2 e 3): MÉDIA E FAIXA ENTRE SEMENTES")
        print("  ════════════════════════════════════════════════")

        df_e = pd.DataFrame([{
            'seed': r.random_seed, 'k': r.k,
            'strategy': r.example_strategy,
            'expert': r.expert_name,
            'accuracy': r.accuracy_llm_vs_expert,
            'kappa': r.kappa_llm_vs_expert,
        } for r in resultados_e])

        # Faixa por (k_exemplos, strategy) para especialista principal
        df_e_main = df_e[df_e['expert'] == df_e['expert'].iloc[0]] if len(df_e) > 0 else df_e
        if len(df_e_main) > 0:
            print(f"\n  Especialista: {df_e_main['expert'].iloc[0]}")
            print(f"\n  {'Estratégia':<12} {'k':>6} {'Acurácia (LLM vs especialista)':>32} {'mín–máx':>20} {'n':>4}")
            print(f"  {'─'*12} {'─'*6} {'─'*10} {'─'*20} {'─'*4}")

            for strategy in sorted(df_e_main['strategy'].unique()):
                for k_exemplos in sorted(df_e_main['k'].unique()):
                    subset = df_e_main[(df_e_main['strategy'] == strategy) & (df_e_main['k'] == k_exemplos)]
                    if len(subset) > 0:
                        values = subset['accuracy'].values
                        mean, lo, hi = resumo_entre_sementes(values)
                        faixa_str = f"[{lo:.3f}, {hi:.3f}]"
                        print(f"  {strategy:<12} {k_exemplos:>6} {mean:>32.3f} {faixa_str:>20} {len(values):>4}")
                print()

        # Teste: easy vs. hard (pareado por seed) para cada k_exemplos > 0 observado
        # (K_EXEMPLOS_APRENDIZ; lido dos dados para não importar o runner)
        for n_test in sorted(int(n) for n in df_e_main['k'].unique() if n > 0):
            df_easy = df_e_main[(df_e_main['strategy'] == 'easy') & (df_e_main['k'] == n_test)]
            df_hard = df_e_main[(df_e_main['strategy'] == 'hard') & (df_e_main['k'] == n_test)]
            if len(df_easy) >= 3 and len(df_hard) >= 3:
                easy_by_seed = df_easy.groupby('seed')['accuracy'].mean()
                hard_by_seed = df_hard.groupby('seed')['accuracy'].mean()
                common = easy_by_seed.index.intersection(hard_by_seed.index)
                if len(common) >= 3:
                    diff_eh = easy_by_seed.loc[common].values - hard_by_seed.loc[common].values
                    mean_d, lo_d, hi_d = resumo_entre_sementes(diff_eh)
                    print(f"  TESTE easy vs. hard ({n_test}-shot): "
                          f"Δ={mean_d:+.3f} mín–máx=[{lo_d:+.3f}, {hi_d:+.3f}]"
                          f" {'(consistente entre sementes)' if diferenca_consistente(lo_d, hi_d) else '(sinal varia entre sementes)'}")

    print("\n  Nota: faixas = mínimo e máximo observados entre sementes (× repetições); sem bootstrap "
          "(removido em 13/10/2026 a pedido do orientador).")
    print("  Comparações pareadas: diferença por semente (média e mín–máx) e Cohen's d; sem teste de "
          "Wilcoxon (removido em 13/10/2026 por decisão do autor).")
    print("  Uma diferença é 'consistente' quando todas as sementes apontam no mesmo sentido "
          "(faixa não contém zero); com 3 sementes isso é indicativo, não confirmatório.\n")


PROBLEMA_LINEAR_R3R4 = "A_linear"  # rótulo das linhas sintéticas do CSV cruzado
PROBLEMA_MEIALUA_D = "D_meialua"   # rótulo das linhas da meia-lua (Problema D, Passo 2)


def sintese_r2_r3_r4(
    results_abc_r3: List[dict],
    resultados_fg_passos_2_e_4: List[dict],
    pasta_execucao: str,
    results_abc_r4: Optional[List[dict]] = None,
    model_name: Optional[str] = None,
    results_abc_r2: Optional[List[dict]] = None,
    linhas_passos_2_a_5: Optional[List[dict]] = None,
) -> Optional[str]:
    """Sintetiza ganhos 2→3→4 atributos no Problema A (LINEAR) vs problemas NÃO-LINEARES.

    Síntese cruzada dos 3 blocos. Gera ``final_cross_linearity__<alias>.csv`` (um por
    modelo): para cada (problema, algoritmo, n_features), fidelidade vs LLM e acurácia
    vs rótulo real por seed. Colunas (09/10/2026, I8): ``problem`` (rótulo ESTÁVEL sem
    seed: A_linear, D_meialua, F_meia_lua, G1_homem_mulher, G1_homem_mulher_classesAB…),
    ``problem_name`` (nome completo, ex. F_meia_lua_seed42), ``problem_base`` (= problem),
    ``feature_names`` (x1/x2, peso/altura, peso_kg/altura_m…), ``is_nonlinear``,
    ``algorithm``, ``n_features``, ``fidelity_vs_llm``, ``accuracy_vs_true``, ``seed``.

    Fontes: as linhas ``A_linear`` vêm de ``results_abc_r2/r3/r4`` (dicts com chaves
    ``fidelity``/``fidelity_nnls`` e ``accuracy_vs_true``/``accuracy_vs_true_nnls``, derivados
    do zero-shot do Problema A (Passos 2 e 4); aceita as chaves antigas ``accuracy``/
    ``accuracy_nnls`` como fidelidade). A meia-lua vem do Problema D (linhas zero-shot de
    ``linhas_passos_2_a_5`` com ``problem == "D"``, 150 pontos, R2/R3/R4); o Problema F (105 pontos,
    split) só entra quando não há linhas do D. Os demais externos (G1/G2) vêm de
    ``resultados_fg_passos_2_e_4`` (uma linha por variante de atributo e de classe, NÃO agregadas).
    """
    rows = []

    def _linear_rows(results, n_features):
        for r in results or []:
            for algo, fkey, fkey_old, akey in (('perceptron', 'fidelity', 'accuracy', 'accuracy_vs_true'),
                                               ('nnls', 'fidelity_nnls', 'accuracy_nnls', 'accuracy_vs_true_nnls')):
                fid = r.get(fkey, r.get(fkey_old))
                if fid is None:
                    continue
                rows.append({
                    'problem': PROBLEMA_LINEAR_R3R4,
                    'problem_name': PROBLEMA_LINEAR_R3R4,
                    'problem_base': PROBLEMA_LINEAR_R3R4,
                    'feature_names': 'x1/x2',
                    'is_nonlinear': False,
                    'algorithm': algo,
                    'n_features': n_features,
                    'fidelity_vs_llm': fid,
                    'accuracy_vs_true': r.get(akey),
                    'seed': r.get('seed'),
                })

    _linear_rows(results_abc_r2, 2)   # R2: 2 atributos originais
    _linear_rows(results_abc_r3, 3)   # R3: + x1·x2 (hipérbole)
    _linear_rows(results_abc_r4, 4)   # R4: + x1², x2² (elipse — sanity check)

    # Meia-lua: Problema D (zero-shot, Passos 2 e 4, 150 pontos, R2/R3/R4)
    d_rows = [r for r in (linhas_passos_2_a_5 or [])
              if r.get('problem') == 'D' and r.get('regime') == 'passo2_zero_shot']
    for r in d_rows:
        rows.append({
            'problem': PROBLEMA_MEIALUA_D,
            # Nome completo coerente com os demais (D_meialua), não a letra solta "D".
            'problem_name': PROBLEMA_MEIALUA_D,
            'problem_base': PROBLEMA_MEIALUA_D,
            'feature_names': 'x1/x2',
            'is_nonlinear': True,
            'algorithm': r.get('algorithm'),
            'n_features': r.get('n_features'),
            'fidelity_vs_llm': r.get('fidelity_vs_llm'),
            'accuracy_vs_true': r.get('accuracy_metric_vs_true_full'),
            'seed': r.get('random_seed'),
        })

    # Resultados dos problemas externos não-lineares (F só sem linhas do D)
    for r in resultados_fg_passos_2_e_4:
        pname = str(r.get('problem_name'))
        _pb = pname.rsplit('_seed', 1)[0]   # F_meia_lua_seed42 → F_meia_lua
        if d_rows and pname.startswith('F_'):
            continue
        fn = r.get('feature_names', ('x1', 'x2'))
        fn = '/'.join(fn) if isinstance(fn, (tuple, list)) else str(fn)
        for algo, fkey, akey in (('perceptron', 'fidelity_perc_vs_llm', 'accuracy_perc_vs_true'),
                                 ('nnls', 'fidelity_nnls_vs_llm', 'accuracy_nnls_vs_true')):
            rows.append({
                'problem': _pb,
                'problem_name': pname,
                'problem_base': _pb,
                'feature_names': fn,
                'is_nonlinear': True,
                'algorithm': algo,
                'n_features': r.get('n_features'),
                'fidelity_vs_llm': r.get(fkey),
                'accuracy_vs_true': r.get(akey),
                'seed': r.get('seed'),
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    # Dados derivados de UM LLM → alias do modelo no nome (um CSV por modelo)
    if model_name:
        filename = llm_asset(pasta_execucao, "final_cross_linearity.csv", model_name)
    else:  # só para uso offline sem modelo identificado
        filename = os.path.join(pasta_execucao, "final_cross_linearity.csv")
    df.to_csv(filename, index=False)
    return filename
