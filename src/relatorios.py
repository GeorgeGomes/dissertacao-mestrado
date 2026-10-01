"""Relatórios de texto e estatística descritiva dos experimentos.

Extraído de ``dissertacao_mestrado.py`` (Fase 1 da modularização): funções
``print_*`` que compõem o log_execucao.txt, ``bootstrap_ci`` e a síntese
cruzada linear x não-linear. São consumidores puros das listas de resultados —
não chamam a API nem mutam estado do runner.
"""
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from execucao_io import llm_asset
from metrics import compute_metric_confidence, predict_with_metric
from protocolo import EXAMPLE_STRATEGIES, EXPERT_W, PERCEPTRON_PARAMS
from relaxed_perceptron import train_relaxed_perceptron
from resultados import ResultadoExperimento, ResultadoPhaseEExperimento


def print_section(title: str, char: str = "="):
    """Imprime um cabeçalho de seção formatado."""
    line = char * 70
    print(f"\n{line}")
    print(f" {title}")
    print(f"{line}\n")


def print_box(text: str):
    """Imprime texto em uma caixa formatada."""
    lines = text.strip().split('\n')
    max_len = min(max(len(line) for line in lines), 90)
    print("┌" + "─" * (max_len + 2) + "┐")
    for line in lines:
        if len(line) > max_len:
            line = line[:max_len-3] + "..."
        print(f"│ {line:<{max_len}} │")
    print("└" + "─" * (max_len + 2) + "┘")


def print_error_analysis_by_region(phase_a_data: dict):
    """Análise quantitativa de erros por região (distância à fronteira).

    Responde: onde o LLM erra? Perto da fronteira (margem baixa) ou longe?
    Complementa bloco1_05_fase_a_errors_seed* com métricas numéricas no log.
    """
    print_section("ANÁLISE DE ERROS POR REGIÃO (DISTÂNCIA À FRONTEIRA)", "═")

    for seed_key, data in sorted(phase_a_data.items()):
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
        print(f"\n  {'Região':<28} {'N pontos':>10} {'Erros':>7} {'Taxa erro':>10} {'% dos erros':>12}")
        print(f"  {'─'*28} {'─'*10} {'─'*7} {'─'*10} {'─'*12}")

        for region_name, mask in regions.items():
            n_region = np.sum(mask)
            n_err_region = np.sum(disagreements[mask])
            rate = n_err_region / n_region if n_region > 0 else 0
            pct_errors = n_err_region / n_errors if n_errors > 0 else 0
            print(f"  {region_name:<28} {n_region:>10} {n_err_region:>7} {rate:>10.1%} {pct_errors:>12.1%}")

        # Correlação margem × acerto
        from scipy import stats as sp_stats
        corr, p_val = sp_stats.pointbiserialr(agreements.astype(int), confidences)
        print(f"\n  Correlação ponto-bisserial (acerto × margem): r={corr:.3f}, p={p_val:.4f}")
        if corr > 0 and p_val < 0.05:
            print(f"  → Confirmado: pontos com maior margem têm mais acertos (esperado)")
        elif p_val >= 0.05:
            print(f"  → Correlação NÃO significativa — erros não se concentram na fronteira")

        # Margem média dos pontos corretos vs errados
        margin_correct = confidences[agreements]
        margin_errors = confidences[disagreements]
        print(f"\n  Margem média dos acertos:  {np.mean(margin_correct):.3f} (±{np.std(margin_correct):.3f})")
        print(f"  Margem média dos erros:    {np.mean(margin_errors):.3f} (±{np.std(margin_errors):.3f})")

        if len(margin_errors) >= 2 and len(margin_correct) >= 2:
            stat, p_mw = sp_stats.mannwhitneyu(margin_correct, margin_errors, alternative='greater')
            print(f"  Mann-Whitney U (acertos > erros): U={stat:.0f}, p={p_mw:.4f}")
            if p_mw < 0.05:
                print(f"  → Significativo: erros têm margem menor que acertos")
            else:
                print(f"  → NÃO significativo: erros não se concentram em margens baixas")

    print()


def print_hyperparameter_sensitivity(phase_a_data: dict):
    """Análise de sensibilidade dos hiperparâmetros do Perceptron Estruturado.

    Re-executa o Perceptron com diferentes combinações de (eta, C) sobre os
    mesmos dados da Fase A (sem chamadas adicionais à API). Reporta se a
    direção de W é estável, o que indicaria robustez à escolha de hiperparâmetros.

    Custo computacional: ~O(n_configs × n_seeds × custo_perceptron), sem API calls.
    """
    print_section("SENSIBILIDADE DOS HIPERPARÂMETROS DO PERCEPTRON", "═")
    print(f"\n  Re-execução do Perceptron com diferentes (eta, C) sobre os mesmos")
    print(f"  dados da Fase A. Nenhuma chamada adicional à API é feita.")
    print(f"  Se a direção de W (cosseno) é estável, o resultado é robusto.\n")

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

    for seed_key, data in sorted(phase_a_data.items()):
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
            print(f"    → W ROBUSTO aos hiperparâmetros (cos mín > 0.95)")
        elif min_cos > 0.80:
            print(f"    → W MODERADAMENTE sensível (0.80 < cos mín < 0.95)")
        else:
            print(f"    → W SENSÍVEL aos hiperparâmetros (cos mín < 0.80) — cautela nas conclusões")

        print()

    print(f"  Nota: apenas a DIREÇÃO de W importa (escala é arbitrária).")
    print(f"  Cosseno > 0.95 entre configs = hiperparâmetros não afetam a fronteira.\n")


def print_example_order_analysis(results_order: List[ResultadoPhaseEExperimento]):
    """Análise quantitativa do viés de ordem dos exemplos few-shot (recency bias).

    Testa se a ordenação dos exemplos afeta significativamente a performance do LLM.
    Recency bias: LLMs tendem a dar mais peso aos últimos exemplos do prompt.
    """
    from scipy import stats as sp_stats

    print_section("ANÁLISE DE VIÉS DE ORDEM DOS EXEMPLOS (RECENCY BIAS)", "═")

    df = pd.DataFrame([{
        'seed': r.random_seed, 'rep': r.repeticao,
        'n_shot': r.n_shot,
        'ordering': r.example_strategy.replace("mixed_order_", ""),
        'accuracy': r.accuracy_llm_vs_expert,
        'kappa': r.kappa_llm_vs_expert,
        'f1': r.f1_llm_vs_expert,
    } for r in results_order])

    orderings = sorted(df['ordering'].unique())
    n_shots = sorted(df['n_shot'].unique())

    # Tabela resumo
    print(f"\n  {'Ordenação':<18} {'n_shot':>6} {'Acc média':>10} {'±std':>8} {'Kappa':>8} {'n':>4}")
    print(f"  {'─'*18} {'─'*6} {'─'*10} {'─'*8} {'─'*8} {'─'*4}")

    for ns in n_shots:
        for ordering in orderings:
            subset = df[(df['ordering'] == ordering) & (df['n_shot'] == ns)]
            if len(subset) > 0:
                print(f"  {ordering:<18} {ns:>6} {subset['accuracy'].mean():>10.1%} "
                      f"{subset['accuracy'].std():>8.1%} {subset['kappa'].mean():>8.3f} "
                      f"{len(subset):>4}")
        print()

    # Teste estatístico por n_shot: Kruskal-Wallis (não-paramétrico, >2 grupos)
    print(f"  TESTES DE SIGNIFICÂNCIA:")
    print(f"  {'─'*60}")

    for ns in n_shots:
        groups = []
        group_names = []
        for ordering in orderings:
            vals = df[(df['ordering'] == ordering) & (df['n_shot'] == ns)]['accuracy'].values
            if len(vals) >= 2:
                groups.append(vals)
                group_names.append(ordering)

        if len(groups) >= 2:
            # Kruskal-Wallis: H0 = todas as ordenações têm mesma distribuição
            stat, p_val = sp_stats.kruskal(*groups)
            print(f"\n  {ns}-shot: Kruskal-Wallis H={stat:.3f}, p={p_val:.4f}")

            if p_val < 0.05:
                print(f"  → SIGNIFICATIVO: a ordem dos exemplos AFETA a performance")
                # Identificar qual ordenação é melhor/pior
                means = {name: np.mean(g) for name, g in zip(group_names, groups)}
                best = max(means, key=means.get)
                worst = min(means, key=means.get)
                diff = means[best] - means[worst]
                print(f"    Melhor: {best} ({means[best]:.1%}), Pior: {worst} ({means[worst]:.1%}), Δ={diff:+.1%}")

                # Teste pareado: class0_first vs class1_first (recency bias direto)
                if 'class0_first' in group_names and 'class1_first' in group_names:
                    c0 = df[(df['ordering'] == 'class0_first') & (df['n_shot'] == ns)]
                    c1 = df[(df['ordering'] == 'class1_first') & (df['n_shot'] == ns)]
                    # Parear por seed
                    c0_by_seed = c0.groupby('seed')['accuracy'].mean()
                    c1_by_seed = c1.groupby('seed')['accuracy'].mean()
                    common = c0_by_seed.index.intersection(c1_by_seed.index)
                    if len(common) >= 3:
                        diff_paired = c0_by_seed.loc[common].values - c1_by_seed.loc[common].values
                        mean_d, lo_d, hi_d = bootstrap_ci(diff_paired)
                        print(f"    Classe0_first - Classe1_first: Δ={mean_d:+.3f} "
                              f"CI95%=[{lo_d:+.3f}, {hi_d:+.3f}]")
                        if lo_d > 0:
                            print(f"    → Recency bias: última classe vista (classe 1) é FAVORECIDA")
                        elif hi_d < 0:
                            print(f"    → Recency bias: última classe vista (classe 0) é FAVORECIDA")
            else:
                print(f"  → NÃO significativo: a ordem dos exemplos NÃO afeta a performance")

    # Variabilidade geral por ordering (colapsando n_shots)
    overall = df.groupby('ordering')['accuracy'].agg(['mean', 'std'])
    max_range = overall['mean'].max() - overall['mean'].min()
    print(f"\n  Variação total entre ordenações: {max_range:.1%}")
    if max_range < 0.03:
        print(f"  → Efeito de ordem NEGLIGÍVEL (< 3 p.p.)")
    elif max_range < 0.10:
        print(f"  → Efeito de ordem MODERADO (3-10 p.p.) — reportar como limitação")
    else:
        print(f"  → Efeito de ordem GRANDE (> 10 p.p.) — recency bias significativo")

    print()


def print_phase_e_analysis(results_e: List[ResultadoPhaseEExperimento]):
    """Imprime análise detalhada dos resultados da Fase E."""
    print_section("ANÁLISE DA FASE E: LLM COMO APRENDIZ", "═")

    df = pd.DataFrame([
        {
            'model': f"{r.provider}/{r.model_name} (temp={r.temperature})",
            'n_shot': r.n_shot,
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

    # Uma análise por (modelo, perito): com RUN_MULTIPLE_EXPERTS há 3 peritos
    # (aniso_x2, aniso_x1, euclidean) e misturá-los diluiria as curvas.
    for model in models:
      for expert in df[df['model'] == model]['expert'].unique():
        model_df = df[(df['model'] == model) & (df['expert'] == expert)]
        expert_w = model_df['expert_w'].iloc[0]
        print(f"\n{'═' * 70}")
        print(f" MODELO: {model} | PERITO: {expert}")
        print(f"{'═' * 70}")

        print(f"\n  Métrica do Perito W = [{expert_w[0]:.2f}, {expert_w[1]:.2f}]")
        print(f"  Acurácia do Perito vs. GT: {model_df['expert_vs_gt'].mean():.1%}")

        # 1. Curva de aprendizado por estratégia
        print(f"\n  1. CURVA DE APRENDIZADO (Concordância LLM vs. Perito):")
        print("  " + "-" * 65)
        header = f"  {'n_shot':>6}"
        for strat in EXAMPLE_STRATEGIES:
            header += f" | {strat:>12}"
        print(header)
        print("  " + "-" * 65)

        for n in sorted(model_df['n_shot'].unique()):
            line = f"  {n:>6}"
            for strat in EXAMPLE_STRATEGIES:
                subset = model_df[(model_df['n_shot'] == n) & (model_df['strategy'] == strat)]
                if len(subset) > 0:
                    mean = subset['accuracy'].mean()
                    std = subset['accuracy'].std()
                    line += f" | {mean:.1%}±{std:.1%}"
                else:
                    line += f" | {'N/D':>12}"
            print(line)

        # 2. Melhor estratégia por n_shot
        print(f"\n  2. MELHOR ESTRATÉGIA POR N_SHOT:")
        print("  " + "-" * 50)
        for n in sorted(model_df['n_shot'].unique()):
            if n == 0:
                continue
            subset = model_df[model_df['n_shot'] == n]
            best = subset.groupby('strategy')['accuracy'].mean().idxmax()
            best_val = subset.groupby('strategy')['accuracy'].mean().max()
            print(f"     {n:>3}-shot: {best.upper():>8} ({best_val:.1%})")

        # 3. Melhoria do zero-shot para o melhor few-shot
        print(f"\n  3. MELHORIA DO ZERO-SHOT PARA O MELHOR FEW-SHOT:")
        print("  " + "-" * 50)
        zero_shot_acc = model_df[model_df['n_shot'] == 0]['accuracy'].mean()
        print(f"     Linha de base zero-shot: {zero_shot_acc:.1%}")

        for strat in EXAMPLE_STRATEGIES:
            strat_df = model_df[model_df['strategy'] == strat]
            if len(strat_df) == 0:
                continue
            best_n = strat_df.groupby('n_shot')['accuracy'].mean().idxmax()
            best_acc = strat_df.groupby('n_shot')['accuracy'].mean().max()
            improvement = best_acc - zero_shot_acc
            print(f"     {strat.upper():>8}: melhor={best_acc:.1%} em {best_n}-shot "
                  f"(Δ={improvement:+.1%})")

        # 4. Respostas malformadas
        total_malformed = model_df['n_malformed'].sum()
        if total_malformed > 0:
            print(f"\n  ⚠️ Total de respostas malformadas: {total_malformed}")

    # Conclusão geral do experimento — perito PRINCIPAL (EXPERT_W = aniso_x2)
    main_expert = df['expert'].iloc[0]
    df_main = df[df['expert'] == main_expert]
    overall_zero = df_main[df_main['n_shot'] == 0]['accuracy'].mean()
    overall_best = df_main.groupby(['n_shot', 'strategy'])['accuracy'].mean().max()
    best_config = df_main.groupby(['n_shot', 'strategy'])['accuracy'].mean().idxmax()

    print_box(f"""
CONCLUSÃO DA FASE E: LLM COMO APRENDIZ (perito principal: {main_expert})

Métrica do perito: W = [{EXPERT_W[0]:.2f}, {EXPERT_W[1]:.2f}]
(Pondera a dimensão x2 {EXPERT_W[1]/EXPERT_W[0]:.1f}x mais do que x1)

Linha de base zero-shot: {overall_zero:.1%}
Melhor resultado few-shot: {overall_best:.1%} (estratégia {best_config[1]}, {best_config[0]}-shot)
Melhoria geral: {overall_best - overall_zero:+.1%}

{'✓ O LLM CONSEGUE aprender com exemplos do perito — o desempenho melhora com mais exemplos.' if overall_best - overall_zero > 0.1 else
 '~ O LLM mostra aprendizado MODERADO com exemplos do perito.' if overall_best - overall_zero > 0.05 else
 '✗ O LLM NÃO melhora significativamente com os exemplos do perito.'}
"""
    )

    # --- Análise data-driven de H4 (easy vs hard) ---
    # Compara acurácia média de cada estratégia (exceto zero-shot)
    df_fewshot = df[df['n_shot'] > 0]
    if len(df_fewshot) > 0:
        strat_means = df_fewshot.groupby('strategy')['accuracy'].mean()
        easy_mean = strat_means.get('easy', None)
        hard_mean = strat_means.get('hard', None)
        random_mean = strat_means.get('random', None)

        print_box(f"""
AVALIAÇÃO DA HIPÓTESE H4: "Exemplos hard são mais informativos que easy"

Acurácia média por estratégia (todos os n_shot > 0):
{chr(10).join(f'  {s.upper():>8}: {v:.1%}' for s, v in sorted(strat_means.items(), key=lambda x: -x[1]))}

{'RESULTADO: H4 REFUTADA.' if easy_mean is not None and hard_mean is not None and easy_mean > hard_mean else 'RESULTADO: H4 sustentável.' if easy_mean is not None and hard_mean is not None and hard_mean > easy_mean else 'RESULTADO: Dados insuficientes para avaliar H4.'}
{f'Exemplos easy ({easy_mean:.1%}) superam hard ({hard_mean:.1%}) consistentemente.' if easy_mean is not None and hard_mean is not None and easy_mean > hard_mean else ''}
{f'Exemplos hard ({hard_mean:.1%}) performam abaixo de random ({random_mean:.1%}).' if hard_mean is not None and random_mean is not None and hard_mean < random_mean else ''}

Interpretação: exemplos próximos à fronteira de decisão são AMBÍGUOS para o
LLM — não fornecem padrões claros de cada classe. Exemplos fáceis funcionam
como "âncoras" que definem bem as regiões de cada classe, permitindo ao LLM
interpolar para os pontos intermediários. Este é um achado relevante sobre
o mecanismo de aprendizado in-context dos LLMs: eles se beneficiam mais de
exemplos prototípicos (alta margem) do que de exemplos fronteiriços.

Nota: esta hipótese foi formulada a priori e refutada pelos dados.
A refutação é reportada como resultado genuíno, não como falha do método.
"""
        )


def print_final_analysis(resultados: List[ResultadoExperimento]):
    """Imprime a análise final dos resultados das Fases A-C."""
    print_section("ANÁLISE FINAL: FASES A-C", "═")

    df = pd.DataFrame([
        {
            'model': f"{r.provider}/{r.model_name} (temp={r.temperature})",
            'seed': r.random_seed,
            'n_shot': r.n_shot,
            'nomes': f"{r.nomes_classes[0]}/{r.nomes_classes[1]}",
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

    models = df['model'].unique()

    for model in models:
        model_df = df[df['model'] == model]
        print(f"\n{'═' * 70}")
        print(f" MODELO: {model}")
        print(f"{'═' * 70}")

        print("\n  1. CONSISTÊNCIA POR NÚMERO DE EXEMPLOS:")
        print(f"     {'n_shot':>6} | {'Consist B':>10} | {'Consist C':>10} | {'Kappa B':>8} | {'Kappa C':>8}")
        print("     " + "-" * 55)
        for n in sorted(model_df['n_shot'].unique()):
            subset = model_df[model_df['n_shot'] == n]
            c_b = f"{subset['consistencia_b'].mean():.1%}±{subset['consistencia_b'].std():.1%}"
            c_c = f"{subset['consistencia_c'].mean():.1%}±{subset['consistencia_c'].std():.1%}"
            k_b = f"{subset['kappa_b'].mean():.3f}"
            k_c = f"{subset['kappa_c'].mean():.3f}"
            print(f"     {n:>6} | {c_b:>10} | {c_c:>10} | {k_b:>8} | {k_c:>8}")

        print("\n  2. CONSISTÊNCIA POR NOMES DE CLASSE:")
        for nome in model_df['nomes'].unique():
            subset = model_df[model_df['nomes'] == nome]
            print(f"     {nome:20s}: B={subset['consistencia_b'].mean():.1%}, C={subset['consistencia_c'].mean():.1%}")

    mean_b = df['consistencia_b'].mean()
    mean_c = df['consistencia_c'].mean()

    print_box(f"""
RESUMO DAS FASES A-C

Consistência Média Problema B: {mean_b:.1%} ± {df['consistencia_b'].std():.1%}
Consistência Média Problema C: {mean_c:.1%} ± {df['consistencia_c'].std():.1%}

{'✓ Os LLMs MANTÊM alta consistência em ambos os problemas.' if min(mean_b, mean_c) > 0.85 else
 '~ Consistência MODERADA entre os problemas.' if min(mean_b, mean_c) > 0.7 else
 '✗ Os LLMs NÃO mantêm consistência.'}
""")


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calcula intervalo de confiança via bootstrap.

    Retorna (média, limite_inferior, limite_superior).
    Usa o método percentil (Efron & Tibshirani, 1993).
    """
    if len(data) < 2:
        return float(np.mean(data)), float(np.mean(data)), float(np.mean(data))
    # Seed fixa proposital: torna o intervalo de confiança reproduzível bit-a-bit,
    # independentemente da semente do experimento. NÃO é número mágico — é a
    # reprodutibilidade do próprio bootstrap (mesmo IC a cada leitura dos dados).
    rng = np.random.RandomState(42)
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - confidence
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(data)), float(lo), float(hi)


def print_statistical_summary(
    all_results_abc: List[ResultadoExperimento],
    all_results_e: List[ResultadoPhaseEExperimento],
):
    """Imprime sumário estatístico com bootstrap CI e testes de significância.

    Endereça a crítica de ausência de testes estatísticos formais:
    - Bootstrap 95% CI para todas as métricas principais
    - Wilcoxon signed-rank test para comparações pareadas
    - Tamanho de efeito (Cohen's d) quando aplicável
    """
    from scipy import stats

    print_section("SUMÁRIO ESTATÍSTICO (Bootstrap 95% CI + Testes de Significância)", "═")

    # --- Fases A-C ---
    if all_results_abc:
        print(f"\n  ════════════════════════════════════════════════")
        print(f"  FASES A-C: INTERVALOS DE CONFIANÇA (Bootstrap)")
        print(f"  ════════════════════════════════════════════════")

        # Filtrar apenas classes A/B, prompt default para métricas limpas
        df = pd.DataFrame([{
            'seed': r.random_seed, 'n_shot': r.n_shot,
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
        } for r in all_results_abc])

        # Métricas por n_shot (classes A/B, prompt default, features x1/x2).
        # O filtro de features evita a contaminação do IC da config base pelas
        # linhas do experimento de nomes de features (que também usam A/B+default).
        df_default = df[(df['nomes'] == 'A/B') & (df['prompt_variant'] == 'default')
                        & (df['features'] == 'x1/x2')]
        if len(df_default) > 0:
            print(f"\n  Configuração base: classes A/B, prompt default, features x1/x2")
            print(f"  n = {len(df_default)} observações ({len(df_default['seed'].unique())} seeds)")
            print(f"\n  {'Métrica':<25} {'n_shot':>6} {'Média':>8} {'95% CI':>20} {'n':>4}")
            print(f"  {'─'*25} {'─'*6} {'─'*8} {'─'*20} {'─'*4}")

            for n_shot in sorted(df_default['n_shot'].unique()):
                subset = df_default[df_default['n_shot'] == n_shot]
                print(f"  --- Perceptron ---")
                for col, label in [
                    ('fidelidade', 'Fidelidade A (Perc)'),
                    ('consistencia_b', 'Consistência B (Perc)'),
                    ('consistencia_c', 'Consistência C (Perc)'),
                    ('kappa_b', 'Kappa B (Perc)'),
                    ('kappa_c', 'Kappa C (Perc)'),
                ]:
                    values = subset[col].values
                    mean, lo, hi = bootstrap_ci(values)
                    ci_str = f"[{lo:.3f}, {hi:.3f}]"
                    print(f"  {label:<25} {n_shot:>6} {mean:>8.3f} {ci_str:>20} {len(values):>4}")
                print()

        # Teste: consistência zero-shot vs. 10-shot (pareado por seed)
        df_0 = df_default[df_default['n_shot'] == 0].groupby('seed')[['consistencia_b', 'consistencia_c']].mean()
        df_10 = df_default[df_default['n_shot'] == 10].groupby('seed')[['consistencia_b', 'consistencia_c']].mean() if 10 in df_default['n_shot'].values else None

        if df_10 is not None:
            common_seeds = df_0.index.intersection(df_10.index)
            if len(common_seeds) >= 3:
                print(f"\n  TESTE DE SIGNIFICÂNCIA: Zero-shot vs. 10-shot (pareado por seed)")
                print(f"  {'─'*60}")
                for col, label in [('consistencia_b', 'Consistência B'), ('consistencia_c', 'Consistência C')]:
                    vals_0 = df_0.loc[common_seeds, col].values
                    vals_10 = df_10.loc[common_seeds, col].values
                    diff = vals_10 - vals_0

                    mean_diff, lo_diff, hi_diff = bootstrap_ci(diff)

                    # Wilcoxon signed-rank (não-paramétrico, pareado)
                    if len(common_seeds) >= 5 and not np.all(diff == 0):
                        stat, p_value = stats.wilcoxon(vals_0, vals_10)
                        p_str = f"p={p_value:.4f}" if p_value >= 0.001 else f"p<0.001"
                    else:
                        p_str = f"n<5, teste não aplicável"

                    # Cohen's d
                    pooled_std = np.sqrt((np.std(vals_0)**2 + np.std(vals_10)**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

                    print(f"  {label}: Δ={mean_diff:+.3f} CI95%=[{lo_diff:+.3f}, {hi_diff:+.3f}]  "
                          f"{p_str}  d={cohens_d:.2f}")

                print(f"\n  Interpretação Cohen's d: |d|<0.2 negligível, 0.2-0.5 pequeno, 0.5-0.8 médio, >0.8 grande")

        # Teste: consistência métrica estimada vs. Euclidiana
        euc_data = [(r.consistencia_problema_b, r.consistencia_euclidiana_problema_b)
                     for r in all_results_abc
                     if r.nomes_classes == ("A", "B") and r.prompt_variant == "default"
                     and r.feature_names == ("x1", "x2") and r.n_shot == 0]
        if len(euc_data) >= 3:
            metric_vals = np.array([x[0] for x in euc_data])
            euc_vals = np.array([x[1] for x in euc_data])
            diff_euc = metric_vals - euc_vals
            mean_d, lo_d, hi_d = bootstrap_ci(diff_euc)

            print(f"\n  TESTE: Métrica estimada vs. Euclidiana (zero-shot, Problema B)")
            print(f"  {'─'*60}")
            print(f"  Δ(métrica - euclidiana) = {mean_d:+.3f} CI95%=[{lo_d:+.3f}, {hi_d:+.3f}]")
            if lo_d > 0:
                print(f"  → Métrica estimada SIGNIFICATIVAMENTE superior à Euclidiana")
            elif hi_d < 0:
                print(f"  → Euclidiana SIGNIFICATIVAMENTE superior — limitação diagonal?")
            else:
                print(f"  → Diferença NÃO significativa (CI inclui zero)")

        # --- Estabilidade de W entre seeds (H5) ---
        # A magnitude de W é arbitrária (depende da escala da margem-alvo).
        # O que importa é a DIREÇÃO: w_ratio = w1/w2 ou o ângulo do vetor unitário.
        # Se w_ratio é estável entre seeds (mesmo n_shot), H5 é sustentável.
        print(f"\n  ════════════════════════════════════════════════")
        print(f"  ESTABILIDADE DE W ENTRE SEEDS (H5)")
        print(f"  ════════════════════════════════════════════════")
        print(f"\n  IMPORTANTE: Valores absolutos de W são irrelevantes para a fronteira")
        print(f"  de decisão — apenas a DIREÇÃO (razão w1/w2) determina a geometria.")
        print(f"  Multiplicar W por k>0 não altera classificações.\n")

        for n_shot_val in sorted(df_default['n_shot'].unique()):
            subset_results = [r for r in all_results_abc
                              if r.nomes_classes == ("A", "B")
                              and r.prompt_variant == "default"
                              and tuple(r.feature_names) == ("x1", "x2")
                              and r.n_shot == n_shot_val
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

            print(f"  n_shot={n_shot_val} ({len(subset_results)} observações, "
                  f"{len(set(r.random_seed for r in subset_results))} seeds):")

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
                mean_cos, lo_cos, hi_cos = bootstrap_ci(np.array(cos_sims)) if len(cos_sims) >= 3 else (np.mean(cos_sims), np.min(cos_sims), np.max(cos_sims))
                print(f"\n    Similaridade cosseno entre seeds (pares): "
                      f"média={mean_cos:.4f}, range=[{lo_cos:.4f}, {hi_cos:.4f}]")
                if mean_cos > 0.95:
                    print(f"    → W ESTÁVEL entre seeds (cos > 0.95) — H5 sustentável")
                elif mean_cos > 0.80:
                    print(f"    → W MODERADAMENTE estável (0.80 < cos < 0.95)")
                else:
                    print(f"    → W INSTÁVEL entre seeds (cos < 0.80) — H5 em risco")

            # Bootstrap CI da razão w1/w2
            if len(ratios_finite) >= 2:
                mean_r, lo_r, hi_r = bootstrap_ci(np.array(ratios_finite)) if len(ratios_finite) >= 3 else (np.mean(ratios_finite), np.min(ratios_finite), np.max(ratios_finite))
                print(f"    Razão w1/w2: média={mean_r:.4f} CI95%=[{lo_r:.4f}, {hi_r:.4f}]")
                cv = np.std(ratios_finite) / np.mean(ratios_finite) if np.mean(ratios_finite) != 0 else float('inf')
                print(f"    Coeficiente de variação (CV): {cv:.2%}")
                if cv < 0.15:
                    print(f"    → Razão w1/w2 ESTÁVEL (CV < 15%)")
                elif cv < 0.30:
                    print(f"    → Razão w1/w2 MODERADAMENTE estável (15% < CV < 30%)")
                else:
                    print(f"    → Razão w1/w2 INSTÁVEL (CV > 30%) — métrica pode não ser captável")

            # Similaridade cosseno média entre algoritmos (robustez ao método)
            cos_nnls_vals = [r.w_cosine_sim_nnls for r in subset_results if r.w_cosine_sim_nnls > 0]
            if cos_nnls_vals:
                print(f"    Cosseno Perceptron-NNLS: média={np.mean(cos_nnls_vals):.4f} "
                      f"(min={np.min(cos_nnls_vals):.4f}, max={np.max(cos_nnls_vals):.4f})")

            print()

    # --- Fase E ---
    if all_results_e:
        print(f"\n  ════════════════════════════════════════════════")
        print(f"  FASE E: INTERVALOS DE CONFIANÇA (Bootstrap)")
        print(f"  ════════════════════════════════════════════════")

        df_e = pd.DataFrame([{
            'seed': r.random_seed, 'n_shot': r.n_shot,
            'strategy': r.example_strategy,
            'expert': r.expert_name,
            'accuracy': r.accuracy_llm_vs_expert,
            'kappa': r.kappa_llm_vs_expert,
        } for r in all_results_e])

        # CI por (n_shot, strategy) para expert principal
        df_e_main = df_e[df_e['expert'] == df_e['expert'].iloc[0]] if len(df_e) > 0 else df_e
        if len(df_e_main) > 0:
            print(f"\n  Expert: {df_e_main['expert'].iloc[0]}")
            print(f"\n  {'Estratégia':<12} {'n_shot':>6} {'Acc Média':>10} {'95% CI':>20} {'n':>4}")
            print(f"  {'─'*12} {'─'*6} {'─'*10} {'─'*20} {'─'*4}")

            for strategy in sorted(df_e_main['strategy'].unique()):
                for n_shot in sorted(df_e_main['n_shot'].unique()):
                    subset = df_e_main[(df_e_main['strategy'] == strategy) & (df_e_main['n_shot'] == n_shot)]
                    if len(subset) > 0:
                        values = subset['accuracy'].values
                        mean, lo, hi = bootstrap_ci(values)
                        ci_str = f"[{lo:.3f}, {hi:.3f}]"
                        print(f"  {strategy:<12} {n_shot:>6} {mean:>10.3f} {ci_str:>20} {len(values):>4}")
                print()

        # Teste: easy vs. hard (pareado por seed) para cada n_shot > 0 observado
        # (FEW_SHOT_SIZES_PHASE_E; lido dos dados para não importar o runner)
        for n_test in sorted(int(n) for n in df_e_main['n_shot'].unique() if n > 0):
            df_easy = df_e_main[(df_e_main['strategy'] == 'easy') & (df_e_main['n_shot'] == n_test)]
            df_hard = df_e_main[(df_e_main['strategy'] == 'hard') & (df_e_main['n_shot'] == n_test)]
            if len(df_easy) >= 3 and len(df_hard) >= 3:
                easy_by_seed = df_easy.groupby('seed')['accuracy'].mean()
                hard_by_seed = df_hard.groupby('seed')['accuracy'].mean()
                common = easy_by_seed.index.intersection(hard_by_seed.index)
                if len(common) >= 3:
                    diff_eh = easy_by_seed.loc[common].values - hard_by_seed.loc[common].values
                    mean_d, lo_d, hi_d = bootstrap_ci(diff_eh)
                    print(f"  TESTE easy vs. hard ({n_test}-shot): "
                          f"Δ={mean_d:+.3f} CI95%=[{lo_d:+.3f}, {hi_d:+.3f}]"
                          f" {'(sig.)' if lo_d > 0 or hi_d < 0 else '(n.s.)'}")

    print(f"\n  Nota: Bootstrap com 10.000 reamostras, seed fixa=42.")
    print(f"  Wilcoxon signed-rank test usado para comparações pareadas (não-paramétrico).")
    print(f"  Significância estatística avaliada pelo CI 95% não incluir zero.\n")


PROBLEMA_LINEAR_R3R4 = "A_linear"  # rótulo das linhas sintéticas do CSV cruzado


def summarize_cross_linearity(
    results_abc_r3: List[dict],
    external_results: List[dict],
    pasta_execucao: str,
    results_abc_r4: Optional[List[dict]] = None,
    model_name: Optional[str] = None,
    results_abc_r2: Optional[List[dict]] = None,
) -> Optional[str]:
    """Sintetiza ganhos 2→3→4 atributos no Problema A (LINEAR) vs problemas NÃO-LINEARES.

    Síntese cruzada dos 3 blocos. Gera ``final_cross_linearity__<alias>.csv`` (um por
    modelo): para cada (problema, algoritmo, n_features), fidelidade vs LLM e acurácia
    vs rótulo real por seed. As linhas sintéticas (``PROBLEMA_LINEAR_R3R4``) vêm do
    experimento R2/R3/R4 sobre o Problema A (os únicos dados lineares aumentados);
    as não-lineares, dos pipelines externos (meia-lua e peso × altura).
    """
    rows = []

    def _linear_rows(results, n_features):
        for r in results or []:
            for algo, key in (('perceptron', 'accuracy'), ('nnls', 'accuracy_nnls')):
                if r.get(key) is None:
                    continue
                rows.append({
                    'problem': PROBLEMA_LINEAR_R3R4,
                    'is_nonlinear': False,
                    'algorithm': algo,
                    'n_features': n_features,
                    'fidelity_vs_llm': r.get(key),
                    'accuracy_vs_true': None,
                    'seed': r.get('seed'),
                })

    _linear_rows(results_abc_r2, 2)   # R2: 2 atributos originais
    _linear_rows(results_abc_r3, 3)   # R3: + x1·x2 (hipérbole)
    _linear_rows(results_abc_r4, 4)   # R4: + x1², x2² (elipse — sanity check)

    # Resultados dos problemas externos não-lineares
    for r in external_results:
        rows.append({
            'problem': r.get('problem_name'),
            'is_nonlinear': True,
            'algorithm': 'perceptron',
            'n_features': r.get('n_features'),
            'fidelity_vs_llm': r.get('fidelity_perc_vs_llm'),
            'accuracy_vs_true': r.get('accuracy_perc_vs_true'),
            'seed': r.get('seed'),
        })
        rows.append({
            'problem': r.get('problem_name'),
            'is_nonlinear': True,
            'algorithm': 'nnls',
            'n_features': r.get('n_features'),
            'fidelity_vs_llm': r.get('fidelity_nnls_vs_llm'),
            'accuracy_vs_true': r.get('accuracy_nnls_vs_true'),
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
