"""Testes de fumaça de ``plots.py``: nomes dos painéis e correções de 01/10/2026.

Contratos verificados (sem API, pasta temporária):

1. Painéis individuais seguem ``<base>_<painel>__<alias>_<color|gray>.png`` (painel antes
   do alias, cor por último; 09/10/2026), via ``asset_variant`` + ``salvar_figura``; toda figura
   sai em ``_color`` e ``_gray`` (as de símbolos são desenhadas duas vezes).
2. ``plot_a_passo4_r2_r3_r4`` aceita R2/R3/R4 e gera os painéis
   ``_linear_vs_quadratica`` e ``_algoritmos_r3r4`` (com ou sem R4).
3. ``_dilution_reference`` lê o número de hard fixos do nome da estratégia
   (``dilution_4hard_0easy``) em vez de um k_exemplos fixo inexistente.
4. ``_grouped_bar_offsets`` centra as barras agrupadas no tick.
5. (09/10/2026) Métricas da seção IV.3 do orientador: os textos das figuras (títulos, eixos,
   legendas, ticks) só usam "Fidelidade", "Acurácia", "Consistência", "Cohen's κ" e
   "F1-Macro"; nada de "F1-Score", "Concordância", "Taxa de Erro", "Acerto/Erro", "Kappa" solto.
6. Contratos novos: oráculo lê ``accuracy`` (fallback ``fidelity``) e agrega as 3 seeds;
   R3/R4 lê ``fidelity``/``fidelity_nnls`` (fallback ``accuracy``); o teto da curva dos Passos 2 a 5
   vem de ``accuracy_teto_vs_true``; k-NN é UMA série nos baselines; bloco1_graphic_16 só
   com os 4 pares base; bloco1_graphic_15 pareado normal × invertido; ``plot_oracle_transfer``
   não existe mais.

Execução: .venv/bin/python -m unittest tests.test_plots_nomes
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import plots  # noqa: E402
from execucao_io import llm_asset  # noqa: E402
from resultados import LearnedMetric, ResultadoABC, ResultadoAprendiz  # noqa: E402

# Nomes de métrica PROIBIDOS nos textos das figuras (seção IV.3 do PDF do orientador).
# "Kappa" só pode aparecer como "Cohen's κ"; o grep abaixo pega "Kappa" em qualquer forma.
TEXTOS_PROIBIDOS = ["F1-Score", "F1 Score", "Concordância", "Concordam", "Discordam", "Taxa de Erro",
                    "Taxa de erro", "Taxa de Discord", "Acerto", "Agreement", "Kappa", "kappa="]


def _res_abc(seed=42, k_exemplos=0, nomes=("A", "B"), w=(0.6, 0.3), cons_b=0.8, prompt="default",
             feats=("x1", "x2"), rep=0):
    w = np.asarray(w, float)
    return ResultadoABC(
        provider="openai", model_name="gpt-4o-mini", temperature=0.0, random_seed=seed, k_passo4_bc=k_exemplos,
        nomes_classes=tuple(nomes), repeticao=rep,
        fidelidade_problema_a=0.85, acuracia_llm_vs_gt_problema_a=0.9,
        consistencia_problema_b=cons_b, kappa_problema_b=2 * cons_b - 1, f1_problema_b=cons_b,
        acuracia_llm_vs_gt_problema_b=0.8, acuracia_metrica_vs_gt_problema_b=0.8,
        consistencia_problema_c=cons_b - 0.05, kappa_problema_c=2 * cons_b - 1.1, f1_problema_c=cons_b - 0.05,
        acuracia_llm_vs_gt_problema_c=0.8, acuracia_metrica_vs_gt_problema_c=0.8,
        w_aprendido=w, gamma_otimo=0.1,
        n_classe_0_problema_a=75, n_classe_1_problema_a=75, n_classe_0_problema_b=50, n_classe_1_problema_b=50,
        n_classe_0_problema_c=50, n_classe_1_problema_c=50,
        w_ratio=float(w[0] / w[1]), w_direction=w / np.linalg.norm(w), feature_names=tuple(feats),
        prompt_variant=prompt,
    )


def _textos_da_figura(fig):
    """Todos os textos visíveis de uma figura: suptitle, títulos, eixos, ticks e legendas."""
    textos = [t.get_text() for t in fig.texts]
    for ax in fig.get_axes():
        textos += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        textos += [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
        leg = ax.get_legend()
        if leg is not None:
            textos += [t.get_text() for t in leg.get_texts()]
        textos += [t.get_text() for t in ax.texts]
    return [t for t in textos if t]


class _CapturaTextos:
    """Troca ``plots.salvar_figura`` por um capturador dos textos de cada figura salva."""

    def __init__(self):
        self.figuras = []       # lista de listas de textos
        self.suptitles = []     # supertítulo (fig.suptitle) de cada figura que tem um
        self._orig = plots.salvar_figura

    def __enter__(self):
        def _captura(fig, filename, dpi=150, ja_cinza=False, cor=None):
            self.figuras.append(_textos_da_figura(fig))
            sup = fig.get_suptitle()
            if sup:
                self.suptitles.append(sup)
            self._orig(fig, filename, dpi=dpi, ja_cinza=ja_cinza, cor=cor)
        plots.salvar_figura = _captura
        return self

    def __exit__(self, *exc):
        plots.salvar_figura = self._orig

    @property
    def textos(self):
        return [t for f in self.figuras for t in f]


def _dados_seed(seed=42):
    rng = np.random.RandomState(seed)
    X_a = np.vstack([rng.normal([-2, 0], 1.0, (75, 2)), rng.normal([2, 0], 1.0, (75, 2))])
    y_a = np.array([0] * 75 + [1] * 75)
    X_b = np.vstack([rng.normal([-2, 1.5], 1.0, (50, 2)), rng.normal([2, -1.5], 1.0, (50, 2))])
    y_b = np.array([0] * 50 + [1] * 50)
    X_c = np.vstack([rng.normal([-2, -1.5], 1.0, (50, 2)), rng.normal([2, 1.5], 1.0, (50, 2))])
    y_c = np.array([0] * 50 + [1] * 50)
    X_e = np.vstack([rng.normal([-1.5, 1], 1.0, (75, 2)), rng.normal([1.5, -1], 1.0, (75, 2))])
    y_e = np.array([0] * 75 + [1] * 75)
    w = np.array([0.6, 0.3])
    cents = np.array([X_a[y_a == 0].mean(0), X_a[y_a == 1].mean(0)])
    lm = LearnedMetric(w=w, centroids=cents, gamma=0.1, source_problem="A")

    def _llm(X, y):
        y2 = y.copy(); flip = rng.rand(len(y)) < 0.1; y2[flip] = 1 - y2[flip]; return y2

    y_llm_a, y_llm_b, y_llm_c = _llm(X_a, y_a), _llm(X_b, y_b), _llm(X_c, y_c)
    from metrics import compute_centroids, predict_with_metric
    return {
        'X_a': X_a, 'y_gt_a': y_a, 'X_b': X_b, 'y_gt_b': y_b, 'X_c': X_c, 'y_gt_c': y_c,
        'X_e': X_e, 'y_gt_e': y_e, 'y_expert_e': (X_e[:, 1] < 0).astype(int),
        'y_llm_a': y_llm_a, 'y_llm_b': y_llm_b, 'y_llm_c': y_llm_c,
        'y_metric_a': predict_with_metric(X_a, cents, w),
        'y_metric_b': predict_with_metric(X_b, compute_centroids(X_b, y_llm_b), w),
        'y_metric_c': predict_with_metric(X_c, compute_centroids(X_c, y_llm_c), w),
        'learned_metric': lm, 'w_nnls': np.array([0.5, 0.4]), 'centroids_nnls': cents,
    }


def _linhas_oraculo(seeds=(42, 123, 7), chave='accuracy', expert_b='B', extra=None):
    rows = []
    configs = [("Oráculo sobre B", expert_b, (1.0, 1.0)),
               ("Oráculo sobre E1", "E1", (0.3, 1.5)),
               ("Oráculo sobre E2", "E2", (1.5, 0.3))]
    if extra:   # linha de configuração extra (execuções anteriores a 13/10/2026), deve ser ignorada
        configs.append(("Oráculo sobre aniso " + extra, extra, (4.0, 0.25)))
    for seed in seeds:
        for prob, exp, tw in configs:
            for algo, dv in (("perceptron", 0.02), ("nnls", -0.02)):
                wr = np.array(tw) * (1 + dv * (seed % 5))
                rows.append({'random_seed': seed, 'problem': prob, 'expert_name': exp,
                             'true_w_0': tw[0], 'true_w_1': tw[1], 'true_w_ratio': tw[0] / tw[1],
                             'algorithm': algo, 'recovered_w_0': wr[0], 'recovered_w_1': wr[1],
                             'recovered_w_norm_0': wr[0] / wr.sum(), 'recovered_w_norm_1': wr[1] / wr.sum(),
                             'recovered_w_ratio': wr[0] / wr[1] if algo == 'perceptron' or seed != 7 else float('inf'),
                             'cosine_similarity': 0.99 - 0.01 * (seed % 3),
                             chave: 0.95 + 0.01 * (seed % 3) + dv})
    return rows


def _res_e(k_exemplos, strategy, acc, seed=42):
    return ResultadoAprendiz(
        provider="openai", model_name="gpt-4o-mini", temperature=0.0, random_seed=seed,
        k=k_exemplos, example_strategy=strategy, nomes_classes=("A", "B"), repeticao=0,
        accuracy_llm_vs_expert=acc, kappa_llm_vs_expert=2 * acc - 1, f1_llm_vs_expert=acc,
        accuracy_expert_vs_gt=0.9, accuracy_llm_vs_gt=acc,
        n_classe_0_expert=50, n_classe_1_expert=50, n_classe_0_llm=50, n_classe_1_llm=50,
        n_disagreements=int(100 * (1 - acc)), n_total_test=100,
    )


class TestPlotsNomes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _arquivos(self):
        return sorted(os.listdir(self.tmp))

    def test_r3r4_gera_paineis_com_alias_por_ultimo(self):
        r2 = [{'fidelity': .80, 'fidelity_nnls': .81, 'seed': 42}, {'fidelity': .82, 'fidelity_nnls': .80, 'seed': 7}]
        r3 = [{'fidelity': .85, 'fidelity_nnls': .86, 'seed': 42}, {'fidelity': .84, 'fidelity_nnls': .85, 'seed': 7}]
        r4 = [{'fidelity': .84, 'fidelity_nnls': .88, 'seed': 42}, {'fidelity': .83, 'fidelity_nnls': .86, 'seed': 7}]
        fname = llm_asset(self.tmp, "bloco1_graphic_08_A_passo4_R2_R3_R4.png", "gpt-4o-mini")
        plots.plot_a_passo4_r2_r3_r4(r2, r3, r4, filename=fname)
        self.assertEqual(self._arquivos(), [
            "bloco1_graphic_08_A_passo4_R2_R3_R4__gpt4mini_color.png",
            "bloco1_graphic_08_A_passo4_R2_R3_R4__gpt4mini_gray.png",
            "bloco1_graphic_08_A_passo4_R2_R3_R4_algoritmos_r3r4__gpt4mini_color.png",
            "bloco1_graphic_08_A_passo4_R2_R3_R4_algoritmos_r3r4__gpt4mini_gray.png",
            "bloco1_graphic_08_A_passo4_R2_R3_R4_linear_vs_quadratica__gpt4mini_color.png",
            "bloco1_graphic_08_A_passo4_R2_R3_R4_linear_vs_quadratica__gpt4mini_gray.png",
        ])

    def test_r3r4_sem_r4_e_sem_nnls_em_r2_chaves_antigas(self):
        # execuções até 08/10/2026 gravavam a fidelidade nas chaves 'accuracy'/'accuracy_nnls'
        r2 = [{'accuracy': .80, 'seed': 42}]
        r3 = [{'accuracy': .85, 'accuracy_nnls': .86, 'seed': 42}]
        fname = llm_asset(self.tmp, "bloco1_graphic_08_A_passo4_R2_R3_R4.png", "google/gemini-2.5-flash-lite")
        plots.plot_a_passo4_r2_r3_r4(r2, r3, None, filename=fname)
        self.assertIn("bloco1_graphic_08_A_passo4_R2_R3_R4_algoritmos_r3r4__flashlite_gray.png", self._arquivos())

    def test_dilution_reference_le_n_hard_da_estrategia(self):
        df = pd.DataFrame({
            'k': [4, 4, 6, 14], 'accuracy': [.7, .72, .75, .8], 'kappa': [.4, .44, .5, .6],
            'strategy': ['dilution_4hard_0easy', 'dilution_4hard_0easy',
                         'dilution_4hard_2easy', 'dilution_4hard_10easy'],
        })
        n_hard, ref = plots._dilution_reference(df)
        self.assertEqual(n_hard, 4)
        self.assertEqual(len(ref), 2)
        self.assertAlmostEqual(ref['accuracy'].mean(), .71)

    def test_dilution_reference_fallback_menor_k(self):
        df = pd.DataFrame({'k': [3, 5], 'accuracy': [.6, .7], 'kappa': [.2, .4],
                           'strategy': ['x', 'y']})
        n_hard, ref = plots._dilution_reference(df)
        self.assertEqual((n_hard, len(ref)), (3, 1))

    def test_dilution_plot_gera_paineis(self):
        res = [_res_e(4, 'dilution_4hard_0easy', .7), _res_e(6, 'dilution_4hard_2easy', .75),
               _res_e(14, 'dilution_4hard_10easy', .8)]
        fname = llm_asset(self.tmp, "bloco2_graphic_06_E1_passo4_diluicao.png", "gpt-4o-mini")
        plots.plot_e_passo4_diluicao(res, filename=fname)
        self.assertEqual(self._arquivos(), [
            "bloco2_graphic_06_E1_passo4_diluicao__gpt4mini_color.png",
            "bloco2_graphic_06_E1_passo4_diluicao__gpt4mini_gray.png",
            "bloco2_graphic_06_E1_passo4_diluicao_accuracy__gpt4mini_color.png",
            "bloco2_graphic_06_E1_passo4_diluicao_accuracy__gpt4mini_gray.png",
            "bloco2_graphic_06_E1_passo4_diluicao_kappa__gpt4mini_color.png",
            "bloco2_graphic_06_E1_passo4_diluicao_kappa__gpt4mini_gray.png",
        ])

    def test_grouped_bar_offsets_centrados(self):
        self.assertTrue(np.allclose(plots._grouped_bar_offsets(2, 0.25), [-0.125, 0.125]))
        self.assertTrue(np.allclose(plots._grouped_bar_offsets(3, 0.2), [-0.2, 0.0, 0.2]))
        self.assertTrue(np.allclose(plots._grouped_bar_offsets(1, 0.5), [0.0]))

    def test_labels_symbols_gera_combinada_e_paineis(self):
        rng = np.random.RandomState(0)
        X = np.vstack([rng.normal([-2, 0], 0.8, (30, 2)), rng.normal([2, 0], 0.8, (30, 2))])
        y = np.array([0] * 30 + [1] * 30)
        c = np.array([X[y == 0].mean(0), X[y == 1].mean(0)])
        schemes = [
            {'suffix': 'passo1_rotulo_real', 'title': 'GT', 'y': y},
            {'suffix': 'passo2_llm_zero_shot', 'title': 'LLM', 'y': (X[:, 0] > 0.3).astype(int), 'example_idx': [1, 2, 31]},
            {'suffix': 'passo4_metrica', 'title': 'métrica', 'y': y,
             'boundary': {'w': np.array([1.0, 0.5]), 'centroids': c, 'n_features': 2}},
        ]
        fname = llm_asset(self.tmp, "bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42.png", "gpt-4o-mini")
        plots.plot_rotulos_simbolos(X, schemes, fname, suptitle="t",
                                  reference_centroids={'from': [(-2, 0), (2, 0)], 'to': [(-2, 1.5), (2, -1.5)], 'label': 'rot'})
        self.assertEqual(self._arquivos(), [
            "bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42__gpt4mini_color.png",
            "bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42__gpt4mini_gray.png",
            "bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42_passo1_rotulo_real__gpt4mini_color.png",
            "bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42_passo1_rotulo_real__gpt4mini_gray.png",
            "bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42_passo2_llm_zero_shot__gpt4mini_color.png",
            "bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42_passo2_llm_zero_shot__gpt4mini_gray.png",
            "bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42_passo4_metrica__gpt4mini_color.png",
            "bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42_passo4_metrica__gpt4mini_gray.png",
        ])

    def test_passos_2_a_5_curva_paineis_por_problema(self):
        rows = []
        for prob, nfs in (("A", (2,)), ("D", (2, 3, 4))):
            for nf in nfs:
                for regime, k in (("passo2_zero_shot", 0), ("passo3_exemplos_rotulo_real", 4), ("passo3_exemplos_rotulo_real", 10), ("passo5_teto", 0)):
                    for seed in (42, 7):
                        for algo in ("perceptron", "nnls"):
                            gt = regime == 'passo5_teto'
                            rows.append({'problem': prob, 'n_features': nf, 'regime': regime, 'k': k, 'rep': 0,
                                         'algorithm': algo, 'random_seed': seed, 'collapsed': 0,
                                         'fidelity_vs_llm': float('nan') if gt else 0.8 + 0.01 * k,
                                         'accuracy_metric_vs_true_full': 0.7,
                                         'accuracy_teto_vs_true': 0.97 if gt else float('nan'),
                                         'llm_accuracy_vs_true': float('nan') if gt else 0.6})
        fname = llm_asset(self.tmp, "bloco1_graphic_04_AD_passos_2_a_5_curva.png", "gpt-4o-mini")
        with _CapturaTextos() as cap:
            plots.plot_passos_2_a_5_curva(rows, filename=fname)
        teto = [t for t in cap.textos if t.startswith('teto')]
        self.assertTrue(teto and all('97.0%' in t for t in teto), teto)   # teto lido de accuracy_teto_vs_true
        self.assertTrue(any('mín–máx' in t for t in cap.textos))
        self.assertTrue(any('n − k avaliados' in t for t in cap.textos))
        self.assertEqual(self._arquivos(), [
            "bloco1_graphic_04_AD_passos_2_a_5_curva__gpt4mini_color.png",
            "bloco1_graphic_04_AD_passos_2_a_5_curva__gpt4mini_gray.png",
            "bloco1_graphic_04_AD_passos_2_a_5_curva_problema_a__gpt4mini_color.png",
            "bloco1_graphic_04_AD_passos_2_a_5_curva_problema_a__gpt4mini_gray.png",
            "bloco1_graphic_04_AD_passos_2_a_5_curva_problema_d_r2__gpt4mini_color.png",
            "bloco1_graphic_04_AD_passos_2_a_5_curva_problema_d_r2__gpt4mini_gray.png",
            "bloco1_graphic_04_AD_passos_2_a_5_curva_problema_d_r3__gpt4mini_color.png",
            "bloco1_graphic_04_AD_passos_2_a_5_curva_problema_d_r3__gpt4mini_gray.png",
            "bloco1_graphic_04_AD_passos_2_a_5_curva_problema_d_r4__gpt4mini_color.png",
            "bloco1_graphic_04_AD_passos_2_a_5_curva_problema_d_r4__gpt4mini_gray.png",
        ])

    def test_passos_2_a_5_curva_gray_mantem_tres_formas_distintas(self):
        """Auditoria de 09/10/2026 (rodada 2): as 3 séries pretas (``o-``, ``s--``, ``^:``) do
        gráfico 04 continuam distintas na versão ``_gray`` e a legenda tem 3 handles diferentes."""
        rows = []
        for regime, k in (("passo2_zero_shot", 0), ("passo3_exemplos_rotulo_real", 4), ("passo3_exemplos_rotulo_real", 10), ("passo5_teto", 0)):
            for seed in (42, 7):
                gt = regime == 'passo5_teto'
                rows.append({'problem': 'A', 'n_features': 2, 'regime': regime, 'k': k, 'rep': 0,
                             'algorithm': 'perceptron', 'random_seed': seed, 'collapsed': 0,
                             'fidelity_vs_llm': float('nan') if gt else 0.8 + 0.01 * k,
                             'accuracy_metric_vs_true_full': 0.7 + 0.01 * k,
                             'accuracy_teto_vs_true': 0.97 if gt else float('nan'),
                             'llm_accuracy_vs_true': float('nan') if gt else 0.3 + 0.05 * k})
        fname = llm_asset(self.tmp, "bloco1_graphic_04_AD_passos_2_a_5_curva.png", "gpt-4o-mini")
        formas_gray = []            # (marcador, traço) das linhas de dados dos errorbars, por figura, após a conversão
        orig = plots.salvar_figura

        def _captura(fig, filename, dpi=150, ja_cinza=False, cor=None):
            orig(fig, filename, dpi=dpi, ja_cinza=ja_cinza, cor=cor)   # converte a figura no lugar
            from matplotlib.container import ErrorbarContainer
            for ax in fig.get_axes():
                conts = [c for c in ax.containers if isinstance(c, ErrorbarContainer) and c.lines[0] is not None]
                if conts:
                    leg = ax.get_legend()
                    formas_gray.append(([(c.lines[0].get_marker(), c.lines[0].get_linestyle()) for c in conts],
                                        [t.get_text() for t in leg.get_texts()] if leg else []))
        plots.salvar_figura = _captura
        try:
            plots.plot_passos_2_a_5_curva(rows, filename=fname)
        finally:
            plots.salvar_figura = orig
        self.assertTrue(formas_gray)
        for formas, textos in formas_gray:
            self.assertEqual(len(formas), 3)
            self.assertEqual(len(set(formas)), 3, formas)                       # 3 formas distintas
            self.assertEqual(len({m for m, _ in formas}), 3, formas)            # marcadores distintos
            self.assertEqual(len([t for t in textos if not t.startswith('teto')]), 3)
        self.assertIn("bloco1_graphic_04_AD_passos_2_a_5_curva__gpt4mini_gray.png", self._arquivos())

    def test_main_expert_filtra_para_o_primeiro_especialista(self):
        a = _res_e(4, 'easy', .8); a.expert_name = 'E1'
        b = _res_e(4, 'easy', .6); b.expert_name = 'E3'
        filtrados, nome = plots._main_expert([a, b, a])
        self.assertEqual((nome, len(filtrados)), ('E1', 2))


class TestLegendasGarantidas(unittest.TestCase):
    """Toda figura com séries ganha legenda nos dois arquivos (color e gray)."""

    def test_scatter_por_classe_e_linha_sem_rotulo_ganham_legenda(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from estilo_cinza import garantir_legendas
        rng = np.random.RandomState(1)
        X = rng.normal(size=(40, 2)); y = (X[:, 0] > 0).astype(int)
        fig, (a1, a2, a3) = plt.subplots(1, 3)
        a1.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")          # c=y sem rótulo
        a2.plot(range(10), range(10)); a2.set_ylabel("acurácia")    # linha única sem rótulo
        a3.bar([0, 1], [1, 2]); a3.bar([0, 1], [2, 1])              # duas séries de barras
        pend = garantir_legendas(fig)
        self.assertEqual(pend, [])
        self.assertEqual([t.get_text() for t in a1.get_legend().get_texts()], ["classe 0", "classe 1"])
        self.assertEqual([t.get_text() for t in a2.get_legend().get_texts()], ["acurácia"])
        self.assertEqual([t.get_text() for t in a3.get_legend().get_texts()], ["série 1", "série 2"])
        tmp = tempfile.mkdtemp()
        try:
            plots.salvar_figura(fig, os.path.join(tmp, "x.png"))
            self.assertEqual(sorted(os.listdir(tmp)), ["x_color.png", "x_gray.png"])
            self.assertIsNotNone(a1.get_legend())    # legenda refeita no modo cinza
            self.assertEqual([t.get_text() for t in a1.get_legend().get_texts()], ["classe 0", "classe 1"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        plt.close(fig)


class TestMetricasIV3(unittest.TestCase):
    """Nomes das métricas (seção IV.3 do orientador) nos textos das figuras."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _assert_sem_proibidos(self, textos):
        ruins = [t for t in textos for p in TEXTOS_PROIBIDOS if p in t]
        self.assertEqual(ruins, [], f"nomes de métrica fora do PDF: {ruins}")

    def _abc(self):
        res = []
        for seed in (42, 123, 7):
            for k in (0, 10):
                for par in plots.PARES_NOMES_BASE:
                    res.append(_res_abc(seed, k, par, w=(0.6 + 0.01 * seed % 3, 0.3)))
                    res.append(_res_abc(seed, k, par[::-1], cons_b=0.7))
            for k in (4, 20, 40):
                res.append(_res_abc(seed, k, ("A", "B")))
            for v in ("geometric", "cot", "tabular"):
                res.append(_res_abc(seed, 0, ("A", "B"), prompt=v, w=(0.001, 0.0005)))
            res.append(_res_abc(seed, 0, ("A", "B"), feats=("altura", "peso"), w=(0.2, 0.9)))
        return res

    def test_bloco1_rotulos(self):
        abc = self._abc()
        with _CapturaTextos() as cap:
            plots.plot_bc_passo3_consistencia(abc, llm_asset(self.tmp, "bloco1_graphic_10_BC_passo3_consistencia.png", "gpt-4o-mini"))
            plots.plot_vies_nomes_classe(abc, llm_asset(self.tmp, "bloco1_graphic_16_vies_nomes_classe.png", "gpt-4o-mini"))
            plots.plot_vies_ordem_classes(abc, llm_asset(self.tmp, "bloco1_graphic_15_vies_ordem_classes.png", "gpt-4o-mini"))
            plots.plot_bc_passo3_sementes(abc, llm_asset(self.tmp, "bloco1_graphic_07_BC_passo3_sementes.png", "gpt-4o-mini"))
            plots.plot_a_passo4_w_distribuicao(abc, llm_asset(self.tmp, "bloco1_graphic_05_A_passo4_w_distribuicao.png", "gpt-4o-mini"))
            plots.plot_vies_nomes_atributo(abc, llm_asset(self.tmp, "bloco1_graphic_17_vies_nomes_atributo.png", "gpt-4o-mini"))
            plots.plot_vies_variantes_prompt(abc, llm_asset(self.tmp, "bloco1_graphic_18_vies_variantes_prompt.png", "gpt-4o-mini"))
            plots.plot_a_passo4_perceptron_vs_nnls(abc[:6], abc[6:12], llm_asset(self.tmp, "bloco1_graphic_06_A_passo4_perceptron_vs_nnls.png", "gpt-4o-mini"))
            d = _dados_seed()
            plots.plot_a_passo4_erros(d['X_a'], d['y_llm_a'], d['y_metric_a'], d['learned_metric'].w,
                                             d['learned_metric'].centroids,
                                             llm_asset(self.tmp, "bloco1_graphic_03_A_passo4_erros_seed42.png", "gpt-4o-mini"))
        self._assert_sem_proibidos(cap.textos)
        self.assertTrue(any("Cohen's κ" in t for t in cap.textos))
        self.assertTrue(any("F1-Macro" in t for t in cap.textos))
        # bloco1_graphic_03: eixo de margem não pode ser só "0.0" (F10) e usa 1 − fidelidade
        self.assertTrue(any('1 − fidelidade' in t for t in cap.textos))

    def test_bloco2_rotulos(self):
        res_e = [_res_e(k, st, 0.6 + 0.01 * k, seed) for seed in (42, 7) for k in (0, 4, 10, 20, 40)
                 for st in ("easy", "hard", "mixed", "random")]
        for r in res_e:
            r.expert_name = "E1"
        ordem = [_res_e(k, f"mixed_order_{o}", 0.8, seed) for seed in (42, 7) for k in (4, 10, 20)
                 for o in ("class0_first", "class1_first", "shuffled", "alternating")]
        dil = [_res_e(4 + m, f"dilution_4hard_{m}easy", 0.7 + 0.005 * m) for m in (0, 2, 4, 10)]
        bl = [{'provider': 'classical', 'model': m, 'random_seed': seed, 'k': k, 'example_strategy': 'mixed',
               'expert_name': 'E1', 'accuracy_vs_expert': 0.7, 'kappa_vs_expert': 0.4, 'f1_vs_expert': 0.7,
               'accuracy_vs_gt': 0.7}
              for seed in (42, 7) for k in (4, 10, 20, 40)
              for m in (f"k-NN (k={3 if k == 4 else 5})", "Logistic Regression", "SVM (RBF)")]
        with _CapturaTextos() as cap:
            plots.plot_e_passo3_curva(res_e, llm_asset(self.tmp, "bloco2_graphic_03_E1_passo3_curva.png", "gpt-4o-mini"))
            plots.plot_e_passo3_estrategias(res_e, llm_asset(self.tmp, "bloco2_graphic_04_E1_passo3_estrategias.png", "gpt-4o-mini"))
            plots.plot_e_passo6_baselines(res_e, bl, llm_asset(self.tmp, "bloco2_graphic_05_E1_passo6_baselines.png", "gpt-4o-mini"))
            plots.plot_e_passo4_diluicao(dil, llm_asset(self.tmp, "bloco2_graphic_06_E1_passo4_diluicao.png", "gpt-4o-mini"))
            plots.plot_e_passo5_ordem(ordem, llm_asset(self.tmp, "bloco2_graphic_07_E1_passo5_ordem.png", "gpt-4o-mini"))
        self._assert_sem_proibidos(cap.textos)
        self.assertTrue(any('Acurácia (LLM vs especialista)' in t for t in cap.textos))

    def test_fechamento_rotulos(self):
        d = _dados_seed()
        abc = self._abc()
        res_e = [_res_e(k, st, 0.6 + 0.01 * k) for k in (0, 4, 10) for st in ("easy", "hard")]
        with _CapturaTextos() as cap:
            plots.plot_final_acertos_erros(d, 42, llm_asset(self.tmp, "final_graphic_03_ABC_acertos_erros_seed42.png", "gpt-4o-mini"))
            plots.plot_final_matrizes_confusao(d, 42, llm_asset(self.tmp, "final_graphic_02_ABC_matrizes_confusao_seed42.png", "gpt-4o-mini"))
            plots.plot_final_margem(d, 42, llm_asset(self.tmp, "final_graphic_05_ABC_margem_seed42.png", "gpt-4o-mini"))
            plots.plot_final_painel_resumo(d, 42, abc, res_e, llm_asset(self.tmp, "final_graphic_06_painel_resumo_seed42.png", "gpt-4o-mini"))
            plots.plot_final_dados(d, 42, llm_asset(self.tmp, "final_graphic_01_ABCE_passo1_dados_seed42.png", "gpt-4o-mini"))
            plots.plot_final_comparacao_modelos(abc + [_res_abc(s_, 0, ("A", "B")).__class__(**{**_res_abc(s_, 0, ("A", "B")).__dict__, 'model_name': 'google/gemini-2.5-flash-lite'}) for s_ in (42, 7)],
                                        os.path.join(self.tmp, "final_graphic_08_comparacao_modelos.png"))
        self._assert_sem_proibidos(cap.textos)
        self.assertTrue(any(t.startswith('Problema A: Fidelidade') for t in cap.textos))
        self.assertTrue(any(t.startswith('Problema B: Consistência') for t in cap.textos))
        self.assertTrue(any('Acurácia (vs rótulo real)' in t for t in cap.textos))
        # final_graphic_08: barras B/C com legenda (F6)
        self.assertIn('Problema B', cap.textos); self.assertIn('Problema C', cap.textos)
        # final_graphic_01: painel E com o rótulo do especialista (F8)
        self.assertTrue(any('Problema E, rótulo do especialista' in t for t in cap.textos))

    def test_externos_rotulos(self):
        rng = np.random.RandomState(3)
        X = rng.uniform(-1, 1, (30, 2)); y = (X[:, 0] > 0).astype(int)
        y_llm = y.copy(); y_llm[:3] = 1 - y_llm[:3]
        rows_e = [{'problem_name': p, 'feature_names': ('peso', 'altura'), 'seed': 42, 'k': k, 'rep': 0,
                   'accuracy_llm_vs_expert': 0.8, 'kappa_llm_vs_expert': 0.6, 'f1_llm_vs_expert': 0.8,
                   'accuracy_llm_vs_true': 0.75, 'accuracy_perceptron_baseline_vs_true': 0.7}
                  for p in ("G1_homem_mulher", "G2_homem_mulher") for k in (0, 4, 10)]
        from metrics import augment_features, compute_centroids
        rows_a = [{'problem_name': 'G2_homem_mulher', 'feature_names': ('peso_kg', 'altura_m'), 'seed': 42,
                   'n_features': nf, 'fidelity_perc_vs_llm': 0.9, 'accuracy_perc_vs_true': 0.85,
                   'fidelity_nnls_vs_llm': 0.88, 'accuracy_nnls_vs_true': 0.84,
                   'X_aug': augment_features(X, nf), 'y_llm': y_llm,
                   'centroids': compute_centroids(augment_features(X, nf), y_llm), 'w_perc': np.ones(nf)}
                  for nf in (2, 3, 4)]
        with _CapturaTextos() as cap:
            plots.plot_fg_aprendiz_curva(rows_e, llm_asset(self.tmp, "bloco23_graphic_01_FG_aprendiz_curva.png", "gpt-4o-mini"))
            plots.plot_fg_llm_vs_perceptron(rows_e, llm_asset(self.tmp, "bloco23_graphic_04_FG_llm_vs_perceptron.png", "gpt-4o-mini"))
            plots.plot_fg_passos_2_e_4_r2_r3_r4(rows_a, llm_asset(self.tmp, "bloco23_graphic_02_FG_passos_2_e_4_R2_R3_R4.png", "gpt-4o-mini"))
            plots.plot_fg_fronteiras(rows_a, llm_asset(self.tmp, "bloco23_graphic_03_FG_fronteiras.png", "gpt-4o-mini"))
            plots.plot_fg_rotulos_llm(X, y_llm, y_true=y, y_metric=y, class_names=("Mulher", "Homem"),
                                              title="t", feature_names=("peso", "altura"),
                                              filename=llm_asset(self.tmp, "final_graphic_07_FG_rotulos_llm_G1_seed42_peso_altura_4.png", "gpt-4o-mini"))
            plots.plot_problema_passo1(X, y, "G1", feature_names=("peso", "altura"), class_names=("Mulher", "Homem"),
                                        filename=os.path.join(self.tmp, "bloco3_graphic_01_G1_passo1_dados.png"))
        self._assert_sem_proibidos(cap.textos)
        self.assertIn('Acurácia (LLM vs especialista)', cap.textos)
        self.assertIn('Acurácia (LLM vs rótulo real)', cap.textos)
        self.assertTrue(any('x1 (normalizado)' == t for t in cap.textos))          # eixos do G2 (F14)
        self.assertTrue(any('Homem' in t and 'especialista' in t for t in cap.textos))   # y_metric + class_names
        # fallback da coluna antiga (execuções até 08/10/2026)
        antigos = [{**r, 'accuracy_llm_vs_metric': r.pop('accuracy_llm_vs_expert')} for r in [dict(r) for r in rows_e]]
        with _CapturaTextos() as cap2:
            plots.plot_fg_aprendiz_curva(antigos, llm_asset(self.tmp, "bloco23_graphic_01_FG_aprendiz_curva.png", "google/gemini-2.5-flash-lite"))
        self.assertIn('Acurácia (LLM vs especialista)', cap2.textos)


class TestContratosNovos(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_grafico_20_removido(self):
        self.assertFalse(hasattr(plots, 'plot_oracle_transfer'))

    def test_oraculo_agrega_3_seeds_e_le_accuracy(self):
        rows = _linhas_oraculo(extra='x1_preciso')      # linha extra de execução antiga: ignorada
        labels, agg = plots._agregar_oraculo(rows)
        self.assertEqual(labels, ["Oráculo sobre B", "Oráculo sobre E1", "Oráculo sobre E2"])
        v = agg[("Oráculo sobre E1", "perceptron")]
        self.assertEqual((v['n_seeds'], v['n']), (3, 3))
        esperado = np.mean([0.95 + 0.01 * (s % 3) + 0.02 for s in (42, 123, 7)])
        self.assertAlmostEqual(v['accuracy'][0], esperado)
        self.assertLessEqual(v['accuracy'][1], v['accuracy'][0]); self.assertGreaterEqual(v['accuracy'][2], v['accuracy'][0])
        # NNLS com razão infinita numa seed: contada, não silenciada
        self.assertEqual(agg[("Oráculo sobre E1", "nnls")]['n_ratio_inf'], 1)
        with _CapturaTextos() as cap:
            plots.plot_oraculo_recuperacao_w(rows, os.path.join(self.tmp, "bloco1_graphic_19_oraculo_passos_1_a_3.png"))
        arqs = sorted(os.listdir(self.tmp))
        self.assertIn("bloco1_graphic_19_oraculo_passos_1_a_3_acuracia_gray.png", arqs)
        self.assertNotIn("bloco1_graphic_19_oraculo_passos_1_a_3_fidelidade_gray.png", arqs)
        self.assertTrue(any('3 seed' in t for t in cap.textos))
        self.assertFalse(any('euclidiana' in t and 'Oráculo' in t for t in cap.textos))
        self.assertTrue(any(t.startswith('Acurácia (vs rótulos do oráculo)') for t in cap.textos))

    def test_oraculo_le_fidelity_e_nomes_antigos(self):
        rows = _linhas_oraculo(seeds=(7,), chave='fidelity', expert_b='euclidiana', extra='x2_dom')
        labels, agg = plots._agregar_oraculo(rows)
        self.assertEqual(labels, ["Oráculo sobre B", "Oráculo sobre E1", "Oráculo sobre E2"])
        self.assertFalse(np.isnan(agg[("Oráculo sobre B", "nnls")]['accuracy'][0]))
        self.assertEqual(agg[("Oráculo sobre B", "nnls")]['n_seeds'], 1)

    def test_baselines_knn_em_uma_serie(self):
        res_e = [_res_e(k, 'mixed', 0.7, seed) for seed in (42, 7) for k in (4, 10, 20, 40)]
        bl = [{'provider': 'classical', 'model': m, 'random_seed': seed, 'k': k, 'example_strategy': 'mixed',
               'accuracy_vs_expert': 0.7, 'kappa_vs_expert': 0.4, 'f1_vs_expert': 0.7, 'accuracy_vs_gt': 0.7}
              for seed in (42, 7) for k in (4, 10, 20, 40)
              for m in (f"k-NN (k={3 if k == 4 else 5})", "Logistic Regression", "SVM (RBF)")]
        cap = _CapturaTextos()
        with cap:
            plots.plot_e_passo6_baselines(res_e, bl, llm_asset(self.tmp, "bloco2_graphic_05_E1_passo6_baselines.png", "gpt-4o-mini"))
        legendas_combinada = cap.figuras[0]
        self.assertEqual(sum(1 for t in legendas_combinada if t == 'k-NN'), 3)        # 1 por painel
        self.assertFalse(any(t.startswith('k-NN (k=') for t in legendas_combinada))
        self.assertTrue(any('k-NN: k = 3, 5' in t for t in legendas_combinada))      # k no rodapé

    def test_nomes_classe_sem_pares_invertidos_e_ordem_pareada(self):
        res = []
        for seed in (42, 7):
            for k in (0, 10):
                for par in plots.PARES_NOMES_BASE:
                    res.append(_res_abc(seed, k, par, cons_b=0.8))
                    res.append(_res_abc(seed, k, par[::-1], cons_b=0.6))
        cap = _CapturaTextos()
        with cap:
            plots.plot_vies_nomes_classe(res, llm_asset(self.tmp, "bloco1_graphic_16_vies_nomes_classe.png", "gpt-4o-mini"))
        ticks = {t for t in cap.figuras[0] if '/' in t and ' ' not in t}
        self.assertEqual(ticks, {"A/B", "0/1", "Positivo/Negativo", "Azul/Vermelho"})
        self.assertTrue(any(t == 'zero-shot' for t in cap.figuras[0]) and any(t == '10-shot' for t in cap.figuras[0]))
        cap2 = _CapturaTextos()
        with cap2:
            plots.plot_vies_ordem_classes(res, llm_asset(self.tmp, "bloco1_graphic_15_vies_ordem_classes.png", "gpt-4o-mini"))
        legs = [t for t in cap2.figuras[0] if t.startswith('normal') or t.startswith('invertido')]
        self.assertTrue(legs and any('B/A' in t for t in legs if t.startswith('invertido')))
        self.assertTrue(any('zero-shot' in t for t in cap2.figuras[0]) and any('10-shot' in t for t in cap2.figuras[0]))
        arqs = sorted(os.listdir(self.tmp))
        self.assertIn("bloco1_graphic_15_vies_ordem_classes_consistencia_b_0shot__gpt4mini_gray.png", arqs)
        self.assertIn("bloco1_graphic_16_vies_nomes_classe_10shot__gpt4mini_color.png", arqs)

    def test_w_unitario_usa_direcao(self):
        r = _res_abc(w=(0.001, 0.0005)); r.w_direction = None
        self.assertTrue(np.allclose(plots._w_unitario(r), np.array([2, 1]) / np.sqrt(5)))
        r2 = _res_abc(w=(3.0, 4.0))
        self.assertTrue(np.allclose(plots._w_unitario(r2), [0.6, 0.8]))

    def test_symbols_usa_centroides_da_fronteira_e_rep(self):
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        X = np.vstack([rng.normal([-2, 0], 0.8, (30, 2)), rng.normal([2, 0], 0.8, (30, 2))])
        y = np.array([0] * 30 + [1] * 30)
        c_front = np.array([[-3.0, 1.0], [3.0, -1.0]])    # diferentes dos centróides dos rótulos
        fig, ax = plt.subplots()
        plots._draw_symbols_scheme(ax, X, {'y': y, 'title': 'Melhor métrica do A (10-shot)', 'rep': 2,
                                           'boundary': {'w': np.array([1.0, 0.5]), 'centroids': c_front, 'n_features': 2}})
        legs = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn('centróide c0 (da estimação)', legs)
        self.assertIn('fronteira da métrica (R²)', legs)     # 15/10/2026: R²/R³/R⁴ em todo texto visível
        pts = [c.get_offsets() for c in ax.collections if c.get_offsets().shape[0] == 1]
        self.assertTrue(any(np.allclose(p[0], c_front[0]) for p in pts))
        self.assertIn('(rep 2)', ax.get_title())
        plt.close(fig)


# ───────────────────────── nomenclatura por passos (13/10/2026) ─────────────────────────
# Termos antigos que NÃO podem aparecer em nenhum texto visível das figuras.
TERMOS_ANTIGOS = ["Fase A", "Fase B", "Fase C", "Fase E", "Fases", "pipeline externo", "protocolo único",
                  "rotocolo da Fase", "Phase A", "Phase E"]


def _abc_fixture():
    res = []
    for seed in (42, 123, 7):
        for k in (0, 10):
            for par in plots.PARES_NOMES_BASE:
                res.append(_res_abc(seed, k, par, w=(0.6 + 0.01 * seed % 3, 0.3)))
                res.append(_res_abc(seed, k, par[::-1], cons_b=0.7))
        for k in (4, 20, 40):
            res.append(_res_abc(seed, k, ("A", "B")))
        for v in ("geometric", "cot", "tabular"):
            res.append(_res_abc(seed, 0, ("A", "B"), prompt=v, w=(0.001, 0.0005)))
        res.append(_res_abc(seed, 0, ("A", "B"), feats=("altura", "peso"), w=(0.2, 0.9)))
    return res


def _linhas_ad_passos(problem_extra=None):
    """Linhas de `bloco1_ad_passos_2_a_5` (A em R², D em R²/R³/R⁴); `problem_extra` acrescenta as
    colunas do Problema G (`subproblem`, `class_names`, `feature_names`)."""
    rows = []
    for prob, nfs in (("A", (2,)), ("D", (2, 3, 4))):
        for nf in nfs:
            for regime, k in (("passo2_zero_shot", 0), ("passo3_exemplos_rotulo_real", 4), ("passo3_exemplos_rotulo_real", 10), ("passo5_teto", 0)):
                for seed in (42, 7):
                    for algo in ("perceptron", "nnls"):
                        gt = regime == 'passo5_teto'
                        r = {'problem': prob, 'n_features': nf, 'regime': regime, 'k': k, 'rep': 0,
                             'algorithm': algo, 'random_seed': seed, 'collapsed': 0,
                             'fidelity_vs_llm': float('nan') if gt else 0.8 + 0.01 * k,
                             'accuracy_metric_vs_true_full': 0.7,
                             'accuracy_teto_vs_true': 0.97 if gt else float('nan'),
                             'llm_accuracy_vs_true': float('nan') if gt else 0.6}
                        if problem_extra:
                            r.update(problem_extra)
                        rows.append(r)
    return rows


class TestNomenclaturaPassos(unittest.TestCase):
    """Nomenclatura por problema e passo (13/10/2026): nenhum texto visível das figuras cita
    as fases antigas (Fase A/B/C/E, pipeline externo, protocolo único) e todo supertítulo diz
    de que problema e passo(s) a figura trata ("Problema A, Passo 4", "Problemas B e C, Passo 3"…).
    Cada função de `plots.py` que salva figura é chamada uma vez, com dublês de dados."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _checa(self, cap, nome):
        ruins = [t for t in cap.textos for p in TERMOS_ANTIGOS if p in t]
        self.assertEqual(ruins, [], f"{nome}: termos antigos nos textos da figura: {ruins}")
        self.assertTrue(cap.figuras, f"{nome}: nenhuma figura salva")
        self.assertTrue(any('Passo' in t for t in cap.textos), f"{nome}: nenhum 'Passo' nos textos")
        for sup in cap.suptitles:
            self.assertIn('Passo', sup, f"{nome}: supertítulo sem 'Passo': {sup!r}")

    def test_todas_as_figuras_citam_problema_e_passo(self):
        d = _dados_seed()
        abc = _abc_fixture()
        res_e = [_res_e(k, st, 0.6 + 0.01 * k, seed) for seed in (42, 7) for k in (0, 4, 10, 20, 40)
                 for st in ("easy", "hard", "mixed", "random")]
        for r in res_e:
            r.expert_name = "E1"
        ordem = [_res_e(k, f"mixed_order_{o}", 0.8, seed) for seed in (42, 7) for k in (4, 10, 20)
                 for o in ("class0_first", "class1_first", "shuffled", "alternating")]
        dil = [_res_e(4 + m, f"dilution_4hard_{m}easy", 0.7 + 0.005 * m) for m in (0, 2, 4, 10)]
        bl = [{'provider': 'classical', 'model': m, 'random_seed': seed, 'k': k, 'example_strategy': 'mixed',
               'expert_name': 'E1', 'accuracy_vs_expert': 0.7, 'kappa_vs_expert': 0.4, 'f1_vs_expert': 0.7,
               'accuracy_vs_gt': 0.7}
              for seed in (42, 7) for k in (4, 10, 20, 40)
              for m in (f"k-NN (k={3 if k == 4 else 5})", "Logistic Regression", "SVM (RBF)")]
        r2 = [{'fidelity': .80, 'fidelity_nnls': .81, 'seed': 42}, {'fidelity': .82, 'fidelity_nnls': .80, 'seed': 7}]
        r3 = [{'fidelity': .85, 'fidelity_nnls': .86, 'seed': 42}, {'fidelity': .84, 'fidelity_nnls': .85, 'seed': 7}]
        r4 = [{'fidelity': .84, 'fidelity_nnls': .88, 'seed': 42}, {'fidelity': .83, 'fidelity_nnls': .86, 'seed': 7}]
        gamma = [{'label': 'seed 42', 'gamma_final': 0.1,
                  'history': [{'iter': i, 'gamma': 0.05 * i, 'gamma_lo': 0.02 * i, 'gamma_hi': 1.0,
                               'violations': 0, 'viable': True} for i in range(4)]}]
        ml = [{'algorithm': a, 'n_features': nf, 'accuracy_vs_true': 0.8 + 0.02 * nf, 'random_seed': s}
              for a in ("perceptron", "nnls") for nf in (2, 3, 4) for s in (42, 7)]
        rng = np.random.RandomState(3)
        X = rng.uniform(-1, 1, (40, 2)); y = (X[:, 0] > 0).astype(int)
        y_llm = y.copy(); y_llm[:3] = 1 - y_llm[:3]
        from metrics import augment_features, compute_centroids
        rows_e = [{'problem_name': p, 'feature_names': ('peso', 'altura'), 'seed': 42, 'k': k, 'rep': 0,
                   'accuracy_llm_vs_expert': 0.8, 'kappa_llm_vs_expert': 0.6, 'f1_llm_vs_expert': 0.8,
                   'accuracy_llm_vs_true': 0.75, 'accuracy_perceptron_baseline_vs_true': 0.7}
                  for p in ("G1_homem_mulher", "G2_homem_mulher") for k in (0, 4, 10)]
        rows_a = [{'problem_name': 'G2_homem_mulher', 'feature_names': ('peso_kg', 'altura_m'), 'seed': 42,
                   'n_features': nf, 'fidelity_perc_vs_llm': 0.9, 'accuracy_perc_vs_true': 0.85,
                   'fidelity_nnls_vs_llm': 0.88, 'accuracy_nnls_vs_true': 0.84,
                   'X_aug': augment_features(X, nf), 'y_llm': y_llm,
                   'centroids': compute_centroids(augment_features(X, nf), y_llm), 'w_perc': np.ones(nf)}
                  for nf in (2, 3, 4)]
        linhas_ad = _linhas_ad_passos()
        rows_g = _linhas_ad_passos({'subproblem': 'G1', 'class_names': 'Mulher/Homem', 'feature_names': 'peso/altura'})
        c = np.array([d['X_a'][d['y_gt_a'] == 0].mean(0), d['X_a'][d['y_gt_a'] == 1].mean(0)])
        schemes = [{'suffix': 'passo5_teto', 'title': 'Rótulo real (Passo 1)', 'y': d['y_gt_a']},
                   {'suffix': 'passo2_llm_zero_shot', 'title': 'LLM zero-shot (Passo 2)', 'y': d['y_llm_a']},
                   {'suffix': 'passo4_metrica', 'title': 'Métrica estimada (Passo 4; zero-shot, R²)', 'y': d['y_metric_a'],
                    'boundary': {'w': d['learned_metric'].w, 'centroids': c, 'n_features': 2}}]
        gemini = [_res_abc(s_, 0, ("A", "B")).__class__(**{**_res_abc(s_, 0, ("A", "B")).__dict__,
                                                          'model_name': 'google/gemini-2.5-flash-lite'}) for s_ in (42, 7)]
        A = lambda nome: llm_asset(self.tmp, nome, "gpt-4o-mini")   # noqa: E731
        P = lambda nome: os.path.join(self.tmp, nome)                 # noqa: E731
        chamadas = {
            'bloco1_graphic_01': lambda: plots.plot_abc_passo1(d['X_a'], d['y_gt_a'], d['X_b'], d['y_gt_b'], d['X_c'], d['y_gt_c'],
                                                                      filename=P("bloco1_graphic_01_ABC_passo1_dados.png")),
            'bloco2_graphic_01': lambda: plots.plot_e_passo1(d['X_e'], d['y_gt_e'], d['y_expert_e'], np.array([0.3, 1.5]),
                                                                               np.array([[-1.5, 1.0], [1.5, -1.0]]),
                                                                               filename=P("bloco2_graphic_01_E_passo1_especialista.png")),
            'bloco2_graphic_02': lambda: plots.plot_e_passo3_exemplos(d['X_e'], d['y_expert_e'], np.array([0.3, 1.5]),
                                                                              np.array([[-1.5, 1.0], [1.5, -1.0]]), n_examples=4,
                                                                              filename=P("bloco2_graphic_02_E_passo3_estrategias_exemplos.png")),
            'bloco1_graphic_21': lambda: plots.plot_d_passo5_teto(ml, P("bloco1_graphic_21_D_passo5_teto.png")),
            'bloco1_graphic_19': lambda: plots.plot_oraculo_recuperacao_w(_linhas_oraculo(), P("bloco1_graphic_19_oraculo_passos_1_a_3.png")),
            'bloco1_graphic_10': lambda: plots.plot_bc_passo3_consistencia(abc, A("bloco1_graphic_10_BC_passo3_consistencia.png")),
            'bloco1_graphic_16': lambda: plots.plot_vies_nomes_classe(abc, A("bloco1_graphic_16_vies_nomes_classe.png")),
            'bloco1_graphic_15': lambda: plots.plot_vies_ordem_classes(abc, A("bloco1_graphic_15_vies_ordem_classes.png")),
            'bloco1_graphic_07': lambda: plots.plot_bc_passo3_sementes(abc, A("bloco1_graphic_07_BC_passo3_sementes.png")),
            'bloco1_graphic_05': lambda: plots.plot_a_passo4_w_distribuicao(abc, A("bloco1_graphic_05_A_passo4_w_distribuicao.png")),
            'bloco1_graphic_17': lambda: plots.plot_vies_nomes_atributo(abc, A("bloco1_graphic_17_vies_nomes_atributo.png")),
            'bloco1_graphic_18': lambda: plots.plot_vies_variantes_prompt(abc, A("bloco1_graphic_18_vies_variantes_prompt.png")),
            'bloco1_graphic_06': lambda: plots.plot_a_passo4_perceptron_vs_nnls(abc[:6], abc[6:12], A("bloco1_graphic_06_A_passo4_perceptron_vs_nnls.png")),
            'bloco1_graphic_03': lambda: plots.plot_a_passo4_erros(d['X_a'], d['y_llm_a'], d['y_metric_a'], d['learned_metric'].w,
                                                                          d['learned_metric'].centroids,
                                                                          A("bloco1_graphic_03_A_passo4_erros_seed42.png")),
            'bloco1_graphic_08': lambda: plots.plot_a_passo4_r2_r3_r4(r2, r3, r4, filename=A("bloco1_graphic_08_A_passo4_R2_R3_R4.png")),
            'bloco1_graphic_09': lambda: plots.plot_a_passo4_gamma(gamma, filename=A("bloco1_graphic_09_A_passo4_gamma.png")),
            'bloco1_graphic_14': lambda: plots.plot_d_passo6_svm(X, y, y_llm, seed=42, filename=A("bloco1_graphic_14_D_passo6_svm_seed42.png")),
            'bloco1_graphic_12': lambda: plots.plot_problema_passo1(X, y, "Problema D, Passo 1: meia-lua (seed=42)",
                                                                     filename=P("bloco1_graphic_12_D_passo1_dados_seed42.png")),
            'bloco1_graphic_02': lambda: plots.plot_rotulos_simbolos(d['X_a'], schemes, A("bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42.png"),
                                                                   suptitle="Problema A, Passos 2 a 4, seed 42, gpt-4o-mini"),
            'bloco1_graphic_04': lambda: plots.plot_passos_2_a_5_curva(linhas_ad, filename=A("bloco1_graphic_04_AD_passos_2_a_5_curva.png")),
            'bloco2_graphic_03': lambda: plots.plot_e_passo3_curva(res_e, A("bloco2_graphic_03_E1_passo3_curva.png")),
            'bloco2_graphic_04': lambda: plots.plot_e_passo3_estrategias(res_e, A("bloco2_graphic_04_E1_passo3_estrategias.png")),
            'bloco2_graphic_05': lambda: plots.plot_e_passo6_baselines(res_e, bl, A("bloco2_graphic_05_E1_passo6_baselines.png")),
            'bloco2_graphic_06': lambda: plots.plot_e_passo4_diluicao(dil, A("bloco2_graphic_06_E1_passo4_diluicao.png")),
            'bloco2_graphic_07': lambda: plots.plot_e_passo5_ordem(ordem, A("bloco2_graphic_07_E1_passo5_ordem.png")),
            'bloco3_graphic_01': lambda: plots.plot_problema_passo1(X, y, "Problema G1, Passo 1: peso × altura (valores normalizados), base real",
                                                                     feature_names=("peso", "altura"), class_names=("Mulher", "Homem"),
                                                                     filename=P("bloco3_graphic_01_G1_passo1_dados.png")),
            'bloco3_graphic_03': lambda: plots.plot_g_passos_2_a_5_curva(rows_g, filename=A("bloco3_graphic_03_G_passos_2_a_5_curva.png")),
            'bloco23_graphic_01': lambda: plots.plot_fg_aprendiz_curva(rows_e, A("bloco23_graphic_01_FG_aprendiz_curva.png")),
            'bloco23_graphic_02': lambda: plots.plot_fg_passos_2_e_4_r2_r3_r4(rows_a, A("bloco23_graphic_02_FG_passos_2_e_4_R2_R3_R4.png")),
            'bloco23_graphic_03': lambda: plots.plot_fg_fronteiras(rows_a, A("bloco23_graphic_03_FG_fronteiras.png")),
            'bloco23_graphic_04': lambda: plots.plot_fg_llm_vs_perceptron(rows_e, A("bloco23_graphic_04_FG_llm_vs_perceptron.png")),
            'final_graphic_01': lambda: plots.plot_final_dados(d, 42, A("final_graphic_01_ABCE_passo1_dados_seed42.png")),
            'final_graphic_02': lambda: plots.plot_final_matrizes_confusao(d, 42, A("final_graphic_02_ABC_matrizes_confusao_seed42.png")),
            'final_graphic_03': lambda: plots.plot_final_acertos_erros(d, 42, A("final_graphic_03_ABC_acertos_erros_seed42.png")),
            'final_graphic_04': lambda: plots.plot_final_w_algoritmos(d, 42, A("final_graphic_04_A_w_algoritmos_seed42.png")),
            'final_graphic_05': lambda: plots.plot_final_margem(d, 42, A("final_graphic_05_ABC_margem_seed42.png")),
            'final_graphic_06': lambda: plots.plot_final_painel_resumo(d, 42, abc, res_e, A("final_graphic_06_painel_resumo_seed42.png")),
            'final_graphic_07': lambda: plots.plot_fg_rotulos_llm(
                X, y_llm, y_true=y, y_metric=y, class_names=("Mulher", "Homem"), feature_names=("peso", "altura"),
                title="Rotulação LLM (gpt4mini): Problema G1_homem_mulher, Passo 6 | seed=42 | peso/altura | k_exemplos=4",
                filename=A("final_graphic_07_FG_rotulos_llm_G1_homem_mulher_seed42_peso_altura_4.png")),
            'final_graphic_08': lambda: plots.plot_final_comparacao_modelos(abc + gemini, P("final_graphic_08_comparacao_modelos.png")),
        }
        for nome, fn in chamadas.items():
            with _CapturaTextos() as cap:
                fn()
            self._checa(cap, nome)

    def test_titulos_passados_pelo_runner_citam_o_passo(self):
        """Os títulos que o runner passa a `plot_rotulos_simbolos`, `plot_problema_passo1` e
        `plot_fg_rotulos_llm` (e os rótulos de coleta do log) já vêm com o passo."""
        with open(os.path.join(SRC_DIR, "dissertacao_mestrado.py"), encoding="utf-8") as fh:
            src = fh.read()
        for trecho in ('suptitle=f"Problema A, Passos 2 a 4, seed {seed}',
                       'suptitle=f"Problema {prob}, Passos 2 e 3: seed {seed}',
                       'suptitle=f"Problema D (meia-lua), Passos 2 a 4, seed {seed}',
                       'title=f"Problema D, Passo 1: meia-lua',
                       'title="Problema G1, Passo 1: peso × altura',
                       'title="Problema G2, Passo 1: peso × altura',
                       'f"Passo {_passo_rotulos} | seed={seed_val}',
                       '[Problema {problem}, Passo 2] ', '[Problema {problem}, Passo 3, k={k}, rep {rep + 1}] ',
                       '[Problema E, Passo 2] '):
            self.assertIn(trecho, src, trecho)
        visiveis = [ln for ln in src.splitlines()
                    if ("print" in ln or "suptitle" in ln or "title=" in ln or "label_prefix" in ln)
                    and any(p in ln for p in TERMOS_ANTIGOS)]
        self.assertEqual(visiveis, [], f"textos visíveis do runner com nomenclatura antiga: {visiveis}")


if __name__ == "__main__":
    unittest.main()


class TestSvmQuaseColapso(unittest.TestCase):
    """bloco1_graphic_14 usa a regra única `colapsou` (13/10/2026): 149/1 é colapso e o painel 2 anota, sem SVM."""

    def test_quase_colapso_anota_e_nao_ajusta_svm(self):
        rng = np.random.RandomState(0)
        X = rng.randn(150, 2); y = (X[:, 0] > 0).astype(int)
        y_llm = np.zeros(150, int); y_llm[0] = 1                       # 149/1
        capturadas = []
        orig = plots.salvar_figura
        plots.salvar_figura = lambda fig, *a, **k: capturadas.append([ax.get_title() for ax in fig.axes])
        try:
            tmp = tempfile.mkdtemp()
            ok = plots.plot_d_passo6_svm(X, y, y_llm, seed=1, filename=os.path.join(tmp, "bloco1_graphic_14_D_passo6_svm_seed1.png"))
        finally:
            plots.salvar_figura = orig
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertTrue(ok)
        titulos = " | ".join(capturadas[0])
        self.assertIn("colapso", titulos)
        self.assertNotIn("SVM RBF treinado nos rótulos do LLM", titulos)
