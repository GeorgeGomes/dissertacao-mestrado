"""Testes das funções estatísticas de ``relatorios.py``.

Prioridade: ``resumo_entre_sementes`` (média, mínimo e máximo entre sementes), que
substituiu o bootstrap em 13/10/2026 a pedido do orientador. Contratos verificados:

1. Dados constantes → faixa degenerada no próprio valor (média = mín = máx).
2. Um ponto → devolve (valor, valor, valor); vazio ou só NaN → NaN nos três.
3. `diferenca_consistente`: True só quando a faixa [mín, máx] não contém zero.
4. `bootstrap_ci` não existe mais no módulo.

Execução: .venv/bin/python -m unittest tests.test_relatorios
"""
import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import relatorios  # noqa: E402
from relatorios import (  # noqa: E402
    resumo_entre_sementes, diferenca_consistente, print_box, sintese_r2_r3_r4, PROBLEMA_LINEAR_R3R4,
)


class TestResumoEntreSementes(unittest.TestCase):
    """Substituto do bootstrap (removido em 13/10/2026 a pedido do orientador)."""

    def test_media_min_max(self):
        self.assertEqual(resumo_entre_sementes([0.6, 0.9, 0.75]), (0.75, 0.6, 0.9))

    def test_dados_constantes(self):
        self.assertEqual(resumo_entre_sementes(np.full(30, 0.75)), (0.75, 0.75, 0.75))

    def test_um_ponto(self):
        self.assertEqual(resumo_entre_sementes(np.array([0.4])), (0.4, 0.4, 0.4))

    def test_vazio_e_nan_ignorados(self):
        self.assertTrue(all(np.isnan(v) for v in resumo_entre_sementes([])))
        self.assertEqual(resumo_entre_sementes([0.2, float("nan"), 0.4]), (0.30000000000000004, 0.2, 0.4))

    def test_diferenca_consistente_quando_faixa_nao_contem_zero(self):
        self.assertTrue(diferenca_consistente(0.1, 0.3))
        self.assertTrue(diferenca_consistente(-0.3, -0.1))
        self.assertFalse(diferenca_consistente(-0.1, 0.3))
        self.assertFalse(diferenca_consistente(float("nan"), 0.3))

    def test_bootstrap_removido(self):
        self.assertFalse(hasattr(relatorios, "bootstrap_ci"))


class TestPrintBox(unittest.TestCase):
    def test_linha_longa_e_quebrada_e_nao_truncada(self):
        longo = "H4 REFUTADA: " + " ".join(f"palavra{i}" for i in range(60))
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_box("Veredito\n" + longo)
        saida = buf.getvalue()
        self.assertNotIn("...", saida)
        self.assertIn("palavra59", saida)                       # o fim da frase saiu inteiro
        linhas = [ln for ln in saida.splitlines() if ln.startswith("│")]
        self.assertTrue(len(linhas) >= 3)                       # cabeçalho + frase em várias linhas
        self.assertEqual(len(set(len(ln) for ln in linhas)), 1)  # caixa alinhada
        self.assertTrue(all(len(ln) <= 90 + 4 for ln in linhas))


class TestSummarizeCrossLinearity(unittest.TestCase):
    """O CSV cruzado deve ter R2/R3/R4 do Problema A (perceptron E nnls) com o alias."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_inclui_r2_r3_r4_dois_algoritmos_e_rotulo_problema_a(self):
        # chaves novas (M17): fidelity/fidelity_nnls + accuracy_vs_true(_nnls)
        r2 = [{'fidelity': .80, 'fidelity_nnls': .81, 'accuracy_vs_true': .9, 'accuracy_vs_true_nnls': .91, 'seed': 42}]
        r3 = [{'fidelity': .85, 'fidelity_nnls': .86, 'accuracy_vs_true': .9, 'accuracy_vs_true_nnls': .91, 'seed': 42}]
        r4 = [{'fidelity': .84, 'fidelity_nnls': .88, 'accuracy_vs_true': .9, 'accuracy_vs_true_nnls': .91, 'seed': 42}]
        f = sintese_r2_r3_r4(r3, [], self.tmp, results_abc_r4=r4,
                                      model_name="gpt-4o-mini", results_abc_r2=r2)
        self.assertEqual(os.path.basename(f), "final_cross_linearity__gpt4mini.csv")
        df = pd.read_csv(f)
        self.assertEqual(set(df['problem']), {PROBLEMA_LINEAR_R3R4})
        self.assertEqual(PROBLEMA_LINEAR_R3R4, "A_linear")
        self.assertEqual(sorted(df['n_features'].unique()), [2, 3, 4])
        self.assertEqual(set(df['algorithm']), {"perceptron", "nnls"})
        self.assertEqual(len(df), 6)
        # I8: colunas novas e acurácia vs real preenchida nas linhas A_linear
        for col in ("problem_name", "feature_names", "accuracy_vs_true", "fidelity_vs_llm"):
            self.assertIn(col, df.columns)
        self.assertEqual(set(df['feature_names']), {"x1/x2"})
        self.assertTrue((df['accuracy_vs_true'] > 0).all())

    def test_r2_sem_nnls_gera_so_perceptron_e_aceita_chaves_antigas(self):
        r2 = [{'accuracy': .80, 'accuracy_nnls': None, 'seed': 7}]
        r3 = [{'accuracy': .85, 'seed': 7}]
        f = sintese_r2_r3_r4(r3, [], self.tmp, model_name="google/gemini-2.5-flash-lite",
                                      results_abc_r2=r2)
        self.assertTrue(f.endswith("final_cross_linearity__flashlite.csv"))
        df = pd.read_csv(f)
        self.assertEqual(set(df['algorithm']), {"perceptron"})
        self.assertEqual(sorted(df['n_features'].unique()), [2, 3])

    def test_meia_lua_vem_do_problema_d_e_externos_guardam_variante(self):
        proto = [
            {'problem': 'D', 'regime': 'passo2_zero_shot', 'algorithm': 'perceptron', 'n_features': nf,
             'fidelity_vs_llm': .7 + nf / 100, 'accuracy_metric_vs_true_full': .8, 'random_seed': 42}
            for nf in (2, 3, 4)
        ] + [{'problem': 'D', 'regime': 'passo5_teto', 'algorithm': 'perceptron', 'n_features': 2,
              'fidelity_vs_llm': float('nan'), 'accuracy_metric_vs_true_full': .95, 'random_seed': 42},
             {'problem': 'A', 'regime': 'passo2_zero_shot', 'algorithm': 'perceptron', 'n_features': 2,
              'fidelity_vs_llm': .9, 'accuracy_metric_vs_true_full': .9, 'random_seed': 42}]
        ext = [
            {'problem_name': 'F_meia_lua_seed42', 'feature_names': ('x1', 'x2'), 'n_features': 2, 'seed': 42,
             'fidelity_perc_vs_llm': .6, 'fidelity_nnls_vs_llm': .6, 'accuracy_perc_vs_true': .5, 'accuracy_nnls_vs_true': .5},
            {'problem_name': 'G1_homem_mulher_classesAB', 'feature_names': ('peso', 'altura'), 'n_features': 4, 'seed': 42,
             'fidelity_perc_vs_llm': .8, 'fidelity_nnls_vs_llm': .8, 'accuracy_perc_vs_true': .7, 'accuracy_nnls_vs_true': .7},
        ]
        f = sintese_r2_r3_r4([], ext, self.tmp, model_name="gpt-4o-mini", linhas_passos_2_a_5=proto)
        df = pd.read_csv(f)
        d = df[df['problem'] == 'D_meialua']
        self.assertEqual(sorted(d['n_features']), [2, 3, 4])          # só zero-shot do D (não o teto, não o A)
        self.assertTrue((d['problem_name'] == 'D_meialua').all())   # nome completo, não a letra solta
        self.assertFalse(df['problem'].str.startswith('F_').any())      # F sai quando há D
        g = df[df['problem'] == 'G1_homem_mulher_classesAB']
        self.assertEqual(set(g['feature_names']), {'peso/altura'})
        self.assertEqual(set(g['algorithm']), {'perceptron', 'nnls'})
        # sem linhas do D, o F entra com rótulo estável e nome completo
        f2 = sintese_r2_r3_r4([], ext, self.tmp, model_name="gpt-4o-mini", linhas_passos_2_a_5=None)
        df2 = pd.read_csv(f2)
        self.assertIn('F_meia_lua', set(df2['problem']))
        self.assertIn('F_meia_lua_seed42', set(df2['problem_name']))


if __name__ == "__main__":
    unittest.main()


# ───────────────────────── H1 no log e malformadas do Problema E (09/10/2026) ─────────────────────────
from relatorios import (malformadas_e, print_e_passos_2_3,  # noqa: E402
                        print_sumario_estatistico, texto_h1)
from resultados import ResultadoABC, ResultadoAprendiz  # noqa: E402


def _res_abc(model, seed, kb, kc, k_exemplos=0, nomes=("A", "B"), variant="default", features=("x1", "x2"),
             collapsed=0, collapsed_b=0, collapsed_c=0, rep=0):
    return ResultadoABC(
        provider="openrouter", model_name=model, temperature=0.0, random_seed=seed, k_passo4_bc=k_exemplos,
        nomes_classes=nomes, repeticao=rep, fidelidade_problema_a=0.9, acuracia_llm_vs_gt_problema_a=0.9,
        consistencia_problema_b=0.8, kappa_problema_b=kb, f1_problema_b=0.8, acuracia_llm_vs_gt_problema_b=0.8,
        acuracia_metrica_vs_gt_problema_b=0.8, consistencia_problema_c=0.8, kappa_problema_c=kc,
        f1_problema_c=0.8, acuracia_llm_vs_gt_problema_c=0.8, acuracia_metrica_vs_gt_problema_c=0.8,
        w_aprendido=np.array([0.5, 0.5]), gamma_otimo=0.1, n_classe_0_problema_a=75, n_classe_1_problema_a=75,
        n_classe_0_problema_b=50, n_classe_1_problema_b=50, n_classe_0_problema_c=50, n_classe_1_problema_c=50,
        w_ratio=1.0, w_direction=np.array([0.7, 0.7]), collapsed=collapsed, collapsed_b=collapsed_b,
        collapsed_c=collapsed_c, feature_names=features, prompt_variant=variant)


class TestVereditoH1(unittest.TestCase):
    def test_sustentada_e_nao_sustentada_por_modelo_com_contagem_de_seeds(self):
        res = [
            # modelo 1: B passa em 3/3, C falha (média 0,35; 1/3 passa)
            _res_abc("m1", 42, 0.55, 0.30), _res_abc("m1", 123, 0.50, 0.45), _res_abc("m1", 7, 0.60, 0.30),
            # modelo 2: tudo acima de 0,4
            _res_abc("m2", 42, 0.7, 0.8), _res_abc("m2", 123, 0.75, 0.85), _res_abc("m2", 7, 0.9, 0.95),
            # linhas que NÃO entram: few-shot, outro nome de classe, variante, atributos, colapso
            _res_abc("m1", 42, 0.0, 0.0, k_exemplos=10), _res_abc("m1", 42, 0.0, 0.0, nomes=("0", "1")),
            _res_abc("m1", 42, 0.0, 0.0, variant="cot"), _res_abc("m1", 42, 0.0, 0.0, features=("altura", "peso")),
            _res_abc("m1", 42, float("nan"), float("nan"), collapsed=1),
        ]
        txt = texto_h1(res)
        self.assertIn("MODELO: openrouter/m1", txt)
        self.assertIn("Problema B: κ médio = 0.550 (moderada) | seeds com κ > 0.4: 3/3", txt)
        self.assertIn("Problema C: κ médio = 0.350 (razoável) | seeds com κ > 0.4: 1/3", txt)
        self.assertIn("H1 (κ > 0.4): SUSTENTADA em B (κ médio 0.550, 3/3 seeds passam) | "
                      "NÃO SUSTENTADA em C (κ médio 0.350, 1/3 seeds passam)", txt)
        self.assertIn("SUSTENTADA em B (κ médio 0.783, 3/3 seeds passam) | SUSTENTADA em C (κ médio 0.867, 3/3 seeds passam)", txt)
        self.assertIn("quase perfeita", txt)

    def test_colapso_em_b_fica_fora_e_sem_base_avisa(self):
        res = [_res_abc("m1", 42, float("nan"), 0.5, collapsed_b=1), _res_abc("m1", 123, 0.1, 0.5)]
        txt = texto_h1(res)
        self.assertIn("Problema B: κ médio = 0.100 (fraca) | seeds com κ > 0.4: 0/1", txt)
        self.assertIn("NÃO SUSTENTADA em B", txt)
        self.assertIn("SUSTENTADA em C (κ médio 0.500, 2/2 seeds passam)", txt)
        self.assertIn("sem linhas zero-shot", texto_h1([_res_abc("m1", 42, 0.5, 0.5, k_exemplos=4)]))
        self.assertIn("sem resultados", texto_h1([]))

    def test_print_statistical_summary_imprime_o_veredito(self):
        res = [_res_abc("m1", s, 0.5, 0.2) for s in (42, 123, 7)]
        with redirect_stdout(io.StringIO()) as buf:
            print_sumario_estatistico(res, [])
        out = buf.getvalue()
        self.assertIn("H1 (CONSISTÊNCIA): VEREDITO POR MODELO", out)
        self.assertIn("H1 (κ > 0.4): SUSTENTADA em B (κ médio 0.500, 3/3 seeds passam) | "
                      "NÃO SUSTENTADA em C (κ médio 0.200, 0/3 seeds passam)", out)


def _res_e(model, seed, k_exemplos, strategy, expert, n_malformed, rep=0):
    return ResultadoAprendiz(
        provider="openrouter", model_name=model, temperature=0.0, random_seed=seed, k=k_exemplos,
        example_strategy=strategy, nomes_classes=("A", "B"), repeticao=rep, accuracy_llm_vs_expert=0.8,
        kappa_llm_vs_expert=0.6, f1_llm_vs_expert=0.8, accuracy_expert_vs_gt=0.9, accuracy_llm_vs_gt=0.8,
        n_classe_0_expert=20, n_classe_1_expert=20, n_classe_0_llm=20, n_classe_1_llm=20, n_disagreements=8,
        n_total_test=40, n_malformed_responses=n_malformed, expert_w=np.array([0.3, 1.5]), expert_name=expert)


class TestMalformadasE(unittest.TestCase):
    def _resultados(self):
        res = []
        for seed in (42, 123):
            for expert in ("E1", "E2", "E3"):
                for strat in ("easy", "hard", "mixed", "random"):
                    res.append(_res_e("m1", seed, 0, strat, expert, 2))        # zero-shot: a MESMA coleta, 12 cópias
                    res.append(_res_e("m1", seed, 4, strat, expert, 1))        # few-shot: coleta própria
        return res

    def test_zero_shot_contado_uma_vez_por_modelo_e_seed(self):
        res = self._resultados()
        df = pd.DataFrame([{'model': r.model_name, 'seed': r.random_seed, 'rep': r.repeticao,
                            'k': r.k, 'n_malformed': r.n_malformed_responses} for r in res])
        few, zero = malformadas_e(df)
        self.assertEqual(few, 2 * 3 * 4 * 1)      # 24 coletas few-shot
        self.assertEqual(zero, 2 * 2)             # 2 seeds × 2 malformadas (e não 12 × 2 × 2 = 48)
        self.assertEqual(int(df['n_malformed'].sum()), 24 + 48)   # a soma ingênua contava 12×

    def test_print_e_passos_2_3_usa_a_soma_deduplicada(self):
        with redirect_stdout(io.StringIO()) as buf:
            print_e_passos_2_3(self._resultados())
        out = buf.getvalue()
        self.assertIn("Respostas malformadas no Problema E (openrouter/m1 (temp=0.0)): few-shot (Passo 3) 24 "
                      "+ zero-shot (Passo 2) 4 (contado uma vez por seed) = 28", out)
        self.assertIn("Respostas malformadas (few-shot deste especialista): 8", out)
        self.assertNotIn("Total de respostas malformadas: 72", out)
