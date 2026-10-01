"""Testes de fumaça de ``plots.py``: nomes dos painéis e correções de 01/10/2026.

Contratos verificados (sem API, pasta temporária):

1. Painéis individuais seguem ``<base>_<painel>__<alias>.png`` (alias por último),
   via ``asset_variant`` — e não ``<base>__<alias>_<painel>.png``.
2. ``plot_r3_comparison`` aceita R2/R3/R4 e gera os painéis
   ``_linear_vs_quadratica`` e ``_algoritmos_r3r4`` (com ou sem R4).
3. ``_dilution_reference`` lê o número de hard fixos do nome da estratégia
   (``dilution_4hard_0easy``) em vez de um n_shot fixo inexistente.
4. ``_grouped_bar_offsets`` centra as barras agrupadas no tick.

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
from resultados import ResultadoPhaseEExperimento  # noqa: E402


def _res_e(n_shot, strategy, acc, seed=42):
    return ResultadoPhaseEExperimento(
        provider="openai", model_name="gpt-4o-mini", temperature=0.0, random_seed=seed,
        n_shot=n_shot, example_strategy=strategy, nomes_classes=("A", "B"), repeticao=0,
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
        r2 = [{'accuracy': .80, 'accuracy_nnls': .81, 'seed': 42}, {'accuracy': .82, 'accuracy_nnls': .80, 'seed': 7}]
        r3 = [{'accuracy': .85, 'accuracy_nnls': .86, 'seed': 42}, {'accuracy': .84, 'accuracy_nnls': .85, 'seed': 7}]
        r4 = [{'accuracy': .84, 'accuracy_nnls': .88, 'seed': 42}, {'accuracy': .83, 'accuracy_nnls': .86, 'seed': 7}]
        fname = llm_asset(self.tmp, "bloco1_10_r3r4_comparison.png", "gpt-4o-mini")
        plots.plot_r3_comparison(r2, r3, r4, filename=fname)
        self.assertEqual(self._arquivos(), [
            "bloco1_10_r3r4_comparison__gpt4mini.png",
            "bloco1_10_r3r4_comparison_algoritmos_r3r4__gpt4mini.png",
            "bloco1_10_r3r4_comparison_linear_vs_quadratica__gpt4mini.png",
        ])

    def test_r3r4_sem_r4_e_sem_nnls_em_r2(self):
        r2 = [{'accuracy': .80, 'seed': 42}]
        r3 = [{'accuracy': .85, 'accuracy_nnls': .86, 'seed': 42}]
        fname = llm_asset(self.tmp, "bloco1_10_r3r4_comparison.png", "google/gemini-2.5-flash-lite")
        plots.plot_r3_comparison(r2, r3, None, filename=fname)
        self.assertIn("bloco1_10_r3r4_comparison_algoritmos_r3r4__flashlite.png", self._arquivos())

    def test_dilution_reference_le_n_hard_da_estrategia(self):
        df = pd.DataFrame({
            'n_shot': [4, 4, 6, 14], 'accuracy': [.7, .72, .75, .8], 'kappa': [.4, .44, .5, .6],
            'strategy': ['dilution_4hard_0easy', 'dilution_4hard_0easy',
                         'dilution_4hard_2easy', 'dilution_4hard_10easy'],
        })
        n_hard, ref = plots._dilution_reference(df)
        self.assertEqual(n_hard, 4)
        self.assertEqual(len(ref), 2)
        self.assertAlmostEqual(ref['accuracy'].mean(), .71)

    def test_dilution_reference_fallback_menor_n_shot(self):
        df = pd.DataFrame({'n_shot': [3, 5], 'accuracy': [.6, .7], 'kappa': [.2, .4],
                           'strategy': ['x', 'y']})
        n_hard, ref = plots._dilution_reference(df)
        self.assertEqual((n_hard, len(ref)), (3, 1))

    def test_dilution_plot_gera_paineis(self):
        res = [_res_e(4, 'dilution_4hard_0easy', .7), _res_e(6, 'dilution_4hard_2easy', .75),
               _res_e(14, 'dilution_4hard_10easy', .8)]
        fname = llm_asset(self.tmp, "bloco2_06_dilution.png", "gpt-4o-mini")
        plots.plot_dilution_experiment(res, filename=fname)
        self.assertEqual(self._arquivos(), [
            "bloco2_06_dilution__gpt4mini.png",
            "bloco2_06_dilution_accuracy__gpt4mini.png",
            "bloco2_06_dilution_kappa__gpt4mini.png",
        ])

    def test_grouped_bar_offsets_centrados(self):
        self.assertTrue(np.allclose(plots._grouped_bar_offsets(2, 0.25), [-0.125, 0.125]))
        self.assertTrue(np.allclose(plots._grouped_bar_offsets(3, 0.2), [-0.2, 0.0, 0.2]))
        self.assertTrue(np.allclose(plots._grouped_bar_offsets(1, 0.5), [0.0]))

    def test_main_expert_filtra_para_o_primeiro_perito(self):
        a = _res_e(4, 'easy', .8); a.expert_name = 'aniso_x2'
        b = _res_e(4, 'easy', .6); b.expert_name = 'euclidean'
        filtrados, nome = plots._main_expert([a, b, a])
        self.assertEqual((nome, len(filtrados)), ('aniso_x2', 2))


if __name__ == "__main__":
    unittest.main()
