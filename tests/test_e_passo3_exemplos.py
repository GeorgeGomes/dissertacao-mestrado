"""Testes das funções de seleção de exemplos few-shot e da regra de repetição.

Cobrem três contratos que sustentam números da dissertação:

1. **Anti-leakage** — os exemplos selecionados ficam FORA do conjunto de avaliação
   (o mask construído a partir dos índices selecionados exclui exatamente os
   exemplos), para todas as estratégias.
2. **Sorteio por repetição** — a estratégia ``random`` com ``random_state = seed + rep``
   sorteia exemplos idênticos para a mesma (seed, rep) e distintos entre reps: é o
   que faz a repetição medir variância de amostragem, não ruído de medição.
3. **Regra única de repetição** (``reps_para``) — zero-shot e seleções determinísticas
   rodam 1 coleta; few-shot com sorteio roda N_REPETICOES.

Execução: .venv/bin/python -m unittest tests.test_e_passo3_exemplos
"""
import os
import sys
import unittest

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402


def _dados():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 2)
    y = (X[:, 0] > 0).astype(int)
    w = np.array([1.0, 1.0])
    centroids = np.array([[-1.0, 0.0], [1.0, 0.0]])
    return X, y, w, centroids


class TestSelecaoExemplos(unittest.TestCase):
    def test_exemplos_selecionados_ficam_fora_do_teste(self):
        # Contrato anti-leakage da dissertação: excluir os índices selecionados
        # do conjunto de avaliação remove exatamente os pontos dos exemplos.
        X, y, w, c = _dados()
        for strategy in ("easy", "hard", "mixed", "random"):
            examples, idx = dm.e_passo3_selecionar_exemplos(
                X, y, w, c, n_examples=10, strategy=strategy,
                nome_classe_0="A", nome_classe_1="B",
                random_state=42, verbose=False,
            )
            mask = np.ones(len(X), dtype=bool)
            mask[idx] = False
            self.assertEqual(mask.sum(), len(X) - len(set(np.atleast_1d(idx).tolist())),
                             f"mask inconsistente para {strategy}")
            pontos_teste = {tuple(p) for p in X[mask]}
            for ex in examples:
                self.assertNotIn(
                    (ex[0], ex[1]), pontos_teste,
                    f"exemplo vazou para o conjunto de teste na estratégia {strategy}",
                )

    def test_k_impar_honrado_exatamente(self):
        # Regressão: "5-shot" tinha só 4 exemplos no prompt (n//2 por classe).
        # Contrato atual: n_examples é honrado exatamente; a classe 0 recebe o
        # excedente do ímpar (⌈n/2⌉ vs ⌊n/2⌋), deterministicamente.
        X, y, w, c = _dados()
        for n in (5, 7, 10):
            for strategy in ("easy", "hard", "mixed", "random"):
                examples, idx = dm.e_passo3_selecionar_exemplos(
                    X, y, w, c, n_examples=n, strategy=strategy,
                    nome_classe_0="A", nome_classe_1="B",
                    random_state=42, verbose=False,
                )
                self.assertEqual(len(examples), n,
                                 f"{strategy} com n={n} devolveu {len(examples)} exemplos")
                rotulos = [ex[2] for ex in examples]
                self.assertEqual(rotulos.count("A"), n - n // 2, f"{strategy} n={n}")
                self.assertEqual(rotulos.count("B"), n // 2, f"{strategy} n={n}")

    def test_mixed_e_exatamente_metade_facil_metade_dificil(self):
        # I15: com n=10 a estratégia mixed dá 5 fáceis + 5 difíceis (antes 4 + 6); o
        # excedente de cada classe alterna (classe 0 arredonda easy para cima, classe 1
        # para baixo). Easy = maior margem dentro da classe.
        from metrics import compute_metric_confidence
        X, y, w, c = _dados()
        conf, _ = compute_metric_confidence(X, c, w)
        for n, esperado in ((10, (5, 5)), (4, (2, 2)), (20, (10, 10)), (40, (20, 20))):
            _, idx = dm.e_passo3_selecionar_exemplos(
                X, y, w, c, n_examples=n, strategy="mixed",
                nome_classe_0="A", nome_classe_1="B", random_state=42, verbose=False)
            idx = np.atleast_1d(idx)
            n_easy = 0
            for cls in (0, 1):
                sel = [i for i in idx if y[i] == cls]
                ordem = sorted(np.where(y == cls)[0], key=lambda i: -conf[i])
                # os "easy" são os primeiros da ordem decrescente de margem da classe
                top = set(ordem[:len(sel)])
                n_easy += sum(1 for i in sel if i in top and conf[i] >= np.median(conf[sel]))
            self.assertEqual((n_easy, n - n_easy), esperado, f"n={n}")

    def test_k_impar_honrado_no_aprendizado_ativo(self):
        # Mesmo contrato para o Passo 4 de B e C (bc_passo4_exemplos_da_metrica, que rotula
        # pela métrica aprendida, não pelo y verdadeiro).
        X, _, w, c = _dados()
        for n in (5, 10):
            examples, _, _, idx = dm.bc_passo4_exemplos_da_metrica(
                X, c, w, n, nome_classe_0="A", nome_classe_1="B", verbose=False,
            )
            self.assertEqual(len(examples), n)
            self.assertEqual(len(np.atleast_1d(idx)), n)

    def test_random_state_seed_mais_rep_gera_sorteios_distintos(self):
        X, y, w, c = _dados()

        def sorteio(rs):
            _, idx = dm.e_passo3_selecionar_exemplos(
                X, y, w, c, n_examples=10, strategy="random",
                nome_classe_0="A", nome_classe_1="B",
                random_state=rs, verbose=False,
            )
            return list(np.atleast_1d(idx))

        self.assertEqual(sorteio(42 + 0), sorteio(42 + 0),
                         "mesma (seed, rep) deve sortear os mesmos exemplos")
        self.assertNotEqual(sorteio(42 + 0), sorteio(42 + 1),
                            "reps distintas devem sortear exemplos distintos")


class TestRegraRepeticao(unittest.TestCase):
    def test_zero_shot_roda_uma_vez(self):
        self.assertEqual(dm.reps_para(0), 1)
        self.assertEqual(dm.reps_para(0, sorteio_estocastico=True), 1)

    def test_few_shot_estocastico_repete(self):
        self.assertEqual(dm.reps_para(10), dm.N_REPETICOES)

    def test_selecao_deterministica_roda_uma_vez(self):
        self.assertEqual(dm.reps_para(10, sorteio_estocastico=False), 1)

    def test_model_slug(self):
        self.assertEqual(dm._model_slug("org/modelo-exemplo v1"),
                         "org-modelo-exemplo-v1")
        self.assertEqual(dm._model_slug("google/gemini-2.5-flash-lite"),
                         "google-gemini-2.5-flash-lite")

    def test_model_alias(self):
        # Aliases curtos nos nomes de asset (decisão 03/07/2026).
        self.assertEqual(dm._model_alias("gpt-4o-mini"), "gpt4mini")
        self.assertEqual(dm._model_alias("google/gemini-2.5-flash-lite"), "flashlite")

    def test_model_alias_fallback_para_slug(self):
        # Modelo fora do protocolo cai no slug longo (nunca quebra).
        self.assertEqual(dm._model_alias("org/modelo-desconhecido"),
                         dm._model_slug("org/modelo-desconhecido"))

    def test_todos_os_modelos_do_protocolo_tem_alias(self):
        # Evita asset com slug longo por esquecimento ao trocar um modelo:
        # todo modelo de MODELS_TO_TEST deve ter entrada em MODEL_ALIAS.
        for _prov, model, _temp in dm.MODELS_TO_TEST:   # 3-tuplas desde 09/10/2026 (sem scope)
            self.assertIn(model, dm.MODEL_ALIAS, model)

    def test_llm_asset_injeta_alias_antes_da_extensao(self):
        self.assertEqual(
            dm.llm_asset("/tmp/exec", "final_graphic_07_FG_rotulos_llm_seed42.png", "gpt-4o-mini"),
            "/tmp/exec/final_graphic_07_FG_rotulos_llm_seed42__gpt4mini.png",
        )

    def test_asset_variant_insere_sufixo_antes_do_alias(self):
        # Painéis individuais e cópias corrigidas: "<base>_<sufixo>__<alias>.ext",
        # nunca "<base>__<alias>_<sufixo>.ext" (alias sempre antes da extensão).
        self.assertEqual(
            dm.asset_variant("/tmp/e/bloco1_graphic_05_A_passo4_w_distribuicao__gpt4mini.png", "boxplot"),
            "/tmp/e/bloco1_graphic_05_A_passo4_w_distribuicao_boxplot__gpt4mini.png",
        )
        self.assertEqual(
            dm.asset_variant("/tmp/e/final_cross_linearity__flashlite.csv", "corrigido_hm"),
            "/tmp/e/final_cross_linearity_corrigido_hm__flashlite.csv",
        )

    def test_asset_variant_sem_alias_e_slug_longo(self):
        self.assertEqual(
            dm.asset_variant("/tmp/e/bloco1_graphic_19_oraculo_passos_1_a_3.png", "ratio"),
            "/tmp/e/bloco1_graphic_19_oraculo_passos_1_a_3_ratio.png",
        )
        slug = dm._model_slug("org/modelo-x.1")
        self.assertEqual(
            dm.asset_variant(f"/tmp/e/final_graphic_03_ABC_acertos_erros_seed42__{slug}.png", "problema_a"),
            f"/tmp/e/final_graphic_03_ABC_acertos_erros_seed42_problema_a__{slug}.png",
        )

    def test_asset_variant_compoe_com_llm_asset(self):
        p = dm.llm_asset("/tmp/e", "bloco1_graphic_08_A_passo4_R2_R3_R4.png", "gpt-4o-mini")
        self.assertEqual(
            dm.asset_variant(p, "linear_vs_quadratica"),
            "/tmp/e/bloco1_graphic_08_A_passo4_R2_R3_R4_linear_vs_quadratica__gpt4mini.png",
        )


if __name__ == "__main__":
    unittest.main()


class TestDiluicaoPasso4(unittest.TestCase):
    """Problema E1, Passo 4: 4 difíceis fixos (2 por classe, menor margem) + M fáceis que não repetem os difíceis."""

    def test_dificeis_fixos_e_faceis_sem_repeticao(self):
        X, y_exp, w, c = _dados()
        _, idx0 = dm.e_passo4_exemplos_diluicao(X, y_exp, w, c, 4, 0, "A", "B")
        _, idx4 = dm.e_passo4_exemplos_diluicao(X, y_exp, w, c, 4, 4, "A", "B")
        _, idx4b = dm.e_passo4_exemplos_diluicao(X, y_exp, w, c, 4, 4, "A", "B")
        self.assertEqual(len(idx0), 4); self.assertEqual(len(idx4), 8)
        self.assertEqual(sorted(idx4.tolist()), sorted(idx4b.tolist()))          # determinístico (sem sorteio)
        self.assertEqual(set(idx0.tolist()), set(idx4[:4].tolist()))             # os 4 difíceis são os mesmos
        self.assertEqual(len(set(idx4.tolist())), 8)                             # fáceis não repetem difíceis
        self.assertEqual(sorted(np.bincount(y_exp[idx0], minlength=2).tolist()), [2, 2])
        conf, _ = dm.compute_metric_confidence(X, c, w)
        self.assertLess(conf[idx0].max(), conf[idx4[4:]].min())                  # difíceis = menor margem que os fáceis
