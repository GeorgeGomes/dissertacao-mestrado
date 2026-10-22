"""Problema G, Passos 2 a 5 (mesmos passos do A; Metodologia IV.7 do orientador, 13/10/2026).

Contratos (sem chamar a API: dublê de `collect_llm_decisions` que registra X, exemplos e tag):
(a) `fg_executar(..., com_passos_3_a_5=True)` gera, por variante de atributo,
    linhas dos 3 regimes (passo2_zero_shot, passo3_exemplos_rotulo_real, passo5_teto) em R²/R³/R⁴ com `subproblem`,
    `class_names` e `feature_names`, devolvidas em 'passos_2_a_5';
(b) os exemplos mostrados ao LLM vêm de `X_display` e são k/2 por classe REAL;
(c) as duas variantes de atributo recebem os MESMOS índices de exemplo (mesmo seed + rep);
(d) o Passo 2 reaproveita a coleta já feita (0 chamadas extras);
(e) a tag de auditoria tem os 7 campos fixos + `nomes=`/`atributos=`;
(f) eleição do especialista: zero-shot colapsado + few-shot separável → `best` vem do few-shot,
    o Passo 6 roda e as linhas levam `especialista_regime="passo3_exemplos_rotulo_real"`;
(g) o Problema F (`com_passos_3_a_5=False`) não muda;
(h) tabela `bloco3_table_06_G_passos_2_a_5` e figura `bloco3_graphic_03_G_passos_2_a_5_curva`.

Execução: .venv/bin/python -m unittest tests.test_g_passos_2_a_5
"""
import io
import os
import re
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402
import data_problems as dp  # noqa: E402
import gerar_tabelas as gt  # noqa: E402
import plots  # noqa: E402
from execucao_io import llm_asset  # noqa: E402

_COLS_PROTO_B1 = {"provider", "model", "temperature", "random_seed", "problem", "n_features", "regime", "k", "rep",
                  "random_state_sorteio", "n_total", "n_examples", "n_eval", "n_classe_0_llm", "n_classe_1_llm",
                  "llm_accuracy_vs_true", "n_malformed", "example_indices", "algorithm", "collapsed",
                  "fidelity_vs_llm", "accuracy_metric_vs_true_eval", "accuracy_metric_vs_true_full",
                  "accuracy_teto_vs_true", "fidelity_teto_vs_llm_passo2", "n_errors_metric_vs_true_eval",
                  "w_ratio", "gamma", "cos_sim_perc_nnls"} | {f"w_{i}" for i in range(4)} | {f"w_unit_{i}" for i in range(4)}
_COLS_FG_APRENDIZ_ANTES = {"problem_name", "seed", "rep", "feature_names", "n_features", "k", "especialista_origem",
                     "accuracy_llm_vs_expert", "accuracy_llm_vs_true", "kappa_llm_vs_expert", "f1_llm_vs_expert",
                     "accuracy_perceptron_baseline_vs_true", "n_test", "n_errors_llm_vs_true",
                     "n_errors_llm_vs_expert", "n_malformed", "example_indices"}


def _dados(n=80, seed=0):
    """Duas gaussianas separáveis em x1 (rótulo real = x1 > 0)."""
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.normal([-2, 0], 0.7, size=(n // 2, 2)), rng.normal([2, 0], 0.7, size=(n // 2, 2))])
    y = (X[:, 0] > 0).astype(int)
    return X, y


class _Base(unittest.TestCase):
    """Dublê de `collect_llm_decisions`: regra x1 > 0 (vale também para X_display = X·100);
    `colapso_passo2` faz o zero-shot (Passo 2) responder só a classe 0."""

    def setUp(self):
        self.X, self.y = _dados()
        self.chamadas = []
        self._saved = {"collect": dm.collect_llm_decisions, "k": list(dm.K_EXEMPLOS_ROTULO_REAL),
                       "reps": dm.N_REPETICOES, "cache": dict(dm.CACHE_ROTULOS_LLM),
                       "model": dm.MODEL_NAME, "provider": dm.CURRENT_PROVIDER}
        dm.K_EXEMPLOS_ROTULO_REAL = [4]
        dm.N_REPETICOES = 1
        dm.CACHE_ROTULOS_LLM.clear()
        dm.MODEL_NAME = "modelo-teste"
        dm.CURRENT_PROVIDER = "openrouter"
        self.colapso_passo2 = False

        def dublê(X, c0, c1, examples=None, **kw):
            X = np.asarray(X, dtype=float)
            tag = kw.get("tag") or dm.CURRENT_TAG
            self.chamadas.append({"X": X.copy(), "examples": examples, "tag": tag,
                                  "feat0": kw.get("nome_feature_0"), "feat1": kw.get("nome_feature_1"),
                                  "extras": kw.get("extra_features_matrix")})
            if self.colapso_passo2 and examples is None and "|passo=2|" in tag:
                return np.zeros(len(X), int), 0
            return (X[:, 0] > 0).astype(int), 0
        dm.collect_llm_decisions = dublê

    def tearDown(self):
        dm.collect_llm_decisions = self._saved["collect"]
        dm.K_EXEMPLOS_ROTULO_REAL = self._saved["k"]
        dm.N_REPETICOES = self._saved["reps"]
        dm.CACHE_ROTULOS_LLM.clear(); dm.CACHE_ROTULOS_LLM.update(self._saved["cache"])
        dm.MODEL_NAME = self._saved["model"]
        dm.CURRENT_PROVIDER = self._saved["provider"]

    def _roda(self, **kw):
        args = dict(problem_name="G1_homem_mulher", X=self.X, y_true=self.y, nome_classe_0="Mulher",
                    nome_classe_1="Homem", feature_variants=[("x1", "x2"), ("peso", "altura")], seeds=[5],
                    ks_aprendiz=[4], n_train_ratio=0.7, verbose=True, com_passos_3_a_5=True)
        args.update(kw)
        with redirect_stdout(io.StringIO()) as buf:
            out = dm.fg_executar(**args)
        return out, buf.getvalue()

    def _train(self, seed=5):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(self.X))[:int(0.7 * len(self.X))]
        return self.X[idx], self.y[idx]


class TestProtocoloNoG(_Base):
    def test_a_linhas_dos_tres_regimes_em_r2_r3_r4_com_colunas_do_g(self):
        out, log = self._roda()
        rows = out["passos_2_a_5"]
        self.assertTrue(rows)
        self.assertTrue(_COLS_PROTO_B1 <= set(rows[0]), _COLS_PROTO_B1 - set(rows[0]))
        for r in rows:
            self.assertEqual(r["problem_name"], "G1_homem_mulher")
            self.assertEqual(r["problem"], "G1_homem_mulher")
            self.assertEqual(r["subproblem"], "G1")
            self.assertEqual(r["class_names"], "Mulher/Homem")
            self.assertIn(r["feature_names"], ("x1/x2", "peso/altura"))
            self.assertEqual(r["n_total"], 56)                      # 70% de 80
        for fn in ("x1/x2", "peso/altura"):
            sub = [r for r in rows if r["feature_names"] == fn]
            # (zero + 1 rep de k=4 + teto) × 3 espaços × 2 algoritmos
            self.assertEqual(len(sub), 3 * 3 * 2)
            self.assertEqual({(r["regime"], r["k"], r["rep"]) for r in sub},
                             {("passo2_zero_shot", 0, 0), ("passo3_exemplos_rotulo_real", 4, 0), ("passo5_teto", 0, 0)})
            self.assertEqual({r["n_features"] for r in sub}, {2, 3, 4})
            self.assertEqual({r["algorithm"] for r in sub}, {"perceptron", "nnls"})
            self.assertTrue(all(r["collapsed"] == 0 for r in sub))
            few = [r for r in sub if r["regime"] == "passo3_exemplos_rotulo_real"]
            self.assertTrue(all(r["n_eval"] == 56 - 4 and r["n_examples"] == 4 for r in few))
        self.assertIn("Passos 2 a 5 do Problema G (mesmos passos do Problema A)", log)
        self.assertIn("Passos 2 a 5 em G1_homem_mulher", log)
        # o zero-shot clássico (bloco23_fg_passos_2_e_4) continua sendo produzido
        self.assertEqual({r["n_features"] for r in out["passo2"]}, {2, 3, 4})

    def test_b_exemplos_vem_de_x_display_e_k_meio_por_classe_real(self):
        X_disp = self.X * 100.0
        out, _ = self._roda(X_display=X_disp)
        X_tr, y_tr = self._train()
        few = [c for c in self.chamadas if c["examples"] and "|passo=3|" in c["tag"]]
        self.assertEqual(len(few), 2)                                # 1 rep × 2 variantes
        for c in few:
            ex = c["examples"]
            self.assertEqual(len(ex), 4)
            self.assertEqual(sum(1 for e in ex if e[-1] == "Mulher"), 2)
            self.assertEqual(sum(1 for e in ex if e[-1] == "Homem"), 2)
            for e in ex:
                self.assertGreater(abs(e[0]), 20.0)                  # escala de X_display, não de X
                i = int(np.argmin(np.abs(X_tr * 100.0 - np.array(e[:2])).sum(axis=1)))
                np.testing.assert_allclose(e[:2], X_tr[i] * 100.0)
                self.assertEqual(e[-1], "Mulher" if y_tr[i] == 0 else "Homem")   # rótulo REAL
            # os pontos a classificar também vêm de X_display e são os 56 − 4 restantes
            self.assertEqual(len(c["X"]), 56 - 4)
            self.assertGreater(np.abs(c["X"][:, 0]).min(), 1.0)
            self.assertIsNone(c["extras"])                           # R³/R⁴ só na estimação
        # a estimação ficou no espaço de X (normalizado): métrica aplicada a X_train
        rows = [r for r in out["passos_2_a_5"] if r["regime"] == "passo3_exemplos_rotulo_real" and r["algorithm"] == "perceptron"]
        self.assertTrue(all(r["accuracy_metric_vs_true_full"] > 0.9 for r in rows))

    def test_c_mesmos_indices_de_exemplo_nas_duas_variantes(self):
        out, _ = self._roda()
        few = [r for r in out["passos_2_a_5"] if r["regime"] == "passo3_exemplos_rotulo_real"]
        por_var = {}
        for r in few:
            por_var.setdefault((r["k"], r["rep"], r["n_features"], r["algorithm"]), {})[r["feature_names"]] = r["example_indices"]
        for chave, d in por_var.items():
            self.assertEqual(set(d), {"x1/x2", "peso/altura"}, chave)
            self.assertEqual(d["x1/x2"], d["peso/altura"], chave)
            self.assertEqual(len(d["x1/x2"].split(";")), 4)
        ex = [tuple(e[:2] for e in c["examples"]) for c in self.chamadas if c["examples"] and "|passo=3|" in c["tag"]]
        self.assertEqual(ex[0], ex[1])                               # mesmos pontos, mesma ordem

    def test_d_passo2_nao_gera_chamada_extra(self):
        self._roda()
        passos_2_3 = [c for c in self.chamadas if "|passo=2|" in c["tag"] or "|passo=3|" in c["tag"]]
        zero = [c for c in passos_2_3 if c["examples"] is None]
        self.assertEqual(len(zero), 2)                               # 1 por variante (a coleta já existente)
        self.assertTrue(all(len(c["X"]) == 56 for c in zero))
        self.assertEqual(len(passos_2_3), 2 + 2)                     # + 1 coleta do Passo 3 (k=4, 1 rep) por variante
        # os nomes de atributo da variante chegam ao coletor
        self.assertEqual({(c["feat0"], c["feat1"]) for c in passos_2_3}, {("x1", "x2"), ("peso", "altura")})

    def test_e_tag_com_sete_campos_mais_nomes_e_atributos(self):
        self._roda()
        tags = {c["tag"] for c in self.chamadas}
        self.assertIn("bloco3|problema=G1_homem_mulher|passo=3|regime=passo3_exemplos_rotulo_real|k=4|rep=0|seed=5"
                      "|nomes=Mulher/Homem|atributos=peso/altura", tags)
        self.assertIn("bloco3|problema=G1_homem_mulher|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=5"
                      "|nomes=Mulher/Homem|atributos=x1/x2", tags)
        # Passo 6 (14/10/2026): depois de nomes= e atributos= vem o nº de atributos mostrados ao LLM
        self.assertIn("bloco3|problema=G1_homem_mulher|passo=6|regime=exemplos_da_metrica|k=4|rep=0|seed=5"
                      "|nomes=Mulher/Homem|atributos=x1/x2|n_atributos_prompt=2", tags)
        # prefixo de 7 campos que a auditoria lê (problema=, passo=, regime=, k=)
        for t in tags:
            campos = t.split("|")
            self.assertEqual(campos[0], "bloco3")
            self.assertTrue(campos[1].startswith("problema="))
            self.assertTrue(campos[2].startswith("passo=") and campos[3].startswith("regime="))
            self.assertTrue(campos[4].startswith("k=") and campos[5].startswith("rep=") and campos[6].startswith("seed="))

    def test_f_eleicao_vem_do_passo3_quando_o_passo2_colapsa(self):
        self.colapso_passo2 = True
        out, log = self._roda(feature_variants=[("x1", "x2")])
        self.assertEqual(out["passo2"], [])                 # zero-shot colapsou (sem métrica clássica)
        rows = out["passos_2_a_5"]
        zs = [r for r in rows if r["regime"] == "passo2_zero_shot"]
        self.assertTrue(zs and all(r["collapsed"] == 1 for r in zs))
        few = [r for r in rows if r["regime"] == "passo3_exemplos_rotulo_real"]
        self.assertTrue(few and all(r["collapsed"] == 0 for r in few))
        e = out["aprendiz"]
        self.assertTrue(e)                                           # o Passo 6 rodou
        for r in e:
            self.assertEqual(r["especialista_origem"], "metrica")
            self.assertEqual(r["especialista_regime"], "passo3_exemplos_rotulo_real")
            self.assertEqual(r["especialista_k"], 4)
            self.assertEqual(r["especialista_rep"], 0)
            self.assertIn(r["n_features"], (2, 3, 4))
        self.assertIn("Passo 5: especialista eleito: regime=passo3_exemplos_rotulo_real k=4 rep=0", log)

    def test_f2_todos_colapsados_pula_o_passo6(self):
        dm.collect_llm_decisions = lambda X, c0, c1, examples=None, **kw: (np.zeros(len(X), int), 0)
        out, log = self._roda(feature_variants=[("x1", "x2")])
        self.assertEqual(out["aprendiz"], [])
        self.assertTrue(all(r["collapsed"] == 1 for r in out["passos_2_a_5"] if r["regime"] != "passo5_teto"))
        self.assertIn("colapso no zero-shot em todos os regimes", log)

    def test_f3_sem_passos_3_a_5_o_especialista_e_o_passo2(self):
        out, log = self._roda(com_passos_3_a_5=False, feature_variants=[("x1", "x2")])
        self.assertEqual(out["passos_2_a_5"], [])
        for r in out["aprendiz"]:
            self.assertEqual((r["especialista_regime"], r["especialista_k"], r["especialista_rep"]), ("passo2_zero_shot", 0, 0))
        self.assertIn("Passo 5: especialista eleito: regime=passo2_zero_shot k=0 rep=0", log)
        self.assertEqual(len([c for c in self.chamadas if "|passo=2|" in c["tag"]]), 1)

    def test_g_problema_f_nao_muda(self):
        X, y = dp.create_problem_d_meialua(n_samples=60, random_state=3)
        self.X, self.y = X, y
        out, _ = self._roda(problem_name="F_meia_lua_seed3", nome_classe_0="A", nome_classe_1="B",
                            feature_variants=[("x1", "x2")], seeds=[3], ks_aprendiz=[0, 4],
                            expert_labels=y, com_passos_3_a_5=False)
        self.assertEqual(out["passos_2_a_5"], [])
        self.assertFalse([c for c in self.chamadas if c["examples"] and "|passo=2|" in c["tag"]])   # F: sem exemplos no Passo 2
        e = out["aprendiz"]
        self.assertTrue(e)
        self.assertEqual(set(e[0]), _COLS_FG_APRENDIZ_ANTES | {"especialista_regime", "especialista_k", "especialista_rep",
                                                               "especialista_n_features"})   # 14/10/2026
        for r in e:
            self.assertEqual(r["especialista_origem"], "rotulo_real")
            self.assertTrue(all(isinstance(r[c], float) and np.isnan(r[c]) for c in ("especialista_regime", "especialista_k", "especialista_rep", "especialista_n_features")))
            self.assertEqual(r["n_features"], 2)                # F: só x1, x2 no prompt
        tags = {c["tag"] for c in self.chamadas}
        self.assertIn("bloco2|problema=F_meia_lua_seed3|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=3|nomes=A/B|atributos=x1/x2", tags)
        # Passo 3 do F (14/10/2026): a tag registra os 2 atributos mostrados ao LLM, como no Passo 6 do G
        self.assertIn("bloco2|problema=F_meia_lua_seed3|passo=3|regime=zero_shot|k=0|rep=0|seed=3|nomes=A/B|atributos=x1/x2|n_atributos_prompt=2", tags)
        self.assertIn("bloco2|problema=F_meia_lua_seed3|passo=3|regime=exemplos_rotulo_real|k=4|rep=0|seed=3|nomes=A/B|atributos=x1/x2|n_atributos_prompt=2", tags)
        # 13/10/2026: no F os exemplos do Passo 3 e o Perceptron do Passo 4 ficam em R² (x1, x2), mesmo
        # com o Passo 2 sem colapso; a métrica do Passo 2 serve só para medir a fidelidade.
        self.assertTrue(out["passo2"], "o Passo 2 não colapsou neste dublê")
        self.assertEqual({r["n_features"] for r in e}, {2})
        self.assertFalse([c for c in self.chamadas if "|passo=3|" in c["tag"] and c["examples"]
                          and any(len(ex) > 3 for ex in c["examples"])])


class TestHelpers(unittest.TestCase):
    def test_tag_prefixo_de_sete_campos_e_extra(self):
        self.assertEqual(dm._tag("bloco1", "A", 3, "passo3_exemplos_rotulo_real", 4, 1, 42),
                         "bloco1|problema=A|passo=3|regime=passo3_exemplos_rotulo_real|k=4|rep=1|seed=42")
        self.assertEqual(dm._tag("bloco3", "G2_homem_mulher", 2, "passo2_zero_shot", 0, 0, 7,
                                 extra="nomes=A/B|atributos=peso_kg/altura_m"),
                         "bloco3|problema=G2_homem_mulher|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=7|nomes=A/B|atributos=peso_kg/altura_m")

    def test_balanced_gt_examples_mostra_x_display_e_sorteia_em_x(self):
        X, y = _dados()
        ex, idx, mask = dm.passo3_exemplos_rotulo_real(X, y, 6, "A", "B", 42, X_display=X * 10.0)
        ex0, idx0, _ = dm.passo3_exemplos_rotulo_real(X, y, 6, "A", "B", 42)
        self.assertEqual(idx.tolist(), idx0.tolist())               # X_display não altera o sorteio
        for (x1, x2, lab), i in zip(ex, idx):
            self.assertAlmostEqual(x1, X[i, 0] * 10.0); self.assertAlmostEqual(x2, X[i, 1] * 10.0)
            self.assertEqual(lab, "A" if y[i] == 0 else "B")
        self.assertEqual(sum(1 for e in ex if e[-1] == "A"), 3)
        with self.assertRaises(ValueError):
            dm.adg_passos_2_a_5(X, y, "A", "B", "G1_x", 1, "p", "m", 0.0, X_display=X[:5])

    def test_eleicao_prefere_maior_acuracia_e_exclui_teto_e_colapso(self):
        nan = float("nan")
        rows = [dict(algorithm="perceptron", regime="passo2_zero_shot", k=0, rep=0, n_features=2, collapsed=1, accuracy_metric_vs_true_full=nan),
                dict(algorithm="perceptron", regime="passo3_exemplos_rotulo_real", k=4, rep=0, n_features=2, collapsed=0, accuracy_metric_vs_true_full=0.80),
                dict(algorithm="perceptron", regime="passo3_exemplos_rotulo_real", k=4, rep=0, n_features=4, collapsed=0, accuracy_metric_vs_true_full=0.85),
                dict(algorithm="nnls", regime="passo3_exemplos_rotulo_real", k=4, rep=0, n_features=4, collapsed=0, accuracy_metric_vs_true_full=0.99),
                dict(algorithm="perceptron", regime="passo5_teto", k=0, rep=0, n_features=2, collapsed=0, accuracy_metric_vs_true_full=0.95)]
        metrics = {("passo3_exemplos_rotulo_real", 4, 0, 2): dict(w_perc=[1, 1], centroids=[[0, 0], [1, 1]]),
                   ("passo3_exemplos_rotulo_real", 4, 0, 4): dict(w_perc=[1, 2, 3, 4], centroids=[[0] * 4, [1] * 4])}
        best = dm.g_passo5_eleger_especialista({"rows": rows, "metrics": metrics}, [])
        self.assertEqual((best["metrica_regime"], best["metrica_k"], best["metrica_rep"], best["n_features"]), ("passo3_exemplos_rotulo_real", 4, 0, 4))
        self.assertAlmostEqual(best["accuracy_perc_vs_true"], 0.85)
        self.assertEqual(best["w_perc"].tolist(), [1, 2, 3, 4])
        self.assertIsNone(dm.g_passo5_eleger_especialista({"rows": rows[:1] + rows[4:], "metrics": {}}, []))


def _rows_proto_g(model="gpt-4o-mini"):
    """Linhas sintéticas de bloco3_g_passos_2_a_5: 2 grupos (Mulher/Homem peso/altura; A/B x1/x2), 2 sementes."""
    nan = float("nan")
    rows = []
    for sp, cn, fn in (("G1", "Mulher/Homem", "peso/altura"), ("G1", "A/B", "x1/x2"), ("G2", "Mulher/Homem", "peso_kg/altura_m")):
        for seed in (42, 7):
            for regime, k in (("passo2_zero_shot", 0), ("passo3_exemplos_rotulo_real", 4), ("passo3_exemplos_rotulo_real", 10), ("passo5_teto", 0)):
                for nf in (2, 3, 4):
                    for algo in ("perceptron", "nnls"):
                        gt = regime == "passo5_teto"
                        # A/B zero-shot da semente 7 colapsa; Mulher/Homem 10-shot tem γ = 0 (†)
                        colapso = cn == "A/B" and regime == "passo2_zero_shot" and seed == 7
                        acc_llm = 0.3 if cn == "A/B" else 0.8        # literal invertida na variante neutra
                        rows.append(dict(provider="openrouter", model=model, temperature=0.0, random_seed=seed,
                                         problem=f"{sp}_x", problem_name=f"{sp}_x", subproblem=sp, class_names=cn,
                                         feature_names=fn, n_features=nf, regime=regime, k=k, rep=0, algorithm=algo,
                                         n_total=70, n_examples=k, n_eval=70 - k,
                                         collapsed=int(colapso),
                                         fidelity_vs_llm=nan if (gt or colapso) else 0.7 + 0.02 * k + 0.01 * nf,
                                         accuracy_metric_vs_true_full=nan if colapso else 0.75 + 0.01 * k,
                                         accuracy_teto_vs_true=0.9 if gt else nan,
                                         llm_accuracy_vs_true=nan if gt else (0.5 if colapso else acc_llm + 0.01 * k),
                                         n_classe_0_llm=35, n_classe_1_llm=35,
                                         gamma=(0.0 if (cn == "Mulher/Homem" and k == 10 and algo == "perceptron") else 0.2)))
    return rows


class TestTabelaEFigura(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_h_tabela_bloco3_06(self):
        df = pd.DataFrame(_rows_proto_g())
        D = {"g_passos_2_a_5": df}
        t = gt.tab_g_passos_2_a_5(D, "gpt-4o-mini")
        self.assertIsNotNone(t)
        self.assertEqual(t.nome_png(), "bloco3_table_06_G_passos_2_a_5__gpt4mini_gray.png")
        self.assertEqual(t.nome_tex(), "bloco3_table_06_G_passos_2_a_5__gpt4mini.tex")
        self.assertTrue(t.caption and t.rodape)
        self.assertEqual(len(t.rows), 3 * 4)                          # 3 grupos × (zero, 4, 10, teto)
        self.assertEqual(t.headers[6:9], ["R$^2$", "R$^3$", "R$^4$"])
        self.assertEqual(t.rows[0][:3], ["G1 (normalizado)", "Mulher / Homem", "peso/altura"])   # Mulher/Homem antes de A/B; 1ª linha nomeia o grupo
        self.assertEqual(t.rows[1][:3], ["", "", ""])
        regimes = [r[3] for r in t.rows[:4]]
        self.assertEqual(regimes[0], "zero-shot"); self.assertEqual(regimes[-1], "W ajustado ao rótulo real (teto)")
        # teto: sem fidelidade, com acurácia
        self.assertEqual(t.rows[3][6:9], ["—", "—", "—"]); self.assertEqual(t.rows[3][9], "90,0")
        # A/B: zero-shot com a semente colapsada fora da média (fica só a 42) e coluna inv. perm. preenchida
        ab = [r for r in t.rows if r[0].startswith("G1") and r[1] == "A / B"][0]
        self.assertEqual(ab[4], "30,0"); self.assertEqual(ab[5], "70,0")
        # Mulher/Homem (G1): inv. perm. "—"; 10-shot com †
        mh = [r for r in t.rows if r[0].startswith("G1") and r[1] == "Mulher / Homem"][0]
        i0 = t.rows.index(mh)
        self.assertEqual(t.rows[i0][5], "—")
        self.assertTrue(t.rows[i0 + 2][3].endswith("†"), t.rows[i0 + 2][3])
        self.assertFalse(t.rows[i0 + 1][3].endswith("†"))
        self.assertIn("70 pontos de treino", t.rodape); self.assertIn("k = 10", t.rodape); self.assertIn("sobram 60", t.rodape)
        self.assertIn("\\begin{tabular}", t.tex())
        self.assertIsNone(gt.tab_g_passos_2_a_5({"g_passos_2_a_5": df}, "outro-modelo"))
        self.assertIsNone(gt.tab_g_passos_2_a_5({}, "gpt-4o-mini"))
        # entra no catálogo e carregar_tudo lê o CSV
        df.to_csv(os.path.join(self.tmp, "bloco3_g_passos_2_a_5_20261013_000000.csv"), index=False)
        D2 = gt.carregar_tudo(__import__("pathlib").Path(self.tmp))
        self.assertIn("g_passos_2_a_5", D2)
        slugs = [x.slug for x in gt.catalogo(D2, None)]
        self.assertIn("bloco3_G_passos_2_a_5", slugs)

    def test_h_figura_bloco3_03(self):
        rows = _rows_proto_g()
        fname = llm_asset(self.tmp, "bloco3_graphic_03_G_passos_2_a_5_curva.png", "gpt-4o-mini")
        plots.plot_g_passos_2_a_5_curva(rows, filename=fname)
        plots.plot_g_passos_2_a_5_curva(rows, filename=llm_asset(
            self.tmp, "bloco3_graphic_03_G_passos_2_a_5_curva_nnls.png", "gpt-4o-mini"), algorithm="nnls")
        arquivos = sorted(os.listdir(self.tmp))
        esperados = ["bloco3_graphic_03_G_passos_2_a_5_curva__gpt4mini_color.png",
                     "bloco3_graphic_03_G_passos_2_a_5_curva__gpt4mini_gray.png",
                     "bloco3_graphic_03_G_passos_2_a_5_curva_g1_a_b_x1_x2__gpt4mini_color.png",
                     "bloco3_graphic_03_G_passos_2_a_5_curva_g1_a_b_x1_x2__gpt4mini_gray.png",
                     "bloco3_graphic_03_G_passos_2_a_5_curva_g1_mulher_homem_peso_altura__gpt4mini_color.png",
                     "bloco3_graphic_03_G_passos_2_a_5_curva_g1_mulher_homem_peso_altura__gpt4mini_gray.png",
                     "bloco3_graphic_03_G_passos_2_a_5_curva_g2_mulher_homem_peso_kg_altura_m__gpt4mini_color.png",
                     "bloco3_graphic_03_G_passos_2_a_5_curva_g2_mulher_homem_peso_kg_altura_m__gpt4mini_gray.png",
                     "bloco3_graphic_03_G_passos_2_a_5_curva_nnls__gpt4mini_color.png",
                     "bloco3_graphic_03_G_passos_2_a_5_curva_nnls__gpt4mini_gray.png"]
        for e in esperados:
            self.assertIn(e, arquivos)
        self.assertTrue(all(re.match(r"^bloco3_graphic_03_G_passos_2_a_5_curva(_[a-z0-9_]+)?__gpt4mini_(color|gray)\.png$", a)
                            for a in arquivos), arquivos)
        # sem as colunas do G (linhas do A/D) a função não desenha nada
        plots.plot_g_passos_2_a_5_curva([{"problem": "A", "algorithm": "perceptron"}], filename=os.path.join(self.tmp, "x.png"))
        self.assertNotIn("x_color.png", os.listdir(self.tmp))


if __name__ == "__main__":
    unittest.main()
