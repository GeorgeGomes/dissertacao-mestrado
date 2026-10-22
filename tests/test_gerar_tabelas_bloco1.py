"""Testes do gerador de tabelas LaTeX dos Passos 2 a 6 do Problema A (`gerar_tabelas_bloco1`) e do catálogo.

CSVs sintéticos mínimos numa pasta temporária, nos DOIS esquemas de coluna:
- "antigo" (execuções até 09/10/2026): teto gravado em `fidelity_vs_llm`, `accuracy_gt_metric_vs_llm_zero`,
  oráculo com `fidelity`, aprendizado em contexto de F/G com `*_llm_vs_metric`, síntese cruzada sem `feature_names`;
- "novo" (contrato de 10/10/2026): `accuracy_teto_vs_true`, `fidelity_teto_vs_llm_passo2`, oráculo com
  `accuracy`, `*_llm_vs_expert` + `especialista_origem`, síntese cruzada com `problem_name` e `feature_names`.

Verifica: os .tex esperados existem e usam booktabs; `_col` resolve o nome certo; a linha do teto mostra "—"
em Fid. (vs LLM) e o valor em Acur. vs real; a eleição da melhor métrica vem de `eleita_passo6` do CSV de
transferência (regra única do runner); a coluna invariante à permutação max(a, 1−a) aparece nas tabelas de
pares invertidos e de classes neutras; `tab_final_cross` abre uma linha por variante de atributo quando há
`feature_names`; NaN vira "—"; num colapso a acurácia do LLM e a partição também viram "—" (não 50,0 e
150/0); o filtro de k das tabelas de viés segue os k_exemplos presentes no CSV; o módulo do Bloco 1 não
sobrescreve `todas_as_tabelas.tex`.

Execução: .venv/bin/python -m unittest tests.test_gerar_tabelas_bloco1
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import gerar_tabelas_bloco1 as gt  # noqa: E402
import gerar_tabelas as gt2  # noqa: E402

NAN = float("nan")
MODEL = "openai/gpt-4o-mini"


def _rows_a(model, esquema="novo", melhor_k=10):
    """Passos 2 a 5 de A e D (R²/R³/R⁴). Fidelidade = 0,80 + 0,005·k + 0,01·(nf−2); teto = 0,98."""
    rows = []
    for prob, nfs in (("A", (2, 3, 4)), ("D", (2, 3, 4))):
        for nf in nfs:
            for regime, k, reps in (("passo2_zero_shot", 0, [0]), ("passo3_exemplos_rotulo_real", 4, [0, 1]), ("passo3_exemplos_rotulo_real", 10, [0, 1]),
                                    ("passo5_teto", 0, [0])):
                for rep in reps:
                    for seed in (42, 123, 7):
                        for algo in ("perceptron", "nnls"):
                            collapsed = int(prob == "D" and regime == "passo2_zero_shot" and seed == 123)
                            gt_ = regime == "passo5_teto"
                            fid = NAN if collapsed else 0.8 + 0.005 * k + 0.01 * (nf - 2)
                            r = {
                                "provider": "openrouter", "model": model, "temperature": 0.0, "random_seed": seed,
                                "problem": prob, "n_features": nf, "regime": regime, "k": k, "rep": rep,
                                "random_state_sorteio": seed + rep, "algorithm": algo, "collapsed": collapsed,
                                "n_total": 150, "n_examples": k, "n_eval": 150 - k,
                                "n_classe_0_llm": 150 if collapsed else 70, "n_classe_1_llm": 0 if collapsed else 80 - k,
                                "llm_accuracy_vs_true": NAN if gt_ else (0.3 if k == 0 else 0.9),
                                "accuracy_metric_vs_true_eval": NAN if collapsed else 0.7,
                                "accuracy_metric_vs_true_full": NAN if collapsed else (0.98 if gt_ else 0.72),
                                "n_errors_metric_vs_true_eval": NAN if collapsed else 40,
                                "w_0": NAN if collapsed else 0.6, "w_1": NAN if collapsed else 0.8,
                                "w_2": NAN if (collapsed or nf < 3) else 0.1, "w_3": NAN if (collapsed or nf < 4) else 0.05,
                                "w_unit_0": NAN if collapsed else 0.6, "w_unit_1": NAN if collapsed else 0.8,
                                "w_unit_2": NAN if (collapsed or nf < 3) else 0.1, "w_unit_3": NAN if (collapsed or nf < 4) else 0.05,
                                "w_ratio": NAN if collapsed else 0.75,
                                # Perceptron em melhor esforço (γ = 0) só no 4-shot
                                "gamma": (0.0 if k == 4 else 0.1) if algo == "perceptron" else NAN,
                                "cos_sim_perc_nnls": 0.99, "n_malformed": 0, "example_indices": "",
                            }
                            if esquema == "novo":
                                r.update(fidelity_vs_llm=NAN if gt_ else fid,
                                         accuracy_teto_vs_true=0.98 if gt_ else NAN,
                                         fidelity_teto_vs_llm_passo2=0.5 if gt_ else NAN)
                            else:   # antigo: o teto ficava gravado em fidelity_vs_llm
                                r.update(fidelity_vs_llm=0.98 if gt_ else fid,
                                         accuracy_gt_metric_vs_llm_zero=0.5 if gt_ else NAN)
                            rows.append(r)
    return rows


def _rows_bc(model, esquema="novo", melhor=("passo3_exemplos_rotulo_real", 10)):
    rows = []
    for prob in ("B", "C"):
        for regime, k, reps in (("passo2_zero_shot", 0, [0]), ("passo3_exemplos_rotulo_real", 4, [0, 1]), ("passo3_exemplos_rotulo_real", 10, [0, 1]), ("passo5_teto", 0, [0])):
            for rep in reps:
                for seed in (42, 123, 7):
                    for algo in ("perceptron", "nnls"):
                        r = {"provider": "openrouter", "model": model, "temperature": 0.0, "random_seed": seed,
                             "problem": prob, "source_problem": "A", "metrica_regime": regime, "metrica_k": k,
                             "metrica_rep": rep, "n_features": 2, "algorithm": algo, "collapsed_metrica": 0,
                             "consistency": 0.8, "kappa": 0.6, "f1": 0.8, "n_agreements": 80, "n_disagreements": 20,
                             "n_eval": 100, "n_classe_0_llm": 50, "n_classe_1_llm": 50, "llm_accuracy_vs_true": 0.2,
                             "metrica_fidelidade_a": 0.85, "metric_accuracy_vs_true": 0.7,
                             "eleita_passo6": int((regime, k) == melhor)}
                        if esquema == "antigo":
                            r["euclidean_consistency"] = 0.7
                        rows.append(r)
    return rows


def _rows_abc(model, esquema="novo"):
    """Problemas A, B e C: configuração base + par invertido (B/A) + nomes de atributo + variante de prompt."""
    rows = []
    for (c0, c1, f0, f1, pv) in (("A", "B", "x1", "x2", "default"), ("B", "A", "x1", "x2", "default"),
                                 ("A", "B", "altura", "peso", "default"), ("A", "B", "x1", "x2", "cot")):
        for k_exemplos in (0, 10):
            if f0 != "x1" and k_exemplos:
                continue
            for seed in (42, 123, 7):
                inv = (c0, c1) == ("B", "A")
                r = {"collapsed": 0, "provider": "openrouter", "model": model, "temperature": 0.0, "random_seed": seed,
                     "k_passo4_bc": k_exemplos, "class_0": c0, "class_1": c1, "feature_0": f0, "feature_1": f1,
                     "prompt_variant": pv, "rep": 0, "fidelity_problem_a": 0.85,
                     "consistency_problem_b": 0.8, "kappa_problem_b": 0.6, "f1_problem_b": 0.8,
                     "consistency_problem_c": 0.7, "kappa_problem_c": 0.4, "f1_problem_c": 0.7,
                     "llm_accuracy_problem_a": 0.25 if inv else 0.75, "llm_accuracy_problem_b": 0.5,
                     "llm_accuracy_problem_c": 0.5, "metric_accuracy_problem_b": 0.5, "metric_accuracy_problem_c": 0.5,
                     "w_0": 0.6, "w_1": 0.8, "w_ratio": 0.75, "w_direction_0": 0.6, "w_direction_1": 0.8,
                     "w_cosine_sim_nnls": 0.99, "gamma": 0.0 if seed == 7 else 0.1, "n_disagreements_b": 20,
                     "n_disagreements_c": 30, "n_malformed_responses": 0}
                if esquema == "antigo":
                    r.update(euclidean_consistency_b=0.7, euclidean_consistency_c=0.7, diagonal_limitation_flag=0)
                rows.append(r)
    return rows


def _rows_algo(model, esquema="novo"):
    """bloco1_a_passo4_perceptron_vs_nnls: NNLS zero-shot por semente (antigo: recoleta, com euclidean_consistency_*)."""
    rows = []
    for seed in (42, 123, 7):
        r = {"algorithm": "NNLS", "provider": "openrouter", "model": model, "temperature": 0.0, "random_seed": seed,
             "k_passo4_bc": 0, "class_0": "A", "class_1": "B", "rep": 0, "fidelity_problem_a": 0.83,
             "consistency_problem_b": 0.78, "kappa_problem_b": 0.55, "f1_problem_b": 0.78, "consistency_problem_c": 0.7,
             "kappa_problem_c": 0.4, "f1_problem_c": 0.7, "w_0": 0.6, "w_1": 0.8, "w_ratio": 0.75, "gamma": NAN}
        if esquema == "antigo":
            r.update(euclidean_consistency_b=0.7, euclidean_consistency_c=0.7, diagonal_limitation_flag=0)
        rows.append(r)
    return rows


def _rows_oraculo(esquema="novo"):
    rows = []
    cfgs = (["Oráculo sobre B", "Oráculo sobre E1", "Oráculo sobre E2"]
            + ([f"Oráculo sobre aniso {n}" for n in ("x1_preciso", "x2_preciso", "forte_aniso")] if esquema == "novo"
               else [f"Oráculo sobre aniso {n}" for n in ("x2_dom", "x1_dom", "forte_aniso")]))
    for prob in cfgs:
        for seed in (42, 123):
            for algo in ("perceptron", "nnls"):
                r = {"random_seed": seed, "problem": prob, "expert_name": prob.split()[-1], "true_w_0": 1.0, "true_w_1": 1.0,
                     "true_w_ratio": 1.0, "algorithm": algo, "recovered_w_0": 0.7, "recovered_w_1": 0.7,
                     "recovered_w_norm_0": 0.707, "recovered_w_norm_1": 0.707, "recovered_w_ratio": 1.0,
                     "cosine_similarity": 0.99}
                r["accuracy" if esquema == "novo" else "fidelity"] = 0.97
                rows.append(r)
    return rows


def _rows_ext_e(model, esquema="novo"):
    rows = []
    for pb in ("F_meia_lua_seed42", "G1_homem_mulher_classesAB", "G2_homem_mulher"):
        feat = "x1/x2" if pb.startswith("F") else ("x1/x2" if "classesAB" in pb else "peso_kg/altura_m")
        for k_exemplos in (0, 4):
            r = {"problem_name": pb, "seed": 42, "rep": 0, "feature_names": feat, "n_features": 2, "k": k_exemplos,
                 "accuracy_llm_vs_true": 0.15 if "classesAB" in pb else 0.85,
                 "accuracy_perceptron_baseline_vs_true": 0.8, "n_test": 45 if pb.startswith("F") else 30,
                 "n_errors_llm_vs_true": 5, "n_malformed": 0, "provider": "openrouter", "model": model}
            # bloco23_fg_aprendiz é lido SÓ no contrato de 13/10/2026 (sem fallback: migra-se a pasta antes),
            # por isso o esquema "antigo" também grava as colunas novas aqui.
            r.update({"accuracy_llm_vs_expert": 0.9, "kappa_llm_vs_expert": 0.8, "f1_llm_vs_expert": 0.9,
                      "n_errors_llm_vs_expert": 3,
                      "especialista_origem": "rotulo_real" if pb.startswith("F") else "metrica", "example_indices": ""})
            rows.append(r)
    return rows


def _rows_cross(esquema="novo"):
    rows = []
    for pb, feat in (("A_linear", "x1/x2"), ("D_meia_lua_seed42", "x1/x2"), ("G2_homem_mulher", "peso_kg/altura_m"),
                     ("G2_homem_mulher", "x1/x2")):
        for algo in ("perceptron", "nnls"):
            for nf in (2, 3, 4):
                r = {"problem_base": pb.split("_")[0], "is_nonlinear": int(not pb.startswith("A")), "algorithm": algo,
                     "n_features": nf, "fidelity_vs_llm": 0.8 + 0.02 * (nf - 2) + (0.1 if feat == "x1/x2" else 0.0),
                     "accuracy_vs_true": 0.7, "seed": 42}
                if esquema == "novo":
                    r.update(problem_name=pb, problem=pb, feature_names=feat)
                else:
                    r["problem"] = pb
                rows.append(r)
    return rows


def _escrever(tmp: Path, esquema: str, model=MODEL, melhor=("passo3_exemplos_rotulo_real", 10)):
    pd.DataFrame(_rows_a(model, esquema)).to_csv(tmp / "bloco1_ad_passos_2_a_5_20261002_000000.csv", index=False)
    pd.DataFrame(_rows_bc(model, esquema, melhor)).to_csv(tmp / "bloco1_bc_passo3_consistencia_20261002_000000.csv", index=False)
    pd.DataFrame(_rows_abc(model, esquema)).to_csv(tmp / "bloco1_abc_passos_2_a_4_20261002_000000.csv", index=False)
    pd.DataFrame(_rows_algo(model, esquema)).to_csv(tmp / "bloco1_a_passo4_perceptron_vs_nnls_20261002_000000.csv", index=False)
    pd.DataFrame(_rows_oraculo(esquema)).to_csv(tmp / "bloco1_oraculo_passos_1_a_3_20261002_000000.csv", index=False)
    pd.DataFrame(_rows_ext_e(model, esquema)).to_csv(tmp / "bloco23_fg_aprendiz_20261002_000000.csv", index=False)
    pd.DataFrame(_rows_cross(esquema)).to_csv(tmp / "final_cross_linearity__gpt4mini.csv", index=False)


def _ler(tmp: Path, nome: str) -> str:
    return (tmp / "tabelas" / nome).read_text(encoding="utf-8")


def _linha(tex: str, chave: str) -> str:
    return [ln for ln in tex.splitlines() if chave in ln][0]


class TestCol(unittest.TestCase):
    def test_col_resolve_novo_antigo_e_ausente(self):
        df = pd.DataFrame({"accuracy": [1.0], "x": [0]})
        self.assertEqual(gt._col(df, "accuracy", "fidelity"), "accuracy")
        self.assertEqual(gt._col(pd.DataFrame({"fidelity": [1.0]}), "accuracy", "fidelity"), "fidelity")
        self.assertIsNone(gt._col(df, "nada", "tambem_nada"))
        self.assertEqual(gt._col(df.iloc[0], "accuracy", "fidelity"), "accuracy")     # Series (linha)
        self.assertEqual(gt._col({"fidelity": 1}, "accuracy", "fidelity"), "fidelity")  # dict

    def test_agregar_ignora_colunas_ausentes(self):
        df = pd.DataFrame({"g": [1, 1], "a": [0.2, 0.4]})
        ag = gt.agregar(df, ["g"], ["a", "nao_existe"])
        self.assertIn("a_mean", ag.columns)
        self.assertNotIn("nao_existe_mean", ag.columns)
        self.assertEqual(gt.fmt_stat(ag.iloc[0], "nao_existe"), "—")


class _Base(unittest.TestCase):
    esquema = "novo"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _escrever(self.tmp, self.esquema)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestBloco1Novo(_Base):
    esquema = "novo"

    def test_gera_todas_as_tabelas_sem_tocar_no_documento_completo(self):
        escritos = gt.gerar_tabelas(self.tmp, por_regime={
            "openrouter/openai/gpt-4o-mini|A|passo=2|passo2_zero_shot|k=0": {"total_chamadas": 450, "taxa_malformadas": 0.0,
                                                               "queries_repetidas": 150, "taxa_flip_T0": 0.05},
            "openrouter/openai/gpt-4o-mini|-|passo=-|sem_tag|k=-": {"total_chamadas": 10, "taxa_malformadas": 0.0,
                                                             "queries_repetidas": 0, "taxa_flip_T0": 0.0},
        })
        nomes = sorted(f.name for f in escritos)
        self.assertEqual(nomes, sorted(["bloco1_table_01_A_passos_2_a_5__gpt4mini.tex", "bloco1_table_06_BC_passo3_todas_metricas_A__gpt4mini.tex",
                                        "bloco1_table_10_D_passos_2_a_5__gpt4mini.tex", "bloco1_table_13_AD_passo4_estabilidade_W__gpt4mini.tex",
                                        "final_table_04_auditoria_flip_por_regime.tex", "bloco1_table_02_A_passo2_vs_passo3__gpt4mini.tex",
                                        "bloco1_table_03_A_passo6_eleicao__gpt4mini.tex",
                                        "bloco1_table_07_BC_passo3_por_semente__gpt4mini.tex",
                                        "bloco1_table_08_BC_passo3_zero_vs_eleita__gpt4mini.tex", "todas_as_tabelas_bloco1.tex"]))
        self.assertFalse((self.tmp / "tabelas" / "todas_as_tabelas.tex").exists())   # F17: pertence a gerar_tabelas.py
        t1 = _ler(self.tmp, "bloco1_table_01_A_passos_2_a_5__gpt4mini.tex")
        self.assertIn("\\toprule", t1); self.assertIn("\\bottomrule", t1)
        self.assertIn("zero-shot", t1); self.assertIn("10-shot", t1); self.assertIn("teto", t1)
        t2 = _ler(self.tmp, "bloco1_table_06_BC_passo3_todas_metricas_A__gpt4mini.tex")
        self.assertIn("\\textbf{10-shot}", t2)
        self.assertNotIn("\\textbf{zero-shot}", t2)
        self.assertIn("F1-Macro", t2); self.assertNotIn("& F1 &", t2)
        t3 = _ler(self.tmp, "bloco1_table_10_D_passos_2_a_5__gpt4mini.tex")
        self.assertIn("R$^3$", t3)
        t5 = _ler(self.tmp, "final_table_04_auditoria_flip_por_regime.tex")
        self.assertIn("zero\\_shot", t5); self.assertNotIn("sem_tag", t5)
        doc = _ler(self.tmp, "todas_as_tabelas_bloco1.tex")
        self.assertEqual(doc.count("\\input{"), 9)

    def test_linha_do_teto_sem_fidelidade_e_com_acuracia(self):
        gt.gerar_tabelas(self.tmp, por_regime={})
        for nome, chave in (("bloco1_table_01_A_passos_2_a_5__gpt4mini.tex", "teto"),
                            ("bloco1_table_10_D_passos_2_a_5__gpt4mini.tex", "R$^2$ — W ajustado")):
            ln = _linha(_ler(self.tmp, nome), chave)
            cells = [c.strip() for c in ln.split("&")]
            # Regime | Acur. LLM | partição | Fid Perc | Acur Perc | W | Fid NNLS | Acur NNLS | W
            self.assertEqual(cells[3], "—", ln)        # Fid. (vs LLM) do teto = "—"
            self.assertEqual(cells[4], "98,0", ln)     # acurácia (vs rótulo real) do teto
            self.assertEqual(cells[6], "—", ln)
            self.assertEqual(cells[7], "98,0", ln)
        # a linha zero-shot continua com fidelidade e acurácia
        ln = _linha(_ler(self.tmp, "bloco1_table_01_A_passos_2_a_5__gpt4mini.tex"), "zero-shot &")
        cells = [c.strip() for c in ln.split("&")]
        self.assertEqual(cells[3], "80,0"); self.assertEqual(cells[4], "72,0")
        t1 = _ler(self.tmp, "bloco1_table_01_A_passos_2_a_5__gpt4mini.tex")
        self.assertIn("4-shot†", t1)              # γ = 0 no 4-shot (I14)
        self.assertNotIn("10-shot†", t1)
        self.assertIn("max($a$, 1 $-$ $a$)", t1)                 # nota de permutação (I2)
        self.assertIn("Acur.\\ vs real", t1); self.assertNotIn("Métr.", t1)

    def test_eleicao_vem_do_csv_bc_passo3(self):
        gt.gerar_tabelas(self.tmp, por_regime={})
        df_bc = pd.read_csv(self.tmp / "bloco1_bc_passo3_consistencia_20261002_000000.csv")
        self.assertEqual(gt.eleita_passo6(df_bc, MODEL), ("passo3_exemplos_rotulo_real", 10))
        self.assertIsNone(gt.eleita_passo6(df_bc.drop(columns=["eleita_passo6"]), MODEL))
        self.assertIsNone(gt.eleita_passo6(None, MODEL))
        t7 = _ler(self.tmp, "bloco1_table_03_A_passo6_eleicao__gpt4mini.tex")
        self.assertIn("\\textbf{10-shot}", t7); self.assertIn("\\textbf{eleita}", t7)
        self.assertNotIn("\\textbf{4-shot", t7); self.assertIn("eleita\\_passo6", t7)
        t8 = _ler(self.tmp, "bloco1_table_07_BC_passo3_por_semente__gpt4mini.tex")
        self.assertIn("eleita (10-shot)", t8); self.assertIn("seed 42", t8); self.assertIn("F1-Macro", t8)
        t9 = _ler(self.tmp, "bloco1_table_08_BC_passo3_zero_vs_eleita__gpt4mini.tex")
        self.assertIn("B & perceptron & 80,0 & 80,0 & +0,0 & 0,600 & 0,600 & +0,000 & substancial & substancial", t9)
        self.assertIn("F1-Macro (\\%)", t9); self.assertIn("Cohen's $\\kappa$", t9)
        t6 = _ler(self.tmp, "bloco1_table_02_A_passo2_vs_passo3__gpt4mini.tex")
        self.assertIn("perceptron & 4-shot & 80,0", t6)
        self.assertIn("& 82,0† & +2,0 &", t6)    # fid 4-shot = 82,0 com γ = 0, Δ = +2,0

    def test_eleicao_diferente_do_menor_k_e_zero_shot_eleito(self):
        # a regra do runner manda: eleita = 4-shot, mesmo com fidelidade menor que a do 10-shot na fixture
        _escrever(self.tmp, self.esquema, melhor=("passo3_exemplos_rotulo_real", 4))
        gt.gerar_tabelas(self.tmp, por_regime={})
        t7 = _ler(self.tmp, "bloco1_table_03_A_passo6_eleicao__gpt4mini.tex")
        self.assertIn("\\textbf{4-shot", t7); self.assertNotIn("\\textbf{10-shot}", t7)
        self.assertIn("eleita (4-shot)", _ler(self.tmp, "bloco1_table_07_BC_passo3_por_semente__gpt4mini.tex"))
        # zero-shot eleito: T7 marca o zero-shot, T9 fica "indisponível" com o motivo
        _escrever(self.tmp, self.esquema, melhor=("passo2_zero_shot", 0))
        gt.gerar_tabelas(self.tmp, por_regime={})
        t7 = _ler(self.tmp, "bloco1_table_03_A_passo6_eleicao__gpt4mini.tex")
        self.assertIn("\\textbf{zero-shot", t7)
        t9 = _ler(self.tmp, "bloco1_table_08_BC_passo3_zero_vs_eleita__gpt4mini.tex")
        self.assertIn("indisponível", t9); self.assertIn("própria zero-shot", t9)
        self.assertIn("zero-shot (eleita)", _ler(self.tmp, "bloco1_table_07_BC_passo3_por_semente__gpt4mini.tex"))

    def test_catalogo_vieses_oraculo_externos_cross(self):
        D = gt2.carregar_tudo(self.tmp); D["_pasta"] = self.tmp
        tabs = {t.slug: t for t in gt2.catalogo(D, None)}
        # 01_14: coluna invariante à permutação; par invertido B/A com 25% → 75%
        v = tabs["bloco1_vies_nomes_classe"]
        self.assertIn("Acur.\\ vs real (inv.\\ perm.)", v.headers)
        ba = [r for r in v.rows if r[0] == "B / A" and r[1] == "zero-shot"][0]
        self.assertEqual(ba[3], "25,0"); self.assertEqual(ba[4], "75,0")
        self.assertIn("vêm do Problema A (Passo 4 sobre os rótulos do Passo 2), sempre zero-shot", v.rodape)
        self.assertIn("centróides locais", v.rodape)
        # 01_15: só zero-shot, sem coluna invariante
        at = tabs["bloco1_vies_nomes_atributo"]
        self.assertTrue(all(r[1] == "zero-shot" for r in at.rows)); self.assertNotIn("inv.\\ perm.", " ".join(at.headers))
        # 01_17: oráculo lê `accuracy`, mantém Problema 1/2/3 e IGNORA as configurações extras
        # (gaussianas anisotrópicas de execuções anteriores a 13/10/2026)
        o = tabs["bloco1_oraculo_passos_1_a_3"]
        self.assertEqual(o.rows[0][0], "Problema 1"); self.assertEqual(o.rows[0][-1], "97,0")
        probs = [r[0] for r in o.rows[::2]]
        self.assertEqual(probs, ["Problema 1", "Problema 2", "Problema 3"])
        self.assertNotIn("anisotrópic", o.rodape)
        self.assertIn("Problema 1 = Oráculo sobre B", o.rodape); self.assertNotIn("euclidiana", o.rows[0][1])
        # 02_10: aprendizado em contexto do F (Passo 3) lê *_llm_vs_expert e especialista_origem
        f = tabs["bloco2_F_passo3_k"]
        self.assertIn("rótulo real (coluna especialista\\_origem = rotulo\\_real)", f.rodape)
        self.assertEqual(f.rows[0][3], "90,0")           # acurácia (LLM vs especialista), n = 0
        self.assertIn("Perceptron diagonal", f.rodape); self.assertIn("Acur.\\ LLM vs espec.", " ".join(g[0] for g in f.grupos))
        # 03_02: coluna "Prompt" (atributos mostrados ao LLM, 14/10/2026) e coluna invariante nas linhas
        # classesAB (15% → 85%), "—" nas demais
        g = tabs["bloco3_G_passo6_k"]
        self.assertEqual(g.headers[3], "Prompt")
        ab = [r for r in g.rows if r[1] == "A / B"][0]; mh = [r for r in g.rows if r[1] == "Mulher / Homem"][0]
        self.assertTrue(ab[3].startswith("R$^")); self.assertTrue(ab[4].startswith("15,0")); self.assertEqual(ab[6], "85,0")
        self.assertEqual(mh[6], "—")
        # final_01: uma linha por variante de atributo, meia-lua do D
        c = tabs["final_linearidade_cruzada_R2_R3_R4"]
        self.assertIn("Atributos", c.headers)
        g2 = [r for r in c.rows if r[0] == "G2\\_homem\\_mulher"]
        self.assertEqual({r[1] for r in g2}, {"x1/x2", "peso\\_kg/altura\\_m"})
        self.assertEqual([r for r in g2 if r[1] == "x1/x2" and r[2] == "perceptron"][0][3], "90,0")
        self.assertEqual([r for r in g2 if r[1] != "x1/x2" and r[2] == "perceptron"][0][3], "80,0")
        self.assertIn("meia-lua = Problema D", c.rodape); self.assertNotIn("média sobre as variantes", c.rodape)
        # Bloco 2 (01_05 pareado): rodapé do novo esquema
        self.assertIn("PAREADAS", tabs["bloco1_A_passo4_perceptron_vs_nnls"].rodape)
        pn = tabs["bloco1_A_passo4_perceptron_vs_nnls"]
        self.assertIn("†", [r for r in pn.rows if r[0] == "7"][0][1])      # seed 7 tem γ = 0
        self.assertNotIn("†", [r for r in pn.rows if r[0] == "42"][0][1])
        self.assertEqual([r for r in pn.rows if r[0] == "42"][0][2], "83,0")          # NNLS do CSV de comparação
        # nenhum cabeçalho com os nomes proibidos
        for t in tabs.values():
            for h in t.headers:
                self.assertNotIn("Métr.", h, (t.slug, h)); self.assertNotIn("Concord", h, (t.slug, h))
                self.assertNotRegex(h, r"(^|[^-])F1($|[^-])", (t.slug, h))
                self.assertFalse(h.startswith("LLM vs real"), (t.slug, h))   # acurácia sempre nomeada como tal


class TestBloco1Antigo(_Base):
    """Mesmos CSVs no esquema antigo (execuções até 09/10/2026): tudo roda por fallback, sem exceção."""
    esquema = "antigo"

    def test_teto_por_fallback_e_eleicao(self):
        gt.gerar_tabelas(self.tmp, por_regime={})
        ln = _linha(_ler(self.tmp, "bloco1_table_01_A_passos_2_a_5__gpt4mini.tex"), "teto")
        cells = [c.strip() for c in ln.split("&")]
        self.assertEqual(cells[3], "—"); self.assertEqual(cells[4], "98,0")   # teto vindo de fidelity_vs_llm (antigo)
        self.assertIn("\\textbf{10-shot}", _ler(self.tmp, "bloco1_table_03_A_passo6_eleicao__gpt4mini.tex"))

    def test_catalogo_antigo_por_fallback(self):
        D = gt2.carregar_tudo(self.tmp); D["_pasta"] = self.tmp
        tabs = {t.slug: t for t in gt2.catalogo(D, None)}
        o = tabs["bloco1_oraculo_passos_1_a_3"]
        self.assertEqual(o.rows[0][-1], "97,0")          # lê `fidelity`
        self.assertEqual([r[0] for r in o.rows[::2]], ["Problema 1", "Problema 2", "Problema 3"])   # extras antigas ignoradas
        f = tabs["bloco2_F_passo3_k"]
        self.assertEqual(f.rows[0][3], "90,0")           # lê accuracy_llm_vs_expert (sem fallback desde 13/10/2026)
        self.assertIn("rotulo\\_real", f.rodape)
        c = tabs["final_linearidade_cruzada_R2_R3_R4"]
        self.assertIn("média sobre as variantes", c.rodape)
        self.assertEqual(len([r for r in c.rows if r[0] == "G2\\_homem\\_mulher"]), 2)   # uma média por algoritmo
        self.assertIn("recoleta", tabs["bloco1_A_passo4_perceptron_vs_nnls"].rodape)
        ba = [r for r in tabs["bloco1_vies_nomes_classe"].rows if r[0] == "B / A" and r[1] == "zero-shot"][0]
        self.assertEqual(ba[4], "75,0")


class TestViesKDinamico(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_k_vem_do_csv_e_referencia_so_entra_nos_k_das_variantes(self):
        rows = _rows_abc(MODEL, "novo")
        for r in rows:
            if r["k_passo4_bc"] == 10:
                r["k_passo4_bc"] = 4          # smoke: K_VIESES = [0, 4]
        # a referência (A/B, x1/x2, default) roda a grade inteira; os k extras não entram na tabela
        for k in (20, 40):
            for seed in (42, 123, 7):
                r = dict(rows[0]); r.update(k_passo4_bc=k, random_seed=seed); rows.append(r)
        pd.DataFrame(rows).to_csv(self.tmp / "bloco1_abc_passos_2_a_4_20261009_000000.csv", index=False)
        D = gt2.carregar_tudo(self.tmp); D["_pasta"] = self.tmp
        tabs = {t.slug: t for t in gt2.catalogo(D, None)}
        for slug in ("bloco1_vies_nomes_classe", "bloco1_vies_variantes_prompt"):
            t = tabs[slug]
            self.assertEqual(sorted(set(r[1] for r in t.rows)), ["4-shot", "zero-shot"], slug)
            self.assertIn("4-shot: B e C, Passo 4, exemplos rotulados pela métrica zero-shot do A (H2)", t.rodape)
            self.assertIn("(iguais nas linhas zero e 4-shot)", t.rodape)
            self.assertNotIn("10-shot", t.rodape)
        at = tabs["bloco1_vies_nomes_atributo"]
        self.assertTrue(all(r[1] == "zero-shot" for r in at.rows)); self.assertNotIn("-shot:", at.rodape)


class TestColapso(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_nan_total_vira_travessao_particao_e_linha_de_colapso_na_T4(self):
        # modelo em que o zero-shot de D colapsa em TODAS as sementes
        rows = _rows_a("google/gemini-2.5-flash-lite", "novo")
        for r in rows:
            if r["problem"] == "D" and r["regime"] == "passo2_zero_shot":
                r.update(collapsed=1, fidelity_vs_llm=NAN, accuracy_metric_vs_true_full=NAN, w_ratio=NAN,
                         n_classe_0_llm=150, n_classe_1_llm=0, w_unit_0=NAN, w_unit_1=NAN)
        pd.DataFrame(rows).to_csv(self.tmp / "bloco1_ad_passos_2_a_5_20261002_000001.csv", index=False)
        gt.gerar_tabelas(self.tmp, por_regime={})
        t3 = _ler(self.tmp, "bloco1_table_10_D_passos_2_a_5__flashlite.tex")
        # colapso em todas as sementes: acurácia do LLM e partição viram "—" (não 50,0 e 150/0)
        self.assertNotIn("150/0", t3)
        zs = _linha(t3, "R$^2$ — zero-shot")
        self.assertTrue(zs.startswith("R$^2$ — zero-shot & — & — & — & —"), zs)
        self.assertIn("também a acurácia do LLM e a partição", t3)
        # bloco1_table_11 (zero vs few): coluna "zero" da acurácia do LLM também "—"
        D = gt2.carregar_tudo(self.tmp); D["_pasta"] = self.tmp
        t11 = {t.slug: t for t in gt2.catalogo(D, None)}["bloco1_D_passo2_vs_passo3"]
        r2 = [r for r in t11.rows if r[0] == "R$^2$"][0]
        self.assertEqual(r2[8], "—"); self.assertNotEqual(r2[9], "—")
        t4 = _ler(self.tmp, "bloco1_table_13_AD_passo4_estabilidade_W__flashlite.tex")
        ln = [ln for ln in t4.splitlines() if ln.startswith("D & Perceptron & zero-shot")]
        self.assertEqual(len(ln), 1, "a linha de colapso deve aparecer (dropna=False)")
        self.assertIn("— & — & — & —", ln[0])
        # bloco1_table_12 (A × D por passo): a acurácia do LLM colapsado também vira "—" e o rodapé explica (13/10/2026)
        t12 = {t.slug: t for t in gt2.catalogo(D, None)}["bloco1_AD_por_passo"]
        zs12 = [r for r in t12.rows if r[0].startswith("zero-shot")][0]
        self.assertEqual(zs12[6], "—"); self.assertIn("colapso", t12.rodape)
        self.assertFalse((self.tmp / "tabelas" / "final_table_04_auditoria_flip_por_regime.tex").exists())
        self.assertIn("quase perfeita", gt.faixa_kappa(0.85)); self.assertEqual(gt.faixa_kappa(-0.1), "pior que o acaso")
        self.assertEqual(gt.faixa_kappa(float("nan")), "—")


# ───────────────────────── nomenclatura por passos (13/10/2026) ─────────────────────────
TERMOS_ANTIGOS = ["Fase A", "Fase B", "Fase C", "Fase E", "Fases", "pipeline externo", "protocolo único",
                  "rotocolo da Fase", "Phase A", "Phase E"]
MODEL2 = "google/gemini-2.5-flash-lite"


def _rows_bloco2(model):
    """`bloco2_e_passos_2_3`, `bloco2_e_passo4_diluicao`, `bloco2_e_passo5_ordem` e `bloco2_e_passo6_baselines` mínimos."""
    fe, dil, ordem, bl = [], [], [], []
    for ex, w in (("E1", (0.3, 1.5)), ("E2", (1.5, 0.3)), ("E3", (1.0, 1.0))):
        for seed in (42, 7):
            for n in (0, 4, 10):
                for st in ("easy", "hard", "mixed", "random"):
                    fe.append({"provider": "openrouter", "model": model, "random_seed": seed, "rep": 0, "k": n,
                               "example_strategy": st, "expert_name": ex, "expert_w_0": w[0], "expert_w_1": w[1],
                               "accuracy_llm_vs_expert": 0.6 + 0.01 * n + (0.05 if st == "easy" else 0.0),
                               "kappa_llm_vs_expert": 0.4, "f1_llm_vs_expert": 0.7, "accuracy_expert_vs_gt": 0.9,
                               "n_malformed": 0})
            for n in (4, 10):
                for clf in ("k-NN (k=3)", "Logistic Regression", "SVM (RBF)"):
                    bl.append({"provider": "classical", "model": clf, "random_seed": seed, "k": n,
                               "example_strategy": "mixed", "expert_name": ex, "accuracy_vs_expert": 0.7,
                               "kappa_vs_expert": 0.4, "f1_vs_expert": 0.7, "accuracy_vs_gt": 0.7})
    for seed in (42, 7):
        for m in (0, 2, 4):
            dil.append({"provider": "openrouter", "model": model, "random_seed": seed, "k": 4 + m,
                        "example_strategy": f"dilution_4hard_{m}easy", "accuracy_llm_vs_expert": 0.7,
                        "kappa_llm_vs_expert": 0.4, "f1_llm_vs_expert": 0.7})
        for n in (4, 10):
            for o in ("class0_first", "class1_first", "shuffled", "alternating"):
                ordem.append({"provider": "openrouter", "model": model, "random_seed": seed, "k": n,
                              "example_ordering": o, "accuracy_llm_vs_expert": 0.75, "kappa_llm_vs_expert": 0.5})
    return fe, dil, ordem, bl


def _rows_ext_a(model):
    rows = []
    for pb, feats in (("F_meia_lua_seed42", ["x1/x2"]), ("G1_homem_mulher", ["x1/x2", "peso/altura"]),
                      ("G1_homem_mulher_classesAB", ["x1/x2"]), ("G2_homem_mulher", ["x1/x2", "peso_kg/altura_m"])):
        for feat in feats:
            for nf in (2, 3, 4):
                rows.append({"provider": "openrouter", "model": model, "problem_name": pb, "feature_names": feat,
                             "seed": 42, "n_features": nf, "n_samples": 105 if pb.startswith("F") else 70,
                             "llm_accuracy_vs_true": 0.2 if "classesAB" in pb else 0.85, "n_errors_llm_vs_true": 10,
                             "fidelity_perc_vs_llm": 0.9, "fidelity_nnls_vs_llm": 0.88, "accuracy_perc_vs_true": 0.8,
                             "n_errors_perc_vs_true": 14, "accuracy_nnls_vs_true": 0.79})
    return rows


def _rows_proto_g(model):
    rows = []
    for sp, cn, fn in (("G1", "Mulher/Homem", "peso/altura"), ("G2", "A/B", "x1/x2")):
        for r in _rows_a(model, "novo"):
            if r["problem"] != "A":
                continue
            r = dict(r); r.update(problem=f"{sp}_homem_mulher", problem_name=f"{sp}_homem_mulher", subproblem=sp,
                                  class_names=cn, feature_names=fn, n_total=70)
            rows.append(r)
    return rows


class TestNomenclaturaPassosTabelas(unittest.TestCase):
    """Nomenclatura por problema e passo (13/10/2026) em TODO o catálogo: nenhuma caption, rodapé,
    cabeçalho ou grupo cita as fases antigas, e toda caption diz de que problema e passo(s) a tabela
    trata ("Problema A, Passos 2 a 5", "Problemas B e C, Passo 3", "Problema E1, Passo 4"…)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _escrever(self.tmp, "novo")
        # dois modelos no protocolo (comparação de modelos) + Bloco 2, Bloco 3 e teto da meia-lua
        pd.DataFrame(_rows_a(MODEL, "novo") + _rows_a(MODEL2, "novo")).to_csv(
            self.tmp / "bloco1_ad_passos_2_a_5_20261002_000000.csv", index=False)
        fe, dil, ordem, bl = _rows_bloco2(MODEL)
        pd.DataFrame(fe).to_csv(self.tmp / "bloco2_e_passos_2_3_20261002_000000.csv", index=False)
        pd.DataFrame(dil).to_csv(self.tmp / "bloco2_e_passo4_diluicao_20261002_000000.csv", index=False)
        pd.DataFrame(ordem).to_csv(self.tmp / "bloco2_e_passo5_ordem_20261002_000000.csv", index=False)
        pd.DataFrame(bl).to_csv(self.tmp / "bloco2_e_passo6_baselines_20261002_000000.csv", index=False)
        pd.DataFrame(_rows_ext_a(MODEL)).to_csv(self.tmp / "bloco23_fg_passos_2_e_4_20261002_000000.csv", index=False)
        pd.DataFrame(_rows_proto_g(MODEL)).to_csv(self.tmp / "bloco3_g_passos_2_a_5_20261002_000000.csv", index=False)
        pd.DataFrame([{"random_seed": s, "n_features": nf, "algorithm": a, "accuracy_vs_true": 0.9, "n_errors": 15,
                       "w_recovered_0": 1.0, "w_recovered_1": 0.8, "w_recovered_2": NAN if nf < 3 else 0.1,
                       "w_recovered_3": NAN if nf < 4 else 0.05}
                      for s in (42, 7) for nf in (2, 3, 4) for a in ("perceptron", "nnls")]).to_csv(
            self.tmp / "bloco1_d_passo5_teto_20261002_000000.csv", index=False)
        stats = {"total_chamadas": 1000, "malformadas": 1, "taxa_malformadas": 0.001, "queries_repetidas": 100,
                 "queries_divergentes": 2, "taxa_flip_T0": 0.02, "model_resolved": {"m-2026": 1000},
                 "inference_provider": {"OpenAI": 1000}}
        self.audit = {"por_modelo": {f"openrouter/{MODEL}": dict(stats), f"openrouter/{MODEL2}": dict(stats)},
                      "por_regime": {f"openrouter/{MODEL}|A|passo=2|passo2_zero_shot|k=0": dict(stats),
                                     f"openrouter/{MODEL}|A|passo=3|passo3_exemplos_rotulo_real|k=10": dict(stats)}}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_catalogo_inteiro_cita_problema_e_passo(self):
        D = gt2.carregar_tudo(self.tmp); D["_pasta"] = self.tmp
        tabs = gt2.catalogo(D, self.audit)
        slugs = {t.slug for t in tabs}
        for esperado in ("bloco1_A_passos_2_a_5", "bloco1_BC_passo4_exemplos_da_metrica", "bloco1_D_passos_2_a_5",
                         "bloco1_AD_por_passo", "bloco1_vies_variantes_prompt", "bloco1_oraculo_passos_1_a_3",
                         "bloco1_D_passo5_teto", "bloco2_E1_passo3_estrategia_x_k",
                         "bloco2_E_passo2_vs_passo3", "bloco2_E_passo3_H4",
                         "bloco2_E_passo6_baselines", "bloco2_E1_passo4_diluicao", "bloco2_E1_passo5_ordem",
                         "bloco2_F_passo2_R2_R3_R4", "bloco2_F_passo3_k", "bloco3_G_passos_2_e_4_R2_R3_R4",
                         "bloco3_G_passo6_k", "bloco3_G_passo7_LLM_vs_perceptron", "bloco3_G1_vs_G2_passos_2_e_4",
                         "bloco3_G_nomes_semanticos_vs_neutros", "bloco3_G_passos_2_a_5",
                         "final_linearidade_cruzada_R2_R3_R4", "final_comparacao_modelos", "final_auditoria_por_modelo",
                         "final_auditoria_flip_por_regime"):
            self.assertIn(esperado, slugs, esperado)
        self.assertGreaterEqual(len(tabs), 40)
        for t in tabs:
            textos = [t.caption, t.rodape or ""] + list(t.headers) + [g[0] for g in t.grupos]
            ruins = [s for s in textos for p in TERMOS_ANTIGOS if p in s]
            self.assertEqual(ruins, [], f"{t.slug} ({t.modelo}): nomenclatura antiga em {ruins}")
            self.assertIn("Passo", t.caption, f"{t.slug} ({t.modelo}): caption sem 'Passo': {t.caption!r}")
            self.assertTrue(t.rodape and t.rodape.strip(), t.slug)
        # a tabela "indisponível" do comparativo também cita o passo
        _escrever(self.tmp, "novo", melhor=("passo2_zero_shot", 0))
        D = gt2.carregar_tudo(self.tmp); D["_pasta"] = self.tmp
        t9 = {t.slug: t for t in gt2.catalogo(D, None)}["bloco1_BC_passo3_zero_vs_eleita"]
        self.assertIn("Passo 3", t9.caption); self.assertIn("indisponível", t9.caption)

    def test_indice_e_documento_sem_nomenclatura_antiga(self):
        gt2.gerar_tudo(self.tmp, png=False, docx=False, audit=self.audit, verbose=False)
        indice = (self.tmp / "tabelas" / "INDICE.md").read_text(encoding="utf-8")
        ruins = [ln for ln in indice.splitlines() for p in TERMOS_ANTIGOS if p in ln]
        self.assertEqual(ruins, [])
        self.assertIn("Passo", indice)
        self.assertNotIn("Fase", gt.documento([]))
        for mod in (gt, gt2):
            cabecalho = mod.__doc__
            ruins = [ln for ln in cabecalho.splitlines() for p in TERMOS_ANTIGOS if p in ln and "ntiga" not in ln]
            self.assertEqual(ruins, [], mod.__name__)


if __name__ == "__main__":
    unittest.main()
