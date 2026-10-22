"""Testes do script de migração das execuções antigas (``renomear_assets_legado``).

Garante que o mapa legado→novo é o mesmo que o código novo produz
(``llm_asset`` + ``asset_variant``), que assets já corretos não mudam (idempotência),
que a aplicação em pasta temporária (sem git) funciona e que as regras de 13/10/2026
(nomes por passos, colunas/valores dos CSVs, tags dos JSONs, problema_X.csv) migram
uma pasta inteira.

Execução: .venv/bin/python -m unittest tests.test_renomear_assets_legado
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import renomear_assets_legado as rn  # noqa: E402
from execucao_io import asset_variant, llm_asset  # noqa: E402

LEGADO = {
    "bloco1_06_w_distribution__gpt4mini_boxplot.png": "bloco1_06_w_distribution_boxplot__gpt4mini.png",
    "final_04_dataset_overview_seed42__flashlite_problema_d.png": "final_04_dataset_overview_seed42_problema_e__flashlite.png",
    "final_04_dataset_overview_seed7_problema_d.png": "final_04_dataset_overview_seed7_problema_e.png",
    "bloco23_external_svm_meialua_seed42__gpt4mini_gt.png": "bloco1_11_meialua_svm_vs_llm_seed42_gt__gpt4mini.png",
    "bloco23_external_svm_meialua_seed7__flashlite.png": "bloco1_11_meialua_svm_vs_llm_seed7__flashlite.png",
    "final_cross_linearity__gpt4mini_corrigido_hm.csv": "final_cross_linearity_corrigido_hm__gpt4mini.csv",
}
INALTERADOS = [
    # convenção de 09/10/2026: a cor vem DEPOIS do alias e NÃO é painel (F18)
    "bloco1_graphic_05_A_w_distribuicao_boxplot__gpt4mini_color.png",
    "bloco1_graphic_05_A_w_distribuicao_boxplot__gpt4mini_gray.png",
    "final_graphic_03_ABC_acertos_erros_seed42__flashlite_color.png",
    "bloco1_02_problema_d_meialua_seed42_overview.png",
    "bloco1_04_oracle_transfer_euclidean_gt.png",
    "bloco1_06_w_distribution_boxplot__gpt4mini.png",
    "bloco1_10_r3r4_comparison_algoritmos_r3__gpt4mini.png",
    "bloco1_phases_abc_20260705_153049.csv",
    "final_09_model_comparison.png",
]


class TestNovoNome(unittest.TestCase):
    def test_mapa_legado(self):
        for antigo, novo in LEGADO.items():
            with self.subTest(antigo=antigo):
                self.assertEqual(rn.novo_nome(antigo), novo)

    def test_inalterados(self):
        for nome in INALTERADOS:
            with self.subTest(nome=nome):
                self.assertEqual(rn.novo_nome(nome), nome)

    def test_cor_depois_do_alias_nao_e_painel(self):
        from execucao_io import com_cor
        nome = os.path.basename(com_cor(asset_variant(llm_asset("/x", "bloco1_graphic_05_A_w_distribuicao.png",
                                                                "gpt-4o-mini"), "boxplot"), "color"))
        self.assertEqual(nome, "bloco1_graphic_05_A_w_distribuicao_boxplot__gpt4mini_color.png")
        self.assertEqual(rn.novo_nome(nome), nome)
        self.assertIsNone(rn.RE_PAINEL.match(nome))
        # um painel legítimo depois do alias continua sendo corrigido
        self.assertEqual(rn.novo_nome("bloco1_06_w_distribution__gpt4mini_colorido.png"),
                         "bloco1_06_w_distribution_colorido__gpt4mini.png")

    def test_bate_com_asset_variant(self):
        novo = asset_variant(llm_asset("/x", "bloco1_06_w_distribution.png", "gpt-4o-mini"), "boxplot")
        self.assertEqual(rn.novo_nome("bloco1_06_w_distribution__gpt4mini_boxplot.png"),
                         os.path.basename(novo))


class TestPlanejarAplicar(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # (sem o painel já correto de bloco1_06: colidiria com o destino do legado,
        #  e o script aborta de propósito quando o destino existe)
        for n in list(LEGADO) + [i for i in INALTERADOS if not i.startswith("bloco1_06_")]:
            (self.tmp / n).touch()
        (self.tmp / "dados_sinteticos_seed42").mkdir()
        (self.tmp / "dados_sinteticos_seed42" / "problem_D.csv").touch()
        (self.tmp / "dados_sinteticos_seed42" / "problem_A.csv").touch()
        self.esperado_dados = {"problem_D.csv": "problema_E.csv", "problem_A.csv": "problema_A.csv"}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_planejar_e_aplicar_idempotente(self):
        pares = rn.planejar(self.tmp)
        # regras de 01/10 (LEGADO) compostas com as de 13/10 (bloco1_phases_abc → bloco1_abc_passos_2_a_4;
        # problem_D → problem_E (01/10) → problema_E (13/10))
        esperado = {a: rn.regra_passos(n) for a, n in LEGADO.items()}
        esperado.update({n: rn.regra_passos(n) for n in INALTERADOS
                         if not n.startswith("bloco1_06_") and rn.regra_passos(n) != n})
        esperado.update(self.esperado_dados)
        self.assertEqual({a.name: b.name for a, b in pares}, esperado)
        rn.aplicar(pares, usar_git=False)
        self.assertEqual(rn.planejar(self.tmp), [])
        self.assertTrue((self.tmp / "bloco1_11_meialua_svm_vs_llm_seed42_gt__gpt4mini.png").exists())
        self.assertTrue((self.tmp / "dados_sinteticos_seed42" / "problema_E.csv").exists())
        self.assertTrue((self.tmp / "dados_sinteticos_seed42" / "problema_A.csv").exists())
        self.assertFalse((self.tmp / "dados_sinteticos_seed42" / "problem_D.csv").exists())

    def test_problem_d_vira_problema_d_quando_problem_e_ja_existe(self):
        # esquema desde 01/10/2026: problem_D.csv (meia-lua) convive com problem_E.csv; 13/10: problema_X.csv
        (self.tmp / "dados_sinteticos_seed42" / "problem_E.csv").touch()
        pares = {a.name: b.name for a, b in rn.planejar(self.tmp)}
        self.assertEqual(pares["problem_D.csv"], "problema_D.csv")
        self.assertEqual(pares["problem_E.csv"], "problema_E.csv")

    def test_suspeitos_aceitam_cor_depois_do_alias(self):
        corretos = ["bloco1_graphic_05_A_w_distribuicao_boxplot__gpt4mini_color.png",
                    "bloco1_table_01_A_zero_fewshot_teto__flashlite_gray.png",
                    "final_cross_linearity__gpt4mini.csv"]
        for n in corretos:
            (self.tmp / n).touch()
        (self.tmp / "bloco9_coisa__desconhecido.png").touch()
        susp = rn.suspeitos(self.tmp)
        self.assertEqual(susp, ["bloco9_coisa__desconhecido.png"])

    def test_dry_run_no_smoke_migrado_de_2026_10_13_nao_propoe_nada(self):
        smoke = rn.RESULTADOS_DIR / "execucao_2026-10-13_14-14-23_smoke"
        if not smoke.is_dir():
            self.skipTest("smoke de 13/10/2026 ausente")
        self.assertEqual(rn.planejar(smoke), [])
        self.assertEqual(rn.suspeitos(smoke), [])
        self.assertFalse(rn.tem_csv_antigo(smoke))
        self.assertEqual(rn.migrar_conteudo(smoke, aplicar=False), {"csv": {}, "json": {}})

    def test_dry_run_na_completa_de_2026_10_08_lista_sem_alterar(self):
        completa = rn.RESULTADOS_DIR / "execucao_2026-10-08_23-16-54_completa"
        if not completa.is_dir():
            self.skipTest("completa de 08/10/2026 ausente")
        self.assertTrue(rn.tem_csv_antigo(completa))
        pares = rn.planejar(completa)
        self.assertTrue(any(a.name.startswith("bloco1_phases_abc_") for a, _ in pares))
        self.assertTrue(any(a.name.startswith("bloco1_phases_abc_") for a, _ in pares))

    def test_substituicao_em_documento(self):
        doc = self.tmp / "artigo.tex"
        doc.write_text("\\includegraphics{bloco23_external_svm_meialua_seed42__gpt4mini_gt.png}\n"
                       "Imagem: bloco23_external_svm_meialua_seed42.png\n", encoding="utf-8")
        mapa = {"bloco23_external_svm_meialua_seed42__gpt4mini_gt.png":
                "bloco1_11_meialua_svm_vs_llm_seed42_gt__gpt4mini.png",
                "bloco23_external_svm_meialua_seed42.png": "bloco1_11_meialua_svm_vs_llm_seed42.png"}
        self.assertEqual(rn.substituir_em_documento(doc, mapa), 2)
        self.assertNotIn("bloco23_external_svm", doc.read_text(encoding="utf-8"))


PASSOS = {   # regras de 13/10/2026 (mapa renomeacao_passos_2026-10-13.md)
    "bloco1_phases_abc_20261013_142816.csv": "bloco1_abc_passos_2_a_4_20261013_142816.csv",
    "bloco1_fase_a_protocolo_20261013_142816.csv": "bloco1_ad_passos_2_a_5_20261013_142816.csv",
    "bloco23_external_phase_e_20261013_142816.csv": "bloco23_fg_aprendiz_20261013_142816.csv",
    "bloco3_fase_a_protocolo_20261013_142816.csv": "bloco3_g_passos_2_a_5_20261013_142816.csv",
    "bloco23_external_phase_a_20261013_142816.csv": "bloco23_fg_passos_2_e_4_20261013_142816.csv",
    # nomes intermediários de 13/10/2026 (estimação da métrica no G = Passo 4)
    "bloco23_fg_passo2_20261013_142816.csv": "bloco23_fg_passos_2_e_4_20261013_142816.csv",
    "bloco23_graphic_02_FG_R2_R3_R4__gpt4mini_color.png": "bloco23_graphic_02_FG_passos_2_e_4_R2_R3_R4__gpt4mini_color.png",
    "bloco23_graphic_02_FG_passo2_R2_R3_R4__flashlite_gray.png": "bloco23_graphic_02_FG_passos_2_e_4_R2_R3_R4__flashlite_gray.png",
    "bloco3_table_01_G_faseA_R2_R3_R4__gpt4mini_gray.png": "bloco3_table_01_G_passos_2_e_4_R2_R3_R4__gpt4mini_gray.png",
    "bloco3_table_01_G_passo2_R2_R3_R4__gpt4mini.tex": "bloco3_table_01_G_passos_2_e_4_R2_R3_R4__gpt4mini.tex",
    "bloco3_table_04_G1_vs_G2_zero_shot__flashlite_gray.png": "bloco3_table_04_G1_vs_G2_passos_2_e_4__flashlite_gray.png",
    "bloco3_table_04_G1_vs_G2_passo2__flashlite.tex": "bloco3_table_04_G1_vs_G2_passos_2_e_4__flashlite.tex",
    "bloco23_fg_passos_2_e_4_20261013_142816.csv": "bloco23_fg_passos_2_e_4_20261013_142816.csv",
    "final_cross_linearity__gpt4mini.csv": "final_cross_linearity__gpt4mini.csv",
    "bloco1_graphic_02_A_rotulos_simbolos_seed42_llm_4shot__gpt4mini_color.png":
        "bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42_passo3_llm_4shot__gpt4mini_color.png",
    "bloco1_graphic_02_A_rotulos_simbolos_seed42_ground_truth__gpt4mini_gray.png":
        "bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42_passo1_rotulo_real__gpt4mini_gray.png",
    "bloco1_graphic_02_A_rotulos_simbolos_seed42_metrica_zero_shot__gpt4mini_gray.png":
        "bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42_passo4_metrica__gpt4mini_gray.png",
    "bloco1_graphic_11_C_rotulos_simbolos_seed7_metrica_melhor_a__flashlite_gray.png":
        "bloco1_graphic_11_C_passos_2_3_simbolos_seed7_passo3_metrica_eleita__flashlite_gray.png",
    "bloco1_graphic_12_D_meialua_seed42_overview_color.png": "bloco1_graphic_12_D_passo1_dados_seed42_color.png",
    "bloco1_graphic_13_D_rotulos_simbolos_seed42_metrica_r4_zero_shot__gpt4mini_gray.png":
        "bloco1_graphic_13_D_passos_2_a_4_simbolos_seed42_passo4_metrica_r4__gpt4mini_gray.png",
    "bloco1_graphic_06_A_perceptron_vs_nnls_fase_c__gpt4mini_gray.png":
        "bloco1_graphic_06_A_passo4_perceptron_vs_nnls_bc_passo3_c__gpt4mini_gray.png",
    "bloco1_graphic_07_A_sementes__gpt4mini_color.png": "bloco1_graphic_07_BC_passo3_sementes__gpt4mini_color.png",
    "bloco1_graphic_19_oraculo_recuperacao_W_ratio_gray.png": "bloco1_graphic_19_oraculo_passos_1_a_3_ratio_gray.png",
    "bloco1_graphic_21_D_teto_oraculo_meialua_color.png": "bloco1_graphic_21_D_passo5_teto_color.png",
    "bloco2_graphic_01_E_especialista_ground_truth_gray.png": "bloco2_graphic_01_E_passo1_especialista_rotulo_real_gray.png",
    "bloco2_graphic_03_E_curva_aprendizado_kappa__gpt4mini_color.png": "bloco2_graphic_03_E1_passo3_curva_kappa__gpt4mini_color.png",
    "bloco3_graphic_03_G_fase_a_fewshot_curva_nnls_g1_a_b_x1_x2__gpt4mini_gray.png":
        "bloco3_graphic_03_G_passos_2_a_5_curva_nnls_g1_a_b_x1_x2__gpt4mini_gray.png",
    "bloco23_graphic_03_FG_fronteiras_decisao__gpt4mini_color.png": "bloco23_graphic_03_FG_fronteiras__gpt4mini_color.png",
    "final_graphic_01_ABCE_dados_seed42_problema_e__gpt4mini_gray.png": "final_graphic_01_ABCE_passo1_dados_seed42_problema_e__gpt4mini_gray.png",
    "final_graphic_07_FG_rotulos_llm_F_meia_lua_seed42_x1_x2_train__gpt4mini_color.png":
        "final_graphic_07_FG_rotulos_llm_F_meia_lua_seed42_x1_x2_passo2__gpt4mini_color.png",
    "final_graphic_07_FG_rotulos_llm_F_meia_lua_seed42_x1_x2_4__gpt4mini_gray.png":
        "final_graphic_07_FG_rotulos_llm_F_meia_lua_seed42_x1_x2_passo3_k4__gpt4mini_gray.png",
    "final_graphic_07_FG_rotulos_llm_G2_homem_mulher_classesAB_seed7_peso_kg_altura_m_0__flashlite_color.png":
        "final_graphic_07_FG_rotulos_llm_G2_homem_mulher_classesAB_seed7_peso_kg_altura_m_passo6_k0__flashlite_color.png",
    "final_graphic_08_comparacao_modelos_gray.png": "final_graphic_08_comparacao_modelos_gray.png",
    "bloco1_table_01_A_zero_fewshot_teto__flashlite_gray.png": "bloco1_table_01_A_passos_2_a_5__flashlite_gray.png",
    "bloco1_table_12_A_vs_D_por_regime__gpt4mini.tex": "bloco1_table_12_AD_por_passo__gpt4mini.tex",
    "bloco2_table_02_E2_faseE_estrategia_x_nshot__gpt4mini_gray.png": "bloco2_table_02_E2_passo3_estrategia_x_k__gpt4mini_gray.png",
    "bloco3_table_06_G_faseA_zero_fewshot_teto__gpt4mini.tex": "bloco3_table_06_G_passos_2_a_5__gpt4mini.tex",
    "final_table_04_auditoria_flip_por_regime_gray.png": "final_table_04_auditoria_flip_por_regime_gray.png",
    "bloco1_table_14_vies_nomes_classe__gpt4mini_gray.png": "bloco1_table_14_vies_nomes_classe__gpt4mini_gray.png",
}

TAGS = {
    "bloco1|fase=A|problema=A|regime=zero_shot|k=0|rep=0|seed=42": "bloco1|problema=A|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=42",
    "bloco1|fase=A|problema=D|regime=few_shot|k=10|rep=1|seed=42": "bloco1|problema=D|passo=3|regime=passo3_exemplos_rotulo_real|k=10|rep=1|seed=42",
    "bloco1|fase=ABC|problema=ABC|regime=zero_shot|k=0|rep=0|seed=7": "bloco1|problema=ABC|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=7",
    "bloco1|fase=ABC|problema=ABC|regime=few_shot_metrica|k=10|rep=0|seed=7": "bloco1|problema=ABC|passo=4|regime=exemplos_da_metrica|k=10|rep=0|seed=7",
    "bloco1|fase=ABC_aux|problema=ABC|regime=nomes_classe|k=0|rep=0|seed=42|nomes=0/1": "bloco1|problema=ABC|passo=2|regime=vies_nomes_classe|k=0|rep=0|seed=42|nomes=0/1",
    "bloco1|fase=ABC_aux|problema=ABC|regime=inversao_ordem|k=10|rep=0|seed=42|nomes=B/A": "bloco1|problema=ABC|passo=4|regime=vies_ordem_classes|k=10|rep=0|seed=42|nomes=B/A",
    "bloco1|fase=ABC_aux|problema=ABC|regime=variantes_prompt|k=0|rep=0|seed=42|variante=cot": "bloco1|problema=ABC|passo=2|regime=vies_variantes_prompt|k=0|rep=0|seed=42|variante=cot",
    "bloco1|fase=ABC_aux|problema=ABC|regime=nomes_atributos|k=0|rep=0|seed=42|atributos=altura/peso": "bloco1|problema=ABC|passo=2|regime=vies_nomes_atributo|k=0|rep=0|seed=42|atributos=altura/peso",
    "bloco2|fase=E|problema=E|regime=zero_shot|k=0|rep=0|seed=42": "bloco2|problema=E|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=42",
    "bloco2|fase=E|problema=E2|regime=hard|k=4|rep=2|seed=42": "bloco2|problema=E2|passo=3|regime=hard|k=4|rep=2|seed=42",
    "bloco2|fase=E_diluicao|problema=E1|regime=diluicao|k=8|rep=0|seed=42|easy=4": "bloco2|problema=E1|passo=4|regime=diluicao|k=8|rep=0|seed=42|easy=4",
    "bloco2|fase=E_ordem|problema=E1|regime=shuffled|k=4|rep=0|seed=42": "bloco2|problema=E1|passo=5|regime=shuffled|k=4|rep=0|seed=42",
    "bloco2|fase=A|problema=F_meia_lua_seed42|regime=zero_shot|k=0|rep=0|seed=42|nomes=A/B|atributos=x1/x2":
        "bloco2|problema=F_meia_lua_seed42|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=42|nomes=A/B|atributos=x1/x2",
    "bloco2|fase=E|problema=F_meia_lua_seed42|regime=few_shot|k=4|rep=1|seed=42|nomes=A/B|atributos=x1/x2":
        "bloco2|problema=F_meia_lua_seed42|passo=3|regime=exemplos_rotulo_real|k=4|rep=1|seed=42|nomes=A/B|atributos=x1/x2",
    "bloco2|fase=E|problema=F_meia_lua_seed42|regime=zero_shot|k=0|rep=0|seed=42|nomes=A/B|atributos=x1/x2":
        "bloco2|problema=F_meia_lua_seed42|passo=3|regime=zero_shot|k=0|rep=0|seed=42|nomes=A/B|atributos=x1/x2",
    "bloco3|fase=A|problema=G1_homem_mulher|regime=few_shot|k=4|rep=0|seed=42|nomes=Mulher/Homem|atributos=peso/altura":
        "bloco3|problema=G1_homem_mulher|passo=3|regime=passo3_exemplos_rotulo_real|k=4|rep=0|seed=42|nomes=Mulher/Homem|atributos=peso/altura",
    "bloco3|fase=E|problema=G2_homem_mulher_classesAB|regime=few_shot|k=4|rep=0|seed=42|nomes=A/B|atributos=x1/x2":
        "bloco3|problema=G2_homem_mulher_classesAB|passo=6|regime=exemplos_da_metrica|k=4|rep=0|seed=42|nomes=A/B|atributos=x1/x2",
    "": "",
}


class TestRegrasPassos(unittest.TestCase):
    def test_nomes_de_arquivo(self):
        for antigo, novo in PASSOS.items():
            with self.subTest(antigo=antigo):
                self.assertEqual(rn.regra_passos(antigo), novo)
                self.assertEqual(rn.regra_passos(novo), novo)          # idempotente

    def test_nomes_novos_batem_com_o_codigo(self):
        from execucao_io import com_cor
        novo = os.path.basename(com_cor(asset_variant(llm_asset(
            "/x", "bloco1_graphic_02_A_passos_2_a_4_simbolos_seed42.png", "gpt-4o-mini"), "passo3_llm_4shot"), "color"))
        self.assertEqual(rn.regra_passos("bloco1_graphic_02_A_rotulos_simbolos_seed42_llm_4shot__gpt4mini_color.png"), novo)

    def test_tags(self):
        for antigo, novo in TAGS.items():
            with self.subTest(antigo=antigo):
                self.assertEqual(rn.migrar_tag(antigo), novo)
                self.assertEqual(rn.migrar_tag(novo), novo)            # idempotente
        self.assertEqual(rn.migrar_tag("bloco1|problema=A|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=42"),
                         "bloco1|problema=A|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=42")

    def test_migra_csv_e_json_numa_pasta_temporaria(self):
        import json
        import pandas as pd
        tmp = Path(tempfile.mkdtemp())
        try:
            pd.DataFrame([{"provider": "p", "model": "m", "n_shot_phase_bc": 0, "repetition": 0,
                           "consistency_transferred_b": 0.5, "kappa_transferred_c": 0.1}]).to_csv(
                tmp / "bloco1_phases_abc_20261013_000000.csv", index=False)
            pd.DataFrame([{"problem": "A", "regime": "zero_shot", "k": 0, "rep": 0, "accuracy_ceiling_vs_true": "",
                           "fidelity_gt_metric_vs_llm_zero": 0.9},
                          {"problem": "A", "regime": "ground_truth", "k": 0, "rep": 0, "accuracy_ceiling_vs_true": 0.95,
                           "fidelity_gt_metric_vs_llm_zero": ""}]).to_csv(
                tmp / "bloco1_fase_a_protocolo_20261013_000000.csv", index=False)
            pd.DataFrame([{"source_regime": "few_shot", "source_k": 4, "source_rep": 1, "fidelity_source_mean": 0.8,
                           "melhor_metrica_a": 1, "collapsed_source": 0}]).to_csv(
                tmp / "bloco1_transferencia_bc_20261013_000000.csv", index=False)
            pd.DataFrame([{"problem_name": "F_meia_lua_seed42", "n_shot": 4, "expert_source": "ground_truth",
                           "expert_regime": "", "expert_k": "", "expert_rep": ""},
                          {"problem_name": "G1_homem_mulher", "n_shot": 0, "expert_source": "metric",
                           "expert_regime": "few_shot", "expert_k": 4, "expert_rep": 0}]).to_csv(
                tmp / "bloco23_external_phase_e_20261013_000000.csv", index=False)
            with open(tmp / "llm_interactions.json", "w", encoding="utf-8") as fh:
                json.dump([{"tag": "bloco1|fase=A|problema=A|regime=zero_shot|k=0|rep=0|seed=42", "parsed_label": "A"},
                           {"tag": "", "parsed_label": "B"}], fh)
            self.assertTrue(rn.tem_csv_antigo(tmp))
            plano = rn.migrar_conteudo(tmp, aplicar=False)
            self.assertEqual(plano["csv"]["bloco1_phases_abc_20261013_000000.csv"], 4)
            self.assertEqual(plano["json"]["llm_interactions.json"], 1)
            rn.aplicar(rn.planejar(tmp), usar_git=False)
            rn.migrar_conteudo(tmp, aplicar=True)
            self.assertFalse(rn.tem_csv_antigo(tmp))
            abc = pd.read_csv(tmp / "bloco1_abc_passos_2_a_4_20261013_000000.csv")
            self.assertEqual(list(abc.columns), ["provider", "model", "k_passo4_bc", "rep",
                                                 "consistency_centroides_a_b", "kappa_centroides_a_c"])
            ad = pd.read_csv(tmp / "bloco1_ad_passos_2_a_5_20261013_000000.csv")
            self.assertEqual(list(ad["regime"]), ["passo2_zero_shot", "passo5_teto"])
            self.assertIn("accuracy_teto_vs_true", ad.columns); self.assertIn("fidelity_teto_vs_llm_passo2", ad.columns)
            self.assertAlmostEqual(float(ad["accuracy_teto_vs_true"].iloc[1]), 0.95)
            bc = pd.read_csv(tmp / "bloco1_bc_passo3_consistencia_20261013_000000.csv")
            self.assertEqual(list(bc.columns), ["metrica_regime", "metrica_k", "metrica_rep", "metrica_fidelidade_a",
                                                "eleita_passo6", "collapsed_metrica"])
            self.assertEqual(bc["metrica_regime"].iloc[0], "passo3_exemplos_rotulo_real")
            fg = pd.read_csv(tmp / "bloco23_fg_aprendiz_20261013_000000.csv")
            self.assertEqual(list(fg["especialista_origem"]), ["rotulo_real", "metrica"])
            self.assertEqual(fg["especialista_regime"].iloc[1], "passo3_exemplos_rotulo_real")
            self.assertEqual(list(fg["k"]), [4, 0])
            with open(tmp / "llm_interactions.json", encoding="utf-8") as fh:
                recs = json.load(fh)
            self.assertEqual(recs[0]["tag"], "bloco1|problema=A|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=42")
            self.assertEqual(rn.migrar_conteudo(tmp, aplicar=False), {"csv": {}, "json": {}})   # idempotente
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


class TestTagsSemK(unittest.TestCase):
    """Vieses da completa de 08/10/2026 gravavam ``k=-``: o passo vem do prompt (13/10/2026)."""

    def test_k_prompt(self):
        self.assertIsNone(rn.k_prompt(None))
        self.assertEqual(rn.k_prompt("Classify the point.\n\nPoint to classify:\nx1 = 1\n"), 0)
        p = ("Learn from the examples.\n\nExamples:\n  x1 = 2.0, x2 = -4.1 -> 0\n  x1 = 1.0, x2 = 3.0 -> 1\n"
             "  x1 = 0.5, x2 = 2.0 -> 1\n  x1 = -1, x2 = 0 -> 0\n\nAnswer ONLY with \"0\" or \"1\".\n")
        self.assertEqual(rn.k_prompt(p), 4)

    def test_migrar_tag_com_k_desconhecido(self):
        antiga = "bloco1|fase=ABC_aux|problema=ABC|regime=inversao_ordem|k=-|rep=0|seed=42"
        self.assertEqual(rn.migrar_tag(antiga, 0),
                         "bloco1|problema=ABC|passo=2|regime=vies_ordem_classes|k=0|rep=0|seed=42")
        self.assertEqual(rn.migrar_tag(antiga, 10),
                         "bloco1|problema=ABC|passo=4|regime=vies_ordem_classes|k=10|rep=0|seed=42")
        # sem prompt: não inventa passo nem k
        self.assertEqual(rn.migrar_tag(antiga),
                         "bloco1|problema=ABC|passo=-|regime=vies_ordem_classes|k=-|rep=0|seed=42")

    def test_migra_coluna_do_especialista_ausente_no_fg_aprendiz(self):
        import pandas as pd
        tmp = Path(tempfile.mkdtemp())
        try:
            pd.DataFrame([{"problem_name": "F_meia_lua_seed42", "n_shot": 4, "accuracy_llm_vs_metric": 0.9}]).to_csv(
                tmp / "bloco23_external_phase_e_20261008_000000.csv", index=False)
            rn.aplicar(rn.planejar(tmp), usar_git=False)
            rn.migrar_conteudo(tmp, aplicar=True)
            df = pd.read_csv(tmp / "bloco23_fg_aprendiz_20261008_000000.csv")
            self.assertEqual(list(df["especialista_origem"]), ["metrica"])     # até 08/10 o especialista era a métrica
            self.assertIn("accuracy_llm_vs_expert", df.columns); self.assertIn("k", df.columns)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
