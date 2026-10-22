"""Migra execuções JÁ GERADAS (pastas ``execucao_*``) para as convenções atuais de nomes.

Dois conjuntos de regras, aplicados nesta ordem:

A. 01/10/2026 (painel antes do alias; só nomes de arquivo). É ANTERIOR à convenção de
   09/10/2026 (``<bloco>_<graphic|table>_<NN>_…[__<alias>]_<color|gray>``): nomes já nessa
   convenção, em que ``_color``/``_gray`` vêm DEPOIS do alias, não são tocados (a cor não é
   painel). As execuções antigas não foram migradas para a numeração de 09/10.
B. 13/10/2026 (nomenclatura por passos, mapa em ``renomeacao_passos_2026-10-13.md``):
   (a) nomes de CSV, gráfico, painel e slug de tabela (raiz e ``tabelas/``);
   (b) colunas e valores de regime DENTRO dos CSVs (``k_passo4_bc``, ``rep``,
       ``passo2_zero_shot``, ``especialista_origem`` = ``rotulo_real``/``metrica``…);
   (c) tags ``fase=`` → ``problema=|passo=|regime=`` nos ``llm_interactions*.json``;
   (d) ``dados_sinteticos_*/problem_X.csv`` → ``problema_X.csv``.
   ``gerar_tabelas``, ``regerar_simbolos`` e a animação do Perceptron só leem os nomes
   novos: uma pasta anterior a 13/10/2026 precisa passar por aqui (``--aplicar``) antes.

Regras do conjunto A (puras sobre o nome do arquivo):

1. Painel/variante depois do alias → antes do alias (``asset_variant``):
   ``bloco1_06_w_distribution__gpt4mini_boxplot.png``
       → ``bloco1_06_w_distribution_boxplot__gpt4mini.png``
   ``final_cross_linearity__gpt4mini_corrigido_hm.csv``
       → ``final_cross_linearity_corrigido_hm__gpt4mini.csv``
2. Painel "Problema D" do ``final_04_dataset_overview`` mostra o Problema E (especialista):
   ``..._seed42_problema_d[__alias].png`` → ``..._seed42_problema_e[__alias].png``
3. SVM da meia-lua é LLM como especialista no Problema D (Bloco 1), não o Problema F:
   ``bloco23_external_svm_meialua_seedN…`` → ``bloco1_11_meialua_svm_vs_llm_seedN…``
4. ``dados_sinteticos_seed*/problem_D.csv`` guarda o Problema E (especialista linear):
   → ``problem_E.csv``, SÓ se ``problem_E.csv`` ainda não existir na mesma pasta (desde
   01/10/2026 ``problem_D.csv`` é a meia-lua e convive com ``problem_E.csv``).

O painel legado ``_algoritmos_r3`` NÃO é renomeado para ``_algoritmos_r3r4`` (o
conteúdo antigo só tem R3). As pastas ``execucao_*`` são rastreadas pelo git, então
arquivos rastreados são movidos com ``git mv`` (os demais com ``os.rename``).

Uso:
    python src/renomear_assets_legado.py                 # dry-run em todas as *_completa
    python src/renomear_assets_legado.py --aplicar       # executa (arquivos + conteúdo dos CSVs/JSONs)
    python src/renomear_assets_legado.py resultados/execucao_2026-10-13_14-14-23_smoke --aplicar
    python src/renomear_assets_legado.py --aplicar --docs  # também artigo.tex, artigo_pandoc.tex, roteiro
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Optional
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execucao_io import MODEL_ALIAS  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTADOS_DIR = BASE_DIR / "resultados"
DOCS_PADRAO = ["artigo.tex", "artigo_pandoc.tex", "roteiro_apresentacao.txt"]

_ALIAS = "|".join(re.escape(a) for a in sorted(MODEL_ALIAS.values(), key=len, reverse=True))
# Na convenção de 09/10/2026 a COR vem depois do alias (``…__gpt4mini_color.png``):
# ``color``/``gray`` NÃO são painel e esses nomes ficam como estão (F18 da revisão de 09/10).
RE_PAINEL = re.compile(rf"^(?P<base>.+?)__(?P<alias>{_ALIAS})_(?!(?:color|gray)\.)(?P<painel>[A-Za-z0-9_]+)\.(?P<ext>png|csv)$")
RE_PROB_D = re.compile(r"^(final_04_dataset_overview_seed\d+)_problema_d((?:__[a-z0-9.-]+)?)\.png$")
RE_SVM = re.compile(r"^bloco23_external_svm_meialua_seed(\d+)")


def regra_painel(nome: str) -> str:
    m = RE_PAINEL.match(nome)
    return f"{m['base']}_{m['painel']}__{m['alias']}.{m['ext']}" if m else nome


def regra_problema_e(nome: str) -> str:
    return RE_PROB_D.sub(r"\1_problema_e\2.png", nome)


def regra_svm(nome: str) -> str:
    return RE_SVM.sub(r"bloco1_11_meialua_svm_vs_llm_seed\1", nome)


def novo_nome(nome: str) -> str:
    """Nome novo (base, sem pasta) para um asset da raiz da execução."""
    for regra in (regra_painel, regra_problema_e, regra_svm):
        nome = regra(nome)
    return nome


# ═══════════════════════════════════════════════════════════════════════════
# 13/10/2026: NOMENCLATURA POR PASSOS (camada 2; mapa em renomeacao_passos_2026-10-13.md)
# Regras sobre NOMES de arquivo (raiz, tabelas/ e dados_sinteticos_*/) e sobre o
# CONTEÚDO dos CSVs (colunas e valores de regime) e dos llm_interactions*.json (tags).
# Os nomes antigos aparecem aqui de propósito: este módulo é o único lugar do código
# que ainda os conhece, para migrar execuções anteriores a 13/10/2026.
# ═══════════════════════════════════════════════════════════════════════════

CSV_STEMS_2026_10_13 = {
    "bloco1_phases_abc": "bloco1_abc_passos_2_a_4",
    "bloco1_fase_a_protocolo": "bloco1_ad_passos_2_a_5",
    "bloco1_transferencia_bc": "bloco1_bc_passo3_consistencia",
    "bloco1_algorithm_comparison": "bloco1_a_passo4_perceptron_vs_nnls",
    "bloco1_r3r4_comparison": "bloco1_a_passo4_r2_r3_r4",
    "bloco1_oracle_validation": "bloco1_oraculo_passos_1_a_3",
    "bloco1_oracle_meialua": "bloco1_d_passo5_teto",
    "bloco2_phase_e": "bloco2_e_passos_2_3",
    "bloco2_dilution": "bloco2_e_passo4_diluicao",
    "bloco2_example_order": "bloco2_e_passo5_ordem",
    "bloco2_classical_baselines": "bloco2_e_passo6_baselines",
    "bloco23_external_phase_a": "bloco23_fg_passos_2_e_4",
    # nome intermediário de 13/10/2026 (manhã): a estimação da métrica no G é o Passo 4
    "bloco23_fg_passo2": "bloco23_fg_passos_2_e_4",
    "bloco23_external_phase_e": "bloco23_fg_aprendiz",
    "bloco3_fase_a_protocolo": "bloco3_g_passos_2_a_5",
}

GRAFICOS_2026_10_13 = {
    "bloco1_graphic_01_ABC_problemas_lineares": "bloco1_graphic_01_ABC_passo1_dados",
    "bloco1_graphic_02_A_rotulos_simbolos": "bloco1_graphic_02_A_passos_2_a_4_simbolos",
    "bloco1_graphic_03_A_fase_a_erros": "bloco1_graphic_03_A_passo4_erros",
    "bloco1_graphic_04_AD_fase_a_fewshot_curva": "bloco1_graphic_04_AD_passos_2_a_5_curva",
    "bloco1_graphic_05_A_w_distribuicao": "bloco1_graphic_05_A_passo4_w_distribuicao",
    "bloco1_graphic_06_A_perceptron_vs_nnls": "bloco1_graphic_06_A_passo4_perceptron_vs_nnls",
    "bloco1_graphic_07_A_sementes": "bloco1_graphic_07_BC_passo3_sementes",
    "bloco1_graphic_08_A_R2_R3_R4": "bloco1_graphic_08_A_passo4_R2_R3_R4",
    "bloco1_graphic_09_A_gamma_convergencia": "bloco1_graphic_09_A_passo4_gamma",
    "bloco1_graphic_10_BC_consistencia": "bloco1_graphic_10_BC_passo3_consistencia",
    "bloco1_graphic_11_B_rotulos_simbolos": "bloco1_graphic_11_B_passos_2_3_simbolos",
    "bloco1_graphic_11_C_rotulos_simbolos": "bloco1_graphic_11_C_passos_2_3_simbolos",
    "bloco1_graphic_13_D_rotulos_simbolos": "bloco1_graphic_13_D_passos_2_a_4_simbolos",
    "bloco1_graphic_14_D_svm_vs_llm": "bloco1_graphic_14_D_passo6_svm",
    "bloco1_graphic_19_oraculo_recuperacao_W": "bloco1_graphic_19_oraculo_passos_1_a_3",
    "bloco1_graphic_21_D_teto_oraculo_meialua": "bloco1_graphic_21_D_passo5_teto",
    "bloco2_graphic_01_E_especialista": "bloco2_graphic_01_E_passo1_especialista",
    "bloco2_graphic_02_E_estrategias_exemplos": "bloco2_graphic_02_E_passo3_estrategias_exemplos",
    "bloco2_graphic_03_E_curva_aprendizado": "bloco2_graphic_03_E1_passo3_curva",
    "bloco2_graphic_04_E_comparacao_estrategias": "bloco2_graphic_04_E1_passo3_estrategias",
    "bloco2_graphic_05_E_baselines_classicos": "bloco2_graphic_05_E1_passo6_baselines",
    "bloco2_graphic_06_E_diluicao": "bloco2_graphic_06_E1_passo4_diluicao",
    "bloco2_graphic_07_E_ordem_exemplos": "bloco2_graphic_07_E1_passo5_ordem",
    "bloco3_graphic_01_G1_peso_altura": "bloco3_graphic_01_G1_passo1_dados",
    "bloco3_graphic_02_G2_peso_altura_kg_m": "bloco3_graphic_02_G2_passo1_dados",
    "bloco3_graphic_03_G_fase_a_fewshot_curva": "bloco3_graphic_03_G_passos_2_a_5_curva",
    "bloco23_graphic_01_FG_curva_aprendizado": "bloco23_graphic_01_FG_aprendiz_curva",
    "bloco23_graphic_02_FG_R2_R3_R4": "bloco23_graphic_02_FG_passos_2_e_4_R2_R3_R4",
    "bloco23_graphic_02_FG_passo2_R2_R3_R4": "bloco23_graphic_02_FG_passos_2_e_4_R2_R3_R4",   # intermediário de 13/10
    "bloco23_graphic_03_FG_fronteiras_decisao": "bloco23_graphic_03_FG_fronteiras",
    "final_graphic_01_ABCE_dados": "final_graphic_01_ABCE_passo1_dados",
}

# Painéis renomeados (aplicados DEPOIS do stem, só no stem novo indicado).
PAINEIS_2026_10_13 = [
    (r"^(bloco1_graphic_02_A_passos_2_a_4_simbolos_seed\d+|bloco1_graphic_13_D_passos_2_a_4_simbolos_seed\d+"
     r"|bloco1_graphic_11_[BC]_passos_2_3_simbolos_seed\d+)_ground_truth(?=[_.])", r"\1_passo1_rotulo_real"),
    (r"^(bloco1_graphic_02_A_passos_2_a_4_simbolos_seed\d+|bloco1_graphic_13_D_passos_2_a_4_simbolos_seed\d+"
     r"|bloco1_graphic_11_[BC]_passos_2_3_simbolos_seed\d+)_llm_zero_shot(?=[_.])", r"\1_passo2_llm_zero_shot"),
    (r"^(bloco1_graphic_02_A_passos_2_a_4_simbolos_seed\d+|bloco1_graphic_13_D_passos_2_a_4_simbolos_seed\d+)"
     r"_llm_(\d+)shot(?=[_.])", r"\1_passo3_llm_\2shot"),
    (r"^(bloco1_graphic_02_A_passos_2_a_4_simbolos_seed\d+|bloco1_graphic_13_D_passos_2_a_4_simbolos_seed\d+)"
     r"_metrica_zero_shot(?=[_.])", r"\1_passo4_metrica"),
    (r"^(bloco1_graphic_13_D_passos_2_a_4_simbolos_seed\d+)_metrica_r4_zero_shot(?=[_.])", r"\1_passo4_metrica_r4"),
    (r"^(bloco1_graphic_11_[BC]_passos_2_3_simbolos_seed\d+)_metrica_melhor_a(?=[_.])", r"\1_passo3_metrica_eleita"),
    (r"^(bloco1_graphic_06_A_passo4_perceptron_vs_nnls)_fase_([bc])(?=[_.])", r"\1_bc_passo3_\2"),
    (r"^(bloco2_graphic_01_E_passo1_especialista)_ground_truth(?=[_.])", r"\1_rotulo_real"),
]
RE_G11 = re.compile(r"^bloco1_graphic_11_([BC])_rotulos_simbolos(?=[_.])")
RE_G12 = re.compile(r"^bloco1_graphic_12_D_meialua_seed(\d+)_overview(?=[_.])")
# final_graphic_07: <kind> `train` → passo2; k → passo3_k<k> (F) / passo6_k<k> (G)
RE_G07 = re.compile(r"^(final_graphic_07_FG_rotulos_llm_([FG])\w*?_seed\d+_\w+?)_(train|\d+)(?=(?:__[a-z0-9.-]+)?(?:_color|_gray)?\.png$)")

SLUGS_2026_10_13 = {
    "A_zero_fewshot_teto": "A_passos_2_a_5",
    "A_zero_vs_fewshot_comparacao": "A_passo2_vs_passo3",
    "A_escolha_melhor_metrica_transferencia": "A_passo6_eleicao",
    "A_R2_R3_R4_zero_fewshot": "A_passo4_R2_R3_R4",
    "A_perceptron_vs_nnls_por_semente": "A_passo4_perceptron_vs_nnls",
    "BC_transferencia_todas_metricas_A": "BC_passo3_todas_metricas_A",
    "BC_transferencia_zero_vs_melhor_por_semente": "BC_passo3_por_semente",
    "BC_comparativo_zero_vs_melhor": "BC_passo3_zero_vs_eleita",
    "BC_fewshot_exemplos_da_metrica": "BC_passo4_exemplos_da_metrica",
    "D_meialua_R2_R3_R4": "D_passos_2_a_5",
    "D_zero_vs_fewshot_comparacao": "D_passo2_vs_passo3",
    "A_vs_D_por_regime": "AD_por_passo",
    "AD_estabilidade_W_por_semente": "AD_passo4_estabilidade_W",
    "oraculo_recuperacao_W": "oraculo_passos_1_a_3",
    "D_teto_oraculo_meialua": "D_passo5_teto",
    "E1_faseE_estrategia_x_nshot": "E1_passo3_estrategia_x_k",
    "E2_faseE_estrategia_x_nshot": "E2_passo3_estrategia_x_k",
    "E3_faseE_estrategia_x_nshot": "E3_passo3_estrategia_x_k",
    "E_zero_vs_fewshot_por_especialista": "E_passo2_vs_passo3",
    "E_H4_media_fewshot_por_estrategia": "E_passo3_H4",
    "E_LLM_vs_baselines_nmax": "E_passo6_baselines",
    "E1_diluicao": "E1_passo4_diluicao",
    "E1_ordem_exemplos": "E1_passo5_ordem",
    "F_meialua_faseA_R2_R3_R4": "F_passo2_R2_R3_R4",
    "F_meialua_faseE_nshot": "F_passo3_k",
    "G_faseA_R2_R3_R4": "G_passos_2_e_4_R2_R3_R4",
    "G_passo2_R2_R3_R4": "G_passos_2_e_4_R2_R3_R4",          # intermediário de 13/10
    "G_faseE_nshot": "G_passo6_k",
    "G_faseE_LLM_vs_perceptron": "G_passo7_LLM_vs_perceptron",
    "G1_vs_G2_zero_shot": "G1_vs_G2_passos_2_e_4",
    "G1_vs_G2_passo2": "G1_vs_G2_passos_2_e_4",              # intermediário de 13/10
    "G_faseA_zero_fewshot_teto": "G_passos_2_a_5",
}
RE_TABELA = re.compile(r"^(bloco\d+_table_\d{2}_)(?P<assunto>.+?)(?P<resto>(?:__[a-z0-9.-]+)?(?:_gray)?\.(?:png|tex))$")
RE_DADOS = re.compile(r"^problem_([A-E](?:_r[34])?)\.csv$")

# Colunas (todas as famílias de CSV; os nomes são exclusivos de cada família)
COLUNAS_2026_10_13 = {
    "n_shot_phase_bc": "k_passo4_bc", "repetition": "rep",
    "consistency_transferred_b": "consistency_centroides_a_b", "consistency_transferred_c": "consistency_centroides_a_c",
    "kappa_transferred_b": "kappa_centroides_a_b", "kappa_transferred_c": "kappa_centroides_a_c",
    "accuracy_ceiling_vs_true": "accuracy_teto_vs_true", "fidelity_gt_metric_vs_llm_zero": "fidelity_teto_vs_llm_passo2",
    "source_regime": "metrica_regime", "source_k": "metrica_k", "source_rep": "metrica_rep",
    "fidelity_source_mean": "metrica_fidelidade_a", "melhor_metrica_a": "eleita_passo6",
    "collapsed_source": "collapsed_metrica",
    "n_shot": "k",
    "expert_source": "especialista_origem", "expert_regime": "especialista_regime",
    "expert_k": "especialista_k", "expert_rep": "especialista_rep",
    # bloco23_fg_aprendiz (até 09/10/2026: *_llm_vs_metric; o especialista nem sempre é a métrica)
    "accuracy_llm_vs_metric": "accuracy_llm_vs_expert", "kappa_llm_vs_metric": "kappa_llm_vs_expert",
    "f1_llm_vs_metric": "f1_llm_vs_expert", "n_errors_llm_vs_metric": "n_errors_llm_vs_expert",
}
REGIMES_2026_10_13 = {"zero_shot": "passo2_zero_shot", "few_shot": "passo3_exemplos_rotulo_real",
                      "ground_truth": "passo5_teto"}
ORIGENS_2026_10_13 = {"ground_truth": "rotulo_real", "metric": "metrica"}
VALORES_2026_10_13 = {"regime": REGIMES_2026_10_13, "metrica_regime": REGIMES_2026_10_13,
                      "especialista_regime": REGIMES_2026_10_13, "especialista_origem": ORIGENS_2026_10_13}
VIESES_2026_10_13 = {"nomes_classe": "vies_nomes_classe", "inversao_ordem": "vies_ordem_classes",
                     "variantes_prompt": "vies_variantes_prompt", "nomes_atributos": "vies_nomes_atributo"}


def _sub_prefixo(nome: str, mapa: dict) -> str:
    for antigo, novo in sorted(mapa.items(), key=lambda kv: -len(kv[0])):
        if nome.startswith(antigo) and (len(nome) == len(antigo) or nome[len(antigo)] in "_."):
            return novo + nome[len(antigo):]
    return nome


def regra_passos(nome: str) -> str:
    """Nome novo (13/10/2026) de um asset da raiz ou de tabelas/ (CSV, gráfico, painel, tabela)."""
    m = re.match(r"^(?P<stem>[a-z0-9_]+?)_(?P<ts>\d{8}_\d{6})(?P<resto>.*\.csv)$", nome)
    if m and m["stem"] in CSV_STEMS_2026_10_13:
        return f"{CSV_STEMS_2026_10_13[m['stem']]}_{m['ts']}{m['resto']}"
    nome = RE_G11.sub(r"bloco1_graphic_11_\1_passos_2_3_simbolos", nome)
    nome = RE_G12.sub(r"bloco1_graphic_12_D_passo1_dados_seed\1", nome)
    nome = _sub_prefixo(nome, GRAFICOS_2026_10_13)
    for pat, rep in PAINEIS_2026_10_13:
        nome = re.sub(pat, rep, nome)
    m = RE_G07.match(nome)
    if m:
        kind = m.group(3)
        suf = "passo2" if kind == "train" else f"passo{3 if m.group(2) == 'F' else 6}_k{kind}"
        nome = f"{m.group(1)}_{suf}{nome[m.end():]}"
    m = RE_TABELA.match(nome)
    if m and m["assunto"] in SLUGS_2026_10_13:
        nome = f"{m.group(1)}{SLUGS_2026_10_13[m['assunto']]}{m['resto']}"
    return nome


def tem_csv_antigo(pasta: Path) -> bool:
    """True se há CSV com nome anterior a 13/10/2026 (gerar_tabelas/regerar_simbolos exigem migração)."""
    return any(next(Path(pasta).glob(f"{antigo}_*.csv"), None) for antigo in CSV_STEMS_2026_10_13)


def k_prompt(prompt) -> Optional[int]:
    """k inferido do prompt de uma interação antiga sem ``k=`` na tag: nº de linhas de exemplo rotulado
    (``x1 = …, x2 = … -> classe``) no bloco ``Examples:``; 0 sem exemplos; None sem prompt."""
    if not prompt:
        return None
    txt = str(prompt)
    if "Examples:" not in txt:
        return 0
    bloco = txt.split("Examples:", 1)[1].split("\n\n", 1)[0]
    return sum(1 for ln in bloco.splitlines() if "->" in ln)


def migrar_tag(tag: str, k_prompt: Optional[int] = None) -> str:
    """Tag antiga (``bloco|fase=F|problema=P|regime=R|k=K|rep=r|seed=s[|extra]``) → formato de 13/10/2026
    (``bloco|problema=P|passo=N|regime=R'|k=K|rep=r|seed=s[|extra]``). Tags sem ``fase=`` voltam iguais.
    ``k_prompt``: k inferido do prompt, usado só quando a tag antiga traz ``k=-`` (vieses até 08/10/2026)."""
    if not tag or "|fase=" not in tag:
        return tag
    campos = tag.split("|")
    d = dict(c.split("=", 1) for c in campos[1:7])
    fase, prob, regime = d["fase"], d["problema"], d["regime"]
    # Até 08/10/2026 os vieses gravavam `k=-` (k não registrado); fica `k=-` e o passo é inferido do
    # prompt por `migrar_interacoes` (com exemplos → Passo 4 de B/C; sem → Passo 2), ver `k_prompt`.
    k = int(d["k"]) if d["k"].lstrip("-").isdigit() and d["k"] != "-" else (k_prompt if k_prompt is not None else "-")
    zero = regime == "zero_shot"
    if fase == "A":                                   # A, D, F (Passo 2) e G (Passos 2 e 3)
        passo, regime = (2 if zero else 3), REGIMES_2026_10_13.get(regime, regime)
    elif fase == "ABC":                               # A, B, C zero-shot (Passo 2); B/C com exemplos (Passo 4)
        passo, regime = (2, "passo2_zero_shot") if zero else (4, "exemplos_da_metrica")
    elif fase == "ABC_aux":                           # vieses: k=0 → Passo 2; k>0 → Passo 4 de B/C
        passo = 2 if k == 0 else (4 if isinstance(k, int) else "-")
        regime = VIESES_2026_10_13.get(regime, f"vies_{regime}")
    elif fase == "E" and prob.startswith("F_"):       # F, Passo 3 (aprendiz)
        passo, regime = 3, ("zero_shot" if zero else "exemplos_rotulo_real")
    elif fase == "E" and prob.startswith("G"):        # G, Passo 6 (aprendiz)
        passo, regime = 6, ("zero_shot" if zero else "exemplos_da_metrica")
    elif fase == "E":                                 # E/E1/E2/E3: Passo 2 (zero-shot) ou 3 (estratégia)
        passo, regime = (2, "passo2_zero_shot") if zero else (3, regime)
    elif fase == "E_diluicao":
        passo = 4
    elif fase == "E_ordem":
        passo = 5
    else:
        return tag
    novo = [campos[0], f"problema={prob}", f"passo={passo}", f"regime={regime}",
            f"k={k}", f"rep={d['rep']}", f"seed={d['seed']}"] + campos[7:]
    return "|".join(novo)


def migrar_csv(caminho: Path, aplicar: bool) -> int:
    """Renomeia colunas e valores de regime/origem de um CSV (texto preservado). Devolve nº de mudanças."""
    import pandas as pd
    df = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    n = sum(1 for c in df.columns if c in COLUNAS_2026_10_13)
    df = df.rename(columns=COLUNAS_2026_10_13)
    if (caminho.name.startswith("bloco23_fg_aprendiz") or caminho.name.startswith("bloco23_external_phase_e")) \
            and "especialista_origem" not in df.columns:
        # Execuções até 08/10/2026 não gravavam a origem do especialista: era SEMPRE a melhor métrica
        # (inclusive no F, que só passou a usar o rótulo real em 09/10/2026). A coluna é acrescentada
        # para o leitor estrito (`gerar_tabelas.tab_f_passo3`) descrever a execução como ela foi.
        df["especialista_origem"] = "metrica"
        n += len(df)
    for col, mapa in VALORES_2026_10_13.items():
        if col in df.columns:
            muda = df[col].isin(mapa)
            n += int(muda.sum())
            df[col] = df[col].map(lambda v: mapa.get(v, v))
    if n and aplicar:
        df.to_csv(caminho, index=False)
    return n


def migrar_interacoes(caminho: Path, aplicar: bool) -> int:
    """Reescreve as tags ``fase=`` de um llm_interactions*.json. Devolve nº de registros alterados."""
    import json
    with open(caminho, encoding="utf-8") as fh:
        dados = json.load(fh)
    n = 0
    for r in dados:
        if isinstance(r, dict) and r.get("tag"):
            novo = migrar_tag(str(r["tag"]), k_prompt(r.get("prompt")) if "|k=-|" in str(r["tag"]) else None)
            if novo != r["tag"]:
                r["tag"] = novo
                n += 1
    if n and aplicar:
        with open(caminho, "w", encoding="utf-8") as fh:
            json.dump(dados, fh, ensure_ascii=False, indent=2)
    return n


def migrar_conteudo(pasta: Path, aplicar: bool) -> dict:
    """CSVs (colunas/valores) e JSONs (tags) de ``pasta``. {'csv': {nome: n}, 'json': {nome: n}}."""
    pasta = Path(pasta)
    out = {"csv": {}, "json": {}}
    for f in sorted(pasta.glob("*.csv")):
        if (n := migrar_csv(f, aplicar)):
            out["csv"][f.name] = n
    for f in sorted(pasta.glob("llm_interactions*.json")):
        if (n := migrar_interacoes(f, aplicar)):
            out["json"][f.name] = n
    return out


def planejar(pasta: Path) -> list[tuple[Path, Path]]:
    """Lista (origem, destino) do que mudaria em ``pasta`` (vazia = já nas duas convenções).

    Compõe as regras de 01/10/2026 (``novo_nome``) com as de 13/10/2026 (``regra_passos``).
    """
    pares = []
    for p in sorted(pasta.iterdir()):
        if p.is_file() and (n := regra_passos(novo_nome(p.name))) != p.name:
            pares.append((p, p.with_name(n)))
    tab = pasta / "tabelas"
    if tab.is_dir():
        for p in sorted(tab.iterdir()):
            if p.is_file() and (n := regra_passos(p.name)) != p.name:
                pares.append((p, p.with_name(n)))
    for csv in sorted(pasta.glob("dados_sinteticos_seed*/problem_*.csv")):
        nome = csv.name
        if nome == "problem_D.csv" and not csv.with_name("problem_E.csv").exists():
            nome = "problem_E.csv"                      # regra 4 (01/10/2026), só no esquema antigo
        m = RE_DADOS.match(nome)
        if m:
            nome = f"problema_{m.group(1)}.csv"
        if nome != csv.name:
            pares.append((csv, csv.with_name(nome)))
    return pares


# Nome correto na convenção vigente: ``__<alias>`` imediatamente antes da extensão, com a cor
# (``_color``/``_gray``, 09/10/2026) opcionalmente DEPOIS do alias.
RE_CORRETO = re.compile(rf"__({_ALIAS})(?:_color|_gray)?\.(png|csv)$")


def suspeitos(pasta: Path) -> list[str]:
    """Arquivos da raiz com ``__`` que nenhuma regra renomeia e que não seguem a convenção."""
    return [p.name for p in sorted(pasta.iterdir())
            if p.is_file() and "__" in p.name and novo_nome(p.name) == p.name
            and not RE_CORRETO.search(p.name)]


def rastreado(p: Path) -> bool:
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=BASE_DIR,
                           capture_output=True, env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null"})
        return r.returncode == 0
    except OSError:
        return False


def aplicar(pares: list[tuple[Path, Path]], usar_git: bool = True) -> None:
    destinos = [d for _, d in pares]
    if len(set(destinos)) != len(destinos):
        raise SystemExit("colisão: dois arquivos mapeiam para o mesmo destino")
    for _, dst in pares:
        if dst.exists():
            raise SystemExit(f"destino já existe, abortando antes de mover: {dst}")
    for src, dst in pares:
        if usar_git and rastreado(src):
            subprocess.run(["git", "mv", str(src), str(dst)], cwd=BASE_DIR, check=True,
                           env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null"})
        else:
            os.rename(src, dst)


def substituir_em_documento(caminho: Path, mapa: dict[str, str]) -> int:
    """Troca nomes antigos por novos (substituição literal). Devolve nº de ocorrências."""
    texto = caminho.read_text(encoding="utf-8")
    total = 0
    for antigo, novo in sorted(mapa.items(), key=lambda kv: -len(kv[0])):
        n = texto.count(antigo)
        if n:
            texto = texto.replace(antigo, novo)
            total += n
    if total:
        caminho.write_text(texto, encoding="utf-8")
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pastas", nargs="*", help="pastas execucao_* (padrão: todas as *_completa)")
    ap.add_argument("--aplicar", action="store_true", help="executa (sem a flag é dry-run)")
    ap.add_argument("--docs", action="store_true",
                    help=f"também atualiza referências em {', '.join(DOCS_PADRAO)}")
    args = ap.parse_args(argv)

    pastas = [Path(p) for p in args.pastas] or sorted(RESULTADOS_DIR.glob("execucao_*_completa"))
    pastas = [p if p.is_absolute() else ((BASE_DIR / p) if (BASE_DIR / p).exists() else RESULTADOS_DIR / p) for p in pastas]
    mapa_docs: dict[str, str] = {}
    for pasta in pastas:
        pares = planejar(pasta)
        print(f"\n== {pasta.name}: {len(pares)} arquivo(s) a renomear ==")
        por_regra = {"painel/alias (01/10)": 0, "problema_e (01/10)": 0, "svm→bloco1_11 (01/10)": 0,
                     "passos (13/10)": 0, "dados_sinteticos": 0}
        for src, dst in pares:
            if src.parent.name.startswith("dados_sinteticos"):
                por_regra["dados_sinteticos"] += 1
            elif RE_SVM.match(src.name):
                por_regra["svm→bloco1_11 (01/10)"] += 1
            elif RE_PROB_D.match(regra_painel(src.name)):
                por_regra["problema_e (01/10)"] += 1
            elif novo_nome(src.name) != src.name:
                por_regra["painel/alias (01/10)"] += 1
            else:
                por_regra["passos (13/10)"] += 1
            print(f"  {src.relative_to(pasta)}  ->  {dst.relative_to(pasta)}")
            mapa_docs[src.name] = dst.name
        print("  totais:", ", ".join(f"{k}={v}" for k, v in por_regra.items()))
        susp = suspeitos(pasta)
        if susp:
            print("  ⚠ com '__' mas fora de qualquer regra:", susp)
        if args.aplicar and pares:
            aplicar(pares)
            print("  ✓ arquivos renomeados")
        conteudo = migrar_conteudo(pasta, aplicar=args.aplicar)
        n_csv, n_json = sum(conteudo["csv"].values()), sum(conteudo["json"].values())
        print(f"  conteúdo (13/10): {len(conteudo['csv'])} CSV(s) com {n_csv} coluna(s)/valor(es) a migrar; "
              f"{len(conteudo['json'])} JSON(s) com {n_json} tag(s) a migrar"
              + ("  ✓ aplicado" if args.aplicar else ""))
    if args.docs:
        # Nomes base sem alias (o roteiro cita os assets sem sufixo de modelo)
        extra = {}
        for antigo, novo in mapa_docs.items():
            a0 = re.sub(rf"__({_ALIAS})(?=\.(png|csv)$)", "", antigo)
            n0 = re.sub(rf"__({_ALIAS})(?=\.(png|csv)$)", "", novo)
            if a0 != n0 and a0 != antigo:
                extra[a0] = n0
        mapa_docs.update(extra)
        for doc in DOCS_PADRAO:
            caminho = BASE_DIR / doc
            if not caminho.exists():
                continue
            if args.aplicar:
                n = substituir_em_documento(caminho, mapa_docs)
            else:
                texto = caminho.read_text(encoding="utf-8")
                n = sum(texto.count(a) for a in mapa_docs)
            print(f"  docs: {doc}: {n} ocorrência(s){'' if args.aplicar else ' (dry-run)'}")
    if not args.aplicar:
        print("\n(dry-run: nada foi alterado; use --aplicar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
