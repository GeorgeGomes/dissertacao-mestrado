"""Testes de ``estilo_cinza`` (legenda garantida + versão preto e branco), 09/10/2026.

Contratos verificados (backend Agg, sem API):

1. B2: tampas e hastes de barras de erro (``bar(yerr=…, capsize=…)``, ``errorbar``) nunca viram
   "série k" nem ganham traço/símbolo; um ``errorbar`` rotulado continua com UMA entrada;
   barras com ``label=''`` num eixo que já tem legenda não geram "série k"; a legenda é refeita
   a partir das entradas originais.
2. F15: séries com 2 pontos recebem traços/símbolos distintos; ``axhline`` fica preta sem
   símbolo; scatter ``c=y`` rotulado "Concordam (113)" vira duas entradas com a contagem por
   classe (sem repetir o 113); barra única sem rótulo e sem ylabel não cria legenda; nenhum
   cinza de série passa de 0.6.
3. F5: scatter de cor única sem rótulo vira "amostras (1)", nunca "classe 0/1".

Execução: .venv/bin/python -m unittest tests.test_estilo_cinza
"""
import os
import sys
import unittest

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import estilo_cinza  # noqa: E402
from estilo_cinza import CINZAS_LINHA, garantir_legendas, para_cinza, tem_series  # noqa: E402


def _rotulos(ax):
    leg = ax.get_legend()
    return [t.get_text() for t in leg.get_texts()] if leg is not None else []


def _todos_rotulos(ax):
    """Rótulos de todos os artistas do eixo (linhas, coleções e containers)."""
    arts = list(ax.lines) + list(ax.collections) + list(ax.containers)
    return [str(a.get_label()) for a in arts]


class TestBarrasDeErro(unittest.TestCase):
    """B2: tampas e hastes das barras de erro ficam fora da legenda e sem traço."""

    def tearDown(self):
        plt.close("all")

    def test_bar_yerr_capsize_nao_rotula_tampas(self):
        fig, ax = plt.subplots()
        ax.bar([0, 1, 2, 3, 4], [1, 2, 3, 4, 5], yerr=[.1, .2, .1, .2, .1], capsize=3)
        ax.set_ylabel("acurácia")
        pend = garantir_legendas(fig)
        self.assertEqual(pend, [])
        self.assertFalse(any(r.startswith("série") for r in _todos_rotulos(ax)))
        self.assertEqual(_rotulos(ax), ["acurácia"])
        marcadores_antes = [ln.get_marker() for ln in ax.lines]       # tampas: '_' nativo
        para_cinza(fig)
        self.assertEqual(_rotulos(ax), ["acurácia"])
        self.assertEqual([ln.get_marker() for ln in ax.lines], marcadores_antes)
        for ln in ax.lines:                       # tampas: só cor, sem traço
            self.assertIn(ln.get_linestyle(), ("None", "", " "))
            self.assertEqual(matplotlib.colors.to_hex(ln.get_color()), "#000000")
            self.assertFalse(str(ln.get_label()).startswith("série"))

    def test_bar_yerr_sem_ylabel_e_sem_rotulo_nao_cria_legenda(self):
        fig, ax = plt.subplots()
        ax.bar([0, 1, 2, 3], [1, 2, 3, 4], yerr=[.1] * 4, capsize=3)
        pend = garantir_legendas(fig)
        self.assertEqual(pend, [])                # barra única: nada a legendar (F15c)
        self.assertIsNone(ax.get_legend())
        self.assertNotIn("valor", _todos_rotulos(ax))
        para_cinza(fig)
        self.assertIsNone(ax.get_legend())

    def test_errorbar_rotulado_mantem_uma_entrada(self):
        fig, ax = plt.subplots()
        x = np.arange(5)
        ax.errorbar(x, x * 2.0, yerr=np.full(5, .3), marker="o", capsize=4, label="LLM", color="steelblue")
        garantir_legendas(fig)
        self.assertEqual(_rotulos(ax), ["LLM"])
        marcadores_tampas = [t.get_marker() for t in ax.lines[1:]]
        para_cinza(fig)
        self.assertEqual(_rotulos(ax), ["LLM"])
        dados = ax.lines[0]
        self.assertIn(dados.get_linestyle(), estilo_cinza.TRACOS)      # linha de dados é série
        self.assertIn(dados.get_marker(), estilo_cinza.SIMBOLOS)
        self.assertEqual([t.get_marker() for t in ax.lines[1:]], marcadores_tampas)
        for tampa in ax.lines[1:]:
            self.assertIn(tampa.get_linestyle(), ("None", "", " "))
            self.assertEqual(matplotlib.colors.to_hex(tampa.get_color()), "#000000")
            self.assertEqual(matplotlib.colors.to_hex(tampa.get_markeredgecolor()), "#000000")

    def test_errorbar_sem_rotulo_recebe_ylabel_no_container(self):
        fig, ax = plt.subplots()
        x = np.arange(6)
        ax.errorbar(x, x * .1, yerr=np.full(6, .02), marker="s", capsize=4, color="#3498db")
        ax.set_ylabel("Cohen's κ")
        pend = garantir_legendas(fig)
        self.assertEqual(pend, [])
        self.assertEqual(_rotulos(ax), ["Cohen's κ"])
        para_cinza(fig)
        self.assertEqual(_rotulos(ax), ["Cohen's κ"])

    def test_barras_label_vazio_em_eixo_com_legenda_nao_viram_serie(self):
        fig, ax = plt.subplots()
        for i in range(4):
            ax.bar(i - 0.2, i + 1, 0.4, color="gray", label="Verdadeiro" if i == 0 else "")
            ax.bar(i + 0.2, i + 2, 0.4, color="steelblue", label="PERCEPTRON" if i == 0 else "")
        ax.axhline(1.0, color="green", linestyle="--", label="Perfeito (1.0)")
        ax.legend(fontsize=8)
        originais = _rotulos(ax)                  # ordem do matplotlib: linhas antes dos containers
        self.assertEqual(sorted(originais), ["PERCEPTRON", "Perfeito (1.0)", "Verdadeiro"])
        garantir_legendas(fig)
        self.assertEqual(_rotulos(ax), originais)
        self.assertFalse(any(r.startswith("série") for r in _todos_rotulos(ax)))
        para_cinza(fig)
        self.assertEqual(_rotulos(ax), originais)
        self.assertEqual(len(ax.get_legend().get_texts()), 3)

    def test_barras_label_vazio_com_artistas_rotulados_sem_legenda_previa(self):
        fig, ax = plt.subplots()
        for i in range(3):
            ax.bar(i, i + 1, 0.4, color="coral", label="NNLS" if i == 0 else "")
        garantir_legendas(fig)                    # cria a legenda só com o rotulado
        self.assertEqual(_rotulos(ax), ["NNLS"])

    def test_legenda_refeita_so_com_entradas_originais(self):
        fig, ax = plt.subplots()
        x = np.arange(6)
        ax.plot(x, x, label="A", color="steelblue")
        ax.plot(x, 2 * x, label="B", color="coral")
        ax.legend(["A"])                          # autor escolheu só uma entrada
        garantir_legendas(fig)
        para_cinza(fig)
        self.assertEqual(_rotulos(ax), ["A"])


class TestLinhasEReferencias(unittest.TestCase):
    """F15a: séries com 2 pontos são séries; referências são identificadas pela transformação."""

    def tearDown(self):
        plt.close("all")

    def test_duas_series_de_2_pontos_tem_formas_diferentes(self):
        for marker in ("o", None):
            fig, ax = plt.subplots()
            l1, = ax.plot([0, 4], [.5, .8], marker=marker, label="easy", color="steelblue")
            l2, = ax.plot([0, 4], [.4, .9], marker=marker, label="hard", color="coral")
            garantir_legendas(fig)
            para_cinza(fig)
            self.assertNotEqual(l1.get_linestyle(), l2.get_linestyle())
            self.assertNotEqual(l1.get_marker(), l2.get_marker())
            self.assertIn(l1.get_marker(), estilo_cinza.SIMBOLOS)
            self.assertEqual(_rotulos(ax), ["easy", "hard"])
            plt.close(fig)

    def test_axhline_e_axvline_ficam_pretas_sem_simbolo(self):
        fig, ax = plt.subplots()
        ax.plot(np.arange(5), np.arange(5) * .1, label="curva", color="steelblue")
        h = ax.axhline(0.5, color="green", linestyle="--", label="Meta")
        v = ax.axvline(2.0, color="red", linestyle=":")
        garantir_legendas(fig)
        para_cinza(fig)
        for ref, ls in ((h, "--"), (v, ":")):
            self.assertEqual(matplotlib.colors.to_hex(ref.get_color()), "#000000")
            self.assertEqual(ref.get_linestyle(), ls)
            self.assertIn(ref.get_marker(), ("", "None", None))
        self.assertEqual(_rotulos(ax), ["curva", "Meta"])

    def test_cinzas_de_serie_nunca_passam_de_0_6(self):
        self.assertTrue(all(float(c) <= 0.6 for c in CINZAS_LINHA))
        fig, ax = plt.subplots()
        cores = ["steelblue", "coral", "green", "purple", "orange", "brown", "pink"]
        for i, c in enumerate(cores):
            ax.plot(np.arange(5), np.arange(5) + i, color=c, label=f"s{i}")
        garantir_legendas(fig)
        para_cinza(fig)
        for ln in ax.lines:
            r, g, b, _ = matplotlib.colors.to_rgba(ln.get_color())
            self.assertLessEqual(max(r, g, b), 0.6 + 1e-9)

    def test_linha_de_marcadores_sem_rotulo_nao_vira_serie(self):
        fig, ax = plt.subplots()
        ax.plot(np.arange(5), np.arange(5), "o", color="red")       # fliers, pontos avulsos
        self.assertFalse(tem_series(ax))
        garantir_legendas(fig)
        self.assertIsNone(ax.get_legend())
        para_cinza(fig)
        self.assertEqual(ax.lines[0].get_linestyle(), "None")


class TestScatters(unittest.TestCase):
    """F5 e F15b: nomes neutros e contagem por classe."""

    def tearDown(self):
        plt.close("all")

    def test_scatter_cor_unica_sem_rotulo_vira_amostras(self):
        fig, ax = plt.subplots()
        rng = np.random.RandomState(0)
        ax.scatter(rng.normal(size=20), rng.normal(size=20), color="steelblue")
        garantir_legendas(fig)
        self.assertEqual(_rotulos(ax), ["amostras (1)"])
        para_cinza(fig)
        self.assertEqual(_rotulos(ax), ["amostras (1)"])

    def test_dois_scatters_cor_unica_viram_amostras_1_2(self):
        fig, ax = plt.subplots()
        rng = np.random.RandomState(0)
        ax.scatter(rng.normal(size=20), rng.normal(size=20), color="green")    # acerto
        ax.scatter(rng.normal(size=20), rng.normal(size=20), color="red")      # erro
        garantir_legendas(fig)
        self.assertEqual(_rotulos(ax), ["amostras (1)", "amostras (2)"])
        self.assertFalse(any("classe" in r for r in _rotulos(ax)))

    def test_scatter_c_y_sem_rotulo_continua_classe_k(self):
        fig, ax = plt.subplots()
        rng = np.random.RandomState(0)
        X = rng.normal(size=(40, 2)); y = (X[:, 0] > 0).astype(int)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
        garantir_legendas(fig)
        self.assertEqual(_rotulos(ax), ["classe 0", "classe 1"])
        para_cinza(fig)
        self.assertEqual(_rotulos(ax), ["classe 0", "classe 1"])

    def test_scatter_c_y_rotulado_divide_sem_repetir_contagem(self):
        fig, ax = plt.subplots()
        rng = np.random.RandomState(0)
        X = rng.normal(size=(113, 2)); y = np.r_[np.zeros(60), np.ones(53)].astype(int)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", label="Concordam (113)")
        garantir_legendas(fig)
        self.assertEqual(_rotulos(ax), ["Concordam (113)"])
        para_cinza(fig)
        self.assertEqual(_rotulos(ax), ["Concordam (60), classe 0", "Concordam (53), classe 1"])

    def test_scatter_c_y_rotulado_com_pts_e_numero_estranho(self):
        self.assertEqual(estilo_cinza._rotulo_por_classe("LLM 113 pts", 113, 60), "LLM 60 pts")
        self.assertEqual(estilo_cinza._rotulo_por_classe("LLM (99)", 113, 60), "LLM")
        self.assertEqual(estilo_cinza._rotulo_por_classe("LLM", 113, 60), "LLM")


class TestMesmaCorFormasDistintas(unittest.TestCase):
    """Auditoria de 09/10/2026 (rodada 2): séries desenhadas na MESMA cor com marcador/traço
    diferentes não podem colapsar no mesmo símbolo (bloco1_graphic_04)."""

    def tearDown(self):
        plt.close("all")

    def test_tres_errorbar_pretos_com_fmt_distintos_continuam_distintos(self):
        fig, ax = plt.subplots()
        x = np.array([0, 4, 10, 20])
        fmts = ("o-", "s--", "^:")
        for fmt, lab in zip(fmts, ("fidelidade", "acurácia métrica", "acurácia LLM")):
            ax.errorbar(x, 0.5 + 0.01 * x + 0.05 * len(lab) / 20, yerr=np.full(4, .02), fmt=fmt, color="k",
                        capsize=3, label=lab)
        ax.legend()
        garantir_legendas(fig)
        para_cinza(fig)
        dados = [c.lines[0] for c in ax.containers]
        formas = {(ln.get_marker(), ln.get_linestyle()) for ln in dados}
        self.assertEqual(formas, {("o", "-"), ("s", "--"), ("^", ":")})     # forma original mantida
        for ln in dados:
            self.assertEqual(matplotlib.colors.to_hex(ln.get_color()), "#000000")
        leg = ax.get_legend()
        self.assertEqual([t.get_text() for t in leg.get_texts()], ["fidelidade", "acurácia métrica", "acurácia LLM"])
        # os três handles da legenda são os containers convertidos, com formas distintas
        handles = estilo_cinza._handles_legenda(leg)
        self.assertEqual(len(handles), 3)
        self.assertEqual(len({(ln.get_marker(), ln.get_linestyle()) for ln in dados}), 3)

    def test_mesma_cor_com_formas_diferentes_recebe_simbolos_diferentes(self):
        fig, ax = plt.subplots()
        x = np.arange(5)
        l1, = ax.plot(x, x, "o-", color="steelblue", label="a")
        l2, = ax.plot(x, x + 1, "s--", color="steelblue", label="b")
        garantir_legendas(fig)
        para_cinza(fig)
        self.assertNotEqual(l1.get_marker(), l2.get_marker())
        self.assertNotEqual(l1.get_linestyle(), l2.get_linestyle())

    def test_mesma_cor_scatter_e_linha_compartilham_simbolo(self):
        fig, ax = plt.subplots()
        x = np.arange(6)
        ln, = ax.plot(x, x, "o-", color="coral", label="curva")
        ax.scatter(x, x + 1, color="coral", label="pontos")
        garantir_legendas(fig)
        para_cinza(fig)
        sc = [c for c in ax.collections][0]
        self.assertEqual(sc.get_paths()[0].vertices.shape, matplotlib.markers.MarkerStyle(ln.get_marker()).get_path().vertices.shape)


class TestErrorbarUmPontoEProxyBranco(unittest.TestCase):
    """Auditoria de 09/10/2026 (rodada 2): ``errorbar(fmt='o')`` de 1 ponto não é tampa e o proxy
    de legenda com linha branca não ganha linha cinza (bloco1_graphic_19, painel _w_scatter)."""

    def tearDown(self):
        plt.close("all")

    def test_linha_de_dados_vira_marcador_e_proxy_branco_fica_sem_linha(self):
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots()
        for xv, yv, cor, mk in ((0.2, 0.25, "steelblue", "o"), (0.8, 0.7, "coral", "s")):
            ax.errorbar([xv], [yv], yerr=[[0.05], [0.05]], fmt=mk, color=cor, markeredgecolor="k",
                        capsize=3)                     # sem rótulo: a legenda vem dos proxies, como no gráfico 19
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="recuperação perfeita")
        proxies = [Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", label="Perceptron"),
                   Line2D([0], [0], marker="s", color="w", markerfacecolor="coral", label="NNLS"),
                   Line2D([0], [0], color="k", linestyle="--", label="recuperação perfeita")]
        ax.legend(handles=proxies)
        garantir_legendas(fig)
        tampas = [ln for c in ax.containers for ln in c.lines[1]]
        marcadores_tampas = [t.get_marker() for t in tampas]             # '_' nativo do yerr
        para_cinza(fig)
        dados = [c.lines[0] for c in ax.containers]
        for ln in dados:
            self.assertIn(ln.get_marker(), estilo_cinza.SIMBOLOS)         # símbolo da cor, não tampa
            self.assertIn(ln.get_linestyle(), ("None", "", " "))         # sem traço
        self.assertNotEqual(dados[0].get_marker(), dados[1].get_marker())
        self.assertEqual([t.get_marker() for t in tampas], marcadores_tampas)   # tampas continuam tampas
        for t in tampas:
            self.assertEqual(matplotlib.colors.to_hex(t.get_color()), "#000000")
        leg = ax.get_legend()
        self.assertEqual([t.get_text() for t in leg.get_texts()], ["Perceptron", "NNLS", "recuperação perfeita"])
        hs = estilo_cinza._handles_legenda(leg)
        self.assertIn(hs[0].get_linestyle(), ("None", "", " "))          # proxy branco: sem linha
        self.assertIn(hs[1].get_linestyle(), ("None", "", " "))
        self.assertEqual(hs[2].get_linestyle(), "--")                     # referência: traço original
        self.assertEqual(matplotlib.colors.to_hex(hs[2].get_color()), "#000000")
        self.assertIn(hs[2].get_marker(), ("", "None", None))
        # a diagonal preta de 2 pontos rotulada mantém traço e cor, sem símbolo
        diag = [ln for ln in ax.lines if ln.get_label() == "recuperação perfeita"][0]
        self.assertEqual(diag.get_linestyle(), "--")
        self.assertEqual(matplotlib.colors.to_hex(diag.get_color()), "#000000")
        self.assertIn(diag.get_marker(), ("", "None", None))


class TestContornoEProxy(unittest.TestCase):
    """Auditoria de 09/10/2026 (rodada 2): o proxy de fronteira na legenda tem o mesmo traço do
    contorno, é preto e não ganha marcador (bloco1_graphic_14, bloco3_graphic_01/02)."""

    def tearDown(self):
        plt.close("all")

    def _malha(self):
        xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 60), np.linspace(-1.5, 1.5, 60))
        return xx, yy, xx ** 2 + yy ** 2 - 1.0

    def test_proxy_explicito_tracejado(self):
        xx, yy, zz = self._malha()
        fig, ax = plt.subplots()
        rng = np.random.RandomState(0)
        ax.scatter(rng.normal(size=20), rng.normal(size=20), color="tab:blue", label="Mulher")
        ax.scatter(rng.normal(size=20), rng.normal(size=20), color="tab:red", label="Homem")
        ax.contour(xx, yy, zz, levels=[0.0], colors="black", linewidths=1.8, linestyles="--")
        proxy, = ax.plot([], [], color="black", linestyle="--", label="Fronteira ótima")
        ax.legend()
        garantir_legendas(fig)
        para_cinza(fig)
        self.assertEqual(proxy.get_linestyle(), "--")
        self.assertIn(proxy.get_marker(), ("", "None", None))
        self.assertEqual(matplotlib.colors.to_hex(proxy.get_color()), "#000000")
        cs = [c for c in ax.collections if isinstance(c, matplotlib.contour.ContourSet)][0]
        self.assertEqual(estilo_cinza._traco_contorno(cs), "--")
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()], ["Mulher", "Homem", "Fronteira ótima"])
        h = estilo_cinza._handles_legenda(ax.get_legend())[2]
        self.assertEqual(h.get_linestyle(), "--")
        self.assertIn(h.get_marker(), ("", "None", None))

    def test_proxy_automatico_segue_o_contorno_solido(self):
        xx, yy, zz = self._malha()
        fig, ax = plt.subplots()
        rng = np.random.RandomState(1)
        ax.scatter(rng.normal(size=20), rng.normal(size=20), color="green")
        ax.contour(xx, yy, zz, levels=[0.0], colors="green", linewidths=1.5)
        garantir_legendas(fig)
        textos = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn("fronteira de decisão", textos)
        proxy = [ln for ln in ax.lines if ln.get_label() == "fronteira de decisão"][0]
        self.assertEqual(proxy.get_linestyle(), "-")
        para_cinza(fig)
        self.assertEqual(proxy.get_linestyle(), "-")
        self.assertIn(proxy.get_marker(), ("", "None", None))
        self.assertEqual(matplotlib.colors.to_hex(proxy.get_color()), "#000000")
        cs = [c for c in ax.collections if isinstance(c, matplotlib.contour.ContourSet)][0]
        self.assertEqual(estilo_cinza._traco_contorno(cs), "-")

    def test_contorno_herda_traco_do_proxy(self):
        xx, yy, zz = self._malha()
        fig, ax = plt.subplots()
        ax.scatter(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), color="tab:blue", label="pts")
        ax.contour(xx, yy, zz, levels=[0.0], colors="green", linewidths=1.5)        # sólido
        ax.plot([], [], color="green", linestyle="dashed", label="métrica diagonal")  # proxy tracejado
        ax.legend()
        garantir_legendas(fig)
        para_cinza(fig)
        cs = [c for c in ax.collections if isinstance(c, matplotlib.contour.ContourSet)][0]
        self.assertEqual(estilo_cinza._traco_contorno(cs), "--")                  # contorno = proxy


class TestBoxplotSemPatchArtist(unittest.TestCase):
    """Auditoria de 09/10/2026 (rodada 2): caixa de ``boxplot`` sem ``patch_artist`` (linha
    fechada de 5 pontos) não é série e não vira ``série k``."""

    def tearDown(self):
        plt.close("all")

    def test_boxplot_sem_patch_artist_nao_gera_serie(self):
        fig, ax = plt.subplots()
        rng = np.random.RandomState(0)
        ax.boxplot([rng.normal(size=30), rng.normal(size=30) + 1], patch_artist=False)
        ax.set_ylabel("valor")
        self.assertFalse(tem_series(ax))
        garantir_legendas(fig)
        self.assertFalse(any(r.startswith("série") for r in _todos_rotulos(ax)))
        self.assertIsNone(ax.get_legend())
        para_cinza(fig)
        for ln in ax.lines:
            self.assertEqual(matplotlib.colors.to_hex(ln.get_color()), "#000000")
            self.assertNotIn(ln.get_marker(), estilo_cinza.SIMBOLOS)


class TestLegendaForaDoEixo(unittest.TestCase):
    """Auditoria de 09/10/2026 (rodada 2): legenda ancorada fora do eixo (``bbox_to_anchor``)
    continua fora depois de refeita pela conversão (bloco23_graphic_01)."""

    def tearDown(self):
        plt.close("all")

    def test_bbox_to_anchor_preservado(self):
        fig, ax = plt.subplots()
        x = np.arange(4)
        ax.errorbar(x, x * .1, yerr=np.full(4, .02), marker="o", color="steelblue", label="G1 | peso/altura")
        ax.errorbar(x, x * .12, yerr=np.full(4, .02), marker="s", color="coral", label="G2 | peso_kg/altura_m")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
        garantir_legendas(fig)
        para_cinza(fig)
        leg = ax.get_legend()
        self.assertEqual([t.get_text() for t in leg.get_texts()], ["G1 | peso/altura", "G2 | peso_kg/altura_m"])
        anc = leg.get_bbox_to_anchor()
        b = anc._bbox
        self.assertAlmostEqual(b.x0, 0.5); self.assertAlmostEqual(b.y0, -0.14)
        self.assertEqual(leg._ncols, 2)


if __name__ == "__main__":
    unittest.main()
