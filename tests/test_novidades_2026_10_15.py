"""Novidades de 14 e 15/10/2026 que a suíte não exercitava diretamente.

Cobre, sem nenhuma chamada à API (dublês de `async_llm_classify_point_openai`, como em
tests/test_adg_passos_2_a_5.py, ou de `collect_llm_decisions`):
(a) o lote assíncrono `collect_llm_decisions_lote`/`_async_collect_lote`: ordem dos resultados,
    tag de auditoria de cada job, semáforo compartilhado, um cliente por lote, 402 → exit 3;
(b) `_executar_figura`/`_figuras_em_paralelo`: captura de stdout, falha vira aviso, saída na ordem
    das tarefas, série com FIGURAS_PROCESSOS=1 e fallback em série quando o pool não serializa;
(c) o cronômetro de `execucao_io` (`iniciar_cronometro`, `decorrido`, marca nos cabeçalhos do
    `Tee`) e `relatorios.print_section`;
(d) o Passo 7 do G no MESMO espaço (R², R³, R⁴) que o LLM viu no Passo 6;
(e) `PROBLEM_A_CENTERS` (15/10/2026) e `create_problem_a(centers=)` com os centros legados;
(f) os leitores do oráculo no esquema de 14/10/2026 (Oráculo sobre E3/E1/E2) com leitura legada
    "sobre B" (tabela 17 e figura 19);
(g) a estrutura das figuras `bloco23_graphic_01` (9 painéis, séries por dimensão) e `04`
    (1 + 8 × 3 painéis).

As tarefas do pool de figuras (`_tarefa_ok`, `_tarefa_falha`, `_tarefa_lenta`) são funções de
nível de módulo, logo são pickláveis por nome no contexto `spawn`, tanto sob
`python -m unittest tests.test_novidades_2026_10_15` quanto sob `unittest discover -s tests`
(o filho reimporta este módulo pelo `sys.path` herdado). Testes que mexem em globais
(`execucao_io._T0/_T_MARCA`, `dm.async_client`, `dm.MAX_CONCURRENCY`, `dm.FIGURAS_PROCESSOS`,
`dm.CURRENT_TAG`, `dm.collect_llm_decisions`) salvam e restauram em setUp/tearDown.

Execução: .venv/bin/python -m unittest tests.test_novidades_2026_10_15
"""
import asyncio
import contextlib
import io
import os
import re
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dissertacao_mestrado as dm  # noqa: E402
import execucao_io  # noqa: E402
import relatorios  # noqa: E402
import plots  # noqa: E402
import gerar_tabelas as gt  # noqa: E402
import data_problems as dp  # noqa: E402

META = {"model_resolved": "modelo-teste", "inference_provider": "Teste"}
_RE_X1 = re.compile(r"x1\s*=\s*(-?\d+\.\d+)")


# ─────────────────────────────────────────────────────────────────────────────
# (a) collect_llm_decisions_lote / _async_collect_lote
# ─────────────────────────────────────────────────────────────────────────────
class _FakeLLM:
    """Dublê assíncrono de `async_llm_classify_point_openai`: 'A' se x1 < 0; mede o nº
    máximo de chamadas simultâneas em voo (para verificar o semáforo compartilhado)."""

    def __init__(self, atraso=0.0, erro=None):
        self.calls = 0
        self.em_voo = 0
        self.max_em_voo = 0
        self.atraso = atraso
        self.erro = erro

    async def __call__(self, async_client, model_name, prompt, temperature, system_msg, **kwargs):
        self.calls += 1
        if self.erro is not None:
            raise Exception(self.erro)
        self.em_voo += 1
        self.max_em_voo = max(self.max_em_voo, self.em_voo)
        try:
            if self.atraso:
                await asyncio.sleep(self.atraso)
            x1 = float(_RE_X1.findall(prompt)[-1])
            return ("A" if x1 < 0 else "B"), dict(META)
        finally:
            self.em_voo -= 1


class _FakeClient:
    def __init__(self):
        self.fechado = 0

    async def close(self):
        self.fechado += 1


class _BaseLote(unittest.TestCase):
    def setUp(self):
        self._saved = {k: getattr(dm, k) for k in (
            "INITIAL_BACKOFF", "async_llm_classify_point_openai", "async_client", "MODEL_NAME",
            "CURRENT_PROVIDER", "MAX_CONCURRENCY", "CURRENT_TAG", "collect_llm_decisions", "get_async_client")}
        self._saved["interactions"] = list(dm.LLM_INTERACTIONS)
        self._saved["abort"] = dict(dm._ABORT_CONTEXT)
        dm.INITIAL_BACKOFF = 0
        dm.MODEL_NAME = "modelo-teste"
        dm.CURRENT_PROVIDER = "openrouter"
        dm.LLM_INTERACTIONS.clear()
        dm.CURRENT_TAG = ""

    def tearDown(self):
        for k, v in self._saved.items():
            if k in ("interactions", "abort"):
                continue
            setattr(dm, k, v)
        dm.LLM_INTERACTIONS.clear(); dm.LLM_INTERACTIONS.extend(self._saved["interactions"])
        dm._ABORT_CONTEXT.clear(); dm._ABORT_CONTEXT.update(self._saved["abort"])

    @staticmethod
    def _jobs(tamanhos=(5, 7, 3), seed=0):
        rng = np.random.RandomState(seed)
        jobs = []
        for i, n in enumerate(tamanhos):
            X = rng.uniform(-3, 3, size=(n, 2))
            jobs.append(dict(X=X, nome_classe_0="A", nome_classe_1="B", examples=None,
                             tag=f"bloco1|problema=T|passo=3|regime=r|k=4|rep={i}|seed=1"))
        return jobs


class TestLoteAssincrono(_BaseLote):
    def test_caminho_assincrono_preserva_ordem_tags_e_cliente_injetado(self):
        fake = _FakeLLM(); dm.async_llm_classify_point_openai = fake
        cliente = object(); dm.async_client = cliente
        jobs = self._jobs()
        # sanidade: `collect_llm_decisions` é a original, logo o lote toma o caminho async
        self.assertIs(dm.collect_llm_decisions, dm._COLLECT_LLM_DECISIONS_ORIGINAL)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            res = dm.collect_llm_decisions_lote(jobs, rotulo="[T] ")
        self.assertIn("lote de 3 coletas (15 chamadas)", buf.getvalue())
        self.assertIn("lote concluído", buf.getvalue())
        self.assertEqual(len(res), 3)
        self.assertEqual(fake.calls, 15)
        for j, (y, n_malf) in zip(jobs, res):
            self.assertEqual(len(y), len(j["X"]))
            np.testing.assert_array_equal(y, (j["X"][:, 0] >= 0).astype(int))   # mesma ordem dos jobs
            self.assertEqual(n_malf, 0)
        # cada registro de auditoria leva a tag do SEU job (não a do último)
        por_tag = {}
        for r in dm.LLM_INTERACTIONS:
            por_tag[r["tag"]] = por_tag.get(r["tag"], 0) + 1
        self.assertEqual(por_tag, {j["tag"]: len(j["X"]) for j in jobs})
        # cliente injetado não é substituído nem fechado
        self.assertIs(dm.async_client, cliente)

    def test_semaforo_compartilhado_limita_o_total_em_voo(self):
        fake = _FakeLLM(atraso=0.003); dm.async_llm_classify_point_openai = fake
        dm.async_client = object(); dm.MAX_CONCURRENCY = 3
        jobs = self._jobs((12, 12))
        with contextlib.redirect_stdout(io.StringIO()):
            dm.collect_llm_decisions_lote(jobs)
        self.assertEqual(fake.calls, 24)
        self.assertLessEqual(fake.max_em_voo, 3)       # um semáforo para as duas coletas
        self.assertGreater(fake.max_em_voo, 1)         # e de fato concorrente

    def test_cliente_proprio_e_criado_uma_vez_e_fechado_no_fim(self):
        fake = _FakeLLM(); dm.async_llm_classify_point_openai = fake
        dm.async_client = None
        criados = []

        def _get(provider):
            c = _FakeClient(); criados.append(c); return c
        dm.get_async_client = _get
        with contextlib.redirect_stdout(io.StringIO()):
            res = dm.collect_llm_decisions_lote(self._jobs((4, 4, 4)))
        self.assertEqual(len(res), 3)
        self.assertEqual(len(criados), 1)              # UM cliente para o lote inteiro
        self.assertEqual(criados[0].fechado, 1)        # fechado exatamente uma vez
        self.assertIsNone(dm.async_client)             # global restaurado

    def test_um_unico_job_nao_abre_lote(self):
        fake = _FakeLLM(); dm.async_llm_classify_point_openai = fake
        dm.async_client = object()
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            res = dm.collect_llm_decisions_lote(self._jobs((6,)))
        self.assertEqual(len(res), 1); self.assertEqual(fake.calls, 6)
        self.assertNotIn("lote de", buf.getvalue())

    def test_lista_vazia(self):
        self.assertEqual(dm.collect_llm_decisions_lote([]), [])

    def test_402_no_lote_aborta_com_exit_3_sem_fallback_e_fecha_cliente(self):
        fake = _FakeLLM(erro="Error code: 402 - Insufficient credits")
        dm.async_llm_classify_point_openai = fake
        dm.async_client = None
        criados = []

        def _get(provider):
            c = _FakeClient(); criados.append(c); return c
        dm.get_async_client = _get
        with tempfile.TemporaryDirectory() as tmp:
            dm._ABORT_CONTEXT.clear(); dm._ABORT_CONTEXT.update(pasta=tmp)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
                dm.collect_llm_decisions_lote(self._jobs((3, 3)))
        self.assertEqual(cm.exception.code, 3)
        self.assertEqual(len(dm.LLM_INTERACTIONS), 0)          # nenhum fallback MD5
        self.assertEqual(criados[0].fechado, 1)                # cliente fechado dentro do laço
        self.assertIsNone(dm.async_client)

    def test_duble_sincrono_chama_uma_a_uma_e_cada_chamada_ve_sua_tag(self):
        vistos = []

        def dublê(X, c0, c1, examples=None, **kw):
            vistos.append((len(X), dm.CURRENT_TAG, kw.get("tag")))
            return np.zeros(len(X), int), 0
        dm.collect_llm_decisions = dublê
        jobs = self._jobs()
        res = dm.collect_llm_decisions_lote(jobs)
        self.assertEqual(len(res), 3)
        self.assertEqual([v[0] for v in vistos], [5, 7, 3])                     # ordem
        self.assertEqual([v[1] for v in vistos], [j["tag"] for j in jobs])      # global atualizado por job
        self.assertEqual([v[2] for v in vistos], [j["tag"] for j in jobs])      # e o argumento também


# ─────────────────────────────────────────────────────────────────────────────
# (b) _executar_figura / _figuras_em_paralelo
# Tarefas de NÍVEL DE MÓDULO: o pool `spawn` as pickla por nome (módulo + qualname) e o processo
# filho reimporta este módulo; funções locais ou lambdas não serializam (ver o teste de fallback).
# ─────────────────────────────────────────────────────────────────────────────
def _tarefa_ok(msg, filename=None):
    print(f"saida {msg}")


def _tarefa_falha(filename=None):
    raise RuntimeError("boom")


def _tarefa_lenta(msg, filename=None):
    time.sleep(0.2); print(f"saida {msg}")


class TestFiguras(unittest.TestCase):
    def setUp(self):
        self._procs = dm.FIGURAS_PROCESSOS

    def tearDown(self):
        dm.FIGURAS_PROCESSOS = self._procs

    def test_executar_figura_captura_stdout_e_transforma_excecao_em_aviso(self):
        self.assertEqual(dm._executar_figura((_tarefa_ok, ("um",), {})), "saida um\n")
        txt = dm._executar_figura((_tarefa_falha, (), {"filename": "f.png"}))
        self.assertEqual(txt, "  ⚠ Falha em _tarefa_falha (f.png): boom\n")

    def _roda(self, tarefas, rotulo=""):
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            dm._figuras_em_paralelo(tarefas, rotulo=rotulo)
        return buf.getvalue()

    def test_paralelo_imprime_no_pai_na_ordem_e_falha_nao_derruba_as_outras(self):
        dm.FIGURAS_PROCESSOS = 2
        tarefas = [(_tarefa_lenta, ("um",), {}), (_tarefa_falha, (), {"filename": "x.png"}),
                   (_tarefa_ok, ("tres",), {"filename": "y.png"})]
        out = self._roda(tarefas, rotulo="T: ")
        linhas = [l for l in out.splitlines() if l.strip()]
        self.assertEqual(linhas[0], "saida um")                         # lenta, mas primeira (ordem das tarefas)
        self.assertEqual(linhas[1], "  ⚠ Falha em _tarefa_falha (x.png): boom")
        self.assertEqual(linhas[2], "saida tres")
        self.assertRegex(linhas[3], r"^  T: 3 figuras em \d+\.\ds \(2 processos\)$")
        self.assertNotIn("falharam", out)                                 # o pool funcionou

    def test_serie_quando_um_processo(self):
        dm.FIGURAS_PROCESSOS = 1
        out = self._roda([(_tarefa_ok, ("a",), {}), (_tarefa_ok, ("b",), {})])
        self.assertIn("saida a\nsaida b\n", out)
        self.assertRegex(out, r"2 figuras em \d+\.\ds \(série\)")

    def test_fallback_em_serie_quando_o_pool_nao_consegue_serializar(self):
        dm.FIGURAS_PROCESSOS = 2
        nao_serializavel = (lambda filename=None: print("lambda ok"))     # lambda não pickla no spawn
        out = self._roda([(nao_serializavel, (), {}), (_tarefa_ok, ("b",), {})])
        self.assertIn("Figuras em paralelo falharam", out)
        self.assertIn("lambda ok", out)
        self.assertIn("saida b", out)


# ─────────────────────────────────────────────────────────────────────────────
# (c) Cronômetro: execucao_io.iniciar_cronometro / decorrido / Tee.write / print_section
# ─────────────────────────────────────────────────────────────────────────────
class TestCronometro(unittest.TestCase):
    def setUp(self):
        self._t0, self._marca = execucao_io._T0, execucao_io._T_MARCA

    def tearDown(self):
        execucao_io._T0, execucao_io._T_MARCA = self._t0, self._marca

    def test_sem_cronometro_decorrido_vazio_e_tee_nao_marca(self):
        execucao_io._T0 = execucao_io._T_MARCA = None
        self.assertEqual(execucao_io.decorrido(), "")
        s = io.StringIO(); tee = execucao_io.Tee(s)
        tee.write(">>> Etapa"); tee.write("\n")
        self.assertEqual(tee.getvalue(), ">>> Etapa\n")

    def test_formato_total_e_intervalo_desde_a_marca(self):
        agora = time.time()
        execucao_io._T0 = agora - 3725          # 1:02:05
        execucao_io._T_MARCA = agora - 65       # +0:01:05
        txt = execucao_io.decorrido(marcar=False)
        self.assertRegex(txt, r"^⏱ 1:02:0[56] \(\+0:01:0[56]\)$")
        self.assertAlmostEqual(execucao_io._T_MARCA, agora - 65, places=3)   # marcar=False não move a marca
        execucao_io.decorrido()                                              # marcar=True move
        self.assertGreater(execucao_io._T_MARCA, agora - 1)
        self.assertRegex(execucao_io.decorrido(marcar=False), r"\(\+0:00:00\)$")

    def test_iniciar_zera(self):
        execucao_io.iniciar_cronometro()
        self.assertRegex(execucao_io.decorrido(), r"^⏱ 0:00:00 \(\+0:00:00\)$")

    def test_tee_marca_so_os_cabecalhos_de_etapa(self):
        execucao_io.iniciar_cronometro()
        s = io.StringIO(); tee = execucao_io.Tee(s)
        for msg in (">>> Bloco 1", "\n", "[ETAPA 3] tabelas", "\n", "linha comum", "\n", "\n>>> com quebra antes", "\n"):
            tee.write(msg)
        linhas = tee.getvalue().split("\n")
        self.assertRegex(linhas[0], r"^>>> Bloco 1   ⏱ 0:00:00 \(\+0:00:00\)$")
        self.assertRegex(linhas[1], r"^\[ETAPA 3\] tabelas   ⏱ ")
        self.assertEqual(linhas[2], "linha comum")
        self.assertRegex(linhas[4], r"^>>> com quebra antes   ⏱ ")
        self.assertEqual(s.getvalue(), tee.getvalue())             # terminal e buffer iguais
        # uma só escrita já com "\n" (sys.stdout.write) NÃO recebe a marca: só o par print() → write(msg), write("\n")
        tee2 = execucao_io.Tee(io.StringIO()); tee2.write(">>> direto\n")
        self.assertEqual(tee2.getvalue(), ">>> direto\n")

    def test_print_section_com_e_sem_cronometro(self):
        execucao_io._T0 = None
        with contextlib.redirect_stdout(io.StringIO()) as b1:
            relatorios.print_section("Sem marca")
        self.assertIn("\n Sem marca\n", b1.getvalue()); self.assertNotIn("⏱", b1.getvalue())
        execucao_io.iniciar_cronometro()
        with contextlib.redirect_stdout(io.StringIO()) as b2:
            relatorios.print_section("Com marca")
        self.assertRegex(b2.getvalue(), r"\n Com marca   ⏱ 0:00:00 \(\+0:00:00\)\n")


# ─────────────────────────────────────────────────────────────────────────────
# (d) Passo 7 do G no MESMO espaço (R², R³, R⁴) que o LLM viu no Passo 6
# (a mesma verificação está em tests/test_g_passo6_atributos_prompt.py, junto do Passo 6)
# ─────────────────────────────────────────────────────────────────────────────
def _dados_g(n=80, seed=11):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    return X, (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)


class TestPasso7MesmoEspaco(unittest.TestCase):
    def setUp(self):
        self._saved = {"collect": dm.collect_llm_decisions, "reps": dm.N_REPETICOES,
                       "k": list(dm.K_EXEMPLOS_ROTULO_REAL), "cache": dict(dm.CACHE_ROTULOS_LLM),
                       "model": dm.MODEL_NAME, "provider": dm.CURRENT_PROVIDER,
                       "perc": dm.train_relaxed_perceptron, "pred": dm.predict_with_metric}
        dm.N_REPETICOES = 1; dm.K_EXEMPLOS_ROTULO_REAL = [4]; dm.CACHE_ROTULOS_LLM.clear()
        dm.MODEL_NAME = "modelo-teste"; dm.CURRENT_PROVIDER = "openrouter"
        dm.collect_llm_decisions = lambda X, c0, c1, examples=None, **kw: (
            (np.asarray(X)[:, 0] + 0.3 * np.asarray(X)[:, 1] > 0).astype(int), 0)
        self.treinos, self.predicoes = [], []
        orig_perc, orig_pred = dm.train_relaxed_perceptron, dm.predict_with_metric

        def perc(X, y, c, **kw):
            self.treinos.append((np.asarray(X).shape, np.asarray(c).shape)); return orig_perc(X, y, c, **kw)

        def pred(X, c, w):
            self.predicoes.append((np.asarray(X).shape[1], len(np.asarray(w)))); return orig_pred(X, c, w)
        dm.train_relaxed_perceptron = perc; dm.predict_with_metric = pred

    def tearDown(self):
        dm.collect_llm_decisions = self._saved["collect"]; dm.N_REPETICOES = self._saved["reps"]
        dm.K_EXEMPLOS_ROTULO_REAL = self._saved["k"]
        dm.CACHE_ROTULOS_LLM.clear(); dm.CACHE_ROTULOS_LLM.update(self._saved["cache"])
        dm.MODEL_NAME = self._saved["model"]; dm.CURRENT_PROVIDER = self._saved["provider"]
        dm.train_relaxed_perceptron = self._saved["perc"]; dm.predict_with_metric = self._saved["pred"]

    def test_perceptron_do_passo_7_treina_e_prediz_na_dimensao_do_prompt(self):
        X, y = _dados_g()
        with contextlib.redirect_stdout(io.StringIO()):
            out = dm.fg_executar(problem_name="G1_homem_mulher", X=X, y_true=y, nome_classe_0="Mulher",
                                 nome_classe_1="Homem", feature_variants=[("peso", "altura")], seeds=[5],
                                 ks_aprendiz=[0, 4], n_train_ratio=0.7, verbose=False, com_passos_3_a_5=True,
                                 atributos_prompt_aprendiz=(2, 3, 4))
        # treinos com 4 linhas = Passo 7 (k = 4 exemplos); um por dimensão do prompt, centróides na mesma dimensão
        passo7 = sorted({(xs, cs) for xs, cs in self.treinos if xs[0] == 4})
        self.assertEqual(passo7, [((4, 2), (2, 2)), ((4, 3), (2, 3)), ((4, 4), (2, 4))])
        # nunca prediz com W de dimensão diferente da de X
        self.assertTrue(all(d == dw for d, dw in self.predicoes))
        linhas = {(r["n_features"], r["k"]): r for r in out["aprendiz"]}
        for d in (2, 3, 4):
            self.assertIsNotNone(linhas[(d, 4)]["accuracy_perceptron_baseline_vs_true"])
            self.assertIsNone(linhas[(d, 0)]["accuracy_perceptron_baseline_vs_true"])


# ─────────────────────────────────────────────────────────────────────────────
# (e) PROBLEM_A_CENTERS (15/10/2026) e create_problem_a(centers=)
# ─────────────────────────────────────────────────────────────────────────────
class TestProblemACenters(unittest.TestCase):
    def test_centros_novos_e_legados(self):
        self.assertEqual(dp.PROBLEM_A_CENTERS, [(-2.0, -0.5), (2.0, 0.5)])
        self.assertEqual(dp.PROBLEM_A_CENTERS_ATE_2026_10_14, [(-2.0, 0.0), (2.0, 0.0)])
        self.assertEqual(dp.PROBLEM_B_CENTERS, [(-2.0, 1.5), (2.0, -1.5)])   # B e C não mudaram
        self.assertEqual(dp.PROBLEM_C_CENTERS, [(-2.0, -1.5), (2.0, 1.5)])

    def test_padrao_usa_centros_novos_e_centers_reproduz_o_legado(self):
        X, y = dp.create_problem_a(600, 42)
        for c, alvo in zip((0, 1), dp.PROBLEM_A_CENTERS):
            np.testing.assert_allclose(X[y == c].mean(axis=0), alvo, atol=0.25)
        Xl, yl = dp.create_problem_a(600, 42, centers=dp.PROBLEM_A_CENTERS_ATE_2026_10_14)
        for c, alvo in zip((0, 1), dp.PROBLEM_A_CENTERS_ATE_2026_10_14):
            np.testing.assert_allclose(Xl[yl == c].mean(axis=0), alvo, atol=0.25)
        np.testing.assert_array_equal(y, yl)                       # mesmo sorteio, só o deslocamento muda
        np.testing.assert_allclose(X - Xl, np.where(y[:, None] == 0, [0, -0.5], [0, 0.5]), atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# (f) Leitores do oráculo no esquema de 14/10/2026 (Oráculo sobre E3/E1/E2)
# ─────────────────────────────────────────────────────────────────────────────
def _linhas_oraculo(p1="E3", seeds=(42, 123, 7), legado=False):
    cfg = [(f"Oráculo sobre {p1}", "euclidiana" if legado else p1, (1.0, 1.0)),
           ("Oráculo sobre E1", "E1", (0.3, 1.5)), ("Oráculo sobre E2", "E2", (1.5, 0.3))]
    rows = []
    for seed in seeds:
        for prob, exp, tw in cfg:
            for algo, dv in (("perceptron", 0.02), ("nnls", -0.02)):
                wr = np.array(tw) * (1 + dv * (seed % 5))
                rows.append({"random_seed": seed, "problem": prob, "expert_name": exp, "true_w_0": tw[0],
                             "true_w_1": tw[1], "true_w_ratio": tw[0] / tw[1], "algorithm": algo,
                             "recovered_w_0": wr[0], "recovered_w_1": wr[1],
                             "recovered_w_norm_0": wr[0] / wr.sum(), "recovered_w_norm_1": wr[1] / wr.sum(),
                             "recovered_w_ratio": wr[0] / wr[1], "cosine_similarity": 0.99,
                             "accuracy": 0.95 + dv})
    return rows


class _Captura:
    """Troca plots.salvar_figura por um capturador da figura (eixos visíveis, legendas, títulos)."""

    def __init__(self):
        self.figs = []

    def __enter__(self):
        self._orig = plots.salvar_figura

        def _cap(fig, filename, dpi=150, ja_cinza=False, cor=None):
            eixos = [ax for ax in fig.get_axes() if ax.get_visible()]
            self.figs.append({
                "n_eixos": len(eixos),
                "titulos": [ax.get_title() for ax in eixos],
                "legendas": [[t.get_text() for t in ax.get_legend().get_texts()] if ax.get_legend() else [] for ax in eixos],
                "legendas_fig": [t.get_text() for lg in fig.legends for t in lg.get_texts()],   # fig.legend (15/10/2026)
                "ticks": [t.get_text() for ax in eixos for t in ax.get_xticklabels()],
                "suptitle": fig.get_suptitle(),
            })
        plots.salvar_figura = _cap
        return self

    def __exit__(self, *exc):
        plots.salvar_figura = self._orig


class TestOraculoEsquemaE3(unittest.TestCase):
    def test_tabela_17_le_e3_como_problema_1(self):
        t = gt.tab_oraculo({"oraculo": pd.DataFrame(_linhas_oraculo())})
        self.assertIsNotNone(t)
        self.assertTrue(t.caption.startswith("Oráculo sobre E3, E1 e E2"))
        self.assertIn("Problema 1 = Oráculo sobre E3", t.rodape); self.assertNotIn("sobre B", t.rodape)
        self.assertEqual([r[0] for r in t.rows[::2]], ["Problema 1", "Problema 2", "Problema 3"])
        self.assertEqual(t.rows[0][2], "E (150 pts)")
        self.assertEqual(len(t.rows), 6)

    def test_tabela_17_mantem_leitura_legada_sobre_b(self):
        t = gt.tab_oraculo({"oraculo": pd.DataFrame(_linhas_oraculo(p1="B", legado=True))})
        self.assertTrue(t.caption.startswith("Oráculo sobre B, E1 e E2"))
        self.assertEqual(t.rows[0][2], "B (100 pts)")
        self.assertIn("Problema 1 = Oráculo sobre B", t.rodape)

    def test_figura_19_rotula_e3_e_aceita_legado(self):
        with tempfile.TemporaryDirectory() as tmp, _Captura() as cap:
            plots.plot_oraculo_recuperacao_w(_linhas_oraculo(), os.path.join(tmp, "bloco1_graphic_19_oraculo_passos_1_a_3.png"))
        ticks = {t for f in cap.figs for t in f["ticks"]}
        self.assertIn("Oráculo sobre E3", ticks); self.assertIn("Oráculo sobre E1", ticks)
        self.assertFalse(any("sobre B" in t for t in ticks))
        with tempfile.TemporaryDirectory() as tmp, _Captura() as cap2:
            plots.plot_oraculo_recuperacao_w(_linhas_oraculo(p1="B", legado=True), os.path.join(tmp, "bloco1_graphic_19_oraculo_passos_1_a_3.png"))
        ticks2 = {t for f in cap2.figs for t in f["ticks"]}
        self.assertIn("Oráculo sobre B", ticks2); self.assertFalse(any("euclidiana" in t for t in ticks2))


# ─────────────────────────────────────────────────────────────────────────────
# (g) Estrutura das figuras bloco23_graphic_01 (9 painéis, séries por dimensão) e 04
# ─────────────────────────────────────────────────────────────────────────────
def _linhas_aprendiz(dims=(2, 3, 4), ks=(0, 4, 10), seeds=(42, 7)):
    combos = [("F_meia_lua_seed{s}", ("x1", "x2"))]
    for g in ("G1_homem_mulher", "G1_homem_mulher_classesAB"):
        combos += [(g, ("x1", "x2")), (g, ("peso", "altura"))]
    for g in ("G2_homem_mulher", "G2_homem_mulher_classesAB"):
        combos += [(g, ("x1", "x2")), (g, ("peso_kg", "altura_m"))]
    rows = []
    for pb, feat in combos:
        for s in seeds:
            for d in (dims if pb.startswith("G") else (2,)):
                for k in ks:
                    rows.append({"problem_name": pb.format(s=s), "feature_names": feat, "seed": s, "k": k, "rep": 0,
                                 "n_features": d, "accuracy_llm_vs_expert": 0.7 + 0.01 * k,
                                 "kappa_llm_vs_expert": 0.5, "f1_llm_vs_expert": 0.7,
                                 "accuracy_llm_vs_true": 0.6 + 0.01 * k + 0.02 * d,
                                 "accuracy_perceptron_baseline_vs_true": (0.65 if k else None)})
    return rows


class TestFigurasFG(unittest.TestCase):
    ROTULOS_DIM = ["R² no prompt (x1, x2)", "R³ no prompt (+ x1·x2)", "R⁴ no prompt (+ x1², x2²)"]   # 15/10/2026: R²/R³/R⁴ em todo texto visível
    ROTULOS_METRICA = ["Acurácia (LLM vs rótulo real)", "Acurácia (LLM vs especialista)"]

    def test_bloco23_01_nove_paineis_series_por_dimensao_e_legenda(self):
        with tempfile.TemporaryDirectory() as tmp, _Captura() as cap:
            plots.plot_fg_aprendiz_curva(_linhas_aprendiz(), os.path.join(tmp, "bloco23_graphic_01_FG_aprendiz_curva__gpt4mini.png"))
        f = cap.figs[0]
        self.assertEqual(f["n_eixos"], 9)
        self.assertEqual(f["titulos"][0], "F (meia-lua) | x1/x2")
        self.assertEqual(f["titulos"][1:3], ["G1 Mulher/Homem | peso/altura", "G1 Mulher/Homem | x1/x2"])   # Mulher/Homem antes de A/B
        for i, leg in enumerate(f["legendas"]):
            esperado = (self.ROTULOS_DIM if i > 0 else self.ROTULOS_DIM[:1]) + self.ROTULOS_METRICA
            self.assertEqual(leg, esperado, msg=f"painel {f['titulos'][i]}")
        self.assertIn("(uma série por nº de atributos no prompt)", f["suptitle"])

    def test_bloco23_01_sem_coluna_n_features_tem_uma_serie(self):
        rows = [{k: v for k, v in r.items() if k != "n_features"} for r in _linhas_aprendiz(dims=(2,))]
        with tempfile.TemporaryDirectory() as tmp, _Captura() as cap:
            plots.plot_fg_aprendiz_curva(rows, os.path.join(tmp, "bloco23_graphic_01_FG_aprendiz_curva__gpt4mini.png"))
        f = cap.figs[0]
        self.assertTrue(all(leg == self.ROTULOS_DIM[:1] + self.ROTULOS_METRICA for leg in f["legendas"]))
        self.assertNotIn("uma série por", f["suptitle"])

    def test_bloco23_04_um_painel_por_dimensao_e_rotulos(self):
        with tempfile.TemporaryDirectory() as tmp, _Captura() as cap:
            plots.plot_fg_llm_vs_perceptron(_linhas_aprendiz(), os.path.join(tmp, "bloco23_graphic_04_FG_llm_vs_perceptron__gpt4mini.png"))
        f = cap.figs[0]
        self.assertEqual(f["n_eixos"], 1 + 8 * 3)        # F (R2) + 8 combinações do G × 3 dimensões
        self.assertEqual(sum(" | R³ no prompt" in t for t in f["titulos"]), 8)
        # 15/10/2026: legenda ÚNICA da figura (fig.legend), em vez de uma por painel cobrindo as barras
        self.assertEqual(f["legendas_fig"], ["LLM (em contexto)", "Perceptron (Passo 4 do F / Passo 7 do G) nos mesmos exemplos"])
        self.assertTrue(all(leg == [] for leg in f["legendas"]))
        self.assertTrue(f["suptitle"].startswith("Problema F, Passo 4, e Problema G, Passo 7"))


if __name__ == "__main__":
    unittest.main()
