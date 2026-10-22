"""Testes de infraestrutura do runner (P7 da revisão de 09/10/2026).

(a) Reduções do `--rapido` (`grades_modo_rapido`/`aplicar_modo_rapido`) e nome/sufixo da
    pasta (`nome_pasta_execucao`): `_completa`, `_smoke`, `_apenas-<blocos>[_smoke]`.
(b) `Tee`: depois de `_abortar_por_creditos` sys.stdout/sys.stderr voltam aos streams
    originais e o buffer vira log_execucao.txt.
(c) Ida e volta `salvar_json_em_chunks` → `audit_interactions.load_interactions`, com 1
    parte e com várias.
(d) Ciclo de vida do cliente da API (item 5): `async_collect_llm_decisions` cria o cliente
    dentro do laço e o fecha com o laço AINDA ABERTO; um cliente injetado não é fechado.

Nenhuma chamada à API.
"""
import asyncio
import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402
import execucao_io  # noqa: E402
from audit_interactions import load_interactions  # noqa: E402

_GRADES = ("RANDOM_SEEDS", "N_REPETICOES", "K_EXEMPLOS_APRENDIZ", "K_EXEMPLOS_METRICA_BC",
           "K_EXEMPLOS_ROTULO_REAL", "K_VIESES", "DILUTION_EASY_ADDITIONS", "K_ORDEM_EXEMPLOS")


class TestModoRapidoEPasta(unittest.TestCase):
    def setUp(self):
        self._saved = {k: getattr(dm, k) for k in _GRADES}

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(dm, k, v)

    def test_grades_do_rapido_como_no_claude_md(self):
        g = dm.grades_modo_rapido()
        self.assertEqual(set(g), set(_GRADES))
        self.assertEqual(g["RANDOM_SEEDS"], [42])
        self.assertEqual(g["N_REPETICOES"], 1)
        self.assertEqual(g["K_EXEMPLOS_APRENDIZ"], [0, 4])
        self.assertEqual(g["K_EXEMPLOS_METRICA_BC"], [0, 4])
        self.assertEqual(g["K_EXEMPLOS_ROTULO_REAL"], [4])
        self.assertEqual(g["K_VIESES"], [0, 4])
        self.assertEqual(g["DILUTION_EASY_ADDITIONS"], [0, 4])
        self.assertEqual(g["K_ORDEM_EXEMPLOS"], [4])
        # todo k few-shot do smoke é par (balanço exato de classes no prompt)
        for k in ("K_EXEMPLOS_APRENDIZ", "K_EXEMPLOS_METRICA_BC", "K_EXEMPLOS_ROTULO_REAL", "K_VIESES", "K_ORDEM_EXEMPLOS"):
            self.assertTrue(all(v % 2 == 0 for v in g[k]), k)
        # cópia: mexer no retorno não altera a constante
        g["RANDOM_SEEDS"].append(1)
        self.assertEqual(dm.GRADES_MODO_RAPIDO["RANDOM_SEEDS"], [42])

    def test_grade_completa_difere_do_rapido(self):
        self.assertEqual(dm.RANDOM_SEEDS, [42, 123, 7])
        self.assertEqual(dm.N_REPETICOES, 3)
        self.assertEqual(dm.K_EXEMPLOS_APRENDIZ, [0, 4, 10, 20, 40])
        self.assertEqual(dm.K_EXEMPLOS_ROTULO_REAL, [4, 10, 20, 40])
        self.assertEqual(dm.K_EXEMPLOS_METRICA_BC, [0, 4, 10, 20, 40])
        self.assertEqual(dm.K_VIESES, [0, 10])
        self.assertEqual(dm.DILUTION_EASY_ADDITIONS, [0, 2, 4, 10, 16, 20])
        self.assertEqual(dm.K_ORDEM_EXEMPLOS, [4, 10, 20])

    def test_aplicar_modo_rapido_altera_o_modulo(self):
        g = dm.aplicar_modo_rapido()
        for k, v in g.items():
            self.assertEqual(getattr(dm, k), v, k)
        ns = {"RANDOM_SEEDS": [1, 2, 3]}
        dm.aplicar_modo_rapido(ns)
        self.assertEqual(ns["RANDOM_SEEDS"], [42])

    def test_nome_da_pasta_por_modo(self):
        ts = "2026-10-09_20-00-00"
        self.assertEqual(dm.nome_pasta_execucao(ts), f"execucao_{ts}_completa")
        self.assertEqual(dm.nome_pasta_execucao(ts, None, True), f"execucao_{ts}_smoke")
        self.assertEqual(dm.nome_pasta_execucao(ts, ["bloco3"]), f"execucao_{ts}_apenas-bloco3")
        self.assertEqual(dm.nome_pasta_execucao(ts, ["bloco3", "bloco1"]), f"execucao_{ts}_apenas-bloco1+bloco3")
        self.assertEqual(dm.nome_pasta_execucao(ts, ["oraculo"], True), f"execucao_{ts}_apenas-oraculo_smoke")
        self.assertEqual(dm.sufixo_apenas(["bloco2", "bloco3"]), "apenas-bloco2+bloco3")
        # uma parcial nunca é "_completa"
        for blocos in (["bloco1"], ["oraculo"], ["bloco1", "bloco2", "bloco3", "oraculo"]):
            self.assertNotIn("completa", dm.nome_pasta_execucao(ts, blocos))


class TestTeeNoAbort(unittest.TestCase):
    def setUp(self):
        self._ctx = dict(dm._ABORT_CONTEXT)
        self._inter = list(dm.LLM_INTERACTIONS)
        self._out, self._err = sys.stdout, sys.stderr
        self.tmp = tempfile.mkdtemp(prefix="tee_abort_")

    def tearDown(self):
        sys.stdout, sys.stderr = self._out, self._err
        dm._ABORT_CONTEXT.clear(); dm._ABORT_CONTEXT.update(self._ctx)
        dm.LLM_INTERACTIONS.clear(); dm.LLM_INTERACTIONS.extend(self._inter)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_streams_restaurados_e_buffer_gravado(self):
        base_out, base_err = io.StringIO(), io.StringIO()
        tee = execucao_io.Tee(base_out)
        tee_err = execucao_io.Tee(base_err, tee._buffer)
        sys.stdout, sys.stderr = tee, tee_err
        dm._ABORT_CONTEXT.clear(); dm._ABORT_CONTEXT.update(pasta=self.tmp, tee=tee, tee_err=tee_err)
        dm.LLM_INTERACTIONS.clear(); dm.LLM_INTERACTIONS.append({"prompt": "p", "parsed_label": "A"})
        print("linha antes do abort")
        print("aviso no stderr", file=sys.stderr)
        with self.assertRaises(SystemExit) as cm:
            dm._abortar_por_creditos(RuntimeError("Error code: 402 - Insufficient credits"))
        self.assertEqual(cm.exception.code, 3)
        # streams originais de volta (o `with` acima já saiu: o print do abort foi para o buffer)
        self.assertIs(sys.stdout, base_out)
        self.assertIs(sys.stderr, base_err)
        log = os.path.join(self.tmp, "log_execucao.txt")
        self.assertTrue(os.path.exists(log))
        conteudo = open(log, encoding="utf-8").read()
        self.assertIn("linha antes do abort", conteudo)
        self.assertIn("aviso no stderr", conteudo)
        self.assertIn("CRÉDITOS DO OPENROUTER ESGOTADOS", conteudo)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "llm_interactions_checkpoint.json")))


class TestRoundtripJsonChunks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="roundtrip_json_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _dados(self, n):
        return [{"i": i, "prompt": "p" * 50, "point": {"x1": i / 7, "x2": -i / 3},
                 "parsed_label": "A" if i % 2 else "B", "tag": f"bloco1|k={i}"} for i in range(n)]

    def test_uma_parte(self):
        dados = self._dados(20)
        paths = execucao_io.salvar_json_em_chunks(dados, self.tmp)
        self.assertEqual(len(paths), 1)
        self.assertEqual(load_interactions(self.tmp), dados)

    def test_varias_partes(self):
        dados = self._dados(400)
        paths = execucao_io.salvar_json_em_chunks(dados, self.tmp, limite_bytes=4_000)
        self.assertGreater(len(paths), 3)
        self.assertEqual(load_interactions(self.tmp), dados)        # ordem e conteúdo preservados

    def test_sem_arquivo_levanta(self):
        with self.assertRaises(FileNotFoundError):
            load_interactions(self.tmp)


class _ClienteDublê:
    """Registra se close() foi chamado e se o laço estava aberto nesse momento."""
    def __init__(self):
        self.fechado = False
        self.laco_aberto_no_close = None
        self.laco = None

    async def close(self):
        self.fechado = True
        self.laco = asyncio.get_running_loop()
        self.laco_aberto_no_close = not self.laco.is_closed()


class TestCicloDeVidaDoCliente(unittest.TestCase):
    def setUp(self):
        self._saved = {k: getattr(dm, k) for k in ("async_client", "get_async_client", "async_llm_classify_point_openai",
                                                    "MODEL_NAME", "CURRENT_PROVIDER", "CURRENT_TEMPERATURE")}
        self._inter = list(dm.LLM_INTERACTIONS)
        self.criados = []

        def fabrica(provider):
            self.assertEqual(provider, "openrouter")
            c = _ClienteDublê(); self.criados.append(c); return c
        self.usados = []

        async def api(client, model_name, prompt, temperature, system_msg, **kw):
            self.usados.append(client)
            return "A", {"model_resolved": "m", "inference_provider": "p"}
        dm.get_async_client = fabrica
        dm.async_llm_classify_point_openai = api
        dm.MODEL_NAME = "modelo-teste"; dm.CURRENT_PROVIDER = "openrouter"; dm.CURRENT_TEMPERATURE = 0.0

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(dm, k, v)
        dm.LLM_INTERACTIONS.clear(); dm.LLM_INTERACTIONS.extend(self._inter)

    def test_cliente_criado_por_coleta_e_fechado_com_o_laco_aberto(self):
        dm.async_client = None
        X = np.zeros((5, 2))
        with redirect_stdout(io.StringIO()):
            y1, _ = dm.collect_llm_decisions(X, "A", "B", verbose=False)
            y2, _ = dm.collect_llm_decisions(X, "A", "B", verbose=False)
        self.assertEqual(len(self.criados), 2)                       # um cliente por coleta
        for c in self.criados:
            self.assertTrue(c.fechado)
            self.assertTrue(c.laco_aberto_no_close)                  # aclose() dentro do laço que o criou
            self.assertTrue(c.laco.is_closed())                      # e o laço só fecha depois (asyncio.run)
        self.assertIsNot(self.criados[0].laco, self.criados[1].laco)
        self.assertTrue(all(u is self.criados[0] for u in self.usados[:5]))   # as chamadas usam o cliente da coleta
        self.assertTrue(all(u is self.criados[1] for u in self.usados[5:]))
        self.assertEqual(y1.tolist(), [0] * 5); self.assertEqual(y2.tolist(), [0] * 5)

    def test_cliente_injetado_e_usado_e_nao_fechado(self):
        injetado = _ClienteDublê()
        dm.async_client = injetado
        with redirect_stdout(io.StringIO()):
            dm.collect_llm_decisions(np.zeros((3, 2)), "A", "B", verbose=False)
        self.assertEqual(self.criados, [])
        self.assertFalse(injetado.fechado)
        self.assertTrue(all(u is injetado for u in self.usados))

    def test_fecha_tambem_no_abort_por_402(self):
        dm.async_client = None

        async def api_402(*a, **k):
            raise Exception("Error code: 402 - Insufficient credits")
        dm.async_llm_classify_point_openai = api_402
        ctx = dict(dm._ABORT_CONTEXT)
        try:
            dm._ABORT_CONTEXT.clear()
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                dm.collect_llm_decisions(np.zeros((2, 2)), "A", "B", verbose=False)
        finally:
            dm._ABORT_CONTEXT.clear(); dm._ABORT_CONTEXT.update(ctx)
        self.assertEqual(len(self.criados), 1)
        self.assertTrue(self.criados[0].fechado and self.criados[0].laco_aberto_no_close)


if __name__ == "__main__":
    unittest.main()
