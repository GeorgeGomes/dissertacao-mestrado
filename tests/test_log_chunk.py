"""Testes do fatiamento do log em partes (`salvar_log_em_chunks`).

Garante que:
- log pequeno → um único arquivo `log_execucao.txt` idêntico ao conteúdo;
- log grande → várias partes `log_execucao_parteNNN.txt`, cada uma dentro do limite,
  quebradas em limites de LINHA, e cuja concatenação (sem os cabeçalhos) reconstrói
  exatamente o conteúdo original;
- uma linha isolada maior que o limite não trava e vai sozinha numa parte.
"""
import os
import sys
import tempfile
import shutil
import unittest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402

HEADER_LINHAS = 4  # o cabeçalho de cada parte tem exatamente 4 linhas


def _corpo_sem_cabecalho(caminho):
    """Lê uma parte e devolve o corpo, descartando o cabeçalho de 4 linhas."""
    with open(caminho, encoding="utf-8") as f:
        linhas = f.readlines()
    return "".join(linhas[HEADER_LINHAS:])


class TestSalvarLogEmChunks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="log_chunk_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_log_pequeno_arquivo_unico(self):
        conteudo = "linha 1\nlinha 2\nlinha 3\n"
        paths = dm.salvar_log_em_chunks(conteudo, self.tmp, limite_bytes=1000)
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].endswith("log_execucao.txt"))
        with open(paths[0], encoding="utf-8") as f:
            self.assertEqual(f.read(), conteudo)  # idêntico, sem cabeçalho

    def test_log_grande_fatiado_em_varias_partes(self):
        # ~200 linhas de 100 bytes ≈ 20 KB, com limite de 1 KB → várias partes.
        conteudo = "".join(f"linha {i:04d} " + "x" * 90 + "\n" for i in range(200))
        limite = 1024
        paths = dm.salvar_log_em_chunks(conteudo, self.tmp, limite_bytes=limite)

        self.assertGreater(len(paths), 1)
        for p in paths:
            self.assertRegex(os.path.basename(p), r"log_execucao_parte\d{3}\.txt")

        # Cada CORPO de parte respeita o limite (linhas curtas → nenhuma estoura sozinha).
        for p in paths:
            corpo_bytes = len(_corpo_sem_cabecalho(p).encode("utf-8"))
            self.assertLessEqual(corpo_bytes, limite)

        # Concatenar os corpos reconstrói o conteúdo original exatamente.
        reconstruido = "".join(_corpo_sem_cabecalho(p) for p in paths)
        self.assertEqual(reconstruido, conteudo)

    def test_quebra_em_limite_de_linha(self):
        # Nenhuma linha pode ser cortada no meio: todo corpo termina em '\n'
        # (o conteúdo original termina em '\n'), logo cada linha fica inteira.
        conteudo = "".join(f"registro-{i}\n" for i in range(500))
        paths = dm.salvar_log_em_chunks(conteudo, self.tmp, limite_bytes=200)
        for p in paths:
            self.assertTrue(_corpo_sem_cabecalho(p).endswith("\n"))

    def test_linha_unica_maior_que_limite_nao_trava(self):
        # Uma única linha de 5 KB com limite de 1 KB: não dá para dividir uma linha,
        # então ela fica sozinha em uma parte (excede o limite, mas não entra em loop).
        conteudo = "L" * 5000 + "\n"
        paths = dm.salvar_log_em_chunks(conteudo, self.tmp, limite_bytes=1024)
        self.assertEqual(len(paths), 1)
        self.assertEqual(_corpo_sem_cabecalho(paths[0]), conteudo)


class TestTeeCapturaStderr(unittest.TestCase):
    """O log persistido precisa conter também o que vai para stderr.

    Regressão do smoke de 03/07/2026: 3 avisos de não-convergência de γ
    (warnings.warn → stderr) não apareceram no log_execucao.txt porque o Tee
    só capturava stdout — quem auditasse o log concluiria "zero avisos".
    """

    def test_stdout_e_stderr_compartilham_o_buffer_na_ordem(self):
        import io
        out, err = io.StringIO(), io.StringIO()
        tee = dm.Tee(out)
        tee_err = dm.Tee(err, tee._buffer)

        tee.write("linha stdout\n")
        tee_err.write("aviso stderr\n")
        tee.write("mais stdout\n")

        # cada stream real recebe só o próprio conteúdo...
        self.assertEqual(out.getvalue(), "linha stdout\nmais stdout\n")
        self.assertEqual(err.getvalue(), "aviso stderr\n")
        # ...e o buffer compartilhado tem TUDO, na ordem de chegada
        self.assertEqual(tee.getvalue(), "linha stdout\naviso stderr\nmais stdout\n")
        self.assertEqual(tee_err.getvalue(), tee.getvalue())

    def test_warning_de_estimador_chega_ao_buffer_do_log(self):
        # Integração: warnings.warn escreve em sys.stderr no momento do warn;
        # com o Tee instalado em stderr, o aviso tem que aparecer no buffer.
        import io
        import warnings as w
        tee = dm.Tee(io.StringIO())
        saved_stderr = sys.stderr
        sys.stderr = dm.Tee(io.StringIO(), tee._buffer)
        try:
            with w.catch_warnings():
                w.simplefilter("always")
                w.warn("gamma nao convergiu (teste)")
        finally:
            sys.stderr = saved_stderr
        self.assertIn("gamma nao convergiu", tee.getvalue())


if __name__ == "__main__":
    unittest.main()
