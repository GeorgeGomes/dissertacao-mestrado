"""Testes do fatiamento do JSON de interações (`salvar_json_em_chunks`).

Diferente do log em texto, cada parte precisa ser um JSON VÁLIDO e independente.
Garante que:
- lista pequena → um único `llm_interactions.json` idêntico à lista;
- lista grande → várias partes `llm_interactions_parteNNN.json`, cada uma um array
  JSON válido dentro do limite, cuja concatenação reconstrói a lista original;
- um item isolado maior que o limite não trava e vai sozinho numa parte;
- objeto não-lista (não fatiável por elemento) vira arquivo único.
"""
import json
import os
import sys
import tempfile
import shutil
import unittest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402


def _load(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


class TestSalvarJsonEmChunks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="json_chunk_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_lista_pequena_arquivo_unico(self):
        dados = [{"i": 0, "prompt": "a"}, {"i": 1, "prompt": "b"}]
        paths = dm.salvar_json_em_chunks(dados, self.tmp, limite_bytes=10_000)
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].endswith("llm_interactions.json"))
        self.assertEqual(_load(paths[0]), dados)

    def test_lista_grande_fatiada_em_partes_validas(self):
        # 300 itens de ~100+ bytes cada, limite pequeno → várias partes.
        dados = [{"i": i, "resposta": "x" * 100} for i in range(300)]
        limite = 2048
        paths = dm.salvar_json_em_chunks(dados, self.tmp, limite_bytes=limite)

        self.assertGreater(len(paths), 1)
        for p in paths:
            self.assertRegex(os.path.basename(p), r"llm_interactions_parte\d{3}\.json")
            # Cada parte é um JSON VÁLIDO (json.load não levanta) e é uma lista.
            self.assertIsInstance(_load(p), list)
            # Cada parte respeita o limite (itens pequenos → nenhum estoura sozinho).
            self.assertLessEqual(os.path.getsize(p), limite + 200)  # folga p/ colchetes/indent

        # Concatenar as listas das partes reconstrói a lista original, na ordem.
        reconstruido = []
        for p in paths:
            reconstruido.extend(_load(p))
        self.assertEqual(reconstruido, dados)

    def test_item_unico_maior_que_limite_nao_trava(self):
        dados = [{"i": 0, "resposta": "L" * 5000}]
        paths = dm.salvar_json_em_chunks(dados, self.tmp, limite_bytes=1024)
        self.assertEqual(len(paths), 1)
        self.assertEqual(_load(paths[0]), dados)

    def test_nao_lista_vira_arquivo_unico(self):
        # Um dict grande não pode ser fatiado por elemento → arquivo único.
        dados = {"config": "y" * 5000}
        paths = dm.salvar_json_em_chunks(dados, self.tmp, limite_bytes=1024)
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].endswith("llm_interactions.json"))
        self.assertEqual(_load(paths[0]), dados)


if __name__ == "__main__":
    unittest.main()
