"""Testes de unidade do parser de respostas do LLM (`parse_llm_response`).

O parser é a peça mais frágil do pipeline: decide, em 8 camadas de permissividade
crescente, qual classe uma resposta livre do LLM representa. Quando nenhuma camada
casa, retorna ``(None, False)`` — sinal de resposta malformada que aciona o fallback
determinístico por hash. Estes testes fixam o contrato camada a camada e, em especial,
garantem que respostas ambíguas/inválidas NÃO sejam silenciosamente aceitas.

Execução (a partir de src/, para que os imports por nome simples resolvam):
    cd src && ../.venv/bin/python -m pytest ../tests/test_parser.py -q
ou:
    python -m unittest discover -s tests
"""
import os
import sys
import unittest

# Permite importar o módulo principal independentemente do diretório de invocação.
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402

parse = dm.parse_llm_response


class TestParserCasosValidos(unittest.TestCase):
    """Cada camada do parser deve reconhecer corretamente a classe pretendida."""

    def test_match_exato_case_insensitive(self):
        self.assertEqual(parse("A", "A", "B"), ("A", True))
        self.assertEqual(parse("b", "A", "B"), ("B", True))
        self.assertEqual(parse("POSITIVO", "Positivo", "Negativo"), ("Positivo", True))

    def test_remocao_de_aspas_e_pontuacao(self):
        self.assertEqual(parse('"A".', "A", "B"), ("A", True))
        self.assertEqual(parse("'Negativo'!", "Positivo", "Negativo"), ("Negativo", True))

    def test_classes_numericas_com_prefixo(self):
        self.assertEqual(parse("Class 0", "0", "1"), ("0", True))
        self.assertEqual(parse("a resposta e 1", "0", "1"), ("1", True))

    def test_word_boundary_classe_isolada(self):
        # Quando só UMA das classes aparece como palavra isolada, ela é escolhida.
        self.assertEqual(parse("eu escolho a classe A aqui", "A", "B"), ("A", True))
        self.assertEqual(parse("classe B", "A", "B"), ("B", True))

    def test_preposicao_colide_com_nome_de_uma_letra(self):
        # LIMITAÇÃO CONHECIDA: a preposição portuguesa "a" casa como palavra isolada
        # com a classe "A". Numa frase que também cite "B", as duas classes aparecem
        # como palavra → ambiguidade → malformado. Fixado para tornar explícito que
        # nomes de classe de uma letra são frágeis em texto livre em português.
        self.assertEqual(parse("o ponto pertence a B", "A", "B"), (None, False))

    def test_starts_with_e_guloso_para_nomes_de_uma_letra(self):
        # LIMITAÇÃO CONHECIDA (não um bug deste teste): com nomes de classe de uma
        # única letra, a camada 7 ("starts-with") casa qualquer palavra que COMECE
        # com aquela letra — "Banana" → "B", "Abacaxi" → "A". Na prática o LLM
        # raramente responde com palavras assim, mas o comportamento fica fixado
        # aqui para que qualquer mudança futura no parser seja intencional.
        self.assertEqual(parse("Banana", "A", "B"), ("B", True))
        self.assertEqual(parse("Abacaxi", "A", "B"), ("A", True))

    def test_contains_exclusivo(self):
        self.assertEqual(parse("provavelmente Azul", "Azul", "Vermelho"), ("Azul", True))

    def test_padrao_linguagem_natural(self):
        self.assertEqual(parse("CLASSIFICATION: Positivo", "Positivo", "Negativo"),
                         ("Positivo", True))
        self.assertEqual(parse("This belongs to B", "A", "B"), ("B", True))

    def test_starts_with(self):
        self.assertEqual(parse("Azul, com alta confiança", "Azul", "Vermelho"),
                         ("Azul", True))

    def test_chain_of_thought_marcador_final(self):
        resposta = ("Os centróides estão deslocados em x2, então o ponto cai\n"
                    "mais perto do centróide superior.\nFinal answer: B")
        self.assertEqual(parse(resposta, "A", "B"), ("B", True))

    def test_chain_of_thought_ultima_linha(self):
        resposta = "Raciocínio longo aqui.\nMais raciocínio.\nA"
        self.assertEqual(parse(resposta, "A", "B"), ("A", True))


class TestParserCasosMalformados(unittest.TestCase):
    """Respostas inválidas/ambíguas devem retornar (None, False) — nunca um palpite."""

    def test_resposta_vazia(self):
        self.assertEqual(parse("", "A", "B"), (None, False))
        self.assertEqual(parse("   ", "A", "B"), (None, False))

    def test_nenhuma_classe_presente(self):
        self.assertEqual(parse("não sei dizer", "A", "B"), (None, False))

    def test_recusa_nao_vira_classe_por_substring(self):
        # Regressão da camada 5 (contains): recusas contêm a letra 'a' dentro de
        # palavras comuns ("cannot", "classificar") e viravam classe A com
        # valid=True — viés sistemático pró-classe-A invisível à taxa de
        # malformadas da auditoria. Com nomes de 1 letra o contains é desativado
        # (a camada 4, word-boundary, já cobre esses nomes com precisão).
        self.assertEqual(parse("I cannot classify this point", "A", "B"), (None, False))
        self.assertEqual(parse("não consigo classificar esse ponto", "A", "B"), (None, False))

    def test_hedge_com_padrao_is_nao_vira_classe(self):
        # Regressão da camada 6: "it is ambiguous" continha o padrão "IS A" como
        # substring e virava classe A; o padrão agora exige fronteira de palavra
        # após o nome da classe.
        self.assertEqual(parse("it is ambiguous", "A", "B"), (None, False))

    def test_ambiguo_ambas_as_classes(self):
        # Ambas aparecem como palavra → camadas word-boundary/contains não decidem.
        self.assertEqual(parse("pode ser A ou B", "A", "B"), (None, False))

    def test_retorno_e_sempre_tupla_de_dois(self):
        out = parse("qualquer coisa", "A", "B")
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], bool)


if __name__ == "__main__":
    unittest.main()
