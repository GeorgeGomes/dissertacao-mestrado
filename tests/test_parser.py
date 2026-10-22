"""Testes de unidade do parser de respostas do LLM (`parse_llm_response`).

O parser é a peça mais frágil da coleta: decide, em 8 camadas de permissividade
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


class TestParserCamada6Padroes(unittest.TestCase):
    """Camada 6 (padrões "class X", "belongs to X"…): a classe vem da TUPLA do padrão.

    Regressão de 01/10/2026: a decisão por substring do nome dentro do padrão
    fazia "A" casar dentro de "CLASS" ("Class B" → A) e "B" dentro de "BELONGS".
    """

    def test_padrao_class_x_com_nomes_de_uma_letra(self):
        self.assertEqual(parse("Between A and B, I choose Class B", "A", "B"), ("B", True))
        self.assertEqual(parse("Between A and B, I choose Class A", "A", "B"), ("A", True))

    def test_padrao_com_ordem_invertida(self):
        self.assertEqual(parse("A or B? It belongs to A", "B", "A"), ("A", True))
        self.assertEqual(parse("A or B? It belongs to B", "B", "A"), ("B", True))

    def test_dois_padroes_declarativos_e_ambiguo(self):
        # A ordem dos padrões não decide: duas classes em padrões → malformada.
        self.assertEqual(parse("Class A ... Class B", "A", "B"), (None, False))
        self.assertEqual(parse("The answer is B, not Class A", "A", "B"), (None, False))

    def test_padrao_nao_casa_nome_que_e_prefixo_de_outro(self):
        self.assertEqual(parse("0 or 10? Class 10", "0", "10"), ("10", True))
        self.assertEqual(parse("1 or 10? Class 1", "1", "10"), ("1", True))

    def test_padroes_para_todos_os_pares_do_protocolo(self):
        pares = list(dm.NOMES_CLASSES) + list(dm.NOMES_CLASSES_INVERTIDAS) + [("Mulher", "Homem")]
        for c0, c1 in pares:
            with self.subTest(c0=c0, c1=c1):
                self.assertEqual(parse(f"{c0} or {c1}? Classified as {c1}", c0, c1), (c1, True))
                self.assertEqual(parse(f"{c0} or {c1}? Classified as {c0}", c0, c1), (c0, True))


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
        self.assertEqual(parse("1", "0", "1"), ("1", True))
        self.assertEqual(parse("The answer is 1.", "0", "1"), ("1", True))

    def test_classes_numericas_ignoram_decimais_e_coordenadas(self):
        # P3 (09/10/2026): a camada 3 contava todos os inteiros, inclusive pedaços
        # de coordenadas ("0.3" → "0" e "3"), e a resposta "1" perdia por contagem.
        # Agora só inteiros isolados valem e o ÚLTIMO candidato decide.
        self.assertEqual(parse("x1 = 0.3 is small and x2 = 0.8 is large. Answer: 1", "0", "1"),
                         ("1", True))
        self.assertEqual(parse("x1 = 1.3 and x2 = -1.25, so it is 0", "0", "1"), ("0", True))
        self.assertEqual(parse("0.5", "0", "1"), (None, False))
        self.assertEqual(parse("-1.25", "0", "1"), (None, False))
        self.assertEqual(parse("Class 10", "0", "1"), (None, False))
        self.assertEqual(parse("x1", "0", "1"), (None, False))

    def test_word_boundary_classe_isolada(self):
        # Quando só UMA das classes aparece como palavra isolada, ela é escolhida.
        self.assertEqual(parse("eu escolho a classe A aqui", "A", "B"), ("A", True))
        self.assertEqual(parse("classe B", "A", "B"), ("B", True))

    def test_preposicao_nao_colide_com_nome_de_uma_letra(self):
        # Desde 08/10/2026 nomes de 1 letra casam case-sensitive na camada 4: a
        # preposição portuguesa "a" e o artigo inglês "a" (minúsculos) não valem
        # como classe "A"; só a letra maiúscula isolada decide.
        self.assertEqual(parse("o ponto pertence a B", "A", "B"), ("B", True))
        self.assertEqual(parse("o ponto pertence a classe", "A", "B"), (None, False))

    def test_starts_with_exige_palavra_inteira_para_nomes_de_uma_letra(self):
        # Até 09/10/2026 a camada 7 ("starts-with") casava qualquer palavra que
        # COMEÇASSE com a letra da classe: "Banana" → "B", "Abacaxi" → "A". Agora
        # exige fronteira de palavra, então essas respostas são malformadas.
        self.assertEqual(parse("Banana", "A", "B"), (None, False))
        self.assertEqual(parse("Abacaxi", "A", "B"), (None, False))

    def test_starts_with_formas_validas(self):
        # Formas que começam com a classe como palavra inteira continuam válidas.
        for resposta in ["A", "A.", "Class A", "**A**", "A)"]:
            with self.subTest(resposta=resposta):
                self.assertEqual(parse(resposta, "A", "B"), ("A", True))
        self.assertEqual(parse("B because the point is far from the centre", "A", "B"),
                         ("B", True))

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

    def test_chain_of_thought_usa_o_ultimo_marcador(self):
        # P3 (09/10/2026): a camada 0 usava a PRIMEIRA ocorrência do marcador;
        # "Classification: Class A" no início do raciocínio escondia a resposta final.
        resposta = ("Classification: Class A points have positive x2.\n"
                    "The point has negative x2.\n"
                    "Final classification: B")
        self.assertEqual(parse(resposta, "A", "B"), ("B", True))

    def test_chain_of_thought_limpa_markdown_e_latex(self):
        self.assertEqual(parse("Class A has x2 > 0, Class B has x2 < 0.\n**Final classification: A**",
                               "A", "B"), ("A", True))
        self.assertEqual(parse("Final Answer: The final answer is $\\boxed{A}$", "A", "B"),
                         ("A", True))
        self.assertEqual(parse("Final answer: Class B.", "A", "B"), ("B", True))
        self.assertEqual(parse("Reasoning about A and B.\n`B`", "A", "B"), ("B", True))

    def test_chain_of_thought_contradicao_entre_marcador_e_ultima_linha(self):
        # Visto na completa de 08/10/2026 (Gemini, variante CoT): "Final Answer: A"
        # e, na última linha, "B". Resposta contraditória → malformada (reenvio),
        # sem deixar as camadas seguintes escolherem por padrão de texto.
        self.assertEqual(parse("The point belongs to Class A.\n\nFinal Answer: A\nB", "A", "B"),
                         (None, False))
        # Marcador GENÉRICO ("Classification:") aparece em regras hipotéticas do
        # raciocínio e não veta a última linha; só o marcador final veta.
        self.assertEqual(parse("Final check:\nClassification: A.\n\nB", "A", "B"), ("B", True))
        self.assertEqual(parse("Rule 1: Classification: A.\nRule 2: Classification: B.\n\nA",
                               "A", "B"), ("A", True))
        # Marcador final e última linha concordando, ou só um deles decidindo → válido.
        self.assertEqual(parse("Final Answer: B\n\nB", "A", "B"), ("B", True))
        self.assertEqual(parse("Final Answer: A\nI hope this helps.", "A", "B"), ("A", True))
        # ``[ \t]*`` depois do ``:``: "each class:\nClass A: ..." não é resposta.
        self.assertEqual(parse("the range for each class:\nClass A: x2 > 3.\nClass B: x2 < 0.\n\nB",
                               "A", "B"), ("B", True))

    def test_chain_of_thought_marcador_sem_classe_nao_decide(self):
        self.assertEqual(parse("I have no rule.\nFinal Answer: Cannot classify", "A", "B"),
                         (None, False))
        # As duas classes no trecho final: a camada 0 não decide (e "either" não
        # dispara nenhum padrão da camada 6).
        self.assertEqual(parse("Final answer: either A or B", "A", "B"), (None, False))

    def test_formatos_estruturados(self):
        self.assertEqual(parse('{"class":"A"}', "A", "B"), ("A", True))
        self.assertEqual(parse("Classe: A", "A", "B"), ("A", True))
        self.assertEqual(parse("Answer: B", "A", "B"), ("B", True))

    def test_nomes_longos_e_invertidos(self):
        self.assertEqual(parse("Mulher", "Mulher", "Homem"), ("Mulher", True))
        self.assertEqual(parse("homem", "Mulher", "Homem"), ("Homem", True))
        self.assertEqual(parse("Vermelho.", "Vermelho", "Azul"), ("Vermelho", True))
        self.assertEqual(parse("negativo", "Negativo", "Positivo"), ("Negativo", True))


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
        self.assertEqual(parse("Azul, não Vermelho", "Azul", "Vermelho"), (None, False))

    def test_starts_with_nao_decide_sem_fronteira_ou_com_as_duas_classes(self):
        # P2 (09/10/2026): a camada 7 usava startswith sem fronteira de palavra e
        # rotulava hedges/recusas: "Both…" → B, "Based…" → B, "Ambiguous" → A.
        self.assertEqual(parse("Both A and B are possible", "A", "B"), (None, False))
        self.assertEqual(parse("Based on the position I cannot decide", "A", "B"), (None, False))
        self.assertEqual(parse("Ambiguous", "A", "B"), (None, False))
        self.assertEqual(parse("A or B, hard to say", "A", "B"), (None, False))

    def test_artigo_ingles_nao_e_classe(self):
        self.assertEqual(parse("this is a borderline point", "A", "B"), (None, False))

    def test_retorno_e_sempre_tupla_de_dois(self):
        out = parse("qualquer coisa", "A", "B")
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[1], bool)


if __name__ == "__main__":
    unittest.main()
