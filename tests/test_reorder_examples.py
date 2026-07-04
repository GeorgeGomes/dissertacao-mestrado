"""Testes de `reorder_examples` (viés de ordem dos exemplos few-shot / recency bias).

A função alimenta um experimento REPORTADO na dissertação (bloco2_07_example_order):
um bug silencioso em qualquer das 4 ordenações invalidaria a comparação entre elas.
Contratos verificados, para cada estratégia:

1. **Preservação do multiset** — reordenar não pode criar, duplicar nem perder
   exemplos; só muda a ordem.
2. **Ordem prometida** — class0_first/class1_first agrupam as classes na ordem
   correta; alternating intercala 0,1,0,1,... enquanto houver das duas.
3. **Reprodutibilidade do shuffle** — mesma random_state → mesma permutação;
   random_states distintas → permutações distintas (em lista não-trivial).
4. **Ordenação desconhecida levanta ValueError.**

Execução: .venv/bin/python -m unittest tests.test_reorder_examples
"""
import os
import sys
import unittest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import dissertacao_mestrado as dm  # noqa: E402

C0, C1 = "A", "B"


def _exemplos(n0: int, n1: int):
    """n0 exemplos da classe A e n1 da classe B, coordenadas únicas."""
    ex0 = [(float(i), float(i) + 0.5, C0) for i in range(n0)]
    ex1 = [(10.0 + i, 10.5 + i, C1) for i in range(n1)]
    # Intercala na entrada para a ordem original não coincidir com nenhuma estratégia
    entrada = []
    for a, b in zip(ex0, ex1):
        entrada.extend([b, a])
    entrada.extend(ex0[len(ex1):] or ex1[len(ex0):])
    return entrada


def _labels(examples):
    return [e[-1] for e in examples]


class TestReorderExamples(unittest.TestCase):
    def test_multiset_preservado_em_todas_as_estrategias(self):
        entrada = _exemplos(5, 5)
        for ordering in ("class0_first", "class1_first", "shuffled", "alternating"):
            saida = dm.reorder_examples(entrada, ordering, C0, C1, random_state=42)
            self.assertEqual(sorted(saida), sorted(entrada),
                             f"{ordering} criou/perdeu exemplos")
            self.assertEqual(len(saida), len(entrada), ordering)

    def test_class0_first_agrupa_na_ordem_certa(self):
        saida = dm.reorder_examples(_exemplos(4, 6), "class0_first", C0, C1)
        labels = _labels(saida)
        self.assertEqual(labels, [C0] * 4 + [C1] * 6)

    def test_class1_first_agrupa_na_ordem_inversa(self):
        saida = dm.reorder_examples(_exemplos(4, 6), "class1_first", C0, C1)
        labels = _labels(saida)
        self.assertEqual(labels, [C1] * 6 + [C0] * 4)

    def test_alternating_intercala_e_esgota_a_classe_maior_no_fim(self):
        saida = dm.reorder_examples(_exemplos(3, 5), "alternating", C0, C1)
        labels = _labels(saida)
        # Enquanto há das duas classes: 0,1,0,1,...; o excedente da maior fica no fim
        self.assertEqual(labels, [C0, C1, C0, C1, C0, C1, C1, C1])

    def test_alternating_balanceado_alterna_estritamente(self):
        saida = dm.reorder_examples(_exemplos(5, 5), "alternating", C0, C1)
        labels = _labels(saida)
        self.assertEqual(labels, [C0, C1] * 5)

    def test_shuffled_reprodutivel_por_random_state(self):
        entrada = _exemplos(10, 10)
        a = dm.reorder_examples(entrada, "shuffled", C0, C1, random_state=42)
        b = dm.reorder_examples(entrada, "shuffled", C0, C1, random_state=42)
        c = dm.reorder_examples(entrada, "shuffled", C0, C1, random_state=123)
        self.assertEqual(a, b, "mesma random_state deve dar a mesma permutação")
        self.assertNotEqual(a, c, "random_states distintas devem dar permutações distintas")

    def test_ordenacao_desconhecida_levanta(self):
        with self.assertRaises(ValueError):
            dm.reorder_examples(_exemplos(2, 2), "inexistente", C0, C1)


if __name__ == "__main__":
    unittest.main()
