"""Testes do auditor offline de interações (``src/audit_interactions.py``).

Duas camadas:
1. Testes de UNIDADE da função ``audit`` sobre registros sintéticos de resultado
   conhecido — rodam sempre, sem depender de nenhuma execução real.
2. Testes de INTEGRAÇÃO que rodam a auditoria na execução completa mais recente do
   repositório, se houver. Verificam os invariantes que sustentam κ/consistência:
   parser sem malformadas e temperatura fixa em 0.0. São pulados (skipTest) quando
   nenhuma execução com JSON de interações está presente — assim o CI não quebra sem
   os dados brutos.
"""
import os
import sys
import unittest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resultados"))

from audit_interactions import audit, find_latest_execution, format_report, load_interactions  # noqa: E402


def _rec(point, resp, malformed=False, temp="0.0", retries="0"):
    return {
        "point": point, "prompt": "P", "raw_response": resp, "parsed_label": resp,
        "malformed": str(malformed), "temperature": temp, "format_retries": retries,
    }


class TestAuditUnit(unittest.TestCase):
    def test_conta_malformadas_e_taxa(self):
        recs = [_rec("a", "A"), _rec("b", "B", malformed=True), _rec("c", "A")]
        s = audit(recs)
        self.assertEqual(s["total_chamadas"], 3)
        self.assertEqual(s["malformadas"], 1)
        self.assertAlmostEqual(s["taxa_malformadas"], 1 / 3)

    def test_flip_detecta_divergencia(self):
        # ponto "x" respondido A e depois B (mesmo prompt) -> 1 query divergente
        recs = [_rec("x", "A"), _rec("x", "B"), _rec("y", "A"), _rec("y", "A")]
        s = audit(recs)
        self.assertEqual(s["queries_repetidas"], 2)      # x e y repetem
        self.assertEqual(s["queries_divergentes"], 1)    # só x diverge
        self.assertAlmostEqual(s["taxa_flip_T0"], 0.5)

    def test_deterministico_taxa_flip_zero(self):
        recs = [_rec("x", "A"), _rec("x", "A"), _rec("x", "A")]
        s = audit(recs)
        self.assertEqual(s["queries_divergentes"], 0)
        self.assertEqual(s["taxa_flip_T0"], 0.0)

    def test_point_como_dict_nao_quebra(self):
        # pontos podem vir como dict; a serialização estável deve torná-los hasheáveis
        recs = [_rec({"x1": 1.0, "x2": 2.0}, "A"), _rec({"x1": 1.0, "x2": 2.0}, "A")]
        s = audit(recs)
        self.assertEqual(s["queries_repetidas"], 1)
        self.assertEqual(s["taxa_flip_T0"], 0.0)

    def test_por_regime_separa_passo2_de_passo3_pela_tag(self):
        t0 = "bloco1|problema=A|passo=2|regime=passo2_zero_shot|k=0|rep=0|seed=42"
        t4 = "bloco1|problema=A|passo=3|regime=passo3_exemplos_rotulo_real|k=4|rep=0|seed=42"
        recs = [dict(_rec("x", "A"), tag=t0), dict(_rec("x", "B"), tag=t0),      # flip no zero-shot
                dict(_rec("y", "A"), tag=t4), dict(_rec("y", "A"), tag=t4),      # estável no few-shot
                _rec("z", "A")]                                                  # sem tag
        s = audit(recs)
        pr = s["por_regime"]
        k0 = [k for k in pr if "|A|passo=2|passo2_zero_shot|k=0" in k][0]
        k4 = [k for k in pr if "|A|passo=3|passo3_exemplos_rotulo_real|k=4" in k][0]
        self.assertEqual(pr[k0]["taxa_flip_T0"], 1.0)
        self.assertEqual(pr[k4]["taxa_flip_T0"], 0.0)
        self.assertTrue(any(k.split("|")[3] == "sem_tag" for k in pr))
        self.assertIn("[4] Breakdown por problema, passo e regime", format_report("execucao_x", s))

    def test_temperaturas_agregadas(self):
        s = audit([_rec("a", "A", temp="0.0"), _rec("b", "B", temp="0.0")])
        self.assertEqual(s["temperaturas"], ["0.0"])


class TestAuditIntegracao(unittest.TestCase):
    """Roda na execução completa mais recente, se existir."""

    @classmethod
    def setUpClass(cls):
        cls.exec_dir = find_latest_execution(ROOT, allow_rapido=False)
        if cls.exec_dir is None:
            raise unittest.SkipTest(
                "nenhuma execução completa com llm_interactions_parte*.json (ou llm_interactions.json) encontrada"
            )
        cls.stats = audit(load_interactions(cls.exec_dir))

    def test_parser_sem_malformadas(self):
        # A credibilidade de κ/consistência exige taxa de fallback baixa (< 5%).
        self.assertLessEqual(self.stats["taxa_malformadas"], 0.05,
                             msg=f"taxa de malformadas alta em {self.exec_dir}")

    def test_temperatura_zero(self):
        self.assertEqual(self.stats["temperaturas"], ["0.0"],
                         msg="esperado temperature=0.0 em todas as chamadas")

    def test_reporta_taxa_flip(self):
        # Não impõe limite (não-determinismo de T=0 é esperado); apenas garante que a
        # métrica é computável e está no intervalo [0, 1].
        self.assertGreaterEqual(self.stats["taxa_flip_T0"], 0.0)
        self.assertLessEqual(self.stats["taxa_flip_T0"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
