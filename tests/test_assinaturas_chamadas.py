"""Checagem estática: toda chamada a uma função definida em src/ usa só parâmetros que
existem na assinatura (kwargs inexistentes, posicionais a mais). Pega o erro clássico de
remover um parâmetro e esquecer um call-site (08/10/2026: `runner.run(..., k_exemplos=)`).

Cobre funções/métodos de módulo (nome simples ou `objeto.metodo`) definidos em src/*.py
e chamados em src/*.py ou tests/*.py; funções com *args/**kwargs são ignoradas.
"""
import ast
import glob
import os
import unittest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
GENERICOS = {"run", "write", "read", "get", "update", "open", "close", "fit", "predict",
             "append", "items", "exists", "mkdir", "parse", "load", "dump", "save"}
RECEPTORES_DO_PROJETO = {"runner", "self", "dm", "gt", "ai", "lc", "cb"}
TESTS = os.path.abspath(os.path.dirname(__file__))


def _assinaturas():
    sigs = {}
    for f in glob.glob(os.path.join(SRC, "*.py")):
        tree = ast.parse(open(f, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                if a.vararg or a.kwarg:
                    continue
                pos = [x.arg for x in a.posonlyargs + a.args]
                kwonly = [x.arg for x in a.kwonlyargs]
                if pos and pos[0] in ("self", "cls"):
                    pos = pos[1:]
                n_default = len(a.defaults)
                sigs.setdefault(node.name, []).append({
                    "file": os.path.basename(f), "line": node.lineno,
                    "pos": pos, "kw": set(pos) | set(kwonly),
                    "min_pos": len(pos) - n_default,
                })
    return sigs


class TestAssinaturas(unittest.TestCase):
    def test_chamadas_batem_com_assinaturas(self):
        sigs = _assinaturas()
        problemas = []
        for f in glob.glob(os.path.join(SRC, "*.py")) + glob.glob(os.path.join(TESTS, "*.py")):
            tree = ast.parse(open(f, encoding="utf-8").read())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    nome = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    nome = node.func.attr
                    # Métodos com nome genérico (run, write, get…) colidem com a stdlib
                    # (subprocess.run, zipfile.write): só checa quando o receptor é
                    # claramente um objeto do projeto.
                    receptor = node.func.value.id if isinstance(node.func.value, ast.Name) else ""
                    if nome in GENERICOS and receptor not in RECEPTORES_DO_PROJETO:
                        continue
                else:
                    nome = None
                if nome not in sigs or len(sigs[nome]) != 1:
                    continue            # ambíguo (nome repetido) ou externo: ignora
                sig = sigs[nome][0]
                if any(isinstance(k.arg, type(None)) for k in node.keywords):
                    continue            # **dict na chamada
                if any(isinstance(a, ast.Starred) for a in node.args):
                    continue
                kws = {k.arg for k in node.keywords}
                ruins = kws - sig["kw"]
                n_pos = len(node.args)
                fornecidos = n_pos + len(kws & set(sig["pos"]))
                faltam = fornecidos < sig["min_pos"]
                if ruins or n_pos > len(sig["pos"]) or faltam:
                    problemas.append(f"{os.path.relpath(f)}:{node.lineno} {nome}(): "
                                     f"kwargs inexistentes={sorted(ruins)} posicionais={n_pos}/{len(sig['pos'])} "
                                     f"obrigatórios faltando={faltam} "
                                     f"(def em {sig['file']}:{sig['line']})")
        self.assertEqual(problemas, [], "\n" + "\n".join(problemas))


if __name__ == "__main__":
    unittest.main()
