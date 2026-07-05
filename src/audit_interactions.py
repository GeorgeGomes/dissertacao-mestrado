"""Auditoria offline das interações com o LLM de uma execução.

Lê os arquivos ``llm_interactions_parte*.json`` (ou ``llm_interactions.json``) de uma
pasta ``execucao_*/`` e verifica, SEM chamar a API, duas propriedades que sustentam a
credibilidade das métricas de κ/consistência do trabalho:

1. **Taxa de fallback do parser** — fração de respostas ``malformed=True`` que caíram no
   fallback determinístico por hash MD5. Se essa taxa for alta, os rótulos são ruído e
   κ/consistência ficam comprometidos (ver ressalva em ``dissertacao_mestrado.py``).
2. **Determinismo de ``T=0``** — o mesmo par (prompt, ponto) consultado 2+ vezes deve
   devolver a mesma resposta. ``temperature=0.0`` NÃO garante isso: mede-se a fração de
   pares idênticos cuja resposta divergiu entre repetições ("taxa de flip"). Reportar
   esse número é honestidade experimental, não é falha do código.

Uso:
    python src/audit_interactions.py                 # audita a execução completa mais recente
    python src/audit_interactions.py execucao_2026-06-30_23-46-58
    python src/audit_interactions.py --json          # saída em JSON (para CI/log)

Código de saída: 0 se a taxa de malformadas <= --max-malformed (padrão 5%); 1 caso
contrário. A não-determinância de ``T=0`` é apenas reportada (não afeta o código de
saída), pois é um achado esperado — a menos que ultrapasse --max-flip (padrão 100%,
desligado), útil para detectar regressões grosseiras.

Depende apenas da biblioteca padrão (json, glob, os, argparse, collections).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

Record = Dict[str, object]


def _stable(x: object) -> str:
    """Serializa qualquer valor (dict/list/str) de forma determinística e hasheável."""
    if isinstance(x, (dict, list)):
        return json.dumps(x, sort_keys=True, ensure_ascii=False)
    return str(x)


def load_interactions(exec_dir: str) -> List[Record]:
    """Carrega e concatena os registros de interação de uma pasta de execução.

    Aceita tanto o arquivo único legado ``llm_interactions.json`` quanto o esquema
    particionado ``llm_interactions_parte001.json``, ``...parte002.json``, etc.
    """
    parts = sorted(glob.glob(os.path.join(exec_dir, "llm_interactions_parte*.json")))
    if not parts:
        single = os.path.join(exec_dir, "llm_interactions.json")
        parts = [single] if os.path.exists(single) else []
    if not parts:
        raise FileNotFoundError(
            f"Nenhum llm_interactions[_parte*].json em {exec_dir!r}"
        )
    records: List[Record] = []
    for p in parts:
        with open(p, encoding="utf-8") as fh:
            chunk = json.load(fh)
        records.extend(chunk if isinstance(chunk, list) else list(chunk.values()))
    return records


def is_rapido(exec_dir: str) -> bool:
    """True se a execução foi rodada em modo --rapido (não serve para resultados finais)."""
    log = os.path.join(exec_dir, "log_execucao.txt")
    if not os.path.exists(log):
        return False
    with open(log, encoding="utf-8", errors="ignore") as fh:
        head = fh.read(4000)
    return "MODO RÁPIDO" in head or "--rapido" in head


def find_latest_execution(root: str = ".", allow_rapido: bool = False) -> Optional[str]:
    """Retorna a pasta execucao_* mais recente que tenha JSON de interações.

    Por padrão pula execuções --rapido (curtas, seed única) — elas não servem para
    conclusões finais, mas a auditoria ainda funciona nelas se allow_rapido=True.

    O contrato principal é o SUFIXO no nome da pasta: ``_completa`` (execução
    cheia) vs ``_smoke`` (--rapido). Pastas legadas sem sufixo caem no fareja-log
    ``is_rapido`` (compatibilidade).
    """
    dirs = sorted(glob.glob(os.path.join(root, "execucao_*")), reverse=True)
    for d in dirs:
        has_json = bool(glob.glob(os.path.join(d, "llm_interactions_parte*.json"))) or \
            os.path.exists(os.path.join(d, "llm_interactions.json"))
        if not has_json:
            continue
        nome = os.path.basename(os.path.normpath(d))
        if not allow_rapido:
            if nome.endswith("_smoke"):
                continue
            # Legado sem sufixo: decide pelo cabeçalho do log.
            if not nome.endswith("_completa") and is_rapido(d):
                continue
        return d
    return None


def _stats_basicas(records: List[Record], n_exemplos: int = 5) -> Dict[str, object]:
    """Estatísticas de fallback e flip para um conjunto de registros."""
    total = len(records)
    malformed = sum(1 for r in records if str(r.get("malformed")).lower() == "true")
    retries = sum(1 for r in records if str(r.get("format_retries", "0")) not in ("0", "None"))
    temperatures = sorted({str(r.get("temperature")) for r in records})

    # Determinismo T=0: agrupa por (provider, modelo, prompt, ponto) e mede a
    # divergência de resposta. O modelo ENTRA na chave — sem ele, execuções
    # multi-modelo contariam divergência ENTRE modelos como "flip", inflando a
    # métrica (não-determinismo é o mesmo modelo respondendo diferente).
    by_key: Dict[Tuple[str, str, str, str], List[str]] = defaultdict(list)
    for r in records:
        by_key[(str(r.get("provider")), str(r.get("model")),
                _stable(r.get("prompt")), _stable(r.get("point")))].append(
            _stable(r.get("raw_response"))
        )
    repeated = {k: v for k, v in by_key.items() if len(v) > 1}
    divergent = {k: v for k, v in repeated.items() if len(set(v)) > 1}
    repeated_calls = sum(len(v) for v in repeated.values())

    examples = []
    for k, v in list(divergent.items())[:n_exemplos]:
        examples.append({"model": f"{k[0]}/{k[1]}", "point": k[3],
                         "respostas": dict(Counter(v))})

    return {
        "total_chamadas": total,
        "malformadas": malformed,
        "taxa_malformadas": (malformed / total) if total else 0.0,
        "com_retries": retries,
        "temperaturas": temperatures,
        "queries_unicas": len(by_key),
        "queries_repetidas": len(repeated),
        "chamadas_repetidas": repeated_calls,
        "queries_divergentes": len(divergent),
        "taxa_flip_T0": (len(divergent) / len(repeated)) if repeated else 0.0,
        "exemplos_divergencia": examples,
    }


def audit(records: List[Record]) -> Dict[str, object]:
    """Computa as estatísticas de auditoria — globais e por (provider, modelo).

    O breakdown por modelo permite comparar a taxa de flip a T=0 entre LLMs de
    famílias distintas (resultado por si só) e localizar problemas de parser em
    um modelo específico.
    """
    stats = _stats_basicas(records)

    por_modelo: Dict[str, Dict[str, object]] = {}
    grupos: Dict[Tuple[str, str], List[Record]] = defaultdict(list)
    for r in records:
        grupos[(str(r.get("provider")), str(r.get("model")))].append(r)
    for (prov, model), recs in sorted(grupos.items()):
        m = _stats_basicas(recs, n_exemplos=0)
        # snapshots resolvidos pela API (reprodutibilidade) e provedores de
        # inferência (OpenRouter) observados neste grupo
        m["models_resolved"] = sorted({
            str(r.get("model_resolved")) for r in recs if r.get("model_resolved")
        })
        m["inference_providers"] = sorted({
            str(r.get("inference_provider")) for r in recs if r.get("inference_provider")
        })
        por_modelo[f"{prov}/{model}"] = m

    stats["por_modelo"] = por_modelo
    return stats


def format_report(exec_dir: str, stats: Dict[str, object]) -> str:
    L = []
    L.append("=" * 68)
    L.append(f" AUDITORIA DE INTERAÇÕES — {os.path.basename(os.path.normpath(exec_dir))}")
    L.append("=" * 68)
    L.append(f"  Total de chamadas ao LLM ......... {stats['total_chamadas']:,}")
    L.append(f"  Temperaturas usadas .............. {stats['temperaturas']}")
    L.append("")
    L.append("  [1] Fallback do parser (malformadas)")
    L.append(f"      malformadas ................... {stats['malformadas']:,}"
             f"  ({stats['taxa_malformadas']:.3%})")
    L.append(f"      respostas com format_retries>0  {stats['com_retries']:,}")
    L.append("")
    L.append("  [2] Determinismo de T=0 (mesmo prompt+ponto -> mesma resposta?)")
    L.append(f"      queries repetidas 2+x ......... {stats['queries_repetidas']:,}"
             f"  ({stats['chamadas_repetidas']:,} chamadas)")
    L.append(f"      queries com resposta divergente {stats['queries_divergentes']:,}")
    L.append(f"      TAXA DE FLIP (não-determinismo) {stats['taxa_flip_T0']:.3%}")
    if stats["exemplos_divergencia"]:
        L.append("      exemplos:")
        for ex in stats["exemplos_divergencia"]:
            modelo = f" [{ex['model']}]" if ex.get("model") else ""
            L.append(f"        {ex['point'][:56]}{modelo} -> {ex['respostas']}")

    por_modelo = stats.get("por_modelo") or {}
    if len(por_modelo) > 1 or any(
        m.get("models_resolved") or m.get("inference_providers")
        for m in por_modelo.values()
    ):
        L.append("")
        L.append("  [3] Breakdown por modelo (fallback | flip T=0)")
        for nome, m in por_modelo.items():
            L.append(
                f"      {nome:<44} n={m['total_chamadas']:>7,}  "
                f"malf={m['taxa_malformadas']:.3%}  flip={m['taxa_flip_T0']:.3%}"
            )
            if m.get("models_resolved"):
                L.append(f"        snapshot(s) resolvido(s): {', '.join(m['models_resolved'])}")
            if m.get("inference_providers"):
                L.append(f"        provedor(es) de inferência: {', '.join(m['inference_providers'])}")
    L.append("=" * 68)
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Auditoria offline das interações do LLM.")
    ap.add_argument("exec_dir", nargs="?", default=None,
                    help="Pasta execucao_* (padrão: a completa mais recente).")
    ap.add_argument("--root", default=".", help="Raiz onde procurar execucao_* (padrão: .).")
    ap.add_argument("--allow-rapido", action="store_true",
                    help="Permite auditar execuções em modo --rapido.")
    ap.add_argument("--max-malformed", type=float, default=0.05,
                    help="Taxa máxima de malformadas tolerada antes de falhar (padrão 0.05).")
    ap.add_argument("--max-flip", type=float, default=1.0,
                    help="Taxa de flip acima da qual falhar (padrão 1.0 = desligado).")
    ap.add_argument("--json", action="store_true", help="Emite as estatísticas em JSON.")
    args = ap.parse_args(argv)

    exec_dir = args.exec_dir or find_latest_execution(args.root, args.allow_rapido)
    if not exec_dir:
        print("ERRO: nenhuma execução com JSON de interações encontrada.", file=sys.stderr)
        return 2
    if not args.allow_rapido and is_rapido(exec_dir):
        print(f"AVISO: {exec_dir} é modo --rapido (não usar para resultados finais).",
              file=sys.stderr)

    records = load_interactions(exec_dir)
    stats = audit(records)

    if args.json:
        print(json.dumps({"exec_dir": exec_dir, **stats}, ensure_ascii=False, indent=2))
    else:
        print(format_report(exec_dir, stats))

    failed = False
    if stats["taxa_malformadas"] > args.max_malformed:
        print(f"FALHA: taxa de malformadas {stats['taxa_malformadas']:.3%} "
              f"> limite {args.max_malformed:.3%}", file=sys.stderr)
        failed = True
    if stats["taxa_flip_T0"] > args.max_flip:
        print(f"FALHA: taxa de flip {stats['taxa_flip_T0']:.3%} "
              f"> limite {args.max_flip:.3%}", file=sys.stderr)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
