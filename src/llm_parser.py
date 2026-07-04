"""Parser de respostas livres do LLM para o rótulo de classe correspondente.

Peça mais frágil do pipeline: o LLM responde em texto livre e precisamos mapear
essa resposta para uma das duas classes. O parser tenta 8 camadas em ordem crescente
de permissividade; se nenhuma casar, retorna ``(None, False)`` — sinal de resposta
malformada, tratado pelo chamador via fallback determinístico por hash.

Extraído de ``dissertacao_mestrado.py`` para isolar a lógica e permitir testes de
unidade independentes (ver ``tests/test_parser.py``). É uma função pura — depende
apenas de ``re`` — então pode ser importada sem efeitos colaterais.
"""
import re
from typing import Optional, Tuple


def parse_llm_response(
    label: str,
    nome_classe_0: str,
    nome_classe_1: str
) -> Tuple[Optional[str], bool]:
    """Interpreta a resposta do LLM e verifica se é válida (corresponde a uma das classes).

    Tentativas em ordem crescente de permissividade:
      0. Extração da última linha e marcadores CoT ("Final answer:", etc.)
      1. Match exato (case-insensitive)
      2. Match exato após remoção de aspas e pontuação marginal
      3. Classe numérica: "0"/"1" quando os nomes são dígitos
      4. Word-boundary regex (evita falsos positivos em substrings)
      5. Contains exclusivo (apenas uma das classes aparece no texto)
      6. Padrões comuns de resposta ("Class X", "Answer is X", ...)
      7. Starts-with (a resposta começa com o nome da classe)
    """
    label_clean = label.strip()
    label_upper = label_clean.upper()
    nome_0_upper = nome_classe_0.upper()
    nome_1_upper = nome_classe_1.upper()

    # 0. Extração CoT: tenta marcadores de resposta final e última linha
    # Necessário para variante chain-of-thought que produz raciocínio antes da resposta
    if '\n' in label_clean:
        # Tenta extrair após marcadores comuns de resposta final
        cot_markers = [
            r'(?:final\s+answer|final\s+classification|my\s+(?:final\s+)?answer|therefore|classification)\s*[:=]\s*',
        ]
        for marker_pattern in cot_markers:
            match = re.search(marker_pattern + r'(.+)', label_clean, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip().strip('"\'.,!? ')
                extracted_upper = extracted.upper()
                if extracted_upper == nome_0_upper:
                    return nome_classe_0, True
                if extracted_upper == nome_1_upper:
                    return nome_classe_1, True

        # Tenta a última linha não-vazia como resposta
        last_line = [l.strip() for l in label_clean.split('\n') if l.strip()][-1]
        last_line_clean = re.sub(r'^[\s\'\"]+|[\s\'\".,!?]+$', '', last_line)
        last_upper = last_line_clean.upper()
        if last_upper == nome_0_upper:
            return nome_classe_0, True
        if last_upper == nome_1_upper:
            return nome_classe_1, True

    # 1. Match exato
    if label_upper == nome_0_upper:
        return nome_classe_0, True
    if label_upper == nome_1_upper:
        return nome_classe_1, True

    # 2. Remove aspas/pontuação marginal e tenta novamente
    stripped = re.sub(r'^[\s\'\"]+|[\s\'\".,!?]+$', '', label_clean)
    stripped_upper = stripped.upper()
    if stripped_upper == nome_0_upper:
        return nome_classe_0, True
    if stripped_upper == nome_1_upper:
        return nome_classe_1, True

    # 3. Classes numéricas: se o nome for "0"/"1", aceita variações como " 0 " ou "Class 0"
    if nome_0_upper.isdigit() and nome_1_upper.isdigit():
        nums_found = re.findall(r'\b\d+\b', label_upper)
        hits_0 = nums_found.count(nome_0_upper)
        hits_1 = nums_found.count(nome_1_upper)
        if hits_0 > hits_1:
            return nome_classe_0, True
        if hits_1 > hits_0:
            return nome_classe_1, True

    # 4. Word-boundary regex (mais preciso que contains para nomes curtos como "A"/"B")
    pat_0 = re.compile(r'\b' + re.escape(nome_0_upper) + r'\b')
    pat_1 = re.compile(r'\b' + re.escape(nome_1_upper) + r'\b')
    wb_0 = bool(pat_0.search(label_upper))
    wb_1 = bool(pat_1.search(label_upper))
    if wb_0 and not wb_1:
        return nome_classe_0, True
    if wb_1 and not wb_0:
        return nome_classe_1, True

    # 5. Contains exclusivo (apenas uma das classes aparece em qualquer posição).
    #    Restrito a nomes com 2+ caracteres: com nomes de 1 letra ("A"/"B"), o
    #    substring casa a letra DENTRO de palavras comuns ("cannot"/"classificar"
    #    contêm 'a') e rotularia recusas/hedges como classe válida (valid=True),
    #    invisível à taxa de malformadas — viés sistemático pró-classe-A.
    #    Nomes de 1 letra já são cobertos com precisão pela camada 4 (word-boundary).
    if len(nome_0_upper) > 1 and len(nome_1_upper) > 1:
        contains_0 = nome_0_upper in label_upper
        contains_1 = nome_1_upper in label_upper
        if contains_0 and not contains_1:
            return nome_classe_0, True
        if contains_1 and not contains_0:
            return nome_classe_1, True

    # 6. Padrões comuns de resposta em linguagem natural
    common_patterns = [
        f"CLASS {nome_0_upper}", f"CLASS {nome_1_upper}",
        f"CLASSE {nome_0_upper}", f"CLASSE {nome_1_upper}",
        f"ANSWER IS {nome_0_upper}", f"ANSWER IS {nome_1_upper}",
        f"ANSWER: {nome_0_upper}", f"ANSWER: {nome_1_upper}",
        f"CLASSIFICATION: {nome_0_upper}", f"CLASSIFICATION: {nome_1_upper}",
        f"CLASSIFIED AS {nome_0_upper}", f"CLASSIFIED AS {nome_1_upper}",
        f"BELONGS TO {nome_0_upper}", f"BELONGS TO {nome_1_upper}",
        f"IS {nome_0_upper}", f"IS {nome_1_upper}",
    ]
    # Fronteira de palavra após o nome da classe: sem ela, o padrão "IS A"
    # casaria como substring de "is ambiguous" e rotularia o hedge como classe A.
    for pattern in common_patterns:
        if re.search(re.escape(pattern) + r"\b", label_upper):
            return (nome_classe_0 if nome_0_upper in pattern else nome_classe_1), True

    # 7. Starts-with (a resposta começa diretamente com o nome da classe)
    if label_upper.startswith(nome_0_upper):
        return nome_classe_0, True
    if label_upper.startswith(nome_1_upper):
        return nome_classe_1, True

    return None, False
