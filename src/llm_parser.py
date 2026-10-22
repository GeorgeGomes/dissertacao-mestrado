"""Parser de respostas livres do LLM para o rótulo de classe correspondente.

Peça mais frágil da coleta: o LLM responde em texto livre e precisamos mapear
essa resposta para uma das duas classes. O parser tenta 8 camadas em ordem crescente
de permissividade; se nenhuma casar, retorna ``(None, False)``, sinal de resposta
malformada, tratado pelo chamador via fallback determinístico por hash.

Extraído de ``dissertacao_mestrado.py`` para isolar a lógica e permitir testes de
unidade independentes (ver ``tests/test_parser.py``). É uma função pura, depende
apenas de ``re``, então pode ser importada sem efeitos colaterais.

Política comum a todas as camadas que olham o texto inteiro (4, 5, 6 e 7): quando as
DUAS classes aparecem, a camada não decide. Só a camada 0 (marcadores de resposta
final e última linha) decide a partir de um trecho, porque ali o LLM declara
explicitamente qual é a resposta.

Fronteiras de nome (09/10/2026): cada nome de classe vira um regex com fronteira
adequada ao seu tipo. Nomes numéricos ("0"/"1") só casam inteiros isolados, nunca
pedaços de coordenadas ("0.3", "-1.25", "x1"); nomes de 1 letra ("A"/"B") casam
case-sensitive (o artigo inglês "a" não é a classe A); nomes maiores casam por
palavra inteira, case-insensitive.
"""
import re
from typing import Optional, Tuple


# ``\boxed{A}`` (LaTeX que o Gemini usa no "Final Answer") vira só o conteúdo.
_RE_BOXED = re.compile(r'\\boxed\s*\{([^{}]*)\}')
# Marcadores de resposta (CoT). Sem o ``(.+)`` para que ``finditer`` encontre TODAS
# as ocorrências, inclusive duas na mesma linha; o trecho é o resto da linha.
# ``[ \t]*`` (não ``\s*``) depois do ``:`` para o trecho não pular para a linha
# seguinte ("for each class:\nClass A: ..." não é uma resposta).
# FINAL: declarações explícitas de resposta final, que podem VETAR a última linha
# quando discordam dela. GENÉRICO: inclui marcadores que o raciocínio usa em regras
# hipotéticas ("Rule 1: ... Classification: B"); só valem quando nem a última linha
# nem um marcador final decidem.
_RE_MARCADOR_FINAL = re.compile(
    r'(?:final\s+answer|final\s+classification|my\s+(?:final\s+)?answer)[ \t]*[:=][ \t]*',
    re.IGNORECASE,
)
_RE_MARCADOR_COT = re.compile(
    r'(?:final\s+answer|final\s+classification|my\s+(?:final\s+)?answer|therefore|'
    r'classification|answer|label)[ \t]*[:=][ \t]*',
    re.IGNORECASE,
)


def _trecho_apos_ultimo(regex, texto: str) -> Optional[str]:
    """Resto da linha depois da ÚLTIMA ocorrência do marcador (None se não há)."""
    ultimo = None
    for m in regex.finditer(texto):
        ultimo = m
    if ultimo is None:
        return None
    inicio = ultimo.end()
    fim = texto.find('\n', inicio)
    return texto[inicio:] if fim < 0 else texto[inicio:fim]


# Prefixo "Class X" / "Classe X" / "Class: X" dentro de um trecho de resposta final.
_RE_PREFIXO_CLASSE = re.compile(r'^(?:class|classe)\s*[:=]?\s*(.+)$', re.IGNORECASE)
# Pontuação marginal (aspas, parênteses, pontuação final) que não faz parte do rótulo.
_RE_MARGEM = re.compile(r'^[\s\'"(\[{]+|[\s\'".,!?;:)\]}]+$')


def _limpar(trecho: str) -> str:
    """Remove markdown/LaTeX e pontuação marginal: ``**A**``, ``$\\boxed{A}$``, ``\"A\".``."""
    s = _RE_BOXED.sub(r'\1', trecho)
    s = re.sub(r'[*`$]', '', s)
    s = s.strip().strip('_').strip()
    return _RE_MARGEM.sub('', s)


def _regex_nome(nome: str):
    """Regex (string) e flags do nome de classe com a fronteira adequada ao seu tipo.

    - dígitos: inteiro isolado, não parte de decimal/coordenada/identificador
      (``0.3``, ``-1.25``, ``x1``, ``10`` não casam ``0``/``1``; ``1.`` final casa);
    - 1 caractere: fronteira de palavra, CASE-SENSITIVE;
    - demais: fronteira de palavra, case-insensitive.
    """
    esc = re.escape(nome)
    if nome.isdigit():
        return r'(?<![\w.\-])' + esc + r'(?!\w|\.\d)', 0
    if len(nome) == 1:
        return r'\b' + esc + r'\b', 0
    return r'\b' + esc + r'\b', re.IGNORECASE


def parse_llm_response(
    label: str,
    nome_classe_0: str,
    nome_classe_1: str
) -> Tuple[Optional[str], bool]:
    """Interpreta a resposta do LLM e verifica se é válida (corresponde a uma das classes).

    Tentativas em ordem crescente de permissividade:
      0. Resposta no fim: a última linha não-vazia (se houver quebra de linha), o
         ÚLTIMO marcador final ("Final answer:", "Final classification:") e, se
         nenhum dos dois decidir, o ÚLTIMO marcador genérico ("Classification:",
         "Therefore:", "Answer:"). Cada trecho é limpo de markdown/LaTeX e aceito se
         for o nome exato, "Class X" ou o único nome presente nele; última linha e
         marcador final decidindo classes diferentes = contraditória: (None, False)
      1. Match exato (case-insensitive)
      2. Match exato após remoção de aspas, markdown e pontuação marginal
      3. Classe numérica: inteiros isolados quando os nomes são dígitos (o ÚLTIMO
         candidato válido decide: a resposta vem no fim)
      4. Word-boundary regex (evita falsos positivos em substrings)
      5. Contains exclusivo (apenas uma das classes aparece no texto)
      6. Padrões comuns de resposta ("Class X", "Answer is X", ...)
      7. Starts-with (a resposta começa com o nome da classe como palavra inteira e a
         outra classe não aparece); rede de segurança, raramente alcançada

    Nomes de classe de 1 letra ("A"/"B"): as camadas 0 (trecho), 4, 6 e 7 casam
    CASE-SENSITIVE (o artigo/preposição "a" minúsculo não é a classe "A"); o match
    exato da resposta inteira (camadas 0 a 2) continua case-insensitive, pois "b"
    sozinho é inequívoco.
    """
    label_clean = label.strip()
    label_upper = label_clean.upper()
    nome_0_upper = nome_classe_0.upper()
    nome_1_upper = nome_classe_1.upper()

    re_0, fl_0 = _regex_nome(nome_classe_0)
    re_1, fl_1 = _regex_nome(nome_classe_1)
    pat_0 = re.compile(re_0, fl_0)
    pat_1 = re.compile(re_1, fl_1)

    def _exato(trecho: str) -> Optional[str]:
        up = trecho.upper()
        if up == nome_0_upper:
            return nome_classe_0
        if up == nome_1_upper:
            return nome_classe_1
        return None

    def _exclusivo(trecho: str) -> Optional[str]:
        h0 = bool(pat_0.search(trecho))
        h1 = bool(pat_1.search(trecho))
        if h0 and not h1:
            return nome_classe_0
        if h1 and not h0:
            return nome_classe_1
        return None

    def _decidir_trecho(trecho: str) -> Optional[str]:
        """Resposta declarada num trecho curto: exato, "Class X" ou único nome presente."""
        limpo = _limpar(trecho)
        if not limpo:
            return None
        hit = _exato(limpo)
        if hit is not None:
            return hit
        m = _RE_PREFIXO_CLASSE.match(limpo)
        if m:
            hit = _exato(_limpar(m.group(1)))
            if hit is not None:
                return hit
        return _exclusivo(limpo)

    # 0. Extração CoT: a resposta vem no FIM. Três leituras, em ordem:
    #    (a) a última linha não-vazia (o prompt CoT pede a classificação final
    #        "on the last line");
    #    (b) o ÚLTIMO marcador explicitamente final ("Final answer:", "Final
    #        classification:"); se (a) e (b) decidem classes DIFERENTES ("Final
    #        Answer: A" e, na última linha, "B"), a resposta é contraditória:
    #        malformada, sem cair nas camadas seguintes (que escolheriam por
    #        padrão de texto);
    #    (c) o ÚLTIMO marcador genérico ("Classification:", "Therefore:", "Answer:"),
    #        só quando (a) e (b) não decidem: o raciocínio usa esses marcadores em
    #        regras hipotéticas ("Rule 1: ... Classification: B"), então ele não
    #        veta a última linha.
    #    Até 09/10/2026 valia o PRIMEIRO marcador de qualquer tipo. Reinterpretando
    #    a completa de 08/10/2026 com esta regra: 7 divergências, todas em respostas
    #    CoT do Gemini: 4 trocas de rótulo (3 A→B, 1 B→A; o marcador de uma regra
    #    hipotética escondia a resposta final), 2 contraditórias (antes aceitas, agora
    #    malformadas) e 1 antes malformada (``\boxed{A}``) agora aceita.
    hit_linha = None
    if '\n' in label_clean:
        last_line = [l.strip() for l in label_clean.split('\n') if l.strip()][-1]
        hit_linha = _decidir_trecho(last_line)
    trecho_final = _trecho_apos_ultimo(_RE_MARCADOR_FINAL, label_clean)
    hit_final = _decidir_trecho(trecho_final) if trecho_final is not None else None
    if hit_linha is not None and hit_final is not None and hit_linha != hit_final:
        return None, False
    if hit_linha is not None:
        return hit_linha, True
    if hit_final is not None:
        return hit_final, True
    trecho_generico = _trecho_apos_ultimo(_RE_MARCADOR_COT, label_clean)
    if trecho_generico is not None:
        hit = _decidir_trecho(trecho_generico)
        if hit is not None:
            return hit, True

    # 1. Match exato
    hit = _exato(label_clean)
    if hit is not None:
        return hit, True

    # 2. Remove aspas/markdown/pontuação marginal e tenta novamente
    stripped = _limpar(label_clean)
    hit = _exato(stripped)
    if hit is not None:
        return hit, True

    # 3. Classes numéricas: inteiros ISOLADOS (não parte de "0.3", "-1.25", "x1",
    #    "10"); entre vários candidatos válidos, o ÚLTIMO decide (resposta no fim).
    #    Até 09/10/2026 contava todos os ``\b\d+\b``, e "x1 = 0.3 ... Answer: 1"
    #    virava classe 0 (dois "0" das coordenadas contra um "1").
    if nome_0_upper.isdigit() and nome_1_upper.isdigit():
        candidatos = re.findall(r'(?<![\w.\-])(\d+)(?!\w|\.\d)', stripped)
        validos = [c for c in candidatos if c in (nome_classe_0, nome_classe_1)]
        if validos:
            return validos[-1], True

    # 4. Word-boundary regex (mais preciso que contains para nomes curtos como "A"/"B").
    #    Fronteira por tipo de nome (``_regex_nome``): nomes de 1 caractere casam
    #    CASE-SENSITIVE no texto original (o artigo inglês "a" não é a classe A;
    #    até 08/10/2026 virava classe A com valid=True); nomes numéricos não casam
    #    dentro de decimais ("0.5" não é a classe 0).
    hit = _exclusivo(label_clean)
    if hit is not None:
        return hit, True

    # 5. Contains exclusivo (apenas uma das classes aparece em qualquer posição).
    #    Restrito a nomes com 2+ caracteres: com nomes de 1 letra ("A"/"B"), o
    #    substring casa a letra DENTRO de palavras comuns ("cannot"/"classificar"
    #    contêm 'a') e rotularia recusas/hedges como classe válida (valid=True),
    #    invisível à taxa de malformadas, viés sistemático pró-classe-A.
    #    Nomes de 1 letra já são cobertos com precisão pela camada 4 (word-boundary).
    if len(nome_0_upper) > 1 and len(nome_1_upper) > 1:
        contains_0 = nome_0_upper in label_upper
        contains_1 = nome_1_upper in label_upper
        if contains_0 and not contains_1:
            return nome_classe_0, True
        if contains_1 and not contains_0:
            return nome_classe_1, True

    # 6. Padrões comuns de resposta em linguagem natural.
    #    Cada padrão carrega a classe que representa (tupla): decidir por
    #    substring do nome dentro do padrão quebrava com nomes de 1 letra
    #    ("A" ⊂ "CLASS B" → classe A; "B" ⊂ "BELONGS TO A" → classe B).
    #    Fronteira nos dois lados do nome (``_regex_nome``): sem ela, "IS A" casaria
    #    como substring de "is ambiguous" (hedge → classe A) e "CLASS 1" casaria
    #    "CLASS 10". Se padrões das DUAS classes casarem ("Class A … Class B"),
    #    a camada não decide, mesma política das camadas 4 e 5.
    #    O prefixo do padrão é sempre case-insensitive; a sensibilidade do nome
    #    segue o tipo ("this IS A point" ≠ "is a point").
    templates = ["CLASS {}", "CLASSE {}", "ANSWER IS {}", "ANSWER: {}",
                 "CLASSIFICATION: {}", "CLASSIFIED AS {}", "BELONGS TO {}", "IS {}"]
    matched = set()
    for tpl in templates:
        prefixo = tpl.replace("{}", "").strip()
        for re_nome, fl_nome, classe in ((re_0, fl_0, nome_classe_0),
                                         (re_1, fl_1, nome_classe_1)):
            pat = r"\b(?i:" + re.escape(prefixo) + r")\s*" + re_nome
            if re.search(pat, label_clean, fl_nome):
                matched.add(classe)
    if len(matched) == 1:
        return matched.pop(), True

    # 7. Starts-with: a resposta começa com o nome da classe como PALAVRA INTEIRA
    #    e a outra classe não aparece no texto. Até 09/10/2026 era ``startswith``
    #    sem fronteira: "Both A and B are possible" → B, "Based on…" → B,
    #    "Ambiguous" → A (nomes "A"/"B"). Com a fronteira e a política das camadas
    #    4/5/6 (não decidir com as duas classes presentes), esta camada é uma rede
    #    de segurança: o que ela aceita, a camada 4 normalmente já aceitou.
    if re.match(re_0, label_clean, fl_0) and not pat_1.search(label_clean):
        return nome_classe_0, True
    if re.match(re_1, label_clean, fl_1) and not pat_0.search(label_clean):
        return nome_classe_1, True

    return None, False
