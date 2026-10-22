# Glossario de Termos Tecnicos (Dissertacao de Mestrado)

## Matematica e Algebra Linear

| Termo | O que e |
|---|---|
| **Distancia de Mahalanobis** | Distancia "pesada" entre um ponto e um centro. Em vez de tratar todas as dimensoes igualmente (Euclidiana), cada dimensao tem um peso W. Formula: d_W(x, c) = soma(w * (x - c)^2) |
| **Matriz Diagonal** | Matriz onde so os elementos da diagonal principal sao diferentes de zero. Simplificacao: em vez de aprender d^2 parametros, aprende apenas d |
| **Semidefinida Positiva** | Propriedade de uma matriz que garante que distancias nunca sao negativas. Necessaria para que a metrica faca sentido |
| **Centroide** | Ponto medio de um grupo de pontos. Media de todas as coordenadas dos pontos de uma classe |
| **Norma (L2)** | "Tamanho" de um vetor: ‖w‖ = raiz(w1^2 + w2^2 + ...). Mede a magnitude |
| **Vetor** | Lista ordenada de numeros. Ex: W = [0.3, 1.5] e um vetor de pesos |
| **Vetor unitario (direcao de W)** | W dividido pela sua norma. Os graficos de W comparam direcoes (vetor unitario ou razao w2/w1), porque a magnitude do W do Perceptron depende de gamma |
| **Projecao R3 / R4** | Transformar dados de 2D para 3D adicionando x3 = x1 * x2 (hiperbole), ou para 4D adicionando x1^2 e x2^2 (elipse). Simulam um kernel quadratico sem sair do mundo linear |
| **Kernel** | Tecnica que captura relacoes nao-lineares entre variaveis sem calcular explicitamente |
| **Espaco de Features** | O "mundo" onde os dados vivem. R2 = 2 dimensoes, R3 = 3 dimensoes |

## Metricas de Avaliacao (secao IV.3 da metodologia do orientador)

Sao exatamente cinco, com estes nomes, em todo o sistema (log, CSVs, figuras e tabelas).

| Termo | O que e |
|---|---|
| **Fidelidade** | F = (1/n) soma 1[y_W(v_i) = y_LLM(v_i)]: fracao dos pontos em que o centroide mais proximo sob o w estimado concorda com o rotulo do LLM. Mede o quanto a metrica reproduz o LLM, nao se o LLM esta certo. Escopo: Problemas A e D (e, por extensao, a estimacao no Passo 2 do F e no Passo 4 do G). Nome exibido: "Fidelidade (metrica vs LLM)" |
| **Acuracia** | A = (1/n) soma 1[y_W = y_GT] ou (1/n) soma 1[y_LLM = y_GT]: fracao de concordancia com o rotulo real (ground truth) ou, no Bloco 2, com o especialista. Escopo: teste-oraculo e LLM como aprendiz (Bloco 2). Nos Blocos 1 e 3 aparece so como contexto, sempre com o nome completo "Acuracia (vs rotulo real)"; no Bloco 2, "Acuracia (LLM vs especialista)". Quando o nome da classe e arbitrario (zero-shot, pares invertidos, classes A/B) as tabelas mostram tambem a versao invariante a permutacao ("inv. perm.") |
| **Consistencia** | Mesma formula da fidelidade, mas y_W e calculado em relacao aos centroides dos NOVOS problemas (os rotulos do LLM em B e em C); a variante com os centroides do A e so ablacao (colunas consistency_centroides_a_b/c e kappa_centroides_a_b/c do CSV, nunca nas figuras); so o vetor w e transferido do Problema A. Escopo: Problemas B e C. A variante com os centroides do A ("consistencia transferida") e ablacao, so no CSV e no log |
| **Cohen's κ (kappa)** | κ = (P0 - Pe) / (1 - Pe), onde P0 e a concordancia observada e Pe = soma_k p_k p'_k a concordancia esperada pelo acaso (p_k = proporcao marginal da metrica na classe k, p'_k = a do LLM). Faixas de Landis e Koch (1977): < 0,2 fraca; 0,2 a 0,4 razoavel; 0,4 a 0,6 moderada; 0,6 a 0,8 substancial; > 0,8 quase perfeita; negativo = pior que o acaso. Calculado junto com as anteriores. **Metrica principal da hipotese H1** (κ > 0,4 em B/C) |
| **F1-Macro** | Para cada classe k: P_k = VP/(VP+FP), R_k = VP/(VP+FN), F1_k = 2 P_k R_k / (P_k + R_k); F1-Macro = media das F1_k entre as classes, de 0 a 1. Calculado junto com as anteriores (`average='macro'` em todo lugar). Nao confundir com "F1-Score" binario ou ponderado, que nao existem mais no sistema |
| **inv. perm. (invariante a permutacao)** | Coluna "Acur. vs real (inv. perm.)" = max(a, 1 - a). Em zero-shot e em classes com nome neutro o LLM pode usar o nome certo com o lado trocado; a acuracia crua ficaria abaixo do acaso embora a particao esteja certa |
| **Matriz de Confusao** | Tabela 2x2 (VP, FP, FN, VN) usada como INSTRUMENTO visual (`final_graphic_02`), nao como metrica |
| **Verdadeiro Positivo (VP)** | Disse "classe 1" e era "classe 1": acertou |
| **Falso Positivo (FP)** | Disse "classe 1" mas era "classe 0": alarme falso |
| **Falso Negativo (FN)** | Disse "classe 0" mas era "classe 1": deixou passar |
| **Verdadeiro Negativo (VN)** | Disse "classe 0" e era "classe 0": acertou |
| **Ground Truth (rotulo real)** | A resposta "verdadeira": os rotulos gerados sinteticamente ou, na base peso x altura, o sexo registrado |
| **Especialista (expert, no ingles dos artigos)** | Quem rotula os exemplos do aprendizado em contexto (Passo 3 de E e do F, Passo 6 de G). Problema E: classificador externo com W conhecido; Problema F: o proprio rotulo real; Problemas G1/G2: a metrica ELEITA no Passo 5 (ver "Eleicao do especialista"). A coluna `especialista_origem` (rotulo_real ou metrica) diz qual foi |
| **Distribuicao Gaussiana** | Distribuicao "sino" (normal). Os dados sinteticos sao gerados assim, com media (centroide) e desvio padrao |
| **Covariancia** | Mede como duas variaveis variam juntas. Positiva = sobem juntas, negativa = uma sobe outra desce |

Nao sao metricas de avaliacao (sao comparacoes de ESTIMADORES e ficam nos CSVs): similaridade de cosseno
entre os W do Perceptron e do NNLS, razao w2/w1, direcao de W, gamma. Removidos das saidas em 09/10/2026:
consistencia euclidiana, precisao e recall isolados, "concordancia" e "taxa de erro".

## Machine Learning e Classificacao

| Termo | O que e |
|---|---|
| **Classificador** | Modelo que atribui uma classe a cada ponto de entrada |
| **Classificacao Binaria** | Problema com apenas 2 classes (0 ou 1, A ou B) |
| **Fronteira de Decisao** | Linha (ou superficie) que separa as duas classes. Pontos de um lado = classe 0, do outro = classe 1 |
| **Separabilidade** | Se e possivel tracar uma fronteira que separa perfeitamente as classes |
| **Rotulo (Label)** | A classe atribuida a um ponto: y=0 ou y=1 |
| **Atributo (feature)** | Uma variavel de entrada. x1 e x2 sao os 2 atributos de cada ponto |
| **Conjunto de Treino** | Dados usados para aprender (Problema A, Passo 2: 150 pontos; Problema F: os 105 de treino, 70% dos 150; Problema G: os 70 de treino) |
| **Conjunto de Teste** | Dados usados para avaliar (Problemas B e C: 100 pontos cada; F: os 45 de teste; G: os 30 de teste) |
| **Overfitting** | Modelo decorou os dados de treino mas nao generaliza. Evitado separando treino/teste |
| **Data Leakage** | Vazamento: usar dados de teste no treino. O codigo exclui exemplos few-shot do teste |
| **Aprendizado Ativo** | Escolher estrategicamente quais exemplos mostrar (os mais informativos) |
| **Anisotropico** | Variancias diferentes em cada eixo. Ex: W=[0.3, 1.5] pesa x2 muito mais que x1 |
| **Isotropico** | Variancias iguais em todas as direcoes. Ex: W=[1.0, 1.0] = distancia Euclidiana |
| **Margem** | Distancia de um ponto ate a fronteira de decisao. Margem alta = confianca alta |
| **Colapso / quase-colapso** | Coleta em que o LLM poe todos os pontos numa classe (colapso) ou deixa menos de 5 pontos na classe minoritaria (quase-colapso, `MIN_PONTOS_CLASSE_MINORITARIA`). Nao gera metrica: vira `collapsed=1` com medidas NaN e fica fora das medias e das figuras |

## Otimizacao

| Termo | O que e |
|---|---|
| **Otimizacao Inversa** | Dado o resultado (classificacoes do LLM), descobrir qual criterio (W) ele usou. O oposto do normal |
| **Problema Direto** | Dado o criterio W, classificar pontos. Normal e facil |
| **Problema Inverso** | Dadas as classificacoes, descobrir W. E isso que a dissertacao faz |
| **Busca Binaria** | Algoritmo que divide o intervalo pela metade a cada passo para encontrar o valor otimo de gamma |
| **Convergencia** | Quando o algoritmo para de melhorar significativamente e encontrou uma solucao boa |
| **Epoca (Epoch)** | Uma passagem completa por todos os dados de treino |
| **Iteracao** | Um passo individual do algoritmo |
| **Taxa de Aprendizado (eta)** | Tamanho do "passo" na atualizacao dos pesos. Muito grande = instavel, muito pequeno = lento |
| **Relaxacao de Margem** | Permitir pequenas violacoes da margem (nem todo ponto precisa estar longe da fronteira) |
| **NNLS** | Non-Negative Least Squares. Resolve min‖Aw - b‖^2 com w >= 0. Alternativa ao Perceptron |
| **Minimos Quadrados** | Encontrar W que minimiza a soma dos erros ao quadrado |
| **Residuo** | O "erro" que sobrou apos a otimizacao. Menor residuo = melhor ajuste |
| **Restricao** | Regra que a solucao deve obedecer. Ex: w >= 0 (pesos nao-negativos) |
| **Viavel/Inviavel** | Viavel = existe solucao que satisfaz todas as restricoes. Inviavel = impossivel |
| **Regularizacao (C)** | Controla o equilibrio entre ajustar os dados e manter pesos "bem-comportados" |
| **Hiperparametro** | Parametro de configuracao que voce define antes de rodar (eta, C, max_epochs) |
| **Clipping** | Limitar valores a um intervalo para evitar explosao numerica |
| **Violacao** | Ponto que nao satisfaz a restricao de margem |
| **Best Effort (melhor esforco)** | Quando nenhum gamma > 0 e viavel, o Perceptron devolve o melhor W parcial com gamma = 0. Nas tabelas essas linhas levam o marcador † |

## Perceptron Estruturado (termos especificos)

| Termo | O que e |
|---|---|
| **Gamma** | Tamanho da margem desejada. O algoritmo busca o maior gamma que ainda tem solucao viavel (comeca em gamma_init = 0,1; no maximo max_iterations = 50 passos da busca) |
| **gamma_lo / gamma_hi** | Limites inferior e superior da busca binaria em gamma |
| **Delta Gamma** | Incremento ao encontrar solucao: tenta gamma maior para ver se ainda e viavel |
| **Lambda** | Coeficiente de regularizacao: lambda = 1/C |
| **Alpha** | Variavel dual do perceptron. Penalidade acumulada por violacoes |
| **Factor (decaimento)** | Multiplicador que encolhe os pesos: factor = 1 - eta*gamma/‖w‖. Evita explosao |
| **Tolerancia (tol)** | Limiar de parada: quando gamma_hi - gamma_lo < tol, para a busca |
| **W* (w_star)** | Melhor vetor de pesos encontrado ate agora |

## Termos do LLM

| Termo | O que e |
|---|---|
| **Prompt** | Texto de instrucao enviado ao LLM. "Classifique este ponto como A ou B" |
| **Zero-Shot** | Sem exemplos. O LLM deve decidir sozinho, usando apenas a instrucao |
| **Few-Shot** | Com exemplos. O LLM recebe N pontos ja classificados antes de decidir |
| **Temperatura** | Controla aleatoriedade: 0.0 = minimiza a aleatoriedade (nao garante a mesma resposta: 1,4% de flip medido no gpt-4o-mini na execucao completa de 10/2026, 0% no Gemini), 1.0 = maximo de aleatoriedade |
| **Flip de T=0** | Mesmo prompt e mesmo ponto devolvendo rotulos diferentes em duas consultas. Medido offline pela auditoria (`src/audit_interactions.py`) |
| **Token** | Unidade de texto que o LLM processa. Uma palavra pode ter 1-3 tokens |
| **API** | Interface para chamar o LLM programaticamente (enviar prompt, receber resposta) |
| **Rate Limit** | Limite de requisicoes por minuto imposto pelo provedor |
| **Backoff Exponencial** | Estrategia de retry: esperar 10s, depois 20s, depois 40s... dobrando a cada tentativa (ate MAX_RETRIES = 5 vezes; cada tentativa e uma chamada ao SDK, que faz ate 3 tentativas proprias) |
| **Resposta Malformada** | Quando o LLM responde algo diferente do esperado (ex: "I think it's A" em vez de "A") ou se contradiz (marcador final e ultima linha com classes diferentes). Reenviada ate 5 vezes; depois cai no fallback por hash MD5 e fica marcada `malformed=True` |
| **Parser** | Codigo que interpreta a resposta do LLM. 8 camadas de tentativa (0 a 7: ultimo marcador "Answer:" com limpeza de markdown/LaTeX, exato, regex, ultimo inteiro, padroes, etc) |
| **Estocasticidade** | Aleatoriedade inerente. Temperatura=0 minimiza para isolar o criterio de decisao |
| **Async / Concorrencia** | Enviar varias requisicoes ao mesmo tempo em vez de uma por vez (ate 50 requisicoes simultaneas (desde 14/10/2026; 40 na execucao completa de 13/10), MAX_CONCURRENCY) |

## Design Experimental

| Termo | O que e |
|---|---|
| **Seed (Semente)** | Numero que inicializa o gerador aleatorio. Mesma seed = mesmos dados = reproducibilidade |
| **Reproducibilidade** | Capacidade de repetir o experimento e obter os mesmos resultados |
| **Robustez** | Resultado se mantem estavel com diferentes seeds/configuracoes |
| **Hipotese** | Afirmacao testavel. H1: LLM tem criterio consistente (Cohen's κ > 0,4 em B/C, faixa moderada de Landis e Koch); o veredito por modelo e calculado e impresso no log (`relatorios.texto_h1`, `LIMIAR_H1_KAPPA = 0.4`: κ medio em B e C na configuracao base zero-shot, seeds acima do limiar e a linha SUSTENTADA/NAO SUSTENTADA) |
| **Baseline** | Referencia para comparacao. No Bloco 2 sao os classificadores classicos (k-NN, regressao logistica, SVM) treinados nos mesmos exemplos de cada estrategia (a tabela e a figura mostram a mista), so no primeiro modelo (nao usam o LLM); nos Problemas F e G, um Perceptron diagonal treinado nos mesmos exemplos (Passo 4 do F, Passo 7 do G) |
| **Ablacao** | Remover/mudar um componente para medir seu efeito isolado |
| **Diluicao** | Experimento: fixar 4 exemplos dificeis e ir adicionando faceis. Testar se "diluem" a informacao |
| **Vies (Bias)** | Tendencia sistematica. Ex: LLM prefere a primeira opcao? ("A ou B" vs "B ou A") |
| **Generalizacao** | Funcionar bem em dados novos, nao vistos durante o treino |
| **Transferencia** | Aplicar o w estimado num problema (A) em outros (B, C), com os centroides dos novos problemas |
| **Repeticao** | Rodar o mesmo experimento varias vezes para medir variabilidade. So onde ha sorteio de exemplos (regra `reps_para`): onde a selecao e deterministica, 1 coleta |
| **Cross-Problem** | Entre problemas diferentes. "O criterio do Problema A transfere para B?" |
| **Teto (regime passo5_teto)** | Passo 5 de A e D, Passo 4 do G: W ajustado diretamente ao rotulo real, sem LLM: o melhor que o estimador consegue naquele espaco (R2/R3/R4). Medido em Acuracia, nunca em Fidelidade |
| **especialista_origem** | Coluna de `bloco23_fg_aprendiz` (Passo 3 do F, Passo 6 do G) que diz quem foi o especialista: `rotulo_real` (Problema F) ou `metrica` (metrica eleita no Passo 5, Problemas G1/G2) |
| **Passos 2 a 5 do Problema G (antes de 13/10/2026: Fase A externa)** | Desde 13/10/2026 (Metodologia IV.7 do orientador), nos 70 pontos de treino de G1 e G2 os Passos 2 a 5 sao os mesmos do Problema A: zero-shot (Passo 2), few-shot com k em {4, 10, 20, 40} exemplos do rotulo real (k/2 por classe, 3 repeticoes; o LLM classifica os 70 - k restantes; Passo 3), estimacao em R2/R3/R4 com Perceptron e NNLS e teto (Passo 4), eleicao da melhor metrica (Passo 5) (`fg_executar(..., com_passos_3_a_5=True)`). O LLM ve so os 2 atributos (kg e m no G2). Saidas: CSV `bloco3_g_passos_2_a_5`, tabela `bloco3_table_06`, figura `bloco3_graphic_03`. O Problema F tem so o zero-shot (Passo 2) |
| **Eleicao do especialista (Problema G, Passo 5)** | Regra de `g_passo5_eleger_especialista` (13/10/2026): entre as metricas do Perceptron dos regimes passo2_zero_shot e passo3_exemplos_rotulo_real (todos os k, repeticoes e R2/R3/R4), sem o teto nem colapsos, vence a de maior acuracia vs rotulo real nos 70 de treino (`accuracy_metric_vs_true_full`); empate: zero-shot, k crescente, repeticao, R2 -> R3 -> R4. Por isso o Passo 6 do G roda mesmo quando o zero-shot colapsa. O log imprime a linha do especialista eleito ("regime=... k=... rep=... R...") |
| **especialista_regime / especialista_k / especialista_rep** | Colunas de `bloco23_fg_aprendiz` (13/10/2026) com o regime (passo2_zero_shot ou passo3_exemplos_rotulo_real), o k e a repeticao da metrica eleita como especialista em G1/G2; NaN no Problema F, cujo especialista e o rotulo real |

## Estrategias de Selecao de Exemplos (Problema E, Passo 3)

| Termo | O que e |
|---|---|
| **Easy (Faceis)** | Pontos longe da fronteira. Alta margem. Obvios, pouca informacao sobre a fronteira |
| **Hard (Dificeis)** | Pontos perto da fronteira. Baixa margem. Ambiguos, mas muito informativos |
| **Mixed (Mistos)** | Exatamente 50% faceis + 50% dificeis (n = 10 -> 5 + 5) |
| **Random (Aleatorio)** | Sem estrategia. Baseline para comparacao |

## Passos por problema (antes de 13/10/2026: fases)

| Termo | O que e |
|---|---|
| **Passo** | Etapa numerada DENTRO de um problema (letra = problema; cada problema tem a sua sequencia, de 1 em diante, so com os passos que existem nele). Escreve-se "Problema A, Passo 3", "Passo 3 do Problema A" ou, em titulos curtos, "A, Passo 3"; varios passos: "Problema A, Passos 2 a 5". Substitui, desde 13/10/2026, os nomes usados antes de 13/10/2026 ("Fases A/B/C/E"; tabela de equivalencia no CLAUDE.md). Na mesma data os identificadores de codigo foram renomeados pelo mesmo criterio (camada 2: funcoes como `adg_passos_2_a_5`, constantes como `K_EXEMPLOS_ROTULO_REAL`, tags de auditoria `passo=N`, colunas e nomes de arquivo; mapa unico em `renomeacao_passos_2026-10-13.md`) |
| **Passos 2 a 6 do Problema A (antes de 13/10/2026: Fase A)** | 1 gerar os 150 pontos; 2 LLM zero-shot; 3 LLM com k exemplos do rotulo real (k em {4, 10, 20, 40}, 3 sorteios); 4 estimar W (Perceptron e NNLS, R2/R3/R4; Fidelidade); 5 teto; 6 eleger a melhor metrica. Pergunta: "Qual criterio o LLM usa?" |
| **Passos 2 a 4 dos Problemas B e C (antes de 13/10/2026: Fases B e C)** | 1 gerar com centroides rotacionados (B horaria, C anti-horaria); 2 LLM zero-shot; 3 aplicar cada metrica do A com centroides locais (Consistencia, Cohen's κ, F1-Macro; H1); 4 LLM com k exemplos rotulados pela metrica do A (H2). Pergunta: "O criterio se mantem em dados novos, virados para um lado e para o outro?" |
| **Passos 2 a 6 do Problema D** | 1 gerar a meia-lua; 2 zero-shot; 3 few-shot do rotulo real; 4 estimar em R2/R3/R4 (Fidelidade); 5 teto; 6 SVM RBF nos rotulos do LLM |
| **Passos 2 a 6 do Problema E (antes de 13/10/2026: Fase E)** | 1 o especialista rotula; 2 LLM zero-shot (uma coleta para E1/E2/E3); 3 LLM com k exemplos do especialista pelas estrategias facil/dificil/mista/aleatoria (Acuracia LLM vs especialista; H3, H4); 4 diluicao (E1); 5 ordem dos exemplos (E1); 6 baselines k-NN/LR/SVM treinados nos mesmos exemplos de cada estrategia (a tabela e a figura mostram a mista), so no primeiro modelo (nao usam o LLM). Pergunta: "O LLM consegue aprender um criterio externo?" |
| **Passos do Problema F** | 1 gerar e dividir 70%/30% (105 e 45 pontos); 2 zero-shot nos 105 de treino e estimar em R2/R3/R4 (so fidelidade); 3 LLM com k exemplos do rotulo real, k em {0, 4, 10, 20, 40} (k = 0 e o LLM classificando o teste sem exemplos), classifica os 45 de teste; 4 Perceptron nos mesmos exemplos |
| **Passos do Problema G (G1 e G2)** | 1 carregar e dividir 70/30; 2 zero-shot nos 70; 3 few-shot do rotulo real nos 70; 4 estimar em R2/R3/R4, fidelidade e teto; 5 eleger a melhor metrica; 6 LLM com k exemplos rotulados pela metrica eleita classifica os 30 de teste, repetido com 2, 3 e 4 atributos mostrados no prompt (x1, x2; + x1·x2; + x1², x2²; fator ATRIBUTOS_PROMPT_PASSO6_G, 14/10/2026, independente da dimensao da metrica eleita; k = 0 e uma coleta so); 7 Perceptron nos mesmos exemplos e no mesmo espaco que o LLM viu |
| **Passos do Oraculo (sobre E3, E1, E2; ate 13/10/2026 sobre B, E1, E2)** | 1 rotular as gaussianas do E pelo centroide mais proximo com W conhecido ([1,1], [0,3;1,5], [1,5;0,3]); 2 Perceptron e NNLS; 3 comparar o W recuperado ao verdadeiro (acuracia e cosseno) |

Letras dos PROBLEMAS (nao confundir com os passos): A, B, C lineares e D meia-lua (Bloco 1);
E especialista linear e F meia-lua (Bloco 2); G = base real peso x altura (Bloco 3).

Regra de nomes (08/10/2026): letra = problema (dados + papel do LLM); numero = SUBPROBLEMA,
mesmos dados com outro rotulo ou outra apresentacao. Subproblemas existentes:
E1 (especialista W=[0,3;1,5], principal), E2 (W=[1,5;0,3]), E3 (W=[1;1], euclidiano);
G1 (peso x altura normalizado em [-1,1]), G2 (mesmas pessoas, prompt em kg e m). Em G1 e G2 os Passos 2 a 5
nos 70 pontos de treino sao os do Problema A (zero-shot, few-shot do rotulo real, estimacao e teto; ver
"Passos 2 a 5 do Problema G"), e o especialista do Passo 6 e eleito entre todas essas metricas (Passo 5).
O oraculo nao e problema: e um teste "sobre" E3 (W=[1,1]), E1 e E2 (sobre B ate 13/10/2026), os Problemas 1/2/3 da secao
IV.5 do PDF do orientador, com rotulos dados pelo centroide mais proximo sob o W verdadeiro e
avaliado em Acuracia. As tres configuracoes anisotropicas extras que existiam ate 13/10/2026 foram removidas do codigo:
nelas qualquer W separava os dados (W nao identificavel, Perceptron devolvia [1,1] com acuracia 100%).
