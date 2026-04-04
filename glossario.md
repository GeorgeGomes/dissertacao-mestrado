# Glossario de Termos Tecnicos — Dissertacao de Mestrado

## Matematica e Algebra Linear

| Termo | O que e |
|---|---|
| **Distancia de Mahalanobis** | Distancia "pesada" entre um ponto e um centro. Em vez de tratar todas as dimensoes igualmente (Euclidiana), cada dimensao tem um peso W. Formula: d_W(x, c) = soma(w * (x - c)^2) |
| **Matriz Diagonal** | Matriz onde so os elementos da diagonal principal sao diferentes de zero. Simplificacao: em vez de aprender d^2 parametros, aprende apenas d |
| **Semidefinida Positiva** | Propriedade de uma matriz que garante que distancias nunca sao negativas. Necessaria para que a metrica faca sentido |
| **Centroide** | Ponto medio de um grupo de pontos. Media de todas as coordenadas dos pontos de uma classe |
| **Norma (L2)** | "Tamanho" de um vetor: ‖w‖ = raiz(w1^2 + w2^2 + ...). Mede a magnitude |
| **Vetor** | Lista ordenada de numeros. Ex: W = [0.3, 1.5] e um vetor de pesos |
| **Projecao R3** | Transformar dados de 2D para 3D adicionando x3 = x1 * x2. Simula um kernel quadratico |
| **Kernel** | Tecnica que captura relacoes nao-lineares entre variaveis sem calcular explicitamente |
| **Espaco de Features** | O "mundo" onde os dados vivem. R2 = 2 dimensoes, R3 = 3 dimensoes |

## Estatistica e Metricas de Avaliacao

| Termo | O que e |
|---|---|
| **Acuracia (Accuracy)** | Porcentagem de acertos: (acertos / total). Simples mas pode enganar com classes desbalanceadas |
| **Cohen's Kappa** | Mede concordancia entre dois avaliadores descontando o acaso. Kappa = 1 = concordancia perfeita, 0 = so acaso, <0 = pior que acaso. **Metrica principal da hipotese H1** |
| **F1-Score** | Media harmonica de precisao e recall. Equilibra os dois erros (falso positivo e falso negativo) |
| **Precisao (Precision)** | Dos que eu disse "sim", quantos realmente sao "sim"? TP / (TP + FP) |
| **Recall (Sensibilidade)** | Dos que realmente sao "sim", quantos eu acertei? TP / (TP + FN) |
| **Matriz de Confusao** | Tabela 2x2 mostrando: acertos classe 0, acertos classe 1, erros tipo 1, erros tipo 2 |
| **Verdadeiro Positivo (TP)** | Disse "classe 1" e era "classe 1" — acertou |
| **Falso Positivo (FP)** | Disse "classe 1" mas era "classe 0" — alarme falso |
| **Falso Negativo (FN)** | Disse "classe 0" mas era "classe 1" — deixou passar |
| **Verdadeiro Negativo (TN)** | Disse "classe 0" e era "classe 0" — acertou |
| **Ground Truth** | A resposta "verdadeira" — os rotulos corretos gerados sinteticamente |
| **Fidelidade** | Quanto a metrica aprendida concorda com o LLM. Alta fidelidade = a metrica capturou bem o criterio do LLM |
| **Consistencia** | O LLM usa o mesmo criterio em problemas diferentes? Medida pela concordancia W_A aplicado em B/C |
| **Distribuicao Gaussiana** | Distribuicao "sino" (normal). Os dados sinteticos sao gerados assim, com media (centroide) e desvio padrao |
| **Covariancia** | Mede como duas variaveis variam juntas. Positiva = sobem juntas, negativa = uma sobe outra desce |

## Machine Learning e Classificacao

| Termo | O que e |
|---|---|
| **Classificador** | Modelo que atribui uma classe a cada ponto de entrada |
| **Classificacao Binaria** | Problema com apenas 2 classes (0 ou 1, A ou B) |
| **Fronteira de Decisao** | Linha (ou superficie) que separa as duas classes. Pontos de um lado = classe 0, do outro = classe 1 |
| **Separabilidade** | Se e possivel tracar uma fronteira que separa perfeitamente as classes |
| **Rotulo (Label)** | A classe atribuida a um ponto: y=0 ou y=1 |
| **Feature (Caracteristica)** | Uma variavel de entrada. x1 e x2 sao as 2 features de cada ponto |
| **Conjunto de Treino** | Dados usados para aprender (Fase A: 150 pontos) |
| **Conjunto de Teste** | Dados usados para avaliar (Fases B/C: 100 pontos cada) |
| **Overfitting** | Modelo decorou os dados de treino mas nao generaliza. Evitado separando treino/teste |
| **Data Leakage** | Vazamento: usar dados de teste no treino. O codigo exclui exemplos few-shot do teste |
| **Aprendizado Ativo** | Escolher estrategicamente quais exemplos mostrar (os mais informativos) |
| **Anisotropico** | Variancias diferentes em cada eixo. Ex: W=[0.3, 1.5] pesa x2 muito mais que x1 |
| **Isotropico** | Variancias iguais em todas as direcoes. Ex: W=[1.0, 1.0] = distancia Euclidiana |
| **Margem** | Distancia de um ponto ate a fronteira de decisao. Margem alta = confianca alta |

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
| **Best Effort** | Quando nao encontra solucao perfeita, retorna a melhor parcial |

## Perceptron Estruturado (termos especificos)

| Termo | O que e |
|---|---|
| **Gamma** | Tamanho da margem desejada. O algoritmo busca o maior gamma que ainda tem solucao viavel |
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
| **Temperatura** | Controla aleatoriedade: 0.0 = deterministico (sempre mesma resposta), 1.0 = maximo de aleatoriedade |
| **Token** | Unidade de texto que o LLM processa. Uma palavra pode ter 1-3 tokens |
| **API** | Interface para chamar o LLM programaticamente (enviar prompt, receber resposta) |
| **Rate Limit** | Limite de requisicoes por minuto imposto pelo provedor (OpenAI, etc) |
| **Backoff Exponencial** | Estrategia de retry: esperar 10s, depois 20s, depois 40s... dobrando a cada tentativa |
| **Resposta Malformada** | Quando o LLM responde algo diferente do esperado (ex: "I think it's A" em vez de "A") |
| **Parser** | Codigo que interpreta a resposta do LLM. 7 camadas de tentativa (exato, regex, contains, etc) |
| **Estocasticidade** | Aleatoriedade inerente. Temperatura=0 minimiza para isolar o criterio de decisao |
| **Async / Concorrencia** | Enviar varias requisicoes ao mesmo tempo em vez de uma por vez (10x mais rapido) |

## Design Experimental

| Termo | O que e |
|---|---|
| **Seed (Semente)** | Numero que inicializa o gerador aleatorio. Mesma seed = mesmos dados = reproducibilidade |
| **Reproducibilidade** | Capacidade de repetir o experimento e obter os mesmos resultados |
| **Robustez** | Resultado se mantem estavel com diferentes seeds/configuracoes |
| **Hipotese** | Afirmacao testavel. H1: LLM tem criterio consistente (Kappa > 0.7 em B/C) |
| **Baseline** | Referencia para comparacao. Metrica Euclidiana (pesos iguais) e o baseline |
| **Ablacao** | Remover/mudar um componente para medir seu efeito isolado |
| **Diluicao** | Experimento: fixar 4 exemplos dificeis e ir adicionando faceis. Testar se "diluem" a informacao |
| **Vies (Bias)** | Tendencia sistematica. Ex: LLM prefere a primeira opcao? ("A ou B" vs "B ou A") |
| **Generalizacao** | Funcionar bem em dados novos, nao vistos durante o treino |
| **Transferencia** | Aplicar W aprendido num problema (A) em outros (B, C) |
| **Repeticao** | Rodar o mesmo experimento varias vezes para medir variabilidade |
| **Cross-Problem** | Entre problemas diferentes. "Os centroides do Problema A transferem para B?" |

## Estrategias de Selecao de Exemplos (Fase D)

| Termo | O que e |
|---|---|
| **Easy (Faceis)** | Pontos longe da fronteira. Alta margem. Obvios, pouca informacao sobre a fronteira |
| **Hard (Dificeis)** | Pontos perto da fronteira. Baixa margem. Ambiguos, mas muito informativos |
| **Mixed (Mistos)** | 50% faceis + 50% dificeis. Equilibrio |
| **Random (Aleatorio)** | Sem estrategia. Baseline para comparacao |
| **Expert (Perito)** | Classificador externo com W conhecido. Na Fase D o LLM tenta aprender o criterio do expert |

## Fases do Experimento

| Fase | O que faz | Pergunta que responde |
|---|---|---|
| **A** | LLM classifica 150 pontos sem exemplos -> aprende W | "Qual criterio o LLM usa?" |
| **B** | Aplica W_A em dados com geometria diferente | "O criterio se mantem em dados novos?" |
| **C** | Aplica W_A em dados com distorcao mais severa | "E em dados ainda mais diferentes?" |
| **D** | LLM recebe exemplos de um expert e tenta aprender | "O LLM consegue aprender um criterio externo?" |
