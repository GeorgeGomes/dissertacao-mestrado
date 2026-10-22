# E-mail para o orientador: o que ajustei para o texto e o sistema baterem

**Assunto:** Metodologia e Resultados: ajustes para o texto e o sistema baterem

---

Raul, boa tarde.

Cruzei a Metodologia, os Resultados e a Fundamentação com o código e com a execução completa de 13/10. A maior parte já bate. Encontrei cinco pontos em que o texto dizia uma coisa e o sistema fazia outra. Resolvi quatro do lado do sistema ou com uma frase pronta para o texto; só um fica a seu critério.

1. Atributos extras no prompt do Problema G. Fiz como está na Metodologia: no aprendizado em contexto do G, o mesmo especialista agora é testado com 2, 3 e 4 atributos no prompt (x1 e x2; mais x1·x2; mais x1² e x2²), como um fator controlado. Antes os atributos extras só apareciam quando a métrica eleita tinha sido estimada em R3 ou R4, o que misturava as duas coisas. A estimativa da métrica em R2, R3 e R4, do e-mail de 05/10, já estava feita.

2. Dados do teste-oráculo. O texto diz que as métricas rotulam os dados do Problema A, mas no A isso não funciona: os dois centróides têm x2 igual a zero, então o peso w2 some da comparação de distâncias e qualquer métrica dá os mesmos rótulos, o mesmo defeito das gaussianas que retiramos. Os três testes agora rodam sobre um único conjunto, as gaussianas do Problema E, só trocando a métrica: Problema 1 com [1; 1], Problema 2 com [0,3; 1,5] e Problema 3 com [1,5; 0,3], como o senhor sugeriu na reunião. No texto, basta trocar "dados do problema A" por "gaussianas do problema E".

3. Métrica que rotula os exemplos em B e C. A melhor métrica do A entra na avaliação de consistência, como está escrito. Os exemplos do few-shot de B e C são rotulados pela métrica zero-shot do A, para que o ganho de consistência meça só o efeito dos exemplos e não a escolha da métrica, que muda de semente para semente. Sugiro uma frase no texto: "os exemplos são rotulados pela métrica zero-shot do Problema A, para isolar o efeito dos exemplos".

4. Quantidade de exemplos do especialista. O texto diz n em {4, 10}; o sistema rodou {4, 10, 20, 40} no few-shot de A, D e G e na transferência de B e C, com as tabelas prontas. Sugiro uma grade única {4, 10, 20, 40} em todo o texto. Se preferir tabelas menores, me diga quais colunas ficam, sem recoletar nada.

5. Divisão 70/30 no Problema E. O texto diz que a divisão vale para E e F, mas no E o LLM é testado nos 150 menos k pontos que não foram mostrados como exemplo; só F e G têm a divisão fixa. A regra que importa, não testar em ponto mostrado como exemplo, já vale no E. Sugiro a frase "no Problema E o teste são os pontos não mostrados como exemplo; a divisão fixa 70/30 vale para F e G". Se o senhor preferir a divisão fixa também no E, eu refaço o Bloco 2; é o único ponto em que preciso da sua palavra.

Como os pontos 1 e 2 mudam resultados, vou rodar uma execução completa nova e mandar as tabelas e figuras atualizadas.

Números da execução de 13/10 para atualizar no texto: 187.328 chamadas; 6 respostas não interpretadas, todas do Gemini e nenhuma do gpt-4o-mini; flip com T = 0 de 1,4% no gpt-4o-mini e 0% no Gemini; 50 consultas em paralelo, não dez; meia-lua, Problema E e Problema F com 150 amostras cada, o F com 105 de treino e 45 de teste; centróides do E em (-1,5; 1) e (1,5; -1) com desvio 1,3; teto da meia-lua entre 84,7% e 87,3% em R2, R3 e R4, nunca 100%; diluição com 4 exemplos difíceis fixos, não 3; R3 e R4 só no Problema A, pois B e C recebem apenas o w transferido; sem intervalo de confiança por bootstrap, como combinamos.

A explicação das quatro variantes de prompt mando em e-mail separado.

Abraços,
George
