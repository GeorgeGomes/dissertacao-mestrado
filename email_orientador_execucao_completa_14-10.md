# E-mail para o orientador: execução completa de 14/10 e o resultado de H1 no Problema B

**Assunto:** Execução completa nova: resultados e uma decisão sobre H1 no Problema B

---

Raul, bom dia.

Rodei a execução completa com os dois ajustes que comentei no e-mail anterior: no Problema G o aprendizado em contexto agora é testado com 2, 3 e 4 atributos no prompt, como fator controlado, e o teste-oráculo roda as três métricas sobre as gaussianas do Problema E. Foram 221.508 consultas aos dois modelos, 6 respostas não interpretadas, todas do Gemini, e 3 h 58 de duração. Conferi a pasta inteira: nada faltando, nomes certos, CSVs batendo com tabelas e log. As tabelas e figuras novas estão prontas para o senhor.

Os resultados principais ficaram como antes: fidelidade no A de 83% (gpt) e 81% (Gemini) em zero-shot, oráculo recuperando as três métricas com 100% no Perceptron e 97% a 100% no NNLS, H4 refutada nos dois modelos, Problema G chegando a 87% a 90% com 40 exemplos no gpt. No G, mostrar os atributos extras no prompt não ajudou de forma consistente: no gpt o melhor foi com x1·x2 (89,6%), no Gemini o melhor foi só com x1 e x2 (88,0%).

Um resultado mudou, e preciso da sua decisão. O veredito de H1 no Problema B para o gpt-4o-mini inverteu entre as duas execuções completas: na de 13/10 o κ médio em B foi 0,46 (sustentada), na de 14/10 foi 0,33 (não sustentada), com o mesmo código, os mesmos prompts e temperatura zero. Em C nada mudou (κ de 0,74 e 0,77). Investiguei a causa na semente 42, onde o κ foi de 0,61 para 0,28.

O que aconteceu: o modelo trocou 9 respostas em 350 entre as execuções (1 no A, 4 no B, 4 no C). Com um rótulo diferente no A, o Perceptron parou numa régua com razão w1/w2 de 0,15 em vez de 0,77, e essa razão é o que decide a consistência em B. O motivo de fundo é a geometria do A: com os centróides em (−2, 0) e (2, 0), a diferença de distâncias não depende de w2, e mesmo com os centróides calculados dos rótulos do LLM a fidelidade no A quase não muda com a razão (de 0,71 a 0,83 variando a razão 400 vezes). Já em B, com os centróides girados, a consistência vai de 0,62 a 0,81 conforme a razão. Ou seja, o A fixa w1 mas deixa w2 solto, e o B testa justamente w2; por isso B é instável e C não.

Duas observações que ajudam. A métrica eleita no A (20 exemplos) é mais estável que a zero-shot: dá κ em B de 0,45 e 0,42 nas duas execuções, acima de 0,4 nas duas, e é a "melhor métrica transferida" que o senhor já descreve na Metodologia. E o Perceptron de 14/10 reproduz melhor os rótulos do LLM no A do que o de 13/10 (85% contra 78%), isto é, a estimativa nova não é pior, só aponta para outro lado numa direção que o A não determina.

Como pretendo fazer: (1) escrever no texto, como achado, que o A com centróides alinhados em x1 identifica w1 mas não w2, e que B e C testam a direção que o A deixou livre, o que explica a diferença entre os dois; (2) reportar H1 em B pela métrica eleita, com o κ zero-shot como faixa entre execuções (0,33 a 0,46) e não como um único valor. Nada disso exige nova execução.

A única mudança que exigiria rodar de novo é gerar o Problema A com centróides que difiram nas duas coordenadas, por exemplo (−2; −0,7) e (2; 0,7), mantendo B e C como rotações desse A. Isso alteraria a Tabela 1 que o senhor já escreveu. Não vou fazer isso sem a sua palavra: acha que vale, ou deixamos o A como está e tratamos a limitação no texto?

Abraços,
George
