# E-mail para o orientador: confirmar o que levar na reunião de sexta

**Assunto:** Re: rescrever

---

Raul, bom dia.

Sexta à tarde está ótimo, pode marcar o horário.

Para me organizar até lá, entendi que o que devo levar é o seguinte. Pode confirmar se é isso?

1. Problema A: estimar a métrica também em few-shot, com exemplos rotulados pelo ground truth (k em 4, 10, 20 e 40), sem split, e apresentar em uma nova tabela ao lado da zero-shot e da estimada pelo ground truth.
2. Problemas B e C: manter a transferência só com a métrica zero-shot e explicar melhor no texto que as amostras são novas e como os centróides são recalculados.
3. Problema D: repetir o protocolo do A, inclusive a estimação em few-shot.
4. Texto: retirar o IC bootstrap, retirar a explicação do fallback e trocar os dois parágrafos pela versão reescrita.
5. Teste do retorno imediato: fica só para o problema real peso × altura, e não é para sexta.

Duas dúvidas para fechar:

- O Cohen's d mede a diferença entre médias em unidades de desvio padrão (aqui, quanto a consistência muda de zero para 10 exemplos). Não é o kappa. Prefere que eu explique isso no texto ou que retire também, como o bootstrap?
- Sobre o Algoritmo 1, é para escrever no artigo conforme o código ou para ajustar o código à sua nova versão?

Abraços,
George
