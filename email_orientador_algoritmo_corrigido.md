# E-mail para o orientador: nova versão do Algoritmo 1

**Assunto:** Re: Nova versão do algoritmo

---

Professor, bom dia.

**Sua mensagem:** "Nova versão do algoritmo. Corrigi o loop de saída."

**Resposta:** Obrigado pela nova versão. O laço de saída ficou igual ao do código: para quando uma época termina sem correções ou quando chega a T épocas.

Comparei o restante com o código e com a Eq. 29 do artigo do CILAMCE 2017. Encontrei quatro diferenças:

1. **Condição da correção.** Na nova versão a correção acontece quando y(i) ≠ ŷ(i). No código e na Eq. 29 ela acontece quando d_w(v(i), c(k)) − d_w(v(i), c(l)) + λ·α_i < γ·‖w‖. Na nova versão o α é atualizado, mas não entra no teste, então a suavização não tem efeito.
2. **Sinal da correção.** A nova versão usa −η·(d_w(v, c(ŷ)) − d_w(v, c(y))). A Eq. 29 e o código usam o contrário, com o centróide correto primeiro: −η·(d_w(v, c(y)) − d_w(v, c(ŷ))).
3. **Projeção.** Falta o passo w ← max(0, w), que garante w ≥ 0.
4. **Valor de γ.** Na nova versão o γ é um parâmetro fixo. No código ele aumenta até ficar inviável e depois é ajustado por busca binária (Eq. 30 e 31).

Os resultados do artigo foram gerados com o código. É para escrever o Algoritmo 1 conforme o código, ou é para mudar o código para seguir a nova versão?

Abraços,
George
