# E-mail para o orientador: término do Perceptron e Algoritmo 1

**Assunto:** Re: Dúvida sobre o término do perceptron

---

Professor, boa tarde.

Sobre a sua dúvida: o perceptron que gerou os resultados faz a suavização da margem. A restrição testada em cada ponto tem o termo λ·α_i, com λ = 1/C e C = 1, como na Eq. 28 do artigo do CILAMCE 2017.

Em problema não separável ele termina em três níveis. Cada tentativa de γ roda no máximo 50 épocas; se nenhuma época fecha sem violação, γ é marcado como inviável e a busca binária reduz γ até o intervalo ficar abaixo de 10⁻⁴, o que leva 11 tentativas. Como nenhum γ foi viável, o retorno é o w da época com menos violações, com γ = 0 e um aviso no log.

A dúvida faz sentido porque o Algoritmo 1 do artigo é um resumo e não mostra essa parte. Ele foi escrito a partir dos slides de abril, de forma condensada, e ficou sem o termo λ·α_i, sem o fator (1 − ηγ/‖w‖) e sem a busca em γ; a "busca binária em f" que aparece ali corresponde, no código, à busca em γ. O código completo está no repositório desde abril e foi ele que produziu todos os números do artigo.

Para facilitar a comparação, preparei uma página HTML (em anexo, abre em qualquer navegador, sem instalar nada). Ela anima passo a passo, sobre os dados reais da Fase A, três versões: o código executado, o Algoritmo 1 do artigo e a sua versão corrigida, com um quadro comparando os resultados das três e o pseudocódigo do código escrito no mesmo formato da sua versão.

Como o senhor ficou com o texto, pergunto: o Algoritmo 1 deve passar a descrever o código executado, ou o senhor prefere que o código siga a sua versão? Se achar melhor, vemos a página juntos na reunião de sexta.

Abraços,
George
