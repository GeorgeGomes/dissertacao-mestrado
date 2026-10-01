# E-mail para o orientador: pseudocódigo do código executado e página animada

**Assunto:** Algoritmo do código executado: pseudocódigo e execução passo a passo

**Anexos:** `algoritmo_codigo_executado_pseudocodigo.docx` e `algoritmo_perceptron_animado.html`

---

Professor, boa tarde.

Envio dois arquivos sobre o Perceptron para olharmos na reunião de sexta.

1. **algoritmo_codigo_executado_pseudocodigo.docx:** pseudocódigo do código que gerou todos os resultados do artigo (relaxed_perceptron.py), escrito linha a linha, na mesma notação da sua versão do Algoritmo 1. No fim há uma tabela que liga cada linha do pseudocódigo à linha correspondente do código, e as observações indicam o que vem das Eq. 28, 29 e 31 do CILAMCE 2017.

2. **algoritmo_perceptron_animado.html:** abre em qualquer navegador, sem instalar nada. Mostra passo a passo, sobre os dados reais da Fase A, três versões: o código executado, o Algoritmo 1 do artigo e a sua versão. Quando as três terminam, a página monta uma comparação dos resultados. A versão do código reproduz exatamente os números do Python.

Sobre a sua dúvida de como o algoritmo termina em problema não separável: o código faz a suavização da margem, com o termo λ.α_i (linha 18 do pseudocódigo). Cada valor de γ tem no máximo 50 épocas; se nenhuma época termina sem violação, γ é considerado inviável e a busca binária reduz γ até a tolerância. Se nenhum γ foi viável, o retorno é o w da época com menos violações, com γ = 0.

Na versão do artigo, o passo 7 (busca binária em f) não diz qual condição é testada. Na página usei f = 1 e expliquei o motivo na própria aba.

O Algoritmo 1 do artigo deve passar a descrever o código executado, como está no Word, ou o senhor prefere que o código siga a sua versão?

Abraços,
George
