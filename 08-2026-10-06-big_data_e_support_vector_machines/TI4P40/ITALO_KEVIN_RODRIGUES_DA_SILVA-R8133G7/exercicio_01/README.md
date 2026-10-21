Exercício 01 - Considere as frases:

```
big data usa dados

dados geram informação

big data usa processamento
```

Usando um algoritmo **MapReduce** e **Python**, descubra quantas vezes cada palavra aparece.

Chave-Valor para este exemplo:

- Chave -> Palavra em sí

- Valor -> quantas vezes a palavra aparece


```
                  DOCUMENTOS
                      |
                      |
                      v
                +-----------+
                |    MAP    |
                +-----------+
                      |
                      v

              (big,1)
              (data,1)
              (dados,1)
              (dados,1)
              (big,1)
              ...

                      |
                      v

              +-------------+
              |   SHUFFLE   |
              +-------------+

                      |
                      v

               big → [1,1]

               data → [1,1]

               dados → [1,1]

                      |
                      v

                +----------+
                |  REDUCE  |
                +----------+

                      |
                      v

               big → 2

               data → 2

               dados → 2
```
