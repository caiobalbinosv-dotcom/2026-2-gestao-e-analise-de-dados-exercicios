# Dilema Viés-Variança

## Exercícios de TREINO-TESTE

### Problema de Classificação (Regressão Logística)

## Exercício 2 — Regressão Logística: aprovação de crédito

Agora queremos trabalhar com **classificação**, não com previsão de um valor contínuo.

Um banco está testando um modelo simplificado para classificar solicitações de crédito.

Considere:

$$
X = \text{pontuação de crédito}
$$

e:

$$
Y =
\begin{cases}
0, & \text{crédito não aprovado}\\
1, & \text{crédito aprovado}
\end{cases}
$$

Temos novamente 10 clientes: 7 para treinamento e 3 para teste.

### Dados de treino

| Cliente | Pontuação \(X\) | Classe \(Y\) |
| ------- | --------------: | -----------: |
| A       |             420 |            0 |
| B       |             460 |            0 |
| C       |             500 |            0 |
| D       |             540 |            0 |
| E       |             600 |            1 |
| F       |             650 |            1 |
| G       |             720 |            1 |

Os alunos devem treinar um modelo de Regressão Logística:

$$
P(Y=1|X)
=
\frac{1}{1+e^{-(wx+b)}}
$$

### Dados de teste

| Cliente | Pontuação \(X\) | Classe real \(Y\) |
| ------- | --------------: | ----------------: |
| H       |             480 |                 0 |
| I       |             580 |                 1 |
| J       |             680 |                 1 |

### Tarefas

1. Crie `X_train`, `y_train`, `X_test` e `y_test`.
2. Treine uma `LogisticRegression`.
3. Descubra os parâmetros:

   $$
   w
   $$

   e

   $$
   b
   $$
4. Calcule as probabilidades:

   $$
   P(Y=1)
   $$

   para os três dados de teste.
5. Classifique os três clientes usando:

   $$
   \text{threshold}=0{,}5
   $$
6. Compare as classes previstas com as classes reais.
7. Calcule a acurácia no conjunto de teste.
8. Use o modelo para responder:

> **Um cliente com pontuação de crédito igual a 560 seria classificado como crédito aprovado ou não aprovado?**

9. Informe também a probabilidade estimada de aprovação.

### Estrutura esperada em Python

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train = np.array([
    [420],
    [460],
    [500],
    [540],
    [600],
    [650],
    [720]
])

y_train = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1
])

X_test = np.array([
    [480],
    [580],
    [680]
])

y_test = np.array([
    0,
    1,
    1
])

modelo = LogisticRegression()

modelo.fit(X_train, y_train)

w = modelo.coef_[0][0]
b = modelo.intercept_[0]

print("w =", w)
print("b =", b)

probabilidades = modelo.predict_proba(X_test)[:, 1]

print("Probabilidades:", probabilidades)

y_pred = modelo.predict(X_test)

print("Classes previstas:", y_pred)

acc = accuracy_score(y_test, y_pred)

print("Acurácia teste =", acc)

novo_cliente = np.array([[560]])

prob = modelo.predict_proba(novo_cliente)[0][1]
classe = modelo.predict(novo_cliente)[0]

print("Probabilidade de aprovação =", prob)
print("Classe prevista =", classe)
```

## Comparação entre os dois exercícios

| Aspecto          | Regressão Linear | Regressão Logística    |
| ---------------- | ---------------- | ---------------------- |
| Entrada \(X\)    | Distância        | Pontuação de crédito   |
| Saída \(Y\)      | Tempo em minutos | Classe 0 ou 1          |
| Tipo de problema | Regressão        | Classificação          |
| Modelo           | \(\hat y=wx+b\)  | Sigmoide               |
| Resultado        | Valor contínuo   | Probabilidade + classe |
| Métrica de teste | MSE              | Acurácia               |
| Inferência final | "Quanto tempo?"  | "Qual classe?"         |

LEMBRE-SE:

#### Treino

$$
\boxed{\text{treino aprende os parâmetros}}
$$

#### Teste

$$
\boxed{\text{teste avalia o modelo sem alterar os parâmetros}}
$$
