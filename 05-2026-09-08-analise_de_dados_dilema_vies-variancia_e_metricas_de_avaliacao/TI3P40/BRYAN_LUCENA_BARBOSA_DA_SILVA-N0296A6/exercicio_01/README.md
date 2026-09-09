## Dilema Viés-Variância

## Exercícios de TREINO e TESTE de Modelos

## Exercício 1 — Regressão Linear: tempo de entrega

Uma empresa de entregas quer estimar o **tempo de entrega** de um pedido a partir da **distância até o cliente**.

Considere:

$$
\text{Entrada } X = \text{distância da entrega, em km}
$$

$$
\text{Saída } Y = \text{tempo de entrega, em minutos}
$$

A empresa coletou dados de 10 entregas. Sete serão usados para treinar o modelo e três para testar.

### Dados de treino

Lembrando que no treino seu modelo é criado através da descoberta dos (melhores) valores dos PARÂMETROS `w`  e `b`.

Segue o dataset de dados de treino do seu modelo. 

| Entrega | Distância \(X\) km | Tempo \(Y\) min |
| ------- | -----------------: | --------------: |
| A       |                  2 |              18 |
| B       |                  3 |              21 |
| C       |                  4 |              25 |
| D       |                  5 |              28 |
| E       |                  6 |              32 |
| F       |                  8 |              38 |
| G       |                 10 |              45 |

Use um modelo linear para resolver o problema. Seu modelo deve ter a "cara" de uma equação de reta:

$$
\hat y = wx+b
$$

Resolva utilizando programação. No python utilize o pacote `LinearRegression` da biblioteca `SciKitLearn`:

```python
from sklearn.linear_model import LinearRegression
```
-------------------------------------------------------
### Dados de teste

Uma vez criado o seu modelo $\hat y = wx+b$, use os dados de teste a seguir

| Entrega | Distância \(X\) km | Tempo real \(Y\) min |
| ------- | -----------------: | -------------------: |
| H       |                3.5 |                   23 |
| I       |                  7 |                   35 |
| J       |                  9 |                   42 |

Descubra:

Qual é a saída prevista pelo seu modelo para a entrega `H`, `I` e `J` ?
Qual é a saída real para a entrega `H`, `I` e `J` ?
Qual é a diferença entre a saída real e a saída prevista pelo seu modelo para `H`, `I` e `J`, também chamado de M.S.E. ?


### Tarefas

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train = np.array([
    [2],
    [3],
    [4],
    [5],
    [6],
    [8],
    [10]
])

y_train = np.array([
    18,
    21,
    25,
    28,
    32,
    38,
    45
])

X_test = np.array([
    [3.5],
    [7],
    [9]
])

y_test = np.array([
    23,
    35,
    42
])

modelo = LinearRegression()

modelo.fit(X_train, y_train)

w = modelo.coef_[0]
b = modelo.intercept_

print("w =", w)
print("b =", b)

y_pred = modelo.predict(X_test)

print("Previsões:", y_pred)

mse = mean_squared_error(y_test, y_pred)

print("MSE teste =", mse)

nova_distancia = np.array([[7.5]])

tempo_estimado = modelo.predict(nova_distancia)

print("Tempo estimado:", tempo_estimado[0])
```

A ideia conceitual do exercício é:

$$
\boxed{
\text{distância}
\rightarrow
\text{modelo linear}
\rightarrow
\text{tempo estimado}
}
$$
