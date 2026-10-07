
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix



dados = pd.DataFrame({

    "Pessoa": [
        "A", "B", "C", "D", "E",
        "F", "G", "H", "I", "J"
    ],

    "Aquario": [
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0
    ],

    "Filhos": [
        1, 1, 0, 1, 0,
        1, 0, 0, 1, 0
    ],

    "Alianca": [
        1, 1, 1, 1, 0,
        1, 0, 0, 0, 0
    ],

    "Casado": [
        1, 1, 1, 1, 0,
        1, 0, 0, 0, 0
    ]

})


print("\n" + "=" * 60)
print("DATASET COMPLETO")
print("=" * 60)

print(dados)



X = dados[
    [
        "Aquario",
        "Filhos",
        "Alianca"
    ]
]

y = dados["Casado"]


print("\n" + "=" * 60)
print("CARACTERÍSTICAS - X")
print("=" * 60)

print(X)


print("\n" + "=" * 60)
print("CLASSE QUE QUEREMOS PREVER - y")
print("=" * 60)

print(y)



X_treino, X_teste, y_treino, y_teste = train_test_split(

    X,
    y,

    test_size=0.30,

    random_state=42,

    stratify=y

)


print("\n" + "=" * 60)
print("DIVISÃO TREINO / TESTE")
print("=" * 60)

print(
    "Quantidade total:",
    len(dados)
)

print(
    "Quantidade para TREINO:",
    len(X_treino)
)

print(
    "Quantidade para TESTE:",
    len(X_teste)
)



print("\n" + "=" * 60)
print("DADOS DE TREINAMENTO")
print("=" * 60)

print(
    dados.loc[X_treino.index]
)


print("\n" + "=" * 60)
print("DADOS DE TESTE")
print("=" * 60)

print(
    dados.loc[X_teste.index]
)


modelo = DecisionTreeClassifier(

    criterion="gini",

    random_state=42

)


modelo.fit(
    X_treino,
    y_treino
)


print("\n" + "=" * 60)
print("MODELO TREINADO")
print("=" * 60)

print("A árvore aprendeu utilizando os dados de treinamento.")


plt.figure(
    figsize=(14, 7)
)

plot_tree(

    modelo,

    feature_names=[
        "Aquário",
        "Filhos",
        "Aliança"
    ],

    class_names=[
        "Não casado",
        "Casado"
    ],

    filled=True,

    rounded=True,

    fontsize=11
)

plt.title(
    "Árvore de Decisão - Exemplo de Joaquim",
    fontsize=16
)

plt.show()




y_pred = modelo.predict(
    X_teste
)



comparacao = pd.DataFrame({

    "Pessoa":
        dados.loc[X_teste.index, "Pessoa"],

    "Real":
        y_teste,

    "Previsto":
        y_pred

})



comparacao["Real"] = comparacao["Real"].map({

    0: "Não casado",

    1: "Casado"

})


comparacao["Previsto"] = comparacao["Previsto"].map({

    0: "Não casado",

    1: "Casado"

})


print("\n" + "=" * 60)
print("RESULTADOS DO TESTE")
print("=" * 60)

print(comparacao)



acuracia = accuracy_score(
    y_teste,
    y_pred
)


print("\n" + "=" * 60)
print("ACURÁCIA")
print("=" * 60)

print(
    f"Acurácia no conjunto de teste: "
    f"{acuracia * 100:.2f}%"
)



# MATRIZ DE CONFUSÃO


matriz = confusion_matrix(
    y_teste,
    y_pred
)


print("\n" + "=" * 60)
print("MATRIZ DE CONFUSÃO")
print("=" * 60)

print(matriz)


# Adicionando gente nova

novos = pd.DataFrame({

    "Aquario": [
        1,
        0
    ],

    "Filhos": [
        0,
        1
    ],

    "Alianca": [
        0,
        1
    ]

},

index=[
    "Joaquim",
    "Manoel"
])


print("\n" + "=" * 60)
print("NOVOS INDIVÍDUOS")
print("=" * 60)

print(novos)




previsoes = modelo.predict(
    novos
)


# calcular probabilidades

probabilidades = modelo.predict_proba(
    novos
)


# tabelar resultados

resultado = novos.copy()


resultado["Previsao"] = previsoes


resultado["Prob_Nao_Casado"] = (
    probabilidades[:, 0]
)


resultado["Prob_Casado"] = (
    probabilidades[:, 1]
)


resultado["Previsao"] = resultado["Previsao"].map({

    0: "Não casado",

    1: "Casado"

})


print("\n" + "=" * 60)
print("PREVISÃO PARA JOAQUIM E MANOEL")
print("=" * 60)

print(resultado)


# resumo

print("\n" + "=" * 60)
print("RESUMO DA AULA")
print("=" * 60)

print("""

X
    Características utilizadas pelo modelo.

y
    Classe que queremos prever.

train_test_split()
    Separa dados de treinamento e teste.

70%
    Dados utilizados para TREINAR a árvore.

30%
    Dados utilizados para TESTAR a árvore.

fit()
    Aprende as regras.

Gini
    Mede a impureza dos grupos.

Gini = 0
    Grupo completamente puro.

predict()
    Utiliza a árvore aprendida para fazer previsões.

accuracy_score()
    Compara as previsões com os valores reais.

""")

