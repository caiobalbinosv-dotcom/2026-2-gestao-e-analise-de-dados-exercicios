from collections import defaultdict


# ==========================================
# DADOS
# ==========================================

documentos = [
    "big data usa dados testes",
    "dados geram informacao",
    "big data usa processamento"
]


# ==========================================
# MAP
# ==========================================

def mapear(texto):

    resultado = []

    palavras = texto.lower().split()

    for palavra in palavras:

        resultado.append(
            (palavra, 1)
        )

    return resultado


mapeados = []

for documento in documentos:

    resultado = mapear(documento)

    mapeados.extend(resultado)


print("MAP")

for item in mapeados:

    print(item)


# ==========================================
# SHUFFLE
# ==========================================

agrupados = defaultdict(list)

for palavra, valor in mapeados:

    agrupados[palavra].append(valor)


print("\nSHUFFLE")

for palavra, valores in agrupados.items():

    print(
        palavra,
        "->",
        valores
    )


# ==========================================
# REDUCE
# ==========================================

def reduzir(palavra, valores):

    return (
        palavra,
        sum(valores)
    )


resultado_final = []

for palavra, valores in agrupados.items():

    resultado = reduzir(
        palavra,
        valores
    )

    resultado_final.append(resultado)


print("\nREDUCE")

for palavra, quantidade in resultado_final:

    print(
        palavra,
        "->",
        quantidade
    )