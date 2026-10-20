# Exercício 01

Transforme a o código da Rede Neural em Notebook 

Reconheça as seguintes etapas:

Imagem RGB
32 × 32 × 3

      ↓ Flatten

3072 entradas

      ↓

MLP

      ↓

10 saídas

      ↓

avião, carro, pássaro, gato,
cervo, cachorro, sapo,
cavalo, navio ou caminhão

Código Modularizado em funções.

```
main()
│
├── obter_dispositivo()
│
├── carregar_datasets()
│
├── mostrar_informacoes_dataset()
│
├── criar_dataloaders()
│
├── mostrar_imagens_dataset()
│
├── criar_modelo()
│   └── MLP
│       └── forward()
│
├── mostrar_parametros()
│
├── treinar_modelo()
│   ├── Feed Forward
│   ├── Loss
│   ├── Backpropagation
│   └── optimizer.step()
│
├── plotar_loss()
│
├── plotar_accuracy()
│
├── avaliar_modelo()
│
├── mostrar_previsoes()
│
├── mostrar_probabilidades()
│
└── salvar_modelo()
```