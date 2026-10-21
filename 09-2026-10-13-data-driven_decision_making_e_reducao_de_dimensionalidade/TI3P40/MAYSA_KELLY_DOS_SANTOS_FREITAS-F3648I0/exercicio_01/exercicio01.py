# ============================================================
# CIFAR-10 com MLP de 10 camadas usando PyTorch
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
DATA_DIR = "./data"


CLASSES = [
    "avião",
    "carro",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão"
]


# ============================================================
# MODELO
# ============================================================

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        # ----------------------------------------------------
        # 10 camadas Linear
        # ----------------------------------------------------

        self.layer1 = nn.Linear(3072, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, 256)
        self.layer5 = nn.Linear(256, 256)
        self.layer6 = nn.Linear(256, 128)
        self.layer7 = nn.Linear(128, 128)
        self.layer8 = nn.Linear(128, 64)
        self.layer9 = nn.Linear(64, 32)
        self.layer10 = nn.Linear(32, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        # ----------------------------------------------------
        # Flatten
        #
        # [batch, 3, 32, 32]
        #
        # vira
        #
        # [batch, 3072]
        # ----------------------------------------------------

        x = self.flatten(x)

        # ----------------------------------------------------
        # Feed Forward
        # ----------------------------------------------------

        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        x = self.relu(self.layer8(x))
        x = self.relu(self.layer9(x))

        # Última camada:
        # não aplicamos Softmax aqui
        x = self.layer10(x)

        return x


# ============================================================
# DISPOSITIVO
# ============================================================

def obter_dispositivo():

    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    print(f"Dispositivo utilizado: {device}")

    return device


# ============================================================
# DATASET
# ============================================================

def carregar_datasets():

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


# ============================================================
# DATALOADERS
# ============================================================

def criar_dataloaders(
    train_dataset,
    test_dataset,
    batch_size=BATCH_SIZE
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


# ============================================================
# INFORMAÇÕES DO DATASET
# ============================================================

def mostrar_informacoes_dataset(
    train_dataset,
    test_dataset
):

    print()
    print("=" * 60)
    print("INFORMAÇÕES DO CIFAR-10")
    print("=" * 60)

    print(
        f"Imagens de treinamento: "
        f"{len(train_dataset)}"
    )

    print(
        f"Imagens de teste: "
        f"{len(test_dataset)}"
    )

    image, label = train_dataset[0]

    print(
        f"Formato de uma imagem: "
        f"{image.shape}"
    )

    print(
        f"Classe da primeira imagem: "
        f"{CLASSES[label]}"
    )

    print()

    print(
        "Número de entradas da MLP:"
    )

    print(
        f"3 × 32 × 32 = "
        f"{3 * 32 * 32}"
    )


# ============================================================
# VISUALIZAÇÃO DO DATASET
# ============================================================

def mostrar_imagens_dataset(
    train_loader,
    quantidade=12
):

    images, labels = next(
        iter(train_loader)
    )

    plt.figure(
        figsize=(10, 7)
    )

    for i in range(quantidade):

        plt.subplot(
            3,
            4,
            i + 1
        )

        image = images[i].permute(
            1,
            2,
            0
        )

        plt.imshow(image)

        plt.title(
            CLASSES[
                labels[i].item()
            ]
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# CRIAÇÃO DO MODELO
# ============================================================

def criar_modelo(device):

    model = MLP()

    model = model.to(device)

    return model


# ============================================================
# CONTAGEM DE PARÂMETROS
# ============================================================

def mostrar_parametros(model):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("=" * 60)
    print("PARÂMETROS DA REDE")
    print("=" * 60)

    print(
        f"Total de parâmetros: "
        f"{total_params:,}"
    )

    print(
        f"Parâmetros treináveis: "
        f"{trainable_params:,}"
    )


# ============================================================
# TREINAMENTO
# ============================================================

def treinar_modelo(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epochs=EPOCHS
):

    historico_loss = []
    historico_accuracy = []

    print()
    print("=" * 60)
    print("TREINAMENTO")
    print("=" * 60)

    for epoch in range(epochs):

        # ----------------------------------------------------
        # Modo de treinamento
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0

        correct = 0
        total = 0

        # ----------------------------------------------------
        # Percorre os batches
        # ----------------------------------------------------

        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)

            # ------------------------------------------------
            # 1. Zerar gradientes anteriores
            # ------------------------------------------------

            optimizer.zero_grad()

            # ------------------------------------------------
            # 2. Feed Forward
            # ------------------------------------------------

            outputs = model(images)

            # ------------------------------------------------
            # 3. Calcular o erro
            # ------------------------------------------------

            loss = criterion(
                outputs,
                labels
            )

            # ------------------------------------------------
            # 4. Backpropagation
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # 5. Gradiente Descendente
            #
            # Atualiza os pesos
            # ------------------------------------------------

            optimizer.step()

            # ------------------------------------------------
            # Estatísticas
            # ------------------------------------------------

            running_loss += (
                loss.item()
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        # ----------------------------------------------------
        # Métricas da época
        # ----------------------------------------------------

        epoch_loss = (
            running_loss
            /
            len(train_loader)
        )

        epoch_accuracy = (
            100
            *
            correct
            /
            total
        )

        historico_loss.append(
            epoch_loss
        )

        historico_accuracy.append(
            epoch_accuracy
        )

        print(
            f"Época "
            f"{epoch + 1:02d}/"
            f"{epochs:02d}"
            f" | "
            f"Loss: "
            f"{epoch_loss:.4f}"
            f" | "
            f"Acurácia: "
            f"{epoch_accuracy:.2f}%"
        )

    return (
        historico_loss,
        historico_accuracy
    )


# ============================================================
# AVALIAÇÃO
# ============================================================

def avaliar_modelo(
    model,
    test_loader,
    device
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            # Feed Forward
            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = (
        100
        *
        correct
        /
        total
    )

    print()
    print("=" * 60)
    print("TESTE")
    print("=" * 60)

    print(
        f"Acurácia no conjunto "
        f"de teste: "
        f"{accuracy:.2f}%"
    )

    return accuracy


# ============================================================
# GRÁFICO DA LOSS
# ============================================================

def plotar_loss(
    historico_loss
):

    epochs = range(
        1,
        len(historico_loss) + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        historico_loss,
        marker="o"
    )

    plt.xlabel(
        "Época"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Erro durante o treinamento"
    )

    plt.grid()

    plt.show()


# ============================================================
# GRÁFICO DA ACURÁCIA
# ============================================================

def plotar_accuracy(
    historico_accuracy
):

    epochs = range(
        1,
        len(historico_accuracy) + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        historico_accuracy,
        marker="o"
    )

    plt.xlabel(
        "Época"
    )

    plt.ylabel(
        "Acurácia (%)"
    )

    plt.title(
        "Acurácia durante o treinamento"
    )

    plt.grid()

    plt.show()


# ============================================================
# MOSTRAR PREVISÕES
# ============================================================

def mostrar_previsoes(
    model,
    test_loader,
    device,
    quantidade=12
):

    images, labels = next(
        iter(test_loader)
    )

    images_device = images.to(
        device
    )

    model.eval()

    with torch.no_grad():

        outputs = model(
            images_device
        )

        _, predicted = torch.max(
            outputs,
            1
        )

    plt.figure(
        figsize=(12, 8)
    )

    for i in range(quantidade):

        plt.subplot(
            3,
            4,
            i + 1
        )

        image = images[i].permute(
            1,
            2,
            0
        )

        plt.imshow(image)

        real = CLASSES[
            labels[i].item()
        ]

        previsto = CLASSES[
            predicted[i].item()
        ]

        plt.title(
            f"Real: {real}\n"
            f"Previsto: {previsto}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# PROBABILIDADES DE UMA IMAGEM
# ============================================================

def mostrar_probabilidades(
    model,
    test_loader,
    device,
    indice=0
):

    images, labels = next(
        iter(test_loader)
    )

    image = images[
        indice
    ].unsqueeze(0)

    image = image.to(
        device
    )

    model.eval()

    with torch.no_grad():

        output = model(
            image
        )

        probabilities = (
            torch.softmax(
                output,
                dim=1
            )
        )

    probabilities = (
        probabilities[0]
        .cpu()
    )

    predicted_class = torch.argmax(
        probabilities
    ).item()

    real_class = labels[
        indice
    ].item()

    print()
    print("=" * 60)
    print("PROBABILIDADES")
    print("=" * 60)

    print(
        f"Classe real: "
        f"{CLASSES[real_class]}"
    )

    print(
        f"Classe prevista: "
        f"{CLASSES[predicted_class]}"
    )

    print()

    for classe, prob in zip(
        CLASSES,
        probabilities
    ):

        print(
            f"{classe:10s}: "
            f"{prob.item() * 100:6.2f}%"
        )


# ============================================================
# SALVAR MODELO
# ============================================================

def salvar_modelo(
    model,
    arquivo="modelo_mlp_cifar10.pth"
):

    torch.save(
        model.state_dict(),
        arquivo
    )

    print()
    print(
        f"Modelo salvo em: "
        f"{arquivo}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Dispositivo
    # --------------------------------------------------------

    device = obter_dispositivo()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        train_dataset,
        test_dataset
    ) = carregar_datasets()

    mostrar_informacoes_dataset(
        train_dataset,
        test_dataset
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    (
        train_loader,
        test_loader
    ) = criar_dataloaders(
        train_dataset,
        test_dataset
    )

    # --------------------------------------------------------
    # Mostrar algumas imagens
    # --------------------------------------------------------

    mostrar_imagens_dataset(
        train_loader
    )

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    model = criar_modelo(
        device
    )

    print()
    print("=" * 60)
    print("ARQUITETURA")
    print("=" * 60)

    print(model)

    mostrar_parametros(
        model
    )

    # --------------------------------------------------------
    # Função de erro
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Otimizador
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Treinamento
    # --------------------------------------------------------

    (
        historico_loss,
        historico_accuracy
    ) = treinar_modelo(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS
    )

    # --------------------------------------------------------
    # Gráficos
    # --------------------------------------------------------

    plotar_loss(
        historico_loss
    )

    plotar_accuracy(
        historico_accuracy
    )

    # --------------------------------------------------------
    # Teste
    # --------------------------------------------------------

    avaliar_modelo(
        model,
        test_loader,
        device
    )

    # --------------------------------------------------------
    # Previsões
    # --------------------------------------------------------

    mostrar_previsoes(
        model,
        test_loader,
        device
    )

    # --------------------------------------------------------
    # Probabilidades
    # --------------------------------------------------------

    mostrar_probabilidades(
        model,
        test_loader,
        device,
        indice=0
    )

    # --------------------------------------------------------
    # Salvar modelo
    # --------------------------------------------------------

    salvar_modelo(
        model
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()
