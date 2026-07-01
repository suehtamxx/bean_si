# Classificação de Doenças em Folhas de Feijão com CNN

> Trabalho Final — Sistemas Inteligentes  
> Classificação automática de imagens de folhas de feijão usando Redes Neurais Convolucionais (CNN) com PyTorch.

---

## Sobre o Projeto

Este projeto aplica técnicas de **Deep Learning** para identificar doenças em folhas de feijão a partir de imagens. O modelo é capaz de classificar uma folha em **4 categorias**:

| Classe | Descrição |
|--------|-----------|
| `als` | *Angular Leaf Spot* — Mancha Angular |
| `bean_rust` | Ferrugem do Feijão |
| `healthy` | Folha Saudável |
| `unknown` | Não identificada |

A aplicação de visão computacional na agricultura permite detectar doenças precocemente, reduzindo perdas na produção e o uso excessivo de defensivos agrícolas.

---

## Base de Dados

- **Nome:** Bean Disease Dataset  
- **Origem:** [AI-Lab-Makerere / Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LJCUXG)  
- **Tipo:** Imagens RGB de folhas de feijão coletadas em campo  
- **Estrutura utilizada:** pasta `Classification/`

| Split | Quantidade |
|-------|-----------|
| Treino | ~6.034 imagens |
| Validação | ~1.293 imagens |
| Teste | ~1.296 imagens |
| **Total** | **~8.623 imagens** |

```
dataverse_files/
└── Classification/
    └── Classification/
        ├── training/
        │   ├── als/
        │   ├── bean_rust/
        │   ├── healthy/
        │   └── unknown/
        ├── validation/
        └── test/
```

### Exemplos do Dataset

A pasta [`dataset_examples/`](./dataset_examples/) contém **apenas 10 imagens por classe por split** para fins de demonstração e referência no repositório. Ela **não representa o dataset completo** e não deve ser usada para treinar o modelo.

Para acessar a base de dados real e completa, utilize o link abaixo:  
🔗 **[Harvard Dataverse — Bean Disease Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TCKVEW)**

---

## Arquitetura do Modelo

O modelo é uma **CNN (Rede Neural Convolucional)** construída do zero com PyTorch, composta por:

```
Entrada: Imagem RGB 224×224

┌─────────────────────────────────────┐
│  Bloco 1: Conv2D(32) + BN + ReLU   │
│           MaxPool 2×2 → 112×112     │
├─────────────────────────────────────┤
│  Bloco 2: Conv2D(64) + BN + ReLU   │
│           MaxPool 2×2 → 56×56       │
├─────────────────────────────────────┤
│  Bloco 3: Conv2D(128) + BN + ReLU  │
│           MaxPool 2×2 → 28×28       │
├─────────────────────────────────────┤
│  Flatten → Dense(256) + ReLU        │
│  Dropout(0.5)                        │
│  Dense(4) + Softmax                  │
└─────────────────────────────────────┘

Saída: 4 classes
```

### Configurações de Treinamento

| Parâmetro | Valor |
|-----------|-------|
| Otimizador | Adam (lr=0.001) |
| Loss | CrossEntropyLoss |
| Épocas | 10 |
| Batch size | 32 |
| Scheduler | StepLR (γ=0.5 a cada 5 épocas) |
| Imagem de entrada | 224×224 px |

### Data Augmentation (Treino)
- Flip horizontal e vertical aleatórios  
- Variação de brilho, contraste e saturação  
- Normalização ImageNet (mean/std)

---

## Como Rodar

### 1. Pré-requisitos

- Python **3.10–3.14**
- pip

---

### 2. Criar e ativar ambiente virtual

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

### 3. Instalar dependências

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```powershell
pip install datasets pillow numpy
```

> Se tiver GPU NVIDIA, substitua `cpu` por `cu121` na URL para usar CUDA.

---

### 4. Baixar o dataset

Faça o download do dataset completo em:  
🔗 **[Harvard Dataverse — Bean Disease Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TCKVEW)**

> A pasta `dataset_examples/` contém apenas amostras para referência. Para treinar o modelo, é necessário o dataset completo.

Extraia o arquivo e coloque a pasta `dataverse_files/` dentro do projeto:

```
trabalho_si_final/
├── dataverse_files/
│   └── Classification/
├── dataset_examples/   # Apenas exemplos (não usar para treino)
├── main.py
└── README.md
```

---

### 5. Executar o treinamento

```powershell
python main.py
```

O script irá:
1. Carregar as imagens do dataset local
2. Aplicar pré-processamento e data augmentation
3. Treinar a CNN por 10 épocas
4. Salvar o melhor modelo em `bean_disease_cnn_model_best.pth`
5. Avaliar no conjunto de teste e exibir a acurácia final

---

## Resultados

| Época | Acc Treino | Acc Validação |
|-------|-----------|--------------|
| 1     | ~34%      | ~39%         |
| 5     | ~41%      | ~43%         |
| 10    | ~42%      | ~43%         |

**Acurácia no Teste: ~43%**

---

## Estrutura do Projeto

```
trabalho_si_final/
├── dataverse_files/                    # Dataset completo (não versionado)
├── dataset_examples/                   # Amostras de exemplo (10 imagens/classe)
├── venv/                               # Ambiente virtual Python
├── main.py                             # Script principal de treinamento
├── bean_disease_cnn_model.pth          # Modelo salvo (último)
├── bean_disease_cnn_model_best.pth     # Melhor modelo (por val acc)
└── README.md                           # Este arquivo
```

---

## Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|-----------|--------|-----|
| Python | 3.14 | Linguagem principal |
| PyTorch | 2.12 | Framework de Deep Learning |
| torchvision | 0.27 | Carregamento de imagens e transforms |
| Pillow | 12.x | Manipulação de imagens |
| NumPy | 2.x | Operações numéricas |

---

## Autores

**Elder Matheus e Hermeson Alves**  
Sistemas de Informação — Trabalho Final de Sistemas Inteligentes  
2026
