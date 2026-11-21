# Classificação de Leucemia Linfoblástica Aguda (C-NMC 2019) com Deep Learning

**SIN 393 – Introdução à Visão Computacional (2026)** · Universidade Federal de Viçosa – Campus Rio Paranaíba
Prof. João Fernando Mari

**Integrantes:** Otavio Silva Pierin - 9339, Breno Silva Pierin - 8154, Lucas Brasil - 8755
**Vídeo de apresentação (YouTube):** _[[inserir link](https://youtu.be/kVp7HuPlALA)]_

---

## 1. Descrição do projeto

Este projeto implementa, em **PyTorch**, um pipeline completo de classificação de imagens para apoiar o diagnóstico de
**Leucemia Linfoblástica Aguda de células B (B-ALL)** a partir de imagens microscópicas de células únicas. Cada imagem é
classificada em uma de duas classes:

| Classe | Rótulo | Descrição |
|---|---|---|
| `hem` | 0 | Célula normal (hematogônia) |
| `all` | 1 | Célula leucêmica (linfoblasto) |

### Dataset

- **Fonte:** [Leukemia Classification – Kaggle (andrewmvd)](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification), versão do desafio **C-NMC 2019** (ISBI 2019, SBILab).
- **Treino:** 10.661 imagens (7.272 ALL / 3.389 HEM) de 73 pacientes, em 3 *folds* (`training_data/fold_{0,1,2}/{all,hem}`).
- **Preliminary test:** 1.867 imagens rotuladas (1.219 ALL / 648 HEM) de 28 pacientes **diferentes** dos de treino (`validation_data/` + CSV de rótulos).
- **Final test:** 2.586 imagens sem rótulos públicos (não utilizado).
- Imagens BMP de 450×450 px, com a célula segmentada sobre fundo preto. As classes estão desbalanceadas, na proporção de ~2:1.

### Questão experimental: comparação entre arquiteturas

> Sob o **mesmo protocolo** (transfer learning do ImageNet, mesmo aumento de dados, hiperparâmetros, semente e divisão de dados),
> qual arquitetura generaliza melhor para **pacientes não vistos**, e a diferença é estatisticamente significativa?

A comparação usa as métricas do Classification Report, a matriz de confusão, a ROC-AUC, o custo computacional (parâmetros, tempo de
treino e de inferência) e o **teste de McNemar exato** par a par sobre as predições no mesmo conjunto de teste.

### Escolha dos modelos (revisão da literatura)

As três arquiteturas do `torchvision` com melhores resultados e uso mais frequente no C-NMC 2019:

| Modelo | Parâmetros | Por que foi escolhido |
|---|---|---|
| **EfficientNet-B3** | ~10,7 M | Melhor F1 relatado com transfer learning no C-NMC (F1 = 94,30%; Ahmed, 2025). Em benchmark com pacientes disjuntos (2026), a família EfficientNet foi a única com sensibilidade e especificidade equilibradas. |
| **ResNet50** | ~23,5 M | *Backbone* da solução vencedora do desafio (ResNet + *neighborhood correction*, F1 ≈ 92,5%) e *baseline* mais usado. |
| **DenseNet121** | ~7,0 M | Muito usada no C-NMC, isolada ou em fusão de características (DenseNet121-ResNet50-MobileNet); reutiliza características com poucos parâmetros. |

VGG16 (~138 M de parâmetros, desempenho semelhante) e ViTs (piores que as CNNs neste dataset pequeno) foram descartados.
O notebook (etapa 0) traz a tabela da revisão e as referências.

### Protocolo experimental

- **Divisão hold-out:**
  - **Teste:** *preliminary test* oficial, com pacientes que não aparecem no treino. Isso evita o **vazamento de paciente** (*data leakage*)
    presente em trabalhos que dividem o dataset por imagem e relatam acurácias de 97–99%. Com `TEST_STRATEGY = "split"`, o teste é obtido por divisão estratificada.
  - **Treino/validação:** `train_test_split` do scikit-learn, **estratificado**, 85%/15%, com **semente fixa 42**.
  - O teste é usado **apenas na avaliação final**.
- **Treinamento:** *fine-tuning* completo com pesos do ImageNet; entrada de 224×224 para todos os modelos; AdamW (lr 1e-4, wd 1e-4);
  *cosine annealing*; batch 32; até 25 épocas com *early stopping* (paciência 6) pelo **F1-macro de validação**; entropia cruzada
  **ponderada pelas classes** por causa do desbalanceamento; precisão mista (AMP) na GPU.
- **Avaliação:** Classification Report completo (precisão, recall, F1 e suporte por classe; médias **macro** e **weighted**), acurácia,
  acurácia balanceada, sensibilidade, especificidade, ROC-AUC, matrizes de confusão (absoluta e normalizada) e curvas ROC.
- **Agregação:** a métrica principal é o **F1-macro**, em que as duas classes têm o mesmo peso e os erros na classe minoritária HEM são
  penalizados. O F1-weighted também é relatado, para comparar com a métrica oficial do desafio.

---

## 2. Aumento de dados (data augmentation) e justificativa

Aplicado **somente ao conjunto de treino**. Validação e teste recebem apenas redimensionamento e normalização ImageNet.

| Transformação | Parâmetros | Justificativa |
|---|---|---|
| `RandomHorizontalFlip` + `RandomVerticalFlip` | p = 0,5 | Uma célula na lâmina não tem orientação canônica; espelhar não muda a morfologia diagnóstica (forma do núcleo, relação núcleo/citoplasma, cromatina). |
| `RandomRotation` | 0–180°, preenchimento preto | Mesma invariância à rotação. O preenchimento preto é igual ao fundo da imagem segmentada, então não cria bordas artificiais. |
| `RandomAffine` | translação ±5%, escala 0,95–1,05 | Simula pequenas diferenças de centralização e enquadramento da segmentação. A escala é **limitada de propósito** porque o tamanho da célula é uma característica discriminativa (linfoblastos tendem a ser maiores). |
| `ColorJitter` (p = 0,8) | brilho/contraste/saturação 0,2; matiz 0,02 | O C-NMC tem grande **variação de coloração** entre lâminas, lotes e pacientes. O jitter torna o modelo mais robusto a essa variação e ajuda na generalização para pacientes novos. A matiz varia pouco para não descaracterizar a coloração. |
| `GaussianBlur` (p = 0,2) | kernel 3, σ 0,1–1,0 | Simula pequenas variações de **foco do microscópio** durante a aquisição. |

**Transformações evitadas de propósito:**
- `RandomResizedCrop` agressivo: cortaria partes da célula e mudaria o tamanho aparente, que é uma pista diagnóstica;
- deformações elásticas fortes, *cutout* e *random erasing*: alterariam ou apagariam o contorno e a textura do núcleo, a principal informação para distinguir as classes.

O aumento de dados combate o sobreajuste (milhares de imagens de poucos pacientes), aumenta a variabilidade efetiva da classe minoritária e
deixa o modelo invariante a fatores que não têm relação com o diagnóstico (orientação, cor, foco).

---

## 3. Estrutura do repositório

```
.
├── SIN393_Leukemia_Classification.ipynb   # notebook principal (etapas 0–15)
├── README.md
└── outputs/                               # criado na execução
    ├── models/      # <modelo>_best.pth (pesos do melhor checkpoint + metadados)
    ├── reports/     # classification_report_<modelo>.txt/.csv, comparacao_modelos.csv,
    │                # mcnemar_pares.csv, predicoes_teste_<modelo>.csv
    ├── figures/     # distribuição, augmentation, curvas, matrizes de confusão, ROC,
    │                # amostras_corretas_<modelo>.png, amostras_incorretas_<modelo>.png, ...
    ├── samples/     # <modelo>/corretas/*.bmp e <modelo>/incorretas/*.bmp
    └── logs/        # config.json, history_<modelo>.csv, split_{train,val,test}.csv
```

### Etapas do notebook

| # | Etapa |
|---|---|
| 0 | Revisão da literatura e escolha dos modelos |
| 1 | Instalação de dependências e importações |
| 2 | Configurações e reprodutibilidade (semente, hiperparâmetros, dispositivo) |
| 3 | Download do dataset (`kagglehub`) |
| 4 | Indexação e análise exploratória (classes, pacientes, exemplos) |
| 5 | Divisão treino / validação / teste (`train_test_split` estratificado) |
| 6 | Aumento de dados e pré-processamento (com visualização) |
| 7 | Datasets, DataLoaders e pesos de classe |
| 8 | Definição das arquiteturas (ResNet50, DenseNet121, EfficientNet-B3) |
| 9 | Funções de treino e validação (AMP, early stopping, checkpoint) |
| 10 | Treinamento e salvamento dos modelos (`.pth`) |
| 11 | Curvas de aprendizado |
| 12 | Avaliação no teste: Classification Report, matriz de confusão, ROC |
| 13 | Análise qualitativa: amostras corretas/incorretas por arquitetura, casos difíceis e erro por paciente |
| 14 | Questão experimental: comparação entre arquiteturas + teste de McNemar |
| 15 | Discussão e conclusão |

---

## 4. Instruções de execução

### Requisitos

- Python ≥ 3.10
- GPU NVIDIA com CUDA (recomendado; testado para Google Colab com GPU **T4**). Em CPU funciona, mas o treino completo leva muitas horas.
- ~2 GB de disco para o dataset e ~250 MB para os modelos salvos.
- Acesso à internet para baixar o dataset (Kaggle) e os pesos pré-treinados (download.pytorch.org).

### Dependências

| Pacote | Versão mínima |
|---|---|
| torch | 2.1 |
| torchvision | 0.16 |
| scikit-learn | 1.3 |
| numpy, pandas, scipy | — |
| matplotlib, seaborn | — |
| pillow, tqdm | — |
| kagglehub | 0.2 |
| jupyter / ipykernel | (execução local) |

```bash
pip install torch torchvision scikit-learn numpy pandas scipy matplotlib seaborn pillow tqdm kagglehub jupyter
```

### Opção A — Google Colab (recomendado)

1. Abra o notebook no Colab (*File → Upload notebook* ou pelo link do GitHub).
2. *Runtime → Change runtime type → **T4 GPU***.
3. *Runtime → Run all*. O notebook instala o que faltar e baixa o dataset automaticamente com o `kagglehub`.
4. Ao final, baixe a pasta `outputs/` (por exemplo, com `!zip -r outputs.zip outputs` e o painel de arquivos), ou monte o Google Drive para salvar os resultados.

### Opção B — Execução local

```bash
git clone <url-do-repositorio>
cd <repositorio>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # ajuste à sua versão de CUDA
pip install scikit-learn numpy pandas scipy matplotlib seaborn pillow tqdm kagglehub jupyter
jupyter notebook SIN393_Leukemia_Classification.ipynb
```

**Dataset já baixado?** Aponte para a pasta que contém `C-NMC_Leukemia/` antes de abrir o notebook:

```bash
export LEUKEMIA_DATA_DIR=/caminho/para/leukemia-classification   # Windows: set LEUKEMIA_DATA_DIR=...
```

> No Windows, se houver erro nos DataLoaders, defina `NUM_WORKERS = 0` na etapa 2.

### Teste rápido do pipeline

Para verificar se tudo funciona antes do treino completo (subconjunto pequeno e 1 época):

```bash
QUICK_RUN=1 jupyter nbconvert --to notebook --execute SIN393_Leukemia_Classification.ipynb --output teste_rapido.ipynb
```

### Parâmetros configuráveis (etapa 2)

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `SEED` | 42 | Semente global (divisão, inicialização, DataLoaders) |
| `IMG_SIZE` | 224 | Resolução de entrada (igual para os 3 modelos) |
| `BATCH_SIZE` / `EPOCHS` / `LR` | 32 / 25 / 1e-4 | Hiperparâmetros de treino |
| `PATIENCE` | 6 | Early stopping |
| `VAL_SIZE` | 0,15 | Fração do treino usada na validação |
| `TEST_STRATEGY` | `"prelim"` | `"prelim"` (pacientes não vistos) ou `"split"` (divisão estratificada por imagem) |
| `MODEL_NAMES` | 3 modelos | Lista das arquiteturas a treinar |

### Tempo estimado (Colab T4)

Tempo medido: ResNet50 ≈ 16 min (parou na época 11 por early stopping), DenseNet121 ≈ 37 min e EfficientNet-B3 ≈ 37 min (25 épocas cada).
O experimento completo leva cerca de **1h30**.

### Carregar um modelo treinado

```python
import torch
from torchvision import models
ckpt = torch.load("outputs/models/efficientnet_b3_best.pth", map_location="cpu")
model = models.efficientnet_b3()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(ckpt["state_dict"]); model.eval()
print(ckpt["class_names"], ckpt["epoch"], ckpt["val_f1_macro"])
```

Cada `.pth` guarda `state_dict`, nome do modelo, classes, tamanho de entrada, média/desvio de normalização, época e F1 de validação.

---

## 5. Resultados

Os resultados abaixo são do conjunto de teste independente (*preliminary test*: 1.867 imagens, 1.219 ALL e 648 HEM, de 28 pacientes não vistos
no treino). O treinamento foi feito no Google Colab com GPU T4. Os arquivos de origem são `outputs/reports/comparacao_modelos.csv` e
`outputs/reports/mcnemar_pares.csv`.

### 5.1 Métricas de classificação

| Modelo | Acurácia | Acur. balanceada | Precisão (macro) | Recall (macro) | **F1-macro** | F1-weighted | Sensibilidade (ALL) | Especificidade (HEM) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| ResNet50 | 0,8018 | 0,7517 | 0,7971 | 0,7517 | 0,7655 | 0,7937 | **0,9155** | 0,5880 | 0,8732 |
| DenseNet121 | **0,8259** | **0,7940** | 0,8139 | **0,7940** | **0,8021** | **0,8231** | 0,8983 | **0,6898** | **0,8942** |
| EfficientNet-B3 | 0,8254 | 0,7922 | **0,8141** | 0,7922 | 0,8009 | 0,8222 | 0,9007 | 0,6836 | 0,8864 |

### 5.2 Custo computacional

| Modelo | Parâmetros (M) | Tempo de treino (min) | Melhor época | Inferência (ms/imagem, GPU T4, batch 1) |
|---|---|---|---|---|
| ResNet50 | 23,51 | 15,8 | 5 (early stopping na época 11) | **6,4** |
| DenseNet121 | **6,96** | 37,1 | 25 | 18,1 |
| EfficientNet-B3 | 10,70 | 37,0 | 24 | 14,0 |

### 5.3 Questão experimental: teste de McNemar (exato, α = 0,05)

| Modelo A | Modelo B | A acerta / B erra | A erra / B acerta | p-valor | Diferença significativa? |
|---|---|---|---|---|---|
| ResNet50 | DenseNet121 | 126 | 171 | 0,0106 | **Sim**: DenseNet121 é melhor |
| ResNet50 | EfficientNet-B3 | 125 | 169 | 0,0120 | **Sim**: EfficientNet-B3 é melhor |
| DenseNet121 | EfficientNet-B3 | 115 | 114 | 1,0000 | Não |

### 5.4 Discussão

- **Resposta à questão experimental:** **DenseNet121** (F1-macro 0,802) e **EfficientNet-B3** (F1-macro 0,801) superaram a ResNet50
  (F1-macro 0,765), e o teste de McNemar confirma que essa diferença é significativa (p ≈ 0,01 nas duas comparações). Entre DenseNet121
  e EfficientNet-B3 não há diferença significativa: cada uma acerta cerca de 115 imagens que a outra erra (p = 1,0). Na prática, as duas
  empatam no topo.
- **Eficiência:** a DenseNet121 chegou ao melhor resultado com o **menor número de parâmetros** (6,96 M, cerca de 3,4× menos que a
  ResNet50), mas teve a inferência mais lenta (18,1 ms/imagem), por causa das muitas concatenações de mapas de características. A
  EfficientNet-B3 fica no meio-termo: desempenho equivalente, 10,7 M de parâmetros e inferência mais rápida que a DenseNet. A ResNet50 é a
  mais rápida (6,4 ms), mas é a menos precisa.
- **Sensibilidade x especificidade:** os três modelos detectam bem as células leucêmicas (sensibilidade de 0,90–0,92), mas têm
  especificidade bem menor (0,59–0,69). Ou seja, boa parte das células normais (HEM) é classificada como ALL. A ResNet50 tem o viés mais
  forte para a classe majoritária: maior sensibilidade (0,916) e menor especificidade (0,588). Em triagem clínica, perder um caso de
  leucemia (falso negativo) é mais grave que gerar um alarme falso, então esse viés é aceitável. Ainda assim, a especificidade pode
  melhorar com ajuste do limiar de decisão pela curva ROC (AUC de 0,87–0,89).
- **Comparação com a literatura:** os resultados (acurácia de cerca de 82–83%) estão abaixo dos 92–98% relatados em vários trabalhos, mas
  na mesma faixa do benchmark *leakage-aware* (arXiv:2606.24944), que obteve 81,8% de acurácia com EfficientNet-B0 em avaliação com
  pacientes disjuntos. A diferença em relação aos números mais altos vem principalmente do protocolo: aqui o teste usa **pacientes não
  vistos**, enquanto os trabalhos com 97–99% dividem as imagens aleatoriamente e misturam células do mesmo paciente no treino e no teste.
- **Convergência:** a DenseNet121 e a EfficientNet-B3 tiveram o melhor resultado nas últimas épocas (25 e 24 de 25), então ainda estavam
  melhorando quando o treino terminou. Mais épocas poderiam elevar o desempenho delas. A ResNet50 atingiu o melhor resultado de validação
  cedo (época 5) e depois sobreajustou, o que é compatível com sua maior capacidade (23,5 M de parâmetros) para o tamanho do dataset.
- **Principais fontes de erro** (ver `outputs/figures/amostras_incorretas_*.png` e `casos_dificeis_todos_modelos.png`): semelhança
  morfológica entre linfoblastos e hematogônias, variação de coloração entre lâminas e pacientes, artefatos de segmentação e ruído de
  rótulo, já que o rótulo vem do diagnóstico do paciente e não de cada célula.

### 5.5 Trabalhos futuros

Treinar por mais épocas (DenseNet121 e EfficientNet-B3), ajustar o limiar de decisão para equilibrar sensibilidade e especificidade,
normalizar a coloração (Macenko/Reinhard), fazer um *ensemble* DenseNet121 + EfficientNet-B3, usar validação cruzada por *fold* de
paciente e aplicar Grad-CAM para interpretar as regiões usadas pelos modelos.

---

## 6. Referências

1. Gupta, A.; Gupta, R. *ALL Challenge dataset of ISBI 2019 (C-NMC 2019)*. The Cancer Imaging Archive, 2019. DOI: 10.7937/tcia.2019.dc64i46r
2. Ahmed, F. *Transfer Learning with EfficientNet for Accurate Leukemia Cell Classification*. arXiv:2508.06535, 2025.
3. *A Leakage-Aware Comparative Benchmark of Machine Learning, Deep Learning, and Transformer Models for Reliable Leukemia Detection*. arXiv:2606.24944, 2026.
4. Prellberg, J.; Kramer, O. *Acute Lymphoblastic Leukemia Classification from Microscopic Images using Convolutional Neural Networks*. arXiv:1906.09020, 2019.
5. Ahmed, I. A. et al. *Hybrid Techniques for Diagnosis with WSIs for Early Detection of Acute Lymphoblastic Leukemia (C-NMC 2019 / ALL-IDB2)*. Diagnostics, 13(6):1026, 2023.
6. Tan, M.; Le, Q. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML, 2019.
7. He, K. et al. *Deep Residual Learning for Image Recognition*. CVPR, 2016.
8. Huang, G. et al. *Densely Connected Convolutional Networks*. CVPR, 2017.
