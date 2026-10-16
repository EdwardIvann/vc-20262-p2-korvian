# Korvian — Classificação de Dígitos de Hidrômetros com AlexNet, VGG16 e ResNet

**Disciplina:** Visão Computacional (VC001) — CESAR School, 2026.2
**Integrantes:** Ivan Edward (iers@cesar.school) · Davi Melo (dam3@cesar.school) · Elizabete Barbosa (eba@cesar.school)

## Fonte dos dados e licença

Dataset: [Yandex.Toloka Water Meters](https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset),
licença **CC BY-NC-ND 4.0**. Este repositório **não** contém imagens do dataset, recortes gerados nem checkpoints.
O uso é restrito a fins acadêmicos e não comerciais.

## Como reproduzir

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate   |   Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # GPU NVIDIA (omitir para CPU/MPS)
pip install -r requirements.txt
# opcional: kaggle.json em ~/.kaggle/ (sem credencial, o notebook usa o endpoint público da API do Kaggle)
jupyter nbconvert --to notebook --execute --inplace pratica_watermeters.ipynb   # ou "Run All" no Jupyter
```

Execute a partir da raiz do repositório (o notebook usa caminhos relativos; no Colab: `!git clone <url>` e
`%cd <repositório>` antes de rodar). O notebook baixa o dataset para `data/` (se ainda não existir), gera `digits/` (recortes + `index.csv`),
`checkpoints/`, `history_*.csv`, `results.csv`, `figures/` e `reports/`. As pastas `data/`, `digits/` e
`checkpoints/` estão no `.gitignore`. As células marcadas com a tag `dataset-image` exibem imagens do dataset e
têm a saída limpa antes de qualquer commit.

- Ambiente usado: local, Windows 10, Python 3.13, PyTorch 2.11 + CUDA 12.8, GPU **NVIDIA GeForce RTX 4070 SUPER (12 GB)**
- Tempo total de execução do notebook: **≈ 9,5 min** (treino dos quatro modelos: ≈ 7 min)
- Split utilizado: `splits.csv` **oficial** fornecido pelo professor (não alterado; SHA-256 `ca4c5d8a…abf2b`, idêntico ao do repositório do professor)

## Protocolo de treino

| Modelo | Pesos | Entrada | Batch | LR cabeça | LR backbone | Épocas máx. | Melhor época |
|---|---|---|---|---|---|---|---|
| smallcnn | do zero | 64×48 | 256 | 1e-3 | — | 10 | 10 |
| alexnet | ImageNet (IMAGENET1K_V1) | 128×128 | 128 | 1e-3 | 1e-4 | 8 | 7 |
| vgg16_bn | ImageNet (IMAGENET1K_V1) | 128×128 | 32 | 1e-3 | 1e-5 | 8 | 7 |
| resnet18 | ImageNet (IMAGENET1K_V1) | 128×128 | 128 | 1e-3 | 1e-4 | 8 | 7 |

Comum a todos: `AdamW` (weight decay 0,01) + `OneCycleLR` (`max_lr` = LR de cada grupo, passo por batch),
`CrossEntropyLoss(label_smoothing=0.1)`, 1ª época com backbone congelado (pré-treinados), early stopping pelo F1-macro
da validação com paciência 3, `seed=42` com cuDNN determinístico, augmentation
`RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))` + `ColorJitter(brightness=0.3, contrast=0.3)`,
sem flip, normalização com média/desvio do ImageNet.

Alterações em relação ao protocolo oficial (e justificativa): **nenhuma nos hiperparâmetros**. Decisões de
implementação, iguais para os quatro modelos:

- "cabeça" = a última camada linear (nova, 10 saídas); "backbone" = todo o resto da rede. A `smallcnn` não tem
  backbone pré-treinado, então todos os seus parâmetros usam o LR da cabeça e ela não passa pela época congelada;
- a `smallcnn` também é normalizada com as estatísticas do ImageNet, para manter um único pré-processamento;
- o augmentation é aplicado no recorte original 64×48 e só depois a imagem é redimensionada (bilinear) para 128×128;
- na época congelada, os pesos do backbone não recebem gradiente, mas as estatísticas das BatchNorms (VGG16-BN,
  ResNet18) continuam sendo atualizadas, porque o modelo roda em modo `train()`;
- `seconds` no `history_*.csv` mede apenas o treino da época (sem validação). A mediana de `s_per_epoch` inclui a
  época congelada, que é mais rápida;
- o `AdaptiveAvgPool2d` das redes pré-treinadas foi trocado por uma implementação matematicamente idêntica (verificada
  com `assert`) com backward determinístico na GPU. Sem isso, AlexNet e VGG variavam até 0,3 p.p. entre execuções com
  a mesma seed. Com a troca, duas execuções completas do notebook produziram `history_*.csv` e métricas idênticos
  (só os tempos variam). O notebook também carrega cada checkpoint na arquitetura padrão do torchvision, com o pooling
  original, e confirma que as predições no teste são idênticas: os modelos são exatamente os citados no roteiro.

## Resultados (conjunto de teste)

| model | acc_digit | f1_macro | acc_reading | params_M | s_per_epoch | ms_per_image | ckpt_MB | best_epoch |
|---|---|---|---|---|---|---|---|---|
| smallcnn | 0.7894 | 0.7373 | 0.2567 | 0.2887 | 6.2596 | 1.0196 | 1.1719 | 10 |
| alexnet | 0.9218 | 0.9029 | 0.631 | 57.0448 | 6.547 | 1.384 | 228.1851 | 7 |
| vgg16_bn | 0.9372 | 0.9264 | 0.7005 | 134.31 | 29.944 | 3.3699 | 537.306 | 7 |
| resnet18 | 0.9606 | 0.9535 | 0.8449 | 11.1816 | 7.1998 | 2.7438 | 44.8059 | 7 |

`ms_per_image`: GPU RTX 4070 SUPER, batch 1, mediana de 200 execuções após 20 de warm-up, só o forward do modelo
(pré-processamento fora da medida), com cuDNN em modo determinístico (igual para os quatro modelos). Como referência para a
discussão de implantação, a mesma medida em **CPU** está em `reports/latency_cpu.csv`
(smallcnn 2,0 ms · alexnet 10,3 ms · vgg16_bn 49,9 ms · resnet18 8,2 ms).

Figuras (pasta `figures/`):

- Matrizes de confusão normalizadas: `figures/confusion_matrices.png`
- Acurácia por posição: `figures/accuracy_by_position.png`
- Curvas de treino: `figures/training_curves.png`
- Desempenho × custo: `figures/performance_vs_cost.png`
- (extra) Frequência dos dígitos por posição: `figures/eda_digits_by_position.png`

Relatórios (pasta `reports/`): `classification_report_{modelo}.txt`, `latency_cpu.csv`, `top_confused_pairs.csv`,
`reading_analysis.csv`, `params_by_part.csv`, `strips_meta.csv` (decisão de orientação por imagem),
`crop_audit.csv` (auditoria dos recortes) e `metrics_without_bad_crops.csv`.

## Respostas às questões

**1.1** O dataset tem três pastas com os mesmos 1244 nomes de arquivo e um `data.csv`:
- `images/`: as fotos originais dos hidrômetros (JPG; a maioria, 913 de 1244, tem 1333×1000 px, altura×largura, em retrato);
- `masks/`: máscaras binárias do visor (região do contador em branco, fundo preto), salvas em JPEG, por isso têm
  ruído de compressão;
- `collage/`: a foto com o polígono do visor desenhado por cima. Serve só para visualizar.

O `data.csv` tem três colunas: `photo_name` (o nome do arquivo, que também carrega o id e a leitura, ex.:
`id_53_value_595_825.jpg`), `value` (a leitura em m³, como float) e `location` (um polígono com 4, 6 ou 8 vértices em
coordenadas normalizadas de 0 a 1, que marca o visor). Para treinar, usamos `images/` junto com o `data.csv`: o
polígono localiza o visor e o rótulo vem do nome/valor. `masks/` também poderia treinar um segmentador/detector do
visor, mas é redundante com o polígono, que é mais preciso que uma máscara JPEG. `collage/` não deve ser usada, porque
o desenho do polígono "contamina" a imagem.

**1.2** O rótulo é **por imagem**: uma leitura inteira (ex.: 595,825), não um rótulo por dígito, e não existe caixa
por dígito. Para classificar dígitos, temos que **inferir** o rótulo de cada recorte a partir da posição. Isso supõe
que todo visor tem 5+3 rodas e que dividir a faixa em 8 partes iguais cai em cima de cada dígito. Quando essa
suposição falha, o recorte recebe o rótulo errado, mesmo com a leitura correta. Ou seja, a tarefa vira classificação
com **rótulos fracos/ruidosos**, e parte do erro medido vem do pipeline de recorte, não do classificador.

**2.1** A fonte mais confiável é o **nome do arquivo**, completado com zeros e conferido contra `value`. As evidências:
- os dois valores numéricos batem em 1244/1244 imagens, mas só 944 batem como texto: 300 valores do `data.csv` têm
  artefato de ponto flutuante (`305.16200000000003`). Extrair dígitos dessa string geraria rótulos errados;
- as duas fontes perdem o zero final dos decimais (128 imagens têm 1–2 casas). Conferimos visualmente as faixas e a
  última roda mostra mesmo um 0 (`70_94` → `0007094|0`), então completar com zeros à direita está certo;
- nos 36 rótulos com decimal `0`, inspecionamos as faixas: 33 têm polígono que cobre só a parte inteira (decimais não
  anotados). Nesses casos nenhuma das fontes tem a parte decimal verdadeira.

O rótulo é ruidoso em uma fração pequena, mas identificável, das imagens.

**2.2** O **0**: são 3549 de 9944 recortes (35,7%; no teste, 515 de 1496 = 34,4%), contra 590–1003 para os outros
dígitos. O motivo são os **zeros à esquerda**: o visor tem 5 rodas inteiras, mas a leitura típica é pequena (mediana de
188,8 m³; 95,9% abaixo de 1000 m³). Por isso a posição 0 é zero em 99,8% das imagens (só 3 leituras têm 5 dígitos inteiros), a posição 1 em
96% e a posição 2 em 32% (`figures/eda_digits_by_position.png`). Nas posições 3–7 a distribuição é próxima da
uniforme: cada dígito fica entre 7% e 13%.
O 1 vem em segundo (1003) pelo mesmo efeito na posição 2 (leituras entre 100 e 199).

**3.1** Recortes da mesma foto compartilham iluminação, modelo do hidrômetro, fonte dos dígitos, desfoque e fundo. Os
zeros à esquerda de uma mesma imagem são quase cópias uns dos outros. Se dividíssemos por recorte, o modelo veria no
treino "irmãos" dos recortes de teste e a métrica mediria memorização da foto, não generalização para hidrômetros
novos (vazamento). Além disso, `acc_reading` só faz sentido se os 8 dígitos de uma imagem estiverem no mesmo split.
Todos os grupos usam o mesmo `splits.csv` para que os resultados sejam **comparáveis**: entre os quatro modelos, entre
grupos e com o gabarito do professor. Também evita que alguém escolha (mesmo sem querer) uma divisão favorável, e
torna o experimento reprodutível.

**3.2** Quantidade de faixas corrigidas: **113 de 1243**. Foram 105 visores verticais girados 90° e 45 faixas giradas
180°: 8 horizontais e 37 verticais (algumas faixas receberam as duas correções). A decisão de 180° usa o degrau do
polígono, que funciona porque a moldura dos decimais é mais alta. Isso cobre 729 faixas. O vermelho dos decimais entra
só nos verticais (30 faixas). As duas pistas concordam em 99,7% dos 607 casos em que ambas são decisivas.

Tipos de recorte errado e quantidades. Auditamos com o melhor modelo (resnet18), comparando os acertos na faixa como
está e girada 180° (`reports/crop_audit.csv`):
- **polígono só com a parte inteira: 33 imagens (264 recortes; 4 imagens no teste).** O rótulo `xxx.000` é imposto a
  uma faixa que tem só 5 dígitos, e os 8 recortes ficam desalinhados. Parte deles ainda coincide com o rótulo por
  acaso, sobretudo os zeros à esquerda: a resnet18 acerta 190 dos 264 (no teste, 16 de 32);
- **orientação errada: 12 imagens (96 recortes; 1 no teste).** As 12 são faixas em que nenhuma pista foi usada:
  polígono de 4 vértices e, nas verticais, vermelho abaixo do limiar. São 7 verticais e 5 horizontais. Em 2 das
  horizontais o vermelho teria acertado, mas ele não é usado em visores horizontais (ver notebook, Parte 3);
- **outro desalinhamento/ilegível: 2 imagens (1 no teste).**

No total, 47 imagens (3,8%) têm recortes sistematicamente errados. Há ainda um erro difuso, que a auditoria não
conta: rodas em transição (metade de um dígito, metade do próximo) e o último decimal parcialmente visível. Isso
aparece na acurácia da posição 7, a pior para todos os modelos (resnet18: 0,914 contra 0,95–0,99 nas outras posições).

**3.3** Dividir em 8 larguras iguais supõe que o polígono é justo ao visor e que as rodas são igualmente espaçadas,
com 1/8 da largura cada. Na prática:
1. muitos polígonos (6/8 vértices) incluem a moldura maior dos decimais, e margens laterais;
2. há um espaço/separador entre a parte inteira e a decimal;
3. a perspectiva e o `minAreaRect` distorcem as extremidades;
4. o último decimal costuma estar meio escondido.

O resultado é um **deslocamento acumulado**: o recorte sai descentrado, corta o dígito ou pega metade do vizinho,
pior nas pontas (posição 7, e a 0/1 em alguns visores). Correções possíveis:
- segmentar pelo perfil de projeção vertical (os separadores escuros entre rodas aparecem como vales) ou por
  componentes conectados;
- usar o degrau do polígono para separar inteiros (5 células) e decimais (3 células) e dividir cada bloco à parte;
- usar homografia nos 4 cantos reais do polígono em vez do retângulo mínimo;
- recortar com margem e deixar a CNN se acomodar;
- ou abandonar a divisão e ler a faixa inteira com um modelo de sequência (CRNN+CTC).

**4.1** Os pesos pré-treinados foram aprendidos com entradas normalizadas pela média/desvio do ImageNet. Os filtros da
primeira camada e as estatísticas acumuladas das BatchNorms (VGG16-BN, ResNet) esperam essa escala. Sem a mesma
normalização, as ativações ficam deslocadas e as features transferidas perdem qualidade logo no início do
fine-tuning. Na `smallcnn`, treinada do zero, a escolha é indiferente, mas usamos a mesma normalização para manter um
pré-processamento único.

**4.2** Parâmetros: smallcnn 0,29 M · resnet18 11,18 M · alexnet 57,04 M · vgg16_bn 134,31 M, uma diferença de 465×
entre a menor e a maior (`reports/params_by_part.csv`). Na AlexNet e na VGG16, a maioria está no **classificador
totalmente conectado**:
- AlexNet: 95,7% (54,6 M). Só a 1ª camada FC (9216→4096) tem 37,8 M (66%);
- VGG16-BN: 89,0% (119,6 M). Só a 1ª FC (25088→4096) tem 102,8 M (76,5%);
- as convoluções somam apenas 2,5 M (AlexNet) e 14,7 M (VGG).

Na ResNet18 é o oposto: 99,95% está nas convoluções, porque o *global average pooling* deixa a cabeça com só 5130
parâmetros. Por isso a ResNet tem um checkpoint 12× menor que o da VGG.

**4.3** **Treinar só a cabeça** mantém o backbone como extrator de features fixo do ImageNet. É rápido e pouco propenso
a overfitting, mas fica limitado a features genéricas, que não são específicas para dígitos de rolo em baixa resolução.
**Fine-tuning completo** adapta todas as camadas ao domínio. É melhor quando os domínios diferem, mas é mais caro e
tem mais risco de destruir as features pré-treinadas. Congelamos só na 1ª época porque a cabeça nasce aleatória: nos
primeiros passos, seus gradientes são grandes e sem sentido e, propagados ao backbone, estragariam as features. Uma
época congelada "aquece" a cabeça; depois liberamos tudo, com LR menor no backbone. Nos nossos dados, o F1 de
validação com backbone congelado (época 1) foi 0,36 (AlexNet), 0,33 (VGG) e 0,08 (ResNet). Após descongelar, saltou
para 0,75, 0,77 e 0,82 na época 2. Isso sugere que as features do ImageNet sozinhas não bastam e que o fine-tuning é
essencial. Mas é uma evidência limitada: com uma única época e o OneCycleLR ainda em aquecimento (LR baixo), a cabeça
não chega a convergir. A ResNet, por exemplo, termina a época 1 com acurácia de validação de 0,364, praticamente
prevendo só 0 (37% da validação é 0).

**4.4** Espelhar um dígito não produz outro exemplo da mesma classe: produz um símbolo que não existe (um 2, 3, 7 ou 4
espelhado) ou se parece com outra classe. O flip vertical aproxima 6/9 e cria formas ambíguas. A câmera nunca vê um
hidrômetro espelhado, então o flip só adiciona exemplos irreais e confunde as classes. Além disso, a orientação foi
corrigida explicitamente no pipeline; o augmentation deve imitar variações plausíveis (pequena rotação, translação,
escala, brilho e contraste), e o espelhamento não é uma delas.

**5.1** A **resnet18**, com F1-macro de 0,9535, foi a melhor, seguida de vgg16_bn (0,9264), alexnet (0,9029) e
smallcnn (0,7373). Nos nossos números, o ranking pela acurácia é o mesmo (0,9606 > 0,9372 > 0,9218 > 0,7894). Mas a
acurácia é sempre maior que o F1-macro e a diferença varia: 0,7 p.p. na ResNet e 5,2 p.p. na smallcnn. O motivo é o
desbalanceamento: o 0 é 34,4% do teste e é uma das classes mais fáceis (recall 0,98 nos pré-treinados), então a acurácia é "puxada" por
ele. O F1-macro dá o mesmo peso às 10 classes e expõe as classes ruins. A smallcnn tem o maior gap porque usa o 0 como
"resposta padrão": prevê 0 para 32% dos 9, 30% dos 6 e 20% dos 5 (recall do 9 = 0,33). Ela ganha acurácia com isso e
perde F1. O ranking poderia inverter entre modelos próximos se um deles acertasse mais 0s e errasse mais classes raras.

**5.2** Nos modelos pré-treinados, os pares mais confundidos (`reports/top_confused_pairs.csv`) são 0↔1 (ResNet 7,
VGG 12, AlexNet 10 erros), 8↔9 (AlexNet 9), 0↔5 (AlexNet 6, VGG 5), 1↔7, 1↔4, 0↔6 e 6↔8. Na smallcnn, 45% dos
erros (143 de 315) são previsões 0 (6→0: 33; 9→0: 27; 1→0: 20; 5→0: 19). Dois fatores explicam:
1. **rodas em transição.** No contador mecânico, a roda gira e mostra metade do dígito atual e metade do seguinte
   (0→1, 8→9, 9→0, 6→7). O rótulo diz um, a imagem mostra os dois;
2. **semelhança de forma em baixa resolução.** 0/6/8/9 são formas fechadas que diferem por um traço; 1/7/4 compartilham
   o traço vertical. Em recortes de 48×64 px (largura×altura), com desfoque e reflexo do vidro, esses detalhes somem.

O viés da smallcnn para o 0 vem do desbalanceamento (35,7% da base). Também há ruído de rótulo dos recortes
desalinhados: um recorte de "nada" ou de meia roda tende a virar 0.

**5.3** `acc_reading` exige os **8** dígitos certos. Se cada dígito acerta com probabilidade p, a leitura acerta com
cerca de p⁸, e o erro se acumula: na ResNet, 0,9606⁸ ≈ 0,72; na AlexNet, 0,9218⁸ ≈ 0,52; na smallcnn, 0,7894⁸ ≈ 0,15.
Os valores observados são **maiores** que p⁸ (0,845, 0,631 e 0,257, em `reports/reading_analysis.csv`), o que mostra
que os erros **não são independentes**: concentram-se em algumas imagens difíceis ou mal recortadas. Na ResNet, das 29
imagens de teste com erro, 5 têm 4 ou mais dígitos errados (3 de polígono só com inteiros, 1 de orientação errada e 1 de outro desalinhamento), 7 têm 2–3 e
17 têm um único dígito errado. Excluindo só as 4 imagens de teste com polígono "só inteiros" (critério visual, que não
depende do modelo), `acc_digit` da ResNet sobe de 0,961 para 0,971 e `acc_reading` de 0,845 para 0,863. Excluindo
também as 2 imagens que a auditoria com a própria ResNet aponta (orientação e outro desalinhamento), sobem para 0,979
e 0,873. Essa segunda versão é otimista, porque usa os erros do modelo para escolher o que excluir
(`reports/metrics_without_bad_crops.csv`).

**5.4** A **resnet18**: tem o melhor desempenho (F1 0,954, `acc_reading` 0,845) com 11,2 M parâmetros e um
checkpoint de 44,8 MB (estimamos ~11 MB se quantizado em int8; não medimos). Na CPU leva 8,2 ms por dígito, cerca de 65 ms por leitura de 8
dígitos (estimativa com 8 chamadas de batch 1; com batch 8 deve ser menos, mas não medimos). A vgg16_bn é pior e muito mais pesada: 537 MB e 50 ms/dígito na CPU (~400 ms por
leitura). A alexnet tem 228 MB e F1 menor (0,903). A smallcnn seria a mais leve (1,2 MB, 2 ms), mas só acerta 26% das
leituras, o que é inaceitável para o usuário. Ressalva: as latências foram medidas em desktop; num celular seriam
maiores. Em produção, valeria testar a ResNet quantizada/exportada (TFLite/ONNX).

**5.5** Conexões residuais, BatchNorm em toda a rede e *global average pooling* no lugar das FCs gigantes. Com isso, a
ResNet18 tem **12× menos parâmetros** (11,2 M contra 134,3 M), checkpoint 12× menor (45 contra 537 MB), treina 4,2×
mais rápido por época (ver ressalva abaixo) (7,2 s contra 29,9 s) e é 6× mais rápida na CPU (8,2 contra 49,9 ms). Mesmo assim, é **melhor**:
F1 0,954 contra 0,926 e `acc_reading` 0,845 contra 0,701. Os atalhos residuais facilitam o fluxo de gradiente, o que
ajuda a adaptar a rede inteira com poucas épocas: a ResNet chega a F1 de validação de 0,913 na época 3, contra 0,892 da
VGG. Parte dessa diferença pode vir do protocolo, que dá à VGG um LR de backbone 10× menor (1e-5 contra 1e-4). A
ResNet também termina com val_loss menor (0,680 contra 0,712). Na GPU com batch 1, a latência é parecida (2,7 contra
3,4 ms). Nossa hipótese, não medida, é que com uma única imagem pequena o tempo depende mais do número de camadas e
chamadas de kernel do que dos FLOPs. Ressalva sobre `s_per_epoch`: o augmentation roda na CPU, uma amostra por vez, e
sozinho leva ≈4,3 s por época (medido à parte). Isso achata a comparação entre smallcnn, AlexNet e ResNet, que ficam
todas em 6,3–7,2 s; só a VGG mostra um custo de GPU claramente maior.

**5.6** Não. A smallcnn ficou bem atrás: F1 0,737 contra ≥ 0,903, e `acc_reading` 0,257 contra ≥ 0,631. As curvas
indicam que ela **ainda estava aprendendo** quando o treino acabou, ou seja, *underfitting*, não overfitting:
- o F1 de validação subiu em todas as últimas épocas (0,664 → 0,672 → 0,728 → 0,733);
- a loss de treino terminou em 1,19, contra 0,58 da ResNet com o mesmo label smoothing;
- a acurácia de treino (0,764) ficou abaixo da de validação (0,791). Nos pré-treinados a relação se inverte. Esse
  sinal é mais fraco, porque a acurácia de treino é medida com augmentation e dropout ativos. Isso indica que,
com poucos dados (6960 recortes) e um orçamento curto (10 épocas, com aquecimento do OneCycle), o **conhecimento prévio
do ImageNet vale muito**. As redes pré-treinadas já chegam com detectores de bordas e formas: na época 3, já têm F1 de
validação de 0,83–0,91 (94–97% do melhor valor de cada uma).
Não é que uma CNN pequena não consiga ler dígitos (MNIST mostra o contrário). Ela precisa de mais épocas, de mais dados
ou de uma arquitetura mais adequada. Como tem 2,6% dos parâmetros da ResNet e leva 2 ms na CPU, valeria investigar um
treino mais longo, mas isso foge do protocolo único.

## Limitações

- **Ruído de rótulo herdado das anotações:** 33 imagens (2,7%) têm polígono só com a parte inteira e geram 264
  recortes desalinhados no treino/val/teste. Parte deles ainda coincide com o rótulo por acaso (ver 3.2). Mantivemos essas imagens porque as verificações obrigatórias
  (1243/9944) exigem; o impacto está quantificado em `reports/metrics_without_bad_crops.csv`.
- **Orientação:** a heurística (degrau → vermelho) não decide quando não há pista. Sobraram 12 faixas invertidas (1%),
  todas nesse grupo. Uma terceira pista, como um classificador de orientação auto-supervisionado ou o próprio modelo de
  dígitos, resolveria esses casos. Os limiares (5% e 0,05) foram escolhidos inspecionando faixas de todos os splits.
  A decisão nunca usa o rótulo, só a geometria do polígono e as cores da própria imagem, então não há vazamento de
  rótulos, mas é uma escolha de pré-processamento feita com o conjunto inteiro.
- **Divisão em larguras iguais e `minAreaRect`:** causam deslocamento nas pontas (pior acurácia na posição 7) e
  ignoram a perspectiva.
- **Distorção de aspecto:** o recorte 48×64 é esticado para 128×128 nos modelos pré-treinados.
- **Uma única seed e uma única divisão:** não estimamos variância entre execuções. Como ordem de grandeza, o erro-padrão
  é ≈0,5 p.p. em `acc_digit` (1496 recortes, que nem são independentes) e ≈2,6 p.p. em `acc_reading` (187 imagens).
  Assim, a diferença ResNet × VGG em `acc_reading` (14 p.p.) é robusta, e AlexNet × VGG (7 p.p.) é limítrofe.
- **Custo medido em desktop:** RTX 4070 SUPER e CPU de desktop. A latência num celular seria diferente.
  `s_per_epoch` é dominado pelo augmentation na CPU (≈4,3 s por época), então mede mal o custo de GPU dos modelos menores.
- **smallcnn limitada pelo orçamento** do protocolo (ver 5.6).

Estimativa de quanto do erro restante é dado e quanto é modelo: na ResNet, só as 4 imagens de teste com polígono
"só inteiros" (2% do teste, critério independente do modelo) respondem por 16 dos 59 dígitos errados (27%). Somando as
2 imagens apontadas pela auditoria, as 6 imagens respondem por 28 dos 59 (47%), mas esse limite superior é otimista,
porque a seleção usa os erros do próprio modelo. O resto é dominado por rodas em transição e baixa resolução, que são em parte ambiguidade do próprio rótulo. O que
mudaríamos: segmentação por perfil de projeção (desafio extra), filtro ou correção dos polígonos "só inteiros" e uma
terceira pista de orientação.

## Declaração de uso de IA

**Atividade:** Prática 2 — Classificação de Dígitos de Hidrômetros com CNNs · **Disciplina:** Visão Computacional — VC001 ·
**Data:** 15/10/2026

| Integrante | E-mail |
|---|---|
| Ivan Edward | iers@cesar.school |
| Elizabete Barbosa | eba@cesar.school |
| Davi Melo | dam3@cesar.school |

### Seção A — Uso de LLM / Assistente de IA

| Campo | Resposta |
|---|---|
| Utilizou IA? | Sim |
| Modelo(s) usado(s) | Claude Opus 5.5 (`claude-opus-5-5`), via Claude Code (extensão do VS Code), assistente da Anthropic com acesso a terminal e edição de arquivos, operando em modo agente. Commits e publicação no GitHub foram feitos pela equipe. |
| Motivação de uso | Apoio à implementação e à verificação do projeto e apoio ao estudo: tirar dúvidas técnicas, discutir alternativas de solução e de arquitetura e revisar a documentação, para entregar com mais qualidade técnica dentro do prazo, mantendo com a equipe as decisões e a análise. |
| Resumo do uso | A IA atuou como assistente de programação e de estudo. Sob direção da equipe, implementou o código do notebook (aquisição e integridade, retificação da faixa, correção de orientação, treino com o protocolo único, avaliação e figuras), além de um script local de checagem da entrega (não versionado). Também ajudou a investigar os dados: gerou pranchas de inspeção visual e testou e comparou heurísticas de orientação. Explicou conceitos (transfer learning, determinismo na GPU, métricas) e apontou armadilhas, como os artefatos de float no `data.csv` e os polígonos que cobrem só a parte inteira. Por fim, fez verificações independentes da entrega (requisitos do roteiro, números do README contra os artefatos e revisão do código). As decisões ficaram com a equipe: organização do repositório, escopo, interpretação dos pontos ambíguos do roteiro e aceitação ou não de cada sugestão. |
| Trecho/resultado incorporado à entrega | Código do `pratica_watermeters.ipynb`; figuras e relatórios gerados pelo notebook; texto do README, a partir do rascunho revisado pela equipe. |
| Adaptações/validações realizadas pela equipe | A equipe executou o notebook, conferiu os números do README contra `results.csv` e `reports/` (inclusive com um script local de checagem), revisou e reescreveu as respostas às questões, discutiu os pontos de dúvida levantados nas notas técnicas e conferiu que nenhuma imagem do dataset, recorte ou checkpoint foi versionado. |

### Seção B — Uso de Skill (Claude Code / SKILL.md)

| Campo | Resposta |
|---|---|
| Utilizou skill? | Sim, apenas uma skill embutida do Claude Code (`dataviz`, diretrizes de visualização), consultada para padronizar as figuras (paleta, marcadores e legibilidade). Nenhuma skill de projeto (SKILL.md) customizada foi criada ou usada. |

### Seção C — Uso de SDD (System Design Document)

| Campo | Resposta |
|---|---|
| Utilizou SDD? | Não. |

**Declaração de responsabilidade.** Declaramos que as informações prestadas nesta declaração são verdadeiras e completas,
que compreendemos e somos responsáveis por todo o conteúdo entregue, incluindo as partes em que o assistente de IA
gerou ou adaptou código, e que estamos cientes de que a omissão do uso de IA configura conduta academicamente
desonesta, conforme a política da disciplina.

Ivan Edward · Elizabete Barbosa · Davi Melo
