# Backdoor acústico por data poisoning no AST (UrbanSound8K)

## Dependências

- Python 3.13+: `pip install -r requirements-local.txt`
- GPU para treinar (usado: Kaggle, Tesla T4, PyTorch 2.11, Transformers 5.20)
- CLI do Kaggle: `pip install kaggle` e `kaggle auth login`
- R 4.x com `ggplot2`, `jsonlite`, `dplyr`, `tidyr`, `patchwork`, `scales`

## Dados

UrbanSound8K:
- https://urbansounddataset.weebly.com/urbansound8k.html
- https://www.kaggle.com/datasets/chrisfilo/urbansound8k (anexado automaticamente nos kernels)

Local esperado: `~/Downloads/UrbanSound8K` (ou defina `TCC_DATA`). Os hashes SHA-256 dos áudios estão em
`results/wav_sha256.csv` e são conferidos no início de cada kernel.

## Execução

### 1. Pré-processamento e gatilho (CPU)

```
python scripts/s1_data.py
python scripts/s2_trigger.py --snr 40        # repetir para 30, 25, 20, 15, 10, 0
python scripts/s3_poison.py
python scripts/s4_input_delta.py --snr 40    # repetir para as demais SNRs
```

### 2. Treinamento (Kaggle)

```
python scripts/kaggle_push.py <job>
kaggle kernels output <usuário>/<slug> -p results/kaggle/<slug>
```

Jobs, nesta ordem (máximo de 2 em paralelo):

1. `varredura-a`, `varredura-b`: 5% de envenenamento, SNR de 40 a 0 dB, 3 sementes
2. `principal-a`, `principal-b`: 10 dB, taxas 0 / 0,5 / 1 / 2 / 5%, 5 sementes
3. `cv-a`, `cv-b`, `cv-c`: validação cruzada, folds 1–9
4. `deteccao-extra`: métricas do gatilho a 25 e 15 dB (sem GPU)

Para outra conta, altere `USER` em `scripts/kaggle_push.py`.

### 3. Análises (CPU)

```
python scripts/s5_main_analysis.py
python scripts/s6_fft_audit.py
python scripts/s7_bootstrap_origem.py
...
python scripts/s17_checkpoint_vitima.py
```

Saídas em `results/analysis/`. `s12` e `s13` precisam do UrbanSound8K local.

### 4. Figuras

```
R_LIBS_USER=~/R/library Rscript R/figuras.R
```

### 5. Testes

```
pytest tests/
```
