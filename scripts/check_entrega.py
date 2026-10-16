"""Checagem da entrega antes do commit.

Uso:
    python scripts/check_entrega.py           # só verifica
    python scripts/check_entrega.py --strip   # remove as saídas das células com imagens do dataset e verifica

Verifica:
  1. notebook executado em ordem, sem erros e com as células de tag `dataset-image` sem saída;
  2. nenhum arquivo proibido (imagens do dataset, recortes, checkpoints, zip, kaggle.json) fora das pastas ignoradas;
     se a pasta já for um repositório git, confere também o que está no índice (inclusive a pasta docs/, que não vai
     para o repositório);
  3. splits.csv idêntico ao oficial (SHA-256, tolerante a quebra de linha CRLF);
  4. presença dos arquivos obrigatórios e colunas exatas de results.csv e history_*.csv;
  5. tabela de resultados do README idêntica ao results.csv;
  6. README sem pendências (declaração de uso de IA preenchida, sem instruções "A preencher").
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'pratica_watermeters.ipynb'
SPLITS_SHA256 = 'ca4c5d8ad131d63eae3a8acd5bd6ac1091f55e02426d8ad083af98cfa65abf2b'
IGNORED_DIRS = {'data', 'digits', 'checkpoints', '.venv', 'venv', '.git', '__pycache__', '.ipynb_checkpoints'}
NOT_COMMITTED_DIRS = {'docs'}  # existe localmente, mas não vai para o repositório
FORBIDDEN_EXT = {'.jpg', '.jpeg', '.pt', '.pth', '.ckpt', '.zip'}
MODELS = ['smallcnn', 'alexnet', 'vgg16_bn', 'resnet18']
RESULTS_COLS = ['model', 'acc_digit', 'f1_macro', 'acc_reading', 'params_M', 's_per_epoch', 'ms_per_image',
                'ckpt_MB', 'best_epoch']
HISTORY_COLS = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1', 'seconds']
REQUIRED = ['README.md', 'requirements.txt', 'splits.csv', 'results.csv', '.gitignore', NOTEBOOK.name,
            *[f'history_{m}.csv' for m in MODELS],
            'figures/confusion_matrices.png', 'figures/accuracy_by_position.png',
            'figures/training_curves.png', 'figures/performance_vs_cost.png']


def strip_dataset_outputs():
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    n = 0
    for c in nb['cells']:
        if c['cell_type'] == 'code' and 'dataset-image' in c.get('metadata', {}).get('tags', []) and c.get('outputs'):
            c['outputs'] = []
            n += 1
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'[strip] saídas removidas de {n} células com imagens do dataset')


def check_notebook():
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    code = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
    tagged = lambda c: 'dataset-image' in c.get('metadata', {}).get('tags', [])
    dirty = [i for i, c in code if tagged(c) and c.get('outputs')]
    errors = [i for i, c in code if any(o.get('output_type') == 'error' for o in c.get('outputs', []))]
    unexecuted = [i for i, c in code if c.get('execution_count') is None]
    counts = [c['execution_count'] for _, c in code if c.get('execution_count') is not None]
    ok = True
    if dirty:
        print(f'[ERRO] células com imagens do dataset ainda têm saída: {dirty} (rode com --strip)'); ok = False
    if errors:
        print(f'[ERRO] células com erro de execução: {errors}'); ok = False
    if unexecuted:
        print(f'[ERRO] células não executadas: {unexecuted}'); ok = False
    if counts != sorted(counts) or len(set(counts)) != len(counts):
        print('[ERRO] células executadas fora de ordem (rode o notebook inteiro do início)'); ok = False
    if ok:
        print('[ok] notebook executado em ordem, sem erros e sem imagens do dataset nas saídas')
    return ok


def is_forbidden(rel):
    p = Path(rel)
    if p.suffix.lower() in FORBIDDEN_EXT or p.name == 'kaggle.json':
        return True
    return p.suffix.lower() == '.png' and p.parts[0] != 'figures'


def check_forbidden_files():
    bad = []
    for p in ROOT.rglob('*'):
        rel = p.relative_to(ROOT)
        if p.is_file() and not (IGNORED_DIRS | NOT_COMMITTED_DIRS) & set(rel.parts) and is_forbidden(rel):
            bad.append(rel)
    ok = True
    if bad:
        print('[ERRO] arquivos proibidos fora das pastas ignoradas:', *bad, sep='\n  '); ok = False
    if (ROOT / '.git').exists():
        tracked = subprocess.run(['git', 'ls-files', '--cached'], cwd=ROOT, capture_output=True, text=True,
                                 encoding='utf-8').stdout.splitlines()
        in_index_bad = [f for f in tracked if is_forbidden(f) or (IGNORED_DIRS | NOT_COMMITTED_DIRS) & set(Path(f).parts)]
        if in_index_bad:
            print('[ERRO] arquivos no índice do git que não devem ir para o repositório '
                  '(use "git rm --cached <arquivo>"):', *in_index_bad, sep='\n  '); ok = False
    if ok:
        extra = ' (índice do git conferido)' if (ROOT / '.git').exists() else ''
        print(f'[ok] nenhum arquivo do dataset, recorte, checkpoint ou docs/ a ser versionado{extra}')
    return ok


def check_splits():
    data = (ROOT / 'splits.csv').read_bytes().replace(b'\r\n', b'\n')
    if hashlib.sha256(data).hexdigest() != SPLITS_SHA256:
        print('[ERRO] splits.csv difere do oficial'); return False
    print('[ok] splits.csv idêntico ao oficial')
    return True


def check_required():
    missing = [f for f in REQUIRED if not (ROOT / f).exists()]
    if missing:
        print('[ERRO] arquivos obrigatórios ausentes:', missing); return False
    ok = True
    res = pd.read_csv(ROOT / 'results.csv')
    if list(res.columns) != RESULTS_COLS or sorted(res.model) != sorted(MODELS):
        print('[ERRO] results.csv com colunas ou modelos diferentes do roteiro'); ok = False
    for m in MODELS:
        if list(pd.read_csv(ROOT / f'history_{m}.csv').columns) != HISTORY_COLS:
            print(f'[ERRO] history_{m}.csv com colunas diferentes do roteiro'); ok = False
    if ok:
        print('[ok] arquivos obrigatórios presentes, com as colunas do roteiro')
    return ok


def check_readme_results():
    res = pd.read_csv(ROOT / 'results.csv', dtype=str).set_index('model')
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    section = re.search(r'^## Resultados.*?(?=^## |\Z)', readme, re.M | re.S)
    if not section:
        print('[ERRO] seção "## Resultados" não encontrada no README'); return False
    ok = True
    for model in MODELS:
        m = re.search(rf'^\|\s*`?{model}`?\s*\|(.+)\|\s*$', section.group(0), re.M)
        if not m:
            print(f'[ERRO] linha de {model} não encontrada na tabela de resultados do README'); ok = False
            continue
        cells = [c.strip() for c in m.group(1).split('|')]
        expected = list(res.loc[model].values)
        if cells != expected:
            print(f'[ERRO] {model}: README {cells} != results.csv {expected}'); ok = False
    if ok:
        print('[ok] tabela de resultados do README idêntica ao results.csv')
    return ok


def check_readme_pending():
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    ok = True
    ia = re.search(r'^## Declaração de uso de IA\s*(.*?)(?=^## |\Z)', readme, re.M | re.S)
    body = ia.group(1).strip() if ia else ''
    if not ia or not body or re.search(r'a preencher', body, re.I):
        print('[ERRO] declaração de uso de IA ausente ou não preenchida (−20% na nota)'); ok = False
    pending = [line.strip() for line in readme.splitlines()
               if re.search(r'a preencher|\[nome do grupo\]', line, re.I) and not line.startswith('## ')]
    if pending and ok:
        print('[ERRO] README ainda tem trechos pendentes:', *pending, sep='\n  '); ok = False
    if ok:
        print('[ok] README sem pendências')
    return ok


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--strip', action='store_true', help='remove saídas das células com imagens do dataset')
    args = ap.parse_args()
    if args.strip:
        strip_dataset_outputs()
    results = [check_notebook(), check_forbidden_files(), check_splits(), check_required(), check_readme_results(),
               check_readme_pending()]
    print('\nENTREGA OK' if all(results) else '\nHÁ PENDÊNCIAS')
    sys.exit(0 if all(results) else 1)
