# SQAI Quantum Chemistry Hands-on

SQAI「いまさら聞けない量子化学」のハンズオン教材です。

---

## ハンズオンの開始方法

1. 下の **Open in Colab** ボタンをクリックします。
2. Google Colaboratoryが開いたら、Googleアカウントでログインします。
3. 必要に応じて、Notebookを自分のGoogle Driveへ保存します。

   * メニューから **ファイル → ドライブにコピーを保存**
4. 一番上のセットアップセルから順番に実行します。
5. セキュリティ警告が表示された場合は、内容を確認したうえで実行を許可してください。

> GitHub上の元のNotebookは直接変更されません。
> 計算結果や変更内容を保存する場合は、最初にGoogle Driveへコピーしてください。

---

## Notebook一覧

### 0. 分子軌道の可視化

PySCFで計算した分子軌道を、py3Dmolを使って対話的に表示します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiksato/SQAI_handson/blob/main/SQAI_notebooks/0_Orbital_View.ipynb)

---

### 1. H₂分子：PySCF入門

H₂分子を対象に、Hartree–Fock法、CCSD、Full CIなどの電子状態計算を行います。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiksato/SQAI_handson/blob/main/SQAI_notebooks/1_H2_PySCF.ipynb)

---

### 2. H₆分子：PySCFによる電子状態計算

H₆分子を対象に、Hartree–Fock計算、分子軌道、分子積分などを確認します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiksato/SQAI_handson/blob/main/SQAI_notebooks/2_H6_PySCF.ipynb)

---

### 3. H₆分子：Full Configuration Interaction

H₆分子のFull CI計算を通して、配置間相互作用法の基本を学びます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiksato/SQAI_handson/blob/main/SQAI_notebooks/3_H6_FCI.ipynb)

---

### 4. H₆分子：Restricted Active Space CI

H₆分子を対象に、活性空間とRASCIの考え方を学びます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiksato/SQAI_handson/blob/main/SQAI_notebooks/4_H6_RASCI.ipynb)

---

## 使用する主なPythonパッケージ

* NumPy
* SciPy
* Numba
* PySCF
* py3Dmol

必要なパッケージと本リポジトリの自作モジュールは、各Notebookの先頭セルで自動的にインストールされます。

GPUは必要ありません。Google Colaboratoryの標準CPUランタイムを使用してください。

---

## ローカル環境での実行

ローカルのPython環境でも実行できます。

```bash
git clone https://github.com/tiksato/SQAI_handson.git
cd SQAI_handson
python -m pip install -e .
jupyter lab
```

Notebookは次のディレクトリにあります。

```text
SQAI_notebooks/
```

自作モジュールは次のようにimportできます。

```python
from SQAI_modules.misc_utils.mo_viewer import MolecularOrbitalViewer
from SQAI_modules.misc_utils.pyscf_tools import get_integrals_rhf
from SQAI_modules.misc_utils.matrix_utilities import davidson
```

---

## リポジトリ構成

```text
SQAI_handson/
├── pyproject.toml
├── README.md
├── SQAI_modules/
│   └── misc_utils/
│       ├── mo_viewer.py
│       ├── pyscf_tools.py
│       └── matrix_utilities.py
└── SQAI_notebooks/
    ├── 0_Orbital_View.ipynb
    ├── 1_H2_PySCF.ipynb
    ├── 2_H6_PySCF.ipynb
    ├── 3_H6_FCI.ipynb
    └── 4_H6_RASCI.ipynb
```

---

## トラブルシューティング

### セットアップセルでエラーが出る場合

Colabのメニューから、

```text
ランタイム → セッションを再起動
```

を選び、最初のセルから再実行してください。

### MO表示が重い場合

分子軌道表示クラスの `resolution` を大きくすると、cube gridが粗くなり、描画が軽くなります。

```python
viewer = MolecularOrbitalViewer(
    mol,
    mf.mo_coeff,
    orbital_indices=mo_indices,
    resolution=0.30,
    margin=2.0,
)
```

### 計算結果を保存したい場合

Colabのメニューから、

```text
ファイル → ドライブにコピーを保存
```

を選んでください。

---

## License

本教材は教育・研究目的で公開しています。

講義や教育資料として利用する場合は、本リポジトリを出典として示してください。
