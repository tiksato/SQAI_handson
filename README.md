# SQAI Quantum Chemistry Hands-on

SQAI「いまさら聞けない量子化学」の講義スライドとハンズオン教材です。

Google Colaboratory上で実行できるため、受講者のローカル環境にPythonやPySCFを事前インストールする必要はありません。

---

## 本日の講義資料

### 講義スライド：「いまさら聞けない量子化学」

[📖 講義スライドをGitHubで開く](./slides/SQAI_QuantumChemistry.pdf)

[⬇️ PDFを直接開く・保存する](https://raw.githubusercontent.com/tiksato/SQAI_handson/main/slides/SQAI_QuantumChemistry.pdf)

### ハンズオン教材

下の **Open in Colab** ボタンから、各NotebookをGoogle Colaboratoryで開けます。

---

## ハンズオンの開始方法

1. 使用するNotebookの **Open in Colab** ボタンをクリックします。
2. Google Colaboratoryが開いたら、Googleアカウントでログインします。
3. 必要に応じて、Notebookを自分のGoogle Driveへ保存します。
   - メニューから **ファイル → ドライブにコピーを保存**
4. Notebookの一番上にあるセットアップセルから順番に実行します。
5. セキュリティ警告が表示された場合は、内容を確認したうえで実行を許可してください。

> GitHub上の元のNotebookは直接変更されません。  
> 計算結果や変更内容を保存する場合は、最初にGoogle Driveへコピーしてください。

---

## Notebook一覧

### 0. 原子軌道・分子軌道の可視化

PySCFで分子を構築し、原子軌道（AO）とHartree–Fock分子軌道（MO）をpy3Dmolで対話的に表示します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiksato/SQAI_handson/blob/main/SQAI_notebooks/0_Orbital_View.ipynb)

---

### 1. H₂分子：PySCFによるHF・CCSD・Full CI

H₂分子について、原子間距離と基底関数を変えながら、Hartree–Fock、CCSD、Full CIのエネルギーを比較します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiksato/SQAI_handson/blob/main/SQAI_notebooks/1_H2_PySCF.ipynb)

---

### 2. H₆分子：PySCFによるHF・CCSD・Full CI

直線状H₆分子を対象に、Hartree–Fock、CCSD、Full CIのエネルギーを比較し、分子サイズの増加に伴う電子相関計算の難しさを確認します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiksato/SQAI_handson/blob/main/SQAI_notebooks/2_H6_PySCF.ipynb)

---

### 3. H₂分子：自作Direct Full CI

PySCFから分子軌道積分を取得し、自作のFull CI空間とDavidson対角化を用いてH₂分子の基底状態を計算します。CI係数とSlater行列式の対応も確認します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiksato/SQAI_handson/blob/main/SQAI_notebooks/3_H2_FCI.ipynb)

---

### 4. H₄分子：自作Direct Full CI

直線状H₄分子に自作Full CIコードを適用し、基底関数や原子間距離によるCI空間と電子状態の変化を調べます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiksato/SQAI_handson/blob/main/SQAI_notebooks/4_H4_FCI.ipynb)

---

### 5. H₆分子：自作Direct Full CIと組合せ爆発

直線状H₆分子に自作Full CIコードを適用し、軌道数の増加に伴うFull CI次元の急激な増大を確認します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiksato/SQAI_handson/blob/main/SQAI_notebooks/5_H6_FCI.ipynb)

---

### 6. H₆分子：Restricted Active Space CI

直線状H₆分子について、軌道空間をRAS1・RAS2・RAS3に分割し、RAS3への電子数を制限したRASCI計算を行います。Full CIの組合せ爆発を抑える考え方を学びます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiksato/SQAI_handson/blob/main/SQAI_notebooks/6_H6_RASCI.ipynb)

---

## 推奨する実行順序

初めて取り組む場合は、次の順序を推奨します。

1. `0_Orbital_View.ipynb`
2. `1_H2_PySCF.ipynb`
3. `2_H6_PySCF.ipynb`
4. `3_H2_FCI.ipynb`
5. `4_H4_FCI.ipynb`
6. `5_H6_FCI.ipynb`
7. `6_H6_RASCI.ipynb`

Notebook 0–2ではPySCFの標準機能を使い、Notebook 3–6では分子積分からCI空間を構築してDavidson対角化を行う流れを扱います。

---

## 使用する主なPythonパッケージ

- NumPy
- SciPy
- Numba
- PySCF
- py3Dmol

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

Notebookは `SQAI_notebooks/` ディレクトリにあります。

主な自作モジュールは、次のようにimportできます。

```python
from SQAI_modules.misc_utils.mo_viewer import MolecularOrbitalViewer
from SQAI_modules.misc_utils.matrix_utilities import davidson
from SQAI_modules.ormas_tools.rasci import Full_CI, RAS_CI, RHF_CI
```

---

## リポジトリ構成

```text
SQAI_handson/
├── pyproject.toml
├── README.md
├── slides/
│   └── SQAI_QuantumChemistry.pdf
├── SQAI_modules/
│   ├── misc_utils/
│   │   ├── mo_viewer.py
│   │   ├── pyscf_tools.py
│   │   └── matrix_utilities.py
│   └── ormas_tools/
│       └── rasci.py
└── SQAI_notebooks/
    ├── 0_Orbital_View.ipynb
    ├── 1_H2_PySCF.ipynb
    ├── 2_H6_PySCF.ipynb
    ├── 3_H2_FCI.ipynb
    ├── 4_H4_FCI.ipynb
    ├── 5_H6_FCI.ipynb
    └── 6_H6_RASCI.ipynb
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

## 利用について

本教材は教育・研究目的で公開しています。講義や教育資料として利用する場合は、本リポジトリを出典として示してください。
