# torch-foundry

PyTorchでOxford-IIIT Petのセマンティックセグメンテーションを学習するプロジェクトです。
データセットを読み込み、train / validation / testに分け、U-Netを学習し、IoUとDiceで評価します。

## 最初に理解する実行フロー

`main.py`を実行すると、次の順番で処理されます。

```text
config.jsonを読み込む
        ↓
Python / NumPy / PyTorchの乱数seedを固定
        ↓
Oxford-IIIT Petを読み込む
        ↓
trainval splitをtrain / validationに分割
test splitは学習に使わず、最後の評価専用にする
        ↓
画像とマスクを256×256へResize
trainだけRandomHorizontalFlipを適用
画像をfloat32 [0, 1]、マスクをクラスID 0 / 1 / 2へ変換
        ↓
train_loader / val_loader / test_loaderを作成
        ↓
U-Netを作成してdevice（CUDA / MPS / CPU）へ移動
        ↓
CrossEntropyLossとAdamを作成
        ↓
train_loaderで学習
val_loaderで各epochの検証とIoU / Dice計算
        ↓
test_loaderで最終評価
        ↓
model.pt / history.json / config.jsonを保存
```

実際の処理は次のファイルに分かれています。

| 処理 | ファイル |
| --- | --- |
| 設定の読み込み・値の検証 | `vision/segmentation/config.py`、`vision/segmentation/config.json` |
| データセット・DataLoader | `vision/segmentation/datasets/oxford_pet.py` |
| U-Net | `vision/segmentation/models/unet.py` |
| 学習ループ | `vision/segmentation/training/train.py` |
| test評価 | `vision/segmentation/training/evaluation.py` |
| IoU / Dice | `vision/segmentation/metrics/segmentation.py` |
| 実行の入口 | `main.py` |

## ローカルで実行する

### 1. 環境を準備する

このプロジェクトはPython 3.12以上を想定しています。`uv`を使う場合は、プロジェクトのルートで実行します。

```bash
uv sync
```

`uv`を使わない場合は、既存のPython環境に`pyproject.toml`の依存パッケージを用意してください。

### 2. 設定を確認する

設定は [vision/segmentation/config.json](vision/segmentation/config.json) にあります。

```json
{
  "data_root": "./data",
  "image_size": [256, 256],
  "batch_size": 32,
  "val_ratio": 0.2,
  "seed": 42,
  "num_epochs": 2,
  "learning_rate": 0.001,
  "num_classes": 3
}
```

| 設定 | 意味 | 変更例 |
| --- | --- | --- |
| `data_root` | データセットの親ディレクトリ | `"./data"`、`"/kaggle/input/oxford-iiit-pet"` |
| `image_size` | 入力画像とマスクの高さ・幅 | `[256, 256]` |
| `batch_size` | 1回の更新で使う画像枚数 | GPUメモリ不足なら`8`や`4` |
| `val_ratio` | trainvalからvalidationへ分ける割合 | `0.2` |
| `seed` | 分割・shuffle・モデル初期化などの乱数seed | `42` |
| `num_epochs` | 学習するepoch数 | 本番では`20`など |
| `learning_rate` | Adamの学習率 | `0.001` |
| `num_classes` | マスクのクラス数 | Oxford-IIIT Petでは`3` |

`image_size`はU-Netのskip connectionの都合で、縦横とも16の倍数にしてください。`[256, 256]`、`[128, 128]`は使用できますが、`[250, 250]`は使用できません。

`num_epochs: 2`は動作確認用の短い設定です。実際の学習では、lossとvalidation指標を確認しながら増やしてください。

### 3. データを用意する

デフォルトでは`data_root`が`./data`です。`torchvision.datasets.OxfordIIITPet`が、次のような構造でデータを読み込みます。

```text
data/
└── oxford-iiit-pet/
    ├── annotations/
    │   ├── trainval.txt
    │   ├── test.txt
    │   └── trimaps/
    └── images/
```

データが存在しない場合、`OxfordIIITPet(..., download=True)`によってダウンロードされます。ダウンロードを使わない場合は、`data_root`の下に`oxford-iiit-pet`が存在することを確認してください。

### 4. 学習を開始する

```bash
uv run python main.py
```

または、仮想環境のPythonを直接使います。

```bash
.venv/bin/python main.py
```

開始時に、次のような情報が表示されます。

```text
train 92 val 23 test 115
device: mps
total params: 31043651
output directory: .../vision/segmentation/runs/20260826_...
```

`device`が`cuda`ならNVIDIA GPU、`mps`ならApple Silicon GPU、`cpu`ならCPUで実行されています。

## データ分割とラベルの意味

`trainval` splitを`val_ratio`に従ってtrainとvalidationへ分けます。デフォルトでは、trainval 3,680件を次のように分けます。

```text
train      2,944件
validation   736件
test       3,669件
```

test splitは学習やvalidationの設定決定には使いません。学習が終わった後の最終評価だけに使います。

Oxford-IIIT Petの元マスクはクラスID 1 / 2 / 3ですが、モデルのCrossEntropyLossに合わせて、コード内では0 / 1 / 2へ変換しています。

モデルへの入力と出力は次の形です。

```text
image: [batch_size, 3, height, width]  float32
mask:  [batch_size, height, width]      int64
logits: [batch_size, num_classes, height, width]
```

## 学習結果の保存先

実行ごとに、次のディレクトリが作られます。

```text
vision/segmentation/runs/<実行日時>/
├── config.json   # その実行で実際に使った設定
├── history.json  # epochごとのloss / IoU / Dice
└── model.pt      # 最終epochのモデルstate_dict
```

`model.pt`は各epoch終了時に上書きされるため、最後に残るのは最終epochの重みです。ベストモデル、Optimizer、学習再開用の状態は現在保存していません。

保存したモデルを読み込む例です。

```python
import torch

from vision.segmentation.config import Config
from vision.segmentation.models.unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config.from_json("vision/segmentation/config.json")
model = UNet(3, config.num_classes).to(device)

state_dict = torch.load(
    "vision/segmentation/runs/<実行日時>/model.pt",
    map_location=device,
    weights_only=True,
)
model.load_state_dict(state_dict)
model.eval()
```

## Kaggleで使う場合

### 1. リポジトリを`/kaggle/working`へ置く

Kaggleの`/kaggle/input`は読み取り専用です。学習結果を書き込むため、リポジトリは`/kaggle/working`へコピーまたはcloneしてください。

例として、リポジトリをKaggle Datasetとして追加した場合は、Notebookで次のようにします。

```python
!cp -r /kaggle/input/<repository-dataset-name>/torch-foundry /kaggle/working/torch-foundry
%cd /kaggle/working/torch-foundry
```

### 2. KaggleのGPUを有効にする

Notebook右側のSettingsで、AcceleratorをGPUに設定します。コードは自動的にCUDAを選択します。実行ログに次が表示されればGPUが使われています。

```text
device: cuda
```

### 3. Kaggle Datasetのパスを設定する

Oxford-IIIT PetをKaggle Datasetとして追加し、データが次の構造なら、`config.json`を変更します。

```text
/kaggle/input/oxford-iiit-pet/
└── oxford-iiit-pet/
    ├── annotations/
    └── images/
```

```json
{
  "data_root": "/kaggle/input/oxford-iiit-pet",
  "image_size": [256, 256],
  "batch_size": 16,
  "val_ratio": 0.2,
  "seed": 42,
  "num_epochs": 20,
  "learning_rate": 0.001,
  "num_classes": 3
}
```

`data_root`は`oxford-iiit-pet`ディレクトリそのものではなく、その親ディレクトリを指定します。`annotations/trainval.txt`が`data_root/oxford-iiit-pet/annotations/trainval.txt`にあるか確認してください。

### 4. Kaggle Notebookから実行する

```python
%cd /kaggle/working/torch-foundry
!python main.py
```

成果物は次の場所に作られます。

```text
/kaggle/working/torch-foundry/vision/segmentation/runs/<実行日時>/
```

GPUメモリが足りない場合は、まず`batch_size`を`16`、`8`、`4`のように小さくしてください。次に`image_size`を`[128, 128]`へ下げる方法があります。ただし、画像サイズを下げると細かい境界の精度に影響します。

Kaggleで外部データをダウンロードする場合は、NotebookのInternet設定を有効にしてください。Kaggle Datasetを使う場合は、通常はダウンロードせずに`data_root`を指定する方法が安定します。

## よくあるエラー

### `Dataset not found`

`data_root`が正しいか、`data_root/oxford-iiit-pet/images`と`data_root/oxford-iiit-pet/annotations`が存在するか確認してください。

### `Sizes of tensors must match` がU-Netで発生する

`image_size`の縦または横が16の倍数ではありません。`[256, 256]`や`[128, 128]`に変更してください。

### `out of memory` が発生する

`batch_size`を下げてください。改善しない場合は`image_size`も下げてください。

### 同じseedなのにGPU上の結果が完全一致しない

seedはPython・NumPy・PyTorch・DataLoaderに設定していますが、GPUやMPSの一部演算は完全な決定論にならない場合があります。同じ環境、同じ設定、同じseedで比較してください。

## テスト

コード変更後は、プロジェクトルートから次を実行します。

```bash
.venv/bin/python -B -m unittest discover -s vision/segmentation/tests -v
.venv/bin/ruff check main.py vision/segmentation
.venv/bin/ty check vision/segmentation/config.py vision/segmentation/datasets/oxford_pet.py vision/segmentation/training/train.py
```
