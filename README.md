# torch-foundry

Kaggle Notebookで画像セグメンテーションを学ぶための、小さなPyTorchサンプルです。
`main.py`を実行すると、Oxford iSegのデータ取得、Dataset / DataLoader、U-Net、損失、IoU・Dice、短い学習と評価までを一度に確認できます。

## 実行フロー

```text
config.jsonを読み込む
        ↓
乱数seedを固定
        ↓
Oxford iSeg（151組）をdata/oxford-isegへ取得
        ↓
マスクの0 / 128 / 255を0 / 1へ変換
        ↓
先頭20組をtrain / validationへ分割
        ↓
画像とマスクを64×64へResize
trainだけRandomHorizontalFlipを適用
        ↓
U-Netの出力形状、損失、IoU・Diceを確認
        ↓
1 epoch学習、validationを使った評価
        ↓
model.pt / history.json / config.jsonとW&Bログを保存
```

主な処理は次のファイルに分かれています。

| 処理 | ファイル |
| --- | --- |
| 設定の読み込み・値の検証 | `vision/segmentation/config.py`、`vision/segmentation/config.json` |
| Oxford iSegの取得 | `vision/segmentation/datasets/oxford_iseg.py` |
| Dataset、DataLoader | `vision/segmentation/datasets/dataset.py` |
| Oxford-IIIT PetのDataset例 | `vision/segmentation/datasets/oxford_pet.py` |
| U-Net | `vision/segmentation/models/unet.py` |
| 学習ループ | `vision/segmentation/training/train.py` |
| 評価 | `vision/segmentation/training/evaluation.py` |
| IoU / Dice | `vision/segmentation/metrics/segmentation.py` |
| 実行の入口 | `main.py` |
| Kaggleでの一連の実行例 | `notebooks/segmentation_wandb_example.ipynb` |

## 実行する

Python 3.12以上を用意し、プロジェクトのルートで依存パッケージをインストールします。

```bash
uv sync
```

W&B（Weights & Biases、実験ログ管理サービス）へ初めて保存する環境では、先に認証します。

```bash
uv run wandb login
```

その後、保存先のW&Bプロジェクト名を指定して実行します。プロジェクト名に固定の既定値はありません。

```bash
uv run python main.py \
  --wandb-project my-segmentation-project \
  --wandb-run-name baseline
```

`--wandb-run-name`は任意です。省略すると、W&Bが実行名を生成します。既存の仮想環境を使う場合は次のように実行します。

```bash
.venv/bin/python main.py --wandb-project my-segmentation-project
```

初回はOxford VGGの公式ページから`images.tgz`（画像）と`images-gt.tgz`（Ground Truth）をダウンロードします。2回目以降は`data/oxford-iseg`のデータを再利用します。Kaggle Notebookから外部ダウンロードする場合は、NotebookのInternet設定を有効にしてください。

公式ページ: <https://www.robots.ox.ac.uk/~vgg/data/iseg/>

## データの構造とマスク

`prepare_oxford_iseg`は、`root`の下に`oxford-iseg`ディレクトリを作ります。デフォルトでは次の構造です。

```text
data/
└── oxford-iseg/
    ├── images/       # 公式画像（jpg / bmpなど）
    ├── images-gt/    # 公式Ground Truth（0 / 128 / 255）
    └── masks/        # Dataset用に変換したPNG（0 / 1）
```

画像とマスクは拡張子を除いたstemで対応付けます。151組すべてが揃っていること、画像とマスクのサイズが一致することを確認してから返します。不一致がある場合は例外で知らせます。

公式マスクの値は次のように変換します。

```text
0         → 0（背景）
128 / 255 → 1（物体）
```

返されるパスは`images`と変換済み`masks`のリストです。そのまま`SegmentDataset(images, masks)`へ渡せます。

```python
from vision.segmentation.datasets.dataset import SegmentDataset
from vision.segmentation.datasets.oxford_iseg import prepare_oxford_iseg

images, masks = prepare_oxford_iseg("./data")
dataset = SegmentDataset(images[:20], masks[:20], image_size=(64, 64))
```

`data/`と学習結果の`vision/segmentation/runs/`はGit管理対象外です。

## 設定

設定は [vision/segmentation/config.json](vision/segmentation/config.json) にあります。デフォルト値は短い動作確認向けです。

```json
{
  "data_root": "./data",
  "image_size": [64, 64],
  "batch_size": 4,
  "val_ratio": 0.2,
  "seed": 42,
  "num_epochs": 1,
  "learning_rate": 0.001,
  "num_classes": 2,
  "wandb_enabled": true
}
```

| 設定 | 意味 |
| --- | --- |
| `data_root` | Oxford iSegを置く親ディレクトリ |
| `image_size` | Datasetへ入力する高さ・幅。U-Netの都合で縦横とも16の倍数 |
| `batch_size` | 1回の更新で使う画像枚数 |
| `val_ratio` | 20組のうちvalidationへ分ける割合 |
| `seed` | 分割、DataLoader、モデル初期化に使う乱数seed |
| `num_epochs` | 学習epoch数 |
| `learning_rate` | Adamの学習率 |
| `num_classes` | 出力クラス数。Oxford iSegでは2 |
| `wandb_enabled` | W&Bへの保存を有効にするか。`false`ならプロジェクト名なしでローカル保存だけを実行 |

本格的に学習する場合は、`main.py`の先頭20組の制限を外し、データ数・epoch数・画像サイズ・バッチサイズを目的に合わせて変更してください。このリポジトリのデフォルト実行は、関数の動作確認を目的とした短い学習です。

## 入出力の形

```text
image:  [batch_size, 3, height, width]  float32、値は0〜1
mask:   [batch_size, height, width]      int64、値は0 / 1
logits: [batch_size, num_classes, height, width]
```

train Datasetだけ水平反転を使い、validation Datasetでは使いません。

## 学習結果

実行ごとに次のディレクトリへ保存します。

```text
vision/segmentation/runs/<実行日時>/
├── config.json   # 実行時に使った設定
├── history.json  # epochごとのloss / IoU / Dice
└── model.pt      # 最終epochのstate_dict
```

`test_evaluation`は、今回の動作確認ではvalidation loaderを使って呼び出します。これは評価関数の動作確認であり、未知データに対する最終性能評価ではありません。

W&Bには、各epochの次の値を保存します。

- `train/loss`: 学習データの平均損失
- `val/loss`: validationデータの平均損失
- `val/mean_iou`、`val/mean_dice`: validationデータのクラス平均指標
- `val/iou/class_<クラスID>`、`val/dice/class_<クラスID>`: validationデータのクラス別指標
- `results/predictions`: 最大5件の画像、Ground Truth、予測、予測を重ねた画像

W&Bには実行時の設定、データセット名、モデル名、学習・validationのサンプル数、モデル総パラメーター数、使用deviceも保存します。モデルファイル自体はW&Bへアップロードせず、従来どおりローカルへ保存します。

## Kaggle Notebookで使う

一連のセルを含むサンプルは [notebooks/segmentation_wandb_example.ipynb](notebooks/segmentation_wandb_example.ipynb) にあります。このNotebookには、clone、依存パッケージの準備、Kaggle Secretsを使ったW&B認証、学習、指標と比較画像の保存までが含まれます。

手動で準備する場合は、リポジトリをNotebookの作業領域へcloneします。

```python
%cd /kaggle/working
!git clone https://github.com/Kaz0818/torch-foundry.git torch-foundry
%cd /kaggle/working/torch-foundry
!pip install -e .
```

KaggleのAdd-ons > Secretsへ、W&BのAPIキーを`WANDB_API_KEY`という名前で登録してください。NotebookではW&Bプロジェクト名を`WANDB_PROJECT`へ明示的に設定します。

KaggleのGPUを有効にすると、コードがCUDAを選択します。CPUやApple Siliconでは、利用可能なdeviceを自動で選びます。GPUメモリが足りない場合は、まず`batch_size`を下げ、必要なら`image_size`も下げてください。

## テストと静的チェック

プロジェクトルートから次を実行します。

```bash
.venv/bin/python -B -m unittest discover -s vision/segmentation/tests -v
.venv/bin/ruff check main.py vision/segmentation
.venv/bin/ty check main.py vision/segmentation/config.py vision/segmentation/datasets/dataset.py vision/segmentation/datasets/oxford_iseg.py vision/segmentation/datasets/oxford_pet.py vision/segmentation/training/train.py vision/segmentation/utils/visualization.py
```
