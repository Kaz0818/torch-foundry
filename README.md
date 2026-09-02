# torch-foundry

Kaggle Hub の Flood Area Segmentation データセットを使った、二値画像セグメンテーションの最小構成です。
以前の実装は [`archive/segmentation_legacy/`](archive/segmentation_legacy/) に保存しています。

## セットアップ

```bash
uv sync
```

W&Bへログを保存する場合は、先に認証します。

```bash
uv run wandb login
```

認証せずに動作確認する場合は、W&Bを無効にできます。

```bash
WANDB_MODE=disabled uv run python main.py --data-dir data/flood-segmentation --epochs 1
```

## データセット

Notebookでは次のコードでKaggle Hubから取得します。

```python
import kagglehub
from pathlib import Path

data_dir = Path(
    kagglehub.dataset_download(
        "faizalkarim/flood-area-segmentation",
        output_dir="data/flood-segmentation",
    )
)
```

`data_dir`は、`metadata.csv`、`Image/`、`Mask/`を含むディレクトリです。取得済みのデータを使う場合は、同じ構造のディレクトリを`--data-dir`へ指定してください。

## 学習

```bash
uv run python main.py \
  --data-dir data/flood-segmentation \
  --epochs 10 \
  --batch-size 8
```

学習では`segmentation_models_pytorch.Unet`（ResNet-34、ImageNet重み、1チャンネル出力）を使い、BCEWithLogitsLossとDice Lossを合算します。画像は256×256へリサイズし、データは70:15:15で学習・検証・テストへ分割します。検証Diceが更新された時点の重みを`best_model.pt`へ保存します。

再利用可能なモデル、指標、学習処理は [`vision/segmentation/`](vision/segmentation/) に、洪水データセット固有の読み込み処理は [`vision/segmentation/datasets/flood.py`](vision/segmentation/datasets/flood.py) に分けています。別の二値データセットを使う場合は、同じく画像とマスクを返すDatasetアダプターを追加してください。

## Notebook

[`notebooks/flood_seg.ipynb`](notebooks/flood_seg.ipynb)には、データ取得、`.py`を使った学習、検証画像の4列表示（Image / GT / Predict / Overlay）をまとめています。
