Qwen Finetune

環境（uv）
- Python: 3.11+
- GPU: CUDA対応（UnslothはGPU必須）
- パッケージ管理: uv

セットアップ
1. uv をインストール（未導入の場合）: https://docs.astral.sh/uv/
2. 環境作成と依存関係のインストール:
   - `uv sync`

ファインチューニング
- スクリプト: `qwen_finetune.py`
- 出力: `outputs/stage1` → `outputs/stage2`
- 実行:
  - `uv run python3 qwen_finetune.py`
- 設定: `qwen_finetune.py` 冒頭の `CONFIG`

量子化（4bit, merged）
- スクリプト: `finetune_quantize.py`
- 入力: `outputs/stage2`
- 出力: `outputs/stage2_merged_4bit`
- 実行:
  - `uv run python3 finetune_quantize.py`
- 設定: `finetune_quantize.py` の `CONFIG`

推論 + 評価（CSV）
- スクリプト: `finetune_inference.py`
- モデル: `outputs/stage2_merged_4bit`
- テストCSV: `data/test/Hawks_正解データマスター - ver 5.0 csv出力用.csv`
- 出力:
  - 予測CSV: `outputs/test_predictions.csv`
  - メトリクス: `outputs/test_metrics.json`
- 実行:
  - `uv run python3 finetune_inference.py`
- 設定: `finetune_inference.py` の `InferenceConfig`
