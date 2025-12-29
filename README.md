# 日本語犯罪情報抽出 - LoRA ファインチューニング

日本語のニュース・事件記事から犯罪者情報（名前、年齢、住所、所属、役職、違反法、犯行地、警察署）を構造化抽出するための LoRA ファインチューニング実装です。

**ベースモデル**: openai/gpt-oss-20b
**フレームワーク**: Axolotl + DeepSpeed (ZeRO-2)
**トレーニング戦略**: 単一フェーズ（hawks + dummy の混合学習）

---

## 概要

### タスク定義

本プロジェクトは、2つの抽出タスクの統合です：

1. **人物抽出・分類** (`structure_extract_person_prompt.py` で定義)
   - 記事内の全人物を列挙
   - 実在性（is_real）、加害者性（is_offender）、属性を判定
   - 出力: `persons[]` 配列

2. **住所抽出** (`structure_extract_address_prompt.py` で定義)
   - 企業本社、人物居住地、犯行地点の住所を抽出
   - 出力: `occured_locations[]`, `residences[]`, `headquarters_locations[]`

3. **警察署名抽出** (別タスク)
   - 記事から警察署名を抽出
   - 出力: `police_departments[]`

### 訓練戦略

**単一フェーズ**:

- hawks（正例）+ person_dummy（空例）を混合して学習
- 目的: 空例の学習は必要最小限にし、加害者を立てない挙動を防ぐ

---

## ディレクトリ構成

```
.
├── data/
│   ├── train/                          # 訓練データ
│   │   ├── hawks_val_segments.jsonl        # 評価用（segments）
│   │   ├── hawks_train_segments.jsonl      # 訓練（segments）
│   │   ├── person_dummy_segments.jsonl     # 空例（segments）
│   │   ├── hawks_val_messages.jsonl        # 評価用（messages）
│   │   ├── hawks_train_messages.jsonl      # 訓練（messages）
│   │   └── person_dummy_messages.jsonl     # 空例（messages）
│   └── test/                           # 評価データ
│       ├── hawks_eval_input.jsonl          # 推論入力（1,064件）
│       ├── hawks_eval_gold.csv             # 正解ラベル（1,064件）
│       └── hawks_eval_messages.jsonl       # 評価用（messages）
│
├── src/
│   ├── axolotl_configs/
│   │   └── gpt-oss_20b.yml                 # 現行設定
│   └── deepspeed_configs/
│       └── zero2.json                      # DeepSpeed ZeRO-2設定
│
├── download_datasets_from_hf.py        # HFからデータセットをダウンロード
├── upload_to_hf.py                     # ローカルデータセットをHFにアップロード
├── evaluate_model.py                   # モデル評価スクリプト
├── regenerate_jsonl_segments.py        # 訓練データをsegments形式で再生成
│
├── outputs/                            # 訓練出力（.gitignoreに含む）
│   ├── lora-out-phase1/                   # Phase 1チェックポイント
│   │   └── checkpoint-*/
│   └── lora-out-phase2/                   # Phase 2チェックポイント（最終）
│       ├── checkpoint-*/
│       └── merged/                        # （作られる場合のみ）マージ済みフルモデル
│
├── mlruns/                             # MLflow追跡（.gitignoreに含む）
│
└── README.md                           # このファイル
```

---

## セットアップ

> **対象環境**: Ubuntu 22.04.5 LTS（クラウド環境 A100 80GB）
>
> 詳細は `docs/env_snapshot_20251226_085433.txt` を参照

### ステップ1: リポジトリをクローン

```bash
git clone <repo-url>
cd Qwen-finetune
```

### ステップ2: uvをインストール

```bash
# 公式インストーラ（推奨）
curl -LsSf https://astral.sh/uv/install.sh | sh

# または pip でインストール
pip install uv
```

### ステップ3: 依存パッケージをインストール

**ステップ 3-1: 基本パッケージをインストール**

```bash
# uv で依存パッケージをインストール（PyTorch など）
uv sync --frozen
```

このコマンドが以下を自動実行します:
- `pyproject.toml` から依存パッケージを読込
- CUDA 12.8用の PyTorch ホイール（cu128）を取得（torch/torchvision/torchaudio を cu128 で固定）
- HuggingFace Hub、MLflow などをインストール

> **重要**: `pip install torch/torchvision/torchaudio ...` のように手で入れ直すと、CUDA メジャー違い（cu128/cu130 など）が混ざって壊れやすいです。原則 `uv sync --frozen` で復旧・再現してください。

**ステップ 3-2: DeepSpeed をインストール（別途）**

本リポジトリの Axolotl 設定は `deepspeed: src/deepspeed_configs/zero2.json` を使うため、DeepSpeed を別途インストールします（ビルド/環境依存が強いので `uv sync` には含めません）。

```bash
# （Axolotl 公式手順に合わせて）ビルド系ツールを先に揃える
uv pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja

# DeepSpeed をインストール
uv pip install --no-build-isolation deepspeed
```

**ステップ 3-3: Flash-Attention をインストール（別途）**

PyTorch インストール完了後、flash-attn をプリコンパイル済みホイールでインストール（`uv sync` には含めません）：

```bash
# flash-attention のプリコンパイル済みホイール（CUDA 12.8対応）をインストール
# PyTorch をロード状態でインストール（ビルド分離を避ける）
uv pip install --no-build-isolation https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl
```

**ステップ 3-4: Axolotl をインストール（別途）**

Axolotl は依存関係が複雑（複数の torch バージョンに対応）なため、別途インストール：

```bash
# v0.13.0 (PyTorch 2.9 / CUDA 12.8 でも動作可)
uv pip install --no-build-isolation "git+https://github.com/axolotl-ai-cloud/axolotl@v0.13.0"

# Opt-out telemetry（任意）
export AXOLOTL_DO_NOT_TRACK=1
```

> **注**: DeepSpeed / flash-attn / Axolotl はビルド依存関係が複雑なため、PyTorch インストール後に別ステップでインストールしています

**正しい状態の確認（必須）**

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY

nvcc --version
```

期待される結果:
- `torch.__version__` が `2.9.0+cu128`
- `torch.version.cuda` が `12.8`
- `nvcc` が `12.8`

### ステップ4: HF トークンを設定

```bash
# オプション A: .env ファイルに記載
cat > .env << 'EOF'
HF_AUTH_TOKEN=hf_your_token_here
EOF

# オプション B: 環境変数として設定
export HF_AUTH_TOKEN="hf_..."
```

### ステップ5: tmux をインストール

訓練中に接続が切れてもプロセスが継続するよう、tmux をインストール：

```bash
sudo apt-get update && sudo apt-get install -y tmux
```

### ステップ6: セットアップの確認

```bash
# Python バージョン確認
uv run python --version          # Python 3.11.x であること

# CUDA 確認
nvidia-smi                # CUDA Version が 12.8 以上、Driver 580 以降が表示されること

# PyTorch インストール確認
uv run python -c "import torch, torchvision, torchaudio; print(torch.__version__, torch.version.cuda, torchvision.__version__, torchaudio.__version__)"

# Axolotl 確認
uv run python -c "import axolotl; print('✓ Axolotl ready')"

# HF トークン確認
echo $HF_AUTH_TOKEN       # "hf_" で始まる トークンが表示されること

# tmux 確認
tmux -V                   # tmux [version] が表示されること
```

全て ✓ なら準備完了！

---

## 現行トレーニング（gpt-oss-20b）

### ステップ1: データセットをダウンロード

```bash
uv run download_datasets_from_hf.py
```

`data/train/` と `data/test/` にデータセットが配置されます。

### ステップ2: 学習を実行（.sh は使わない）

`src/axolotl_configs/gpt-oss_20b.yml` をそのまま使います。
ログは必ずファイルに落とします。

```bash
mkdir -p logs
accelerate launch --num_processes 4 --multi_gpu \
  -m axolotl.cli.train src/axolotl_configs/gpt-oss_20b.yml \
  2>&1 | tee "logs/gpt-oss_20b_$(date +%Y%m%d_%H%M%S).log"
```

### ステップ3: モデルマージ（Ollama 用）

`merged/` が無い場合は明示的にマージします。

```bash
axolotl merge-lora src/axolotl_configs/gpt-oss_20b.yml \
  --lora-model-dir="./outputs/gpt-oss-out"
```

### ステップ4: HF Hub アップロード（任意）

```bash
uv run --no-sync ./upload_merged_model.py
```

---

### ログの確認

#### リアルタイムモニタリング（訓練中）

```bash
tail -f logs/gpt-oss_20b_*.log
```

#### 訓練完了後

```bash
ls -lh logs/
```

#### tmux セッションの管理

```bash
# セッション一覧確認
tmux list-sessions

# セッションアタッチ（訓練状況をリアルタイム確認）
tmux attach-session -t training

# セッションから デタッチ（Ctrl+B → D）
# Ctrl+B キー → D キー

# セッション終了
tmux kill-session -t training
```

#### MLflow でメトリクス可視化（訓練中）

##### 1. ローカル環境の場合

別ターミナルで：

```bash
mlflow ui --backend-store-uri ./mlruns
# ブラウザで http://localhost:5000 を開く
```

##### 2. クラウド環境の場合（SSH ポート転送）

**ステップ 1**: ローカル端末から、SSH ポート転送して接続：

```bash
# クラウドサーバーの IP アドレスまたはホスト名に置き換える
ssh -L 5000:localhost:5000 ubuntu@<cloud-ip-or-hostname>

# 例：
ssh -L 5000:localhost:5000 ubuntu@203.0.113.42
```

> **確認方法**: クラウドプロバイダーの管理画面でインスタンスの IP アドレスを確認するか、接続情報を確認してください

**ステップ 2**: クラウド側で MLflow UI を起動：

```bash
mlflow ui --backend-store-uri ./mlruns
```

**ステップ 3**: ローカルブラウザで以下にアクセス：

```
http://localhost:5000
```

損失、学習率、勾配ノルムなどをダッシュボードで確認可能

---

## データセット

### 訓練データ形式（segments JSONL）

4つのセグメントで構成：

```json
{
  "segments": [
    {
      "label": false,
      "text": "あなたはニュース記事から犯罪者情報を抽出するアシスタント。出力はJSONのみ。推測禁止。犯罪者が明確でなければ[]。"
    },
    {
      "label": false,
      "text": "ニュース報道から犯罪者の情報を構造化抽出し、以下のJSON形式で出力してください。\n【基本ルール】\n1. 「逮捕」「容疑者」「被告」など、明確な犯罪者のみ抽出対象..."
    },
    {
      "label": false,
      "text": "【記事】\n記事本文（メタデータ削除済み）"
    },
    {
      "label": true,
      "text": "{\"persons\": [{\"name\": \"...\", \"is_real\": true, \"is_offender\": true, \"age\": \"...\", \"position\": \"...\", \"affiliation\": \"...\", \"address\": \"...\", \"category\": \"法令違反\", \"act\": \"詐欺\"}], \"occured_locations\": [...], \"police_departments\": [...]}"
    }
  ]
}
```

**セグメント制御**:
- `label: false` → 前方パスのみ、loss計算から除外（tokens → -100）
- `label: true` → loss計算に含含（訓練対象）

**結果**: モデルは JSON 出力部分のみで学習

### 評価データ

**入力** (`hawks_eval_input.jsonl`):
```json
{"id": "LVP0359", "body": "記事本文..."}
```

**正解ラベル** (`hawks_eval_gold.csv`):
```csv
quality_test_id,person_name,person_age_at_published,person_address,person_belongs_company,person_position_name,person_violated_law,occured_locations,police_departments
LVP0359,,,,,,
LVP0372,,,,,,
```

**統計**:
- 総件数: 1,064 (pick="Pick" のレコード)
- 犯罪記事: 186 件（17%）
- 非犯罪記事: 878 件（83%）

---

## トラブルシューティング

### メモリ不足 (OOM)

A100 80GB でも OOM が発生する場合：

```yaml
# 以下を調整（rakuten_7b_phase*.yml）
micro_batch_size: 4  # 8 → 4
gradient_accumulation_steps: 4  # 2 → 4
sequence_len: 2048  # 4096 → 2048（トレードオフあり）
```

### HF トークン認証エラー

```bash
# トークン設定確認
echo $HF_AUTH_TOKEN

# 再ログイン
huggingface-cli login

# または .env を確認
cat .env
```

### データセット見つからない

```bash
# データセット再ダウンロード
rm -rf data/train data/test
uv run download_datasets_from_hf.py

# 検証
ls -lh data/train/ data/test/
```

### Axolotl エラー

```bash
# Axolotl 再インストール
pip install --upgrade --force-reinstall axolotl

# DeepSpeed 再インストール
pip install deepspeed
```

---

## 参考リンク

- **Axolotl ドキュメント**: https://docs.axolotl.ai/docs/
- **Axolotl Segments形式**: https://docs.axolotl.ai/docs/dataset-formats/template_free.html
- **HF Dataset**: https://huggingface.co/datasets/teru00801/person-finetune-dataset1
- **RakutenAI-7B**: https://huggingface.co/Rakuten/RakutenAI-7B-instruct
- **MLflow**: https://mlflow.org/

---

## ファイル説明

### スクリプト

| ファイル | 用途 |
|---------|------|
| `download_datasets_from_hf.py` | HF Hub からデータセットをダウンロード（訓練＋評価） |
| `upload_to_hf.py` | ローカルファイルを HF Hub にアップロード |
| `regenerate_jsonl_segments.py` | 訓練データを segments JSONL 形式で再生成 |

### 設定

| ファイル | 用途 |
|---------|------|
| `src/axolotl_configs/rakuten_7b_phase1.yml` | Phase 1（負例キャリブレーション）の Axolotl 設定 |
| `src/axolotl_configs/rakuten_7b_phase2.yml` | Phase 2（正例学習）の Axolotl 設定 |
| `deepspeed_configs/zero2.json` | DeepSpeed ZeRO-2 メモリ最適化設定 |

### データ

| ファイル | 件数 | 用途 |
|---------|------|------|
| `data/train/hawks_val_segments.jsonl` | 4,633 | 評価用データ |
| `data/train/hawks_train_segments.jsonl` | 26,221 | Phase 2 訓練データ |
| `data/train/person_dummy_segments.jsonl` | 100,000 | Phase 1 訓練 + Phase 2 リプレイ |
| `data/test/hawks_eval_input.jsonl` | 1,064 | 推論入力 |
| `data/test/hawks_eval_gold.csv` | 1,064 | 正解ラベル |

---

## 更新履歴

- **2025-12-27**:
  - Rakuten-7B ベース実装に変更
  - Segments JSONL 形式での訓練統一
  - Phase 1/Phase 2 訓練フロー確立
  - 評価データセット生成・HF アップロード完了
  - README 完全更新

- **前期**:
  - Qwen3-30B ベース実装
  - 複数形式のデータセット試験

---

## Appendix: 将来用機能

### 推論（将来実装）

マージ版モデルを使用した推論は、標準的な HuggingFace API で対応可能：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("teru00801/rakuten-7b-instruct-person")
tokenizer = AutoTokenizer.from_pretrained("teru00801/rakuten-7b-instruct-person")
```

詳細なプロンプト例や実装については、CLAUDE.md の「タスク概要」を参照。

### 評価（将来実装）

8つの評価項目 (`person_name`, `person_age_at_published`, `person_address`, `person_belongs_company`, `person_position_name`, `person_violated_law`, `occured_locations`, `police_departments`) に対して、正解データとの一致度を採点。詳細は CLAUDE.md の「CSV正解データ（Hawks_...csv）評価への最小マッピング」を参照。

### Phase 1/Phase 2 設定詳細

- `src/axolotl_configs/rakuten_7b_phase1.yml` - Phase 1（負例キャリブレーション）の完全設定
- `src/axolotl_configs/rakuten_7b_phase2.yml` - Phase 2（正例学習）の完全設定
- 詳細は git 管理されるファイルを直接確認
