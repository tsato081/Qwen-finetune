# 日本語犯罪情報抽出 - LoRA ファインチューニング

日本語のニュース・事件記事から犯罪者情報（名前、年齢、住所、所属、役職、違反法、犯行地、警察署）を構造化抽出するための LoRA ファインチューニング実装です。

**ベースモデル**: Rakuten/RakutenAI-7B-instruct (Mistral-7B ベース)
**フレームワーク**: Axolotl + DeepSpeed (ZeRO-2)
**トレーニング戦略**: 2フェーズ課程学習（負例キャリブレーション → 正例学習）

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

**2フェーズ課程学習**:

| フェーズ | データセット | 目的 | ステップ |
|---------|-------------|------|--------|
| **Phase 1** | person_dummy (100k) | 負例キャリブレーション | 1,000 |
| **Phase 2** | hawks (26k) + dummy (20% replay) | 正例学習 + FP抑制 | 5,000 |

---

## ディレクトリ構成

```
.
├── data/
│   ├── train/                          # 訓練データ
│   │   ├── hawks_val_segments.jsonl        # 評価用（4,633件）
│   │   ├── hawks_train_segments.jsonl      # Phase 2訓練（26,221件）
│   │   └── person_dummy_segments.jsonl     # Phase 1訓練＋Phase 2 replay（100,000件）
│   └── test/                           # 評価データ
│       ├── hawks_eval_input.jsonl          # 推論入力（1,064件）
│       └── hawks_eval_gold.csv             # 正解ラベル（1,064件）
│
├── src/
│   ├── axolotl_configs/
│   │   ├── rakuten_7b_phase1.yml           # Phase 1設定
│   │   └── rakuten_7b_phase2.yml           # Phase 2設定
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
│   │   └── checkpoint-final/
│   └── lora-out-phase2/                   # Phase 2チェックポイント（最終）
│       └── checkpoint-final/
│
├── mlruns/                             # MLflow追跡（.gitignoreに含む）
│
└── README.md                           # このファイル
```

---

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <repo-url>
cd Qwen-finetune
```

### 2. 依存パッケージのインストール

```bash
# uv推奨（速い）
uv sync

# または pip の場合
pip install -r requirements.txt

# Axolotlのインストール
pip install axolotl

# MLflow（訓練監視用）
pip install mlflow
```

### 3. HF トークン設定

`.env` ファイルを作成：

```bash
cat > .env << 'EOF'
HF_AUTH_TOKEN=hf_your_token_here
EOF
```

または環境変数で設定：

```bash
export HF_AUTH_TOKEN="hf_..."
```

---

## クイックスタート

### クラウド環境（推奨）

```bash
# 1. データセットをダウンロード
uv run download_datasets_from_hf.py

# 2. Phase 1訓練（約30-60分、A100 80GB）
axolotl train src/axolotl_configs/rakuten_7b_phase1.yml

# 3. Phase 2訓練（約1-2時間、A100 80GB）
axolotl train src/axolotl_configs/rakuten_7b_phase2.yml
```

### ローカル環境

データセットが既に `data/train/` と `data/test/` にある場合：

```bash
# Phase 1
axolotl train src/axolotl_configs/rakuten_7b_phase1.yml

# Phase 2
axolotl train src/axolotl_configs/rakuten_7b_phase2.yml
```

データセットが必要な場合：

```bash
# ローカルでダウンロード
uv run download_datasets_from_hf.py
```

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

## 訓練パイプライン

### Phase 1: 負例キャリブレーション

非犯罪記事のみで事前学習。モデルが過度に犯罪者を検出しないようにキャリブレーション。

```bash
axolotl train src/axolotl_configs/rakuten_7b_phase1.yml
```

**設定** (`rakuten_7b_phase1.yml`):

```yaml
base_model: Rakuten/RakutenAI-7B-instruct
datasets:
  - path: data/train/person_dummy_segments.jsonl
    type: input_output
    train_on_inputs: false

test_datasets:
  - path: data/train/hawks_val_segments.jsonl
    type: input_output

sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true

lora_r: 32
lora_alpha: 16
lora_dropout: 0
lora_target_linear: true
lora_target_modules:
  - gate_proj
  - down_proj
  - up_proj
  - q_proj
  - v_proj
  - k_proj
  - o_proj

micro_batch_size: 8
gradient_accumulation_steps: 2
max_steps: 1000
num_epochs: 1
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002

bf16: auto
gradient_checkpointing: true
flash_attention: true

output_dir: ./outputs/lora-out-phase1
```

**出力**: `./outputs/lora-out-phase1/checkpoint-final/`

### Phase 2: 正例学習 + リプレイ

実際の犯罪記事で訓練しながら、負例を20%の比率でリプレイしてFalse Positive を抑制。

```bash
axolotl train src/axolotl_configs/rakuten_7b_phase2.yml
```

**設定** (`rakuten_7b_phase2.yml`):

```yaml
base_model: Rakuten/RakutenAI-7B-instruct
datasets:
  - path: data/train/hawks_train_segments.jsonl
    type: input_output
    train_on_inputs: false
  - path: data/train/person_dummy_segments.jsonl
    type: input_output
    train_on_inputs: false
    weight: 0.2  # 20%リプレイ

test_datasets:
  - path: data/train/hawks_val_segments.jsonl
    type: input_output

resume_from_checkpoint: ./outputs/lora-out-phase1/checkpoint-final
max_steps: 5000
learning_rate: 0.0001

# その他は Phase 1と同じ
output_dir: ./outputs/lora-out-phase2
```

**出力**: `./outputs/lora-out-phase2/checkpoint-final/` ← **最終モデル**

---

## 訓練モニタリング

### MLflow で可視化

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

http://localhost:5000 でダッシュボードを確認可能。

記録される情報：
- Loss (train/val)
- Learning rate
- Gradient norm
- Generation metrics (Phase 1 & 2)

### ログ出力

訓練中のコンソール出力例：

```
Step [  100 / 1000] - Loss: 0.8234 | LR: 1.8e-4 | Time: 42.3s
Step [  200 / 1000] - Loss: 0.7156 | LR: 1.6e-4 | Time: 41.8s
...
Step [ 1000 / 1000] - Loss: 0.4432 | LR: 1.0e-5 | Time: 40.2s

✓ Phase 1 Training Complete
  Checkpoint saved to: ./outputs/lora-out-phase1/checkpoint-final/
```

---

## 推論（将来用）

Phase 2 checkpoint を使用した推論例：

```python
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# モデルロード
base_model_name = "Rakuten/RakutenAI-7B-instruct"
lora_path = "./outputs/lora-out-phase2/checkpoint-final"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_8bit=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, lora_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# プロンプト構築
SYSTEM_RULES = "あなたはニュース記事から犯罪者情報を抽出するアシスタント。出力はJSONのみ。推測禁止。犯罪者が明確でなければ[]。"

DETAILED_RULES = """ニュース報道から犯罪者の情報を構造化抽出し、以下のJSON形式で出力してください。

【基本ルール】
1. 「逮捕」「容疑者」「被告」など、明確な犯罪者のみ抽出対象
...（詳細ルール）"""

article = "東京のIT企業ABC社の山田太郎社長（45）が、詐欺容疑で逮捕された。"

prompt = f"{SYSTEM_RULES}\n\n{DETAILED_RULES}\n\n【記事】\n{article}"

# 推論
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,  # 確定的出力
        top_p=0.95
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

try:
    result = json.loads(response)
    print(json.dumps(result, indent=2, ensure_ascii=False))
except json.JSONDecodeError:
    print("Failed to parse JSON:", response)
```

---

## 評価（将来用）

### 評価ロジック

**8つの評価項目** (各項目独立採点):

| 項目 | 説明 |
|------|------|
| person_name | 人物のフルネーム |
| person_age_at_published | 逮捕時の年齢 |
| person_address | 住所（都道府県/市/区まで） |
| person_belongs_company | 所属会社 |
| person_position_name | 職位（社長、役員など） |
| person_violated_law | 違反法（詐欺、強盗など） |
| occured_locations | 犯行地点（複数可） |
| police_departments | 警察署名（複数可） |

### 採点ルール

- ✓ **正解**: 内容が完全一致（順番不問）
- ✗ **不正解**: 誤抽出、抽出漏れ、中途半端な抽出
- ∅ **空白正解**: 正解データ＆予測ともに空白（分母除外）

### 正解率計算

```
正解率 = 正解数 / (正解数 + 不正解数)
※空白正解は分母に含まない
```

### 複数値の処理

**区切り文字が異なる**:
- `person_name`, `person_address`, `person_belongs_company` → `/` で区切り
- `police_departments` → `,` で区切り

評価時に、これらを正しく分割して比較する。

**例**:
```
正解: "小林義治/古本辰則"
予測: "小林義治,古本辰則"
→ 不正解（区切り文字が異なる）
```

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
