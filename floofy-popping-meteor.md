# Axolotl移行計画: Unsloth + SFTTrainer → Axolotl

## 概要

現在の `qwen_finetune.py` (Unsloth + SFTTrainerベース) をAxolotl（YAML設定駆動型フレームワーク）に移行します。

### 主な変更点
- **設定方法**: Pythonスクリプト → YAML設定ファイル
- **実行方法**: `torchrun qwen_finetune.py` → `accelerate launch -m axolotl.cli.train config.yml`
- **学習フロー**: 2ステージ (Stage1 → Stage2) → 単一ステージ（**Negative Sampling によるバランス学習**）
- **量子化**: 4bit → **なし（bf16フル精度）** ← **重要な変更**
- **カスタム機能**: GenerationEvalCallback + MLflowLoggerCallback を維持

### ユーザー要件
- ✅ 生成評価（GenerationEvalCallback）とMLflowロギングを維持
- ✅ 2ステージを単一ステージに統合（Negative Sampling で実現）
- ✅ 純粋なYAML設定を優先（axolotl CLIで実行）
- ✅ **QLoRA不要（A100 80GB × 4台でbf16フル精度）**

---

## ディレクトリ構成

```
/Users/terusato/Dev/Qwen-finetune/
├── axolotl_configs/
│   ├── qwen_finetune.yml              # メイン設定ファイル（NEW）
│   └── accelerate_config.yaml         # Accelerate設定（NEW）
├── deepspeed_configs/                  # DeepSpeed設定（NEW）
│   └── zero2.json                     # ZeRO-2設定ファイル（NEW）
├── callbacks/                          # カスタムコールバック（NEW）
│   ├── __init__.py
│   ├── generation_eval.py             # GenerationEvalCallback移植
│   └── mlflow_logger.py               # MLflowLoggerCallback移植
├── scripts/
│   ├── prepare_curriculum_data.py     # データマージスクリプト（NEW）
│   └── run_axolotl_train.sh           # 実行スクリプト（NEW）
├── data/
│   └── train/
│       ├── hawks_train.json           # 既存
│       ├── hawks_val.json             # 既存
│       ├── person_dummy_2025-12-23-1817.json  # 既存
│       └── hawks_train_curriculum.json # マージデータ（NEW）
├── outputs/
│   └── axolotl_qwen_finetune/         # Axolotl出力先（NEW）
└── qwen_finetune.py                    # 既存（バックアップ保持）
```

---

## 主要ファイルの詳細

### 1. Axolotl YAML設定ファイル

**パス**: `axolotl_configs/qwen_finetune.yml`

**主要セクション**:
```yaml
# モデル設定
base_model: unsloth/Qwen3-30B-A3B-Instruct-2507
model_type: qwen3
trust_remote_code: true

# 精度設定（bf16フル精度、量子化なし）
load_in_4bit: false
load_in_8bit: false
bf16: true
fp16: false
tf32: true  # A100でTensor Core最適化を有効化

# LoRA設定（16bit/bf16で学習）
# 重要: MoEモデルはExpert層（MLP）の学習が必須
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.0
# MoEモデル対応: Attention層 + Expert層（MLP）の全てを学習
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - up_proj      # Expert層のUp
  - down_proj    # Expert層のDown
  - gate_proj    # Gating層（重要）
lora_target_linear: true  # 線形層全体を対象に
lora_fan_in_fan_out: false

# データセット（Positive + Negative Sampling）
# Hawks: 正解あり（犯人がいるニュース）
# Dummy: 正解なし（犯人がいないニュース、Negative Sample）
# → 両方をシャッフルして学習することで、ハルシネーション防止
datasets:
  - path: data/train/hawks_train_curriculum.json
    type: custom_instruction
    format: instruction
    field_instruction: instruction
    field_input: input
    field_output: output

# チャットテンプレート
chat_template: qwen3_instruct
train_on_inputs: false

# シーケンス長
sequence_len: 1536
sample_packing: false

# 学習パラメータ
num_epochs: 1.5
max_steps: -1  # epochsで制御
micro_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 0.0002
lr_scheduler: cosine
cosine_min_lr_ratio: 0.05  # 最小LR = 1e-5
warmup_ratio: 0.03
optimizer: adamw_torch  # 通常のAdamW（量子化版ではない）
weight_decay: 0.0
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 1e-8
max_grad_norm: 1.0

# 評価設定
datasets_eval:
  - path: data/train/hawks_val.json
evaluation_strategy: steps
eval_steps: 200
eval_batch_size: 1

# チェックポイント
output_dir: ./outputs/axolotl_qwen_finetune
save_strategy: steps
save_steps: 200
save_total_limit: 3

# 最適化
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
flash_attention: true  # FA2でネイティブ高速化
unsloth: false  # BF16マルチGPU環境での安定性のため無効化
torch_compile: false

# DeepSpeed（ZeRO-2でOptimizer State シャード化）
deepspeed: deepspeed_configs/zero2.json

# カスタムコールバック（プラグインシステム）
plugins:
  - callbacks.generation_eval:GenerationEvalCallback
  - callbacks.mlflow_logger:MLflowLoggerCallback
```

**Negative Sampling による学習**:
- Positive (Hawks ×3, 78K) と Negative (Dummy, 96K) をマージ
- Axolotl がデフォルトでシャッフルして学習
- 各ステップで「抽出すべき」と「無視すべき」を混在させて学習
- 学習率スケジュール: cosine (2e-4 → 1e-5) で段階的に減衰

**VRAMメモリ見積もり（A100 80GB × 4台、DeepSpeed ZeRO-2適用）**:

**GPU単体当たり（GPU 0の例）**:
- モデル重み (bf16, シャード化): ~15GB
- LoRA追加パラメータ (r=16): ~0.5-1GB
- Optimizer states (ZeRO-2でシャード化): ~30GB
  - 本来120GB ÷ 4GPUs ≈ 30GB/GPU
- Gradients (シャード化): ~15GB
- Activations (micro_batch=1): ~15-20GB
- **合計/GPU**: ~70-75GB

**全GPU合計**:
- **~280-300GB → A100 80GB × 4 = 320GB で十分な余裕**

**DeepSpeed ZeRO-2の効果**:
- Optimizer States を全GPU間でシャード化
- Gradient も全GPU間でシャード化
- メモリが逼迫しず、安定した学習が可能

**bf16の利点**:
- A100のTensor Coreでネイティブサポート（高速）
- 4bit量子化より精度が高い
- 数値安定性が高い

---

### 2. データマージスクリプト

**パス**: `scripts/prepare_curriculum_data.py`

**目的**: Positive（Hawks）と Negative（Dummy）のサンプルをマージし、バランスの良いデータセットを作成

**処理フロー**:
1. `hawks_train.json` (26,221件) を3回繰り返し → 78,663件（Positive）
2. `person_dummy_2025-12-23-1817.json` から先頭96,000件を取得（Negative）
3. 順番に結合して `hawks_train_curriculum.json` (174,663件) を生成

**理由**:
- **Hawks (×3)**: 78,663件（正解ラベルあり、犯人がいるニュース）
- **Dummy**: 96,000件（正解なし、犯人がいないニュース）
- **比率**: 約 1:1 のバランス（Negative Sampling の最大効果）
- **Axolotl のシャッフル**: データセットをランダムに混ぜて学習
  - モデルが「ある時は抽出、ある時は無視」という判断を学習
  - **ハルシネーション防止**: いない人を犯人にしてしまう現象が劇的に減少

---

### 3. カスタムコールバック

#### 3.1 GenerationEvalCallback

**パス**: `callbacks/generation_eval.py`

**機能**:
- 評価時にJSON生成を実行
- JSONをパースしてCSV形式フィールドに変換
- F1スコア、パース率、フィールド一致率等を計算
- メトリクスをファイル出力 (`gen_metrics_step{N}.json`)

**主要メトリクス**:
- `gen_offender_presence_f1`: 加害者有無のF1スコア
- `gen_offender_name_set_f1`: 人名セットのF1スコア
- `gen_json_parse_rate`: JSON解析成功率
- `gen_all_fields_exact_match_rate`: 全フィールド完全一致率

**実装**: `qwen_finetune.py` の `GenerationEvalCallback` をそのまま移植

#### 3.2 MLflowLoggerCallback

**パス**: `callbacks/mlflow_logger.py`

**機能**:
- MLflow実験トラッキング
- ハイパーパラメータのログ記録
- メトリクスのステップごとの記録

**実装**: `qwen_finetune.py` の `MLflowLoggerCallback` をそのまま移植

---

### 4. 実行スクリプト

**パス**: `scripts/run_axolotl_train.sh`

**内容**:
```bash
#!/bin/bash
set -e

CONFIG_FILE="${1:-axolotl_configs/qwen_finetune.yml}"
NUM_GPUS="${2:-4}"

# データ準備
if [ ! -f "data/train/hawks_train_curriculum.json" ]; then
    python scripts/prepare_curriculum_data.py
fi

# 学習実行
accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=bf16 \
    -m axolotl.cli.train \
    "$CONFIG_FILE"
```

**使用法**:
```bash
chmod +x scripts/run_axolotl_train.sh
./scripts/run_axolotl_train.sh axolotl_configs/qwen_finetune.yml 4
```

---

## 実装手順

### ステップ1: 環境準備
```bash
# Axolotlインストール
pip install axolotl-ai

# Accelerate設定（初回のみ）
accelerate config
# → MULTI_GPU, 4 processes, bf16 を選択
```

### ステップ2: ディレクトリ作成
```bash
mkdir -p axolotl_configs callbacks scripts
touch callbacks/__init__.py
```

### ステップ3: ファイル作成
以下のファイルを作成:
1. `axolotl_configs/qwen_finetune.yml` - Axolotlメイン設定
2. `callbacks/generation_eval.py` - 生成評価コールバック
3. `callbacks/mlflow_logger.py` - MLflowロガー
4. `scripts/prepare_curriculum_data.py` - データマージスクリプト
5. `scripts/run_axolotl_train.sh` - 実行スクリプト

### ステップ4: データ準備
```bash
python scripts/prepare_curriculum_data.py
# → hawks_train_curriculum.json (174,663件) を生成
```

### ステップ5: テスト実行（小規模）
```bash
# YAML設定を一時的に変更
# max_steps: 50
# eval_steps: 25

# 1 GPUでテスト
accelerate launch --num_processes=1 \
    -m axolotl.cli.train \
    axolotl_configs/qwen_finetune.yml
```

### ステップ6: 本番実行（4 GPU）
```bash
# YAML設定を本番に戻す
# num_epochs: 1.5
# max_steps: -1

# 実行
./scripts/run_axolotl_train.sh
```

### ステップ7: 検証
- 学習速度（steps/sec）を比較
- 評価メトリクス（F1スコア等）を比較
- VRAMピーク使用量を確認

---

## 注意点と対策

### 1. Unslothの無効化（安定性重視）
- **YAML設定: `unsloth: false`**（本来はtrue推奨だが、BF16マルチGPU環境では不安定な報告あり）
- Axolotlのunsloth統合は予測不可能な挙動がある場合がある
- 代わりに Flash Attention 2 (`flash_attention: true`) でネイティブ高速化
- A100 + FA2 で十分なスピード（Unslothなし）
- **理由**: 不確定要素を排除し、マルチGPU学習の安定性を最優先

### 2. DeepSpeed ZeRO-2設定（OOM防止）
- **YAML設定: `deepspeed: deepspeed_configs/zero2.json`**
- 理由: BF16学習でのOptimizer States メモリ削減（120GB → 30GB/GPU）
- ZeRO-2の効果:
  - Optimizer States をGPU間でシャード化
  - Gradients もシャード化
  - メモリ効率が3-4倍改善
- **deepspeed_configs/zero2.json**: 以下の内容で作成
```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 0.0002,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0
    }
  }
}
```

### 3. LoRAターゲット設定（MoE対応）
- **重要**: Qwen3-30B-A3B はMoEモデル（Expert層が複数ある）
- **YAML設定: `lora_target_linear: true` + 明示的に全モジュール指定**
- MoE対応の必須要素:
  - `q_proj, k_proj, v_proj, o_proj` (Attention層)
  - `up_proj, down_proj, gate_proj` (Expert層)
- **理由**: Attention層のみではMoE性能が死ぬ。Expert層の学習が必須

### 4. カスタムコールバックの統合
- Axolotlの `plugins` セクションでコールバック指定
- プラグインシステムが動作しない場合の代替案:
  - カスタムトレーニングスクリプト (`axolotl_train_custom.py`) を作成
  - Axolotlのトレーナーを取得後、手動で `add_callback()` を実行

### 5. データフォーマット
- 現在のJSON形式は `type: custom_instruction` で対応
- フィールド名を明示的に指定: `field_instruction`, `field_input`, `field_output`

### 6. Chat Template
- `chat_template: qwen3_instruct` で自動適用
- カスタマイズが必要な場合、`tokenizer_config` で上書き可能

### 7. 学習率スケジュール
- Stage1 (2e-4) → Stage2 (1e-4) を cosine スケジュールで近似
- `cosine_min_lr_ratio: 0.05` で最小LR = 1e-5
- 必要に応じて調整可能

### 8. 評価データセット
- `datasets_eval` で hawks_val.json を指定
- GenerationEvalCallback は同じデータで生成評価を実行

---

## 期待される成果

### パフォーマンス
- **学習速度**: bf16はA100ネイティブ精度のため、4bit版より**高速化が期待**
- **メモリ効率**: A100 80GB × 4で余裕あり（合計320GB中~270-280GB使用）
- **収束性**: bf16は4bitより**精度が高い**ため、収束が安定・改善する可能性
- **精度向上**: 量子化による精度劣化がないため、最終モデル性能の向上を期待

### メンテナンス性
- **設定管理**: YAML一元管理、Git管理が容易
- **拡張性**: Axolotlの豊富な機能（DeepSpeed, FSDP等）を活用可能
- **再現性**: YAML + seed で完全再現可能

### 評価指標（目標値）
- `gen_offender_presence_f1`: 2ステージ版と同等以上
- `gen_json_parse_rate`: 95%以上
- `gen_all_fields_exact_match_rate`: 向上を期待

---

## バックアップと復旧

### バックアップ
```bash
# 既存スクリプトをバックアップ
cp qwen_finetune.py qwen_finetune_legacy.py
```

### 復旧（問題発生時）
```bash
# Unsloth版に戻す
torchrun --nproc_per_node=4 qwen_finetune_legacy.py
```

---

## 実装優先順位

### 必須（Phase 1）
1. ✅ `axolotl_configs/qwen_finetune.yml` - メイン設定
2. ✅ `scripts/prepare_curriculum_data.py` - データマージ
3. ✅ `scripts/run_axolotl_train.sh` - 実行スクリプト

### 重要（Phase 2）
4. ✅ `callbacks/generation_eval.py` - 生成評価コールバック
5. ✅ `callbacks/mlflow_logger.py` - MLflowロガー

### オプション（Phase 3）
6. ⚪ `axolotl_train_custom.py` - カスタムトレーニングスクリプト（プラグインが動かない場合の代替）

---

## 成功基準

1. **学習の完了**: 4 GPU で問題なく学習が完了
2. **評価メトリクス**: F1スコアが既存実装と同等以上
3. **パフォーマンス**: 学習速度がUnsloth版の80%以上
4. **再現性**: 同じYAML + seedで同じ結果を再現可能

---

## Critical Files

以下のファイルを作成/編集する必要があります:

### 最優先（これらがないと学習できない）
1. **axolotl_configs/qwen_finetune.yml** - Axolotlメイン設定（NEW）
2. **deepspeed_configs/zero2.json** - DeepSpeed ZeRO-2設定（NEW、OOM防止）
3. **scripts/prepare_curriculum_data.py** - データマージスクリプト（NEW）

### 次優先（評価・ロギング）
4. **callbacks/generation_eval.py** - 生成評価コールバック（NEW）
5. **callbacks/mlflow_logger.py** - MLflowロガー（NEW）
6. **scripts/run_axolotl_train.sh** - 実行スクリプト（NEW）

**既存ファイル（参照のみ）**:
- `qwen_finetune.py` - 既存実装（コールバックロジックを移植元として参照）
- `data/train/hawks_train.json`, `data/train/person_dummy_2025-12-23-1817.json` - 既存データ

---

## まとめ: 4つの重要ポイント

### 【重要1】Negative Sampling による学習
**戦略**: Hawks（正解あり）+ Dummy（正解なし）をシャッフル混合学習
**効果**:
- モデルが「ある時は抽出、ある時は無視」という判断を習得
- **ハルシネーション防止**: いない人を犯人にしてしまう現象が劇的に減少
- **比率バランス**: Hawks ×3 (78K) : Dummy (96K) ≈ 1:1 で最大効果
- **重要**: 順序を固定しない（Axolotl のシャッフル機能に任せる）

### 【重要2】MoEモデル対応: LoRAターゲットの明示
**問題**: Attention層のみでは Expert層が学習されない → MoE性能が死ぬ
**修正内容**:
```yaml
lora_target_linear: true
lora_target_modules:
  - q_proj, k_proj, v_proj, o_proj  # Attention
  - up_proj, down_proj, gate_proj    # Expert層（必須）
```
**理由**: Qwen3-30B-A3B の強みは Expert の多様性。Attention だけ学習では意味不明

### 【重要3】DeepSpeed ZeRO-2: OOM防止とメモリ効率化
**問題**: BF16学習でOptimizer States が120GB → GPU単体ではギリギリ
**修正内容**:
```yaml
deepspeed: deepspeed_configs/zero2.json
```
**効果**:
- Optimizer States が 120GB → 30GB/GPU （4分の1）
- Gradients も分散シャード化
- スパイク時の OOM を完全防止
- メモリ効率が 3-4倍改善

### 【重要4】Unsloth無効化: 安定性重視
**問題**: Axolotl の Unsloth統合は BF16 マルチGPU環境で不安定
**修正内容**:
```yaml
unsloth: false        # Unsloth無効化
flash_attention: true # FA2でネイティブ高速化
```
**理由**: 不確定要素排除。A100 + FA2 で十分高速（Unsloth の必要性なし）

### A100 80GB × 4台の環境を最大活用
1. **量子化不要**: QLoRA (4bit) → LoRA (bf16) でフル精度学習
2. **高速化**: A100のTensor Coreがbf16をネイティブサポート → 4bitより高速
3. **高精度**: 量子化による精度劣化なし → モデル性能向上が期待できる
4. **MoE対応**: Expert層を学習対象に → MoE本来の性能を発揮
5. **ハルシネーション防止**: Negative Sampling で「判断」を習得
6. **メモリ安全**: DeepSpeed ZeRO-2で安定運用 → OOM ゼロ

### 期待される改善
- **F1スコア**: 量子化なしで精度向上 + MoE本来性能 + Negative Sampling
- **学習速度**: A100ネイティブ精度 + FA2 で高速化
- **安定性**: DeepSpeed + FA2 による確実な学習
- **頑健性**: Expert 多様性 + Positive-Negative バランス学習
