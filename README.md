# Qwen Finetune

日本語ニュース記事から「人物」と「住所」を構造化抽出するためのLoRA ファインチューニング実装。

Axolotlフレームワークを使用した、YAML駆動型の学習パイプライン。

## プロジェクト概要

本プロジェクトは、Qwen3-30B-A3Bモデルを以下の学習手法で最適化します：

- **フレームワーク**: Axolotl（YAML設定駆動）
- **精度**: bf16フル精度（A100ネイティブ最適化）
- **LoRA**: r=16, MoE対応（Expert層を含む）
- **学習戦略**: Negative Sampling（Positive + Negative混合学習）
- **並列化**: DeepSpeed ZeRO-2（メモリ効率化）
- **評価**: 生成メトリクス（F1スコア等）+ MLflowロギング

---

## ディレクトリ構成

```
Qwen-finetune/
├── src/
│   ├── axolotl_configs/           # Axolotl設定ファイル
│   │   └── qwen_finetune.yml      # メイン学習設定
│   ├── deepspeed_configs/         # DeepSpeed最適化設定
│   │   └── zero2.json             # ZeRO-2メモリ最適化
│   └── callbacks/
│       ├── __init__.py            # モジュール初期化
│       ├── generation_eval.py     # 生成評価コールバック
│       └── mlflow_logger.py       # MLflowロギング
├── scripts/
│   ├── prepare_curriculum_data.py # データマージ（Positive + Negative）
│   └── run_axolotl_train.sh       # 学習ラッパースクリプト
├── data/
│   └── train/
│       ├── hawks_train.json       # 正解ラベル（26,221サンプル）
│       ├── hawks_val.json         # 評価用データ（27,798サンプル）
│       ├── person_dummy_2025-12-23-1817.json  # ネガティブサンプル
│       └── hawks_train_curriculum.json        # マージ済みデータ（自動生成）
├── outputs/                       # 学習出力（.gitignore）
│   └── axolotl_qwen_finetune/
│       ├── checkpoint-*           # 学習チェックポイント
│       ├── gen_metrics_step*.json # 生成評価メトリクス
│       └── adapter_model.safetensors  # LoRAアダプタ（最終モデル）
├── requirements.txt               # Python依存パッケージ
├── qwen_finetune.py              # 既存実装（参照用）
└── README.md                      # このファイル
```

### src/ ディレクトリの詳細

#### `src/axolotl_configs/qwen_finetune.yml`
Axolotlの統一設定ファイル。以下を定義します：

- **モデル設定**: `unsloth/Qwen3-30B-A3B-Instruct-2507`
- **精度**: `bf16: true`（A100ネイティブ、高速かつ高精度）
- **LoRA設定**:
  - r=16, alpha=32
  - **重要**: MoE対応ターゲット
    - Attention層: `q_proj, k_proj, v_proj, o_proj`
    - Expert層: `up_proj, down_proj, gate_proj` ← **必須**
- **学習パラメータ**:
  - 学習率: 2e-4（cosine: 2e-4 → 1e-5）
  - バッチサイズ: 1 × 16積算 = 実効16
  - Epoch: 1.5（～174,663サンプル）
  - 評価/保存: 200ステップごと
- **データセット**: `data/train/hawks_train_curriculum.json`
  - Hawks（正解あり）×3 + Dummy（正解なし）×96,000
  - 比率: ~1:1.2（Negative Sampling効果最大）
- **評価用**: `data/train/hawks_val.json`
- **最適化**:
  - Flash Attention 2（FA2）でネイティブ高速化
  - Gradient Checkpointingでメモリ効率化
  - DeepSpeed ZeRO-2でOptimizer Stateシャード化
- **コールバック**:
  - `GenerationEvalCallback`: JSON生成メトリクス計算
  - `MLflowLoggerCallback`: メトリクスロギング

#### `src/deepspeed_configs/zero2.json`
メモリ効率化設定。以下を提供します：

- **ZeRO-2最適化**:
  - Optimizer States をGPU間でシャード化（120GB → 30GB/GPU）
  - Gradients もシャード化
  - CPU Offload対応（より安全な運用）
- **Optimizerパラメータ**: AdamW（lr=2e-4）

**メモリ見積もり** (A100 80GB):
```
GPU単体当たり:
  - モデル重み (bf16):       ~15GB
  - LoRA:                    ~1GB
  - Optimizer States (ZeRO-2): ~30GB
  - Gradients (ZeRO-2):      ~15GB
  - Activations:             ~15-20GB
  ─────────────────────────
  合計:                       ~70-75GB（80GB以内）
```

#### `src/callbacks/generation_eval.py`
生成タスク評価コールバック（~930行）。以下を実装：

**メトリクス計算**:
- `gen_json_parse_rate`: JSON解析成功率
- `gen_schema_ok_rate`: スキーマ準拠率
- `gen_offender_presence_f1`: 加害者有無のF1スコア
- `gen_offender_name_set_f1`: 人名セットF1スコア
- `gen_all_fields_exact_match_rate`: 全フィールド一致率
- `gen_field_exact_match_rate/*`: フィールド別一致率
- `gen_fp_rate_on_negative`: ネガティブサンプル誤検出率

**実行**: 評価フェーズ（`eval_steps: 200`）で自動実行

#### `src/callbacks/mlflow_logger.py`
MLflow実験トラッキング。以下を記録：

- 学習メトリクス（loss, learning_rate等）
- 生成評価メトリクス（F1スコア等）
- ステップごとのメトリック推移

**MLflow UI**:
```bash
mlflow ui --host 0.0.0.0 --port 5000
# http://localhost:5000 でダッシュボード確認
```

---

### scripts/ ディレクトリの詳細

#### `scripts/prepare_curriculum_data.py`
Positive（Hawks）とNegative（Dummy）サンプルをマージするデータ準備スクリプト。

**目的**:
- Hawks（正解ラベルあり、犯人がいるニュース）×3回繰り返し
- Dummy（正解なし、犯人がいないニュース）の先頭96,000件
- → バランスの取れた学習データを生成

**入力ファイル**:
```
data/train/
  ├── hawks_train.json (26,221サンプル)
  └── person_dummy_2025-12-23-1817.json (100,000サンプル)
```

**処理フロー**:
1. `hawks_train.json` を3回繰り返す → 78,663サンプル
2. `person_dummy_2025-12-23-1817.json` から先頭96,000件取得
3. 順序を保ったまま結合
4. `hawks_train_curriculum.json` に保存（174,663サンプル）

**出力ファイル**:
```
data/train/hawks_train_curriculum.json (174,663サンプル, ~468MB)
```

**実行**:
```bash
python scripts/prepare_curriculum_data.py

# 出力例:
# ✅ Hawks数: 26,221 → ×3 = 78,663
# ✅ Dummy数: 96,000
# ✅ 合計: 174,663 (比率: 1:1.2)
# ✅ 保存: data/train/hawks_train_curriculum.json
```

**重要な設計選択**:
- **順序固定しない**: マージ後、Axolotlがランダムシャッフルする
- **比率バランス**: Hawks ×3 : Dummy = 1:1.2 で Negative Sampling効果最大
- **既存ファイル不使用**: `hawks_train_plus_person_dummy_2025-12-23-1817.json` (126,221サンプル) は不均衡（1:3.8）のため使用しない

**トラブルシューティング**:
- エラー: `FileNotFoundError: hawks_train.json`
  → `data/train/` に入力ファイルを配置してから実行
- 出力サイズ確認:
  ```bash
  du -sh data/train/hawks_train_curriculum.json
  # → 約468MB が期待値
  ```

#### `scripts/run_axolotl_train.sh`
Axolotl学習パイプラインのラッパースクリプト。

**目的**:
- データ準備の自動実行
- Axolotlトレーニング起動の簡素化
- GPU数動的指定に対応

**使用法**:
```bash
# 基本（4GPU、デフォルト設定）
chmod +x scripts/run_axolotl_train.sh
./scripts/run_axolotl_train.sh

# 設定ファイル指定
./scripts/run_axolotl_train.sh src/axolotl_configs/qwen_finetune.yml

# GPU数指定（例: 8GPU）
./scripts/run_axolotl_train.sh src/axolotl_configs/qwen_finetune.yml 8
```

**処理フロー**:
```
1. 入力チェック
   - CONFIG_FILE が存在するか
   - デフォルト: src/axolotl_configs/qwen_finetune.yml
   - デフォルト NUM_GPUS: 4

2. データ準備
   - hawks_train_curriculum.json が存在するか
   - 未生成なら prepare_curriculum_data.py を実行

3. 学習実行
   - accelerate launch コマンド実行
   - --num_processes: NUM_GPUS
   - --mixed_precision: bf16
   - -m axolotl.cli.train: Axolotl CLIで実行

4. 終了
   - 学習完了を表示
```

**出力ディレクトリ**:
```
outputs/axolotl_qwen_finetune/
  ├── checkpoint-200/               # ステップ200チェックポイント
  ├── checkpoint-400/               # ステップ400チェックポイント
  ├── checkpoint-600/               # ... (最新3つ保持)
  ├── adapter_model.safetensors     # LoRAアダプタ（最終モデル）
  ├── adapter_config.json           # LoRA設定メタデータ
  ├── gen_metrics_step200.json      # ステップ200生成メトリクス
  ├── gen_metrics_step400.json      # ステップ400生成メトリクス
  ├── training_args.bin             # 学習引数
  ├── optimizer.pt                  # 最後のOptimizer状態
  └── trainer_state.json            # 学習状態（再開に必要）
```

**トラブルシューティング**:
- 実行権限エラー: `./scripts/run_axolotl_train.sh: Permission denied`
  ```bash
  chmod +x scripts/run_axolotl_train.sh
  ```
- CONFIG_FILE エラー: `Error: Config file not found`
  ```bash
  # パスを完全に指定
  ./scripts/run_axolotl_train.sh /absolute/path/to/qwen_finetune.yml
  ```
- GPU認識エラー: `RuntimeError: Expected all tensors to be on the same device`
  ```bash
  nvidia-smi  # GPU × 4が表示されるか確認
  accelerate env  # num_processes=4 が表示されるか確認
  ```

**ログ確認**:
```bash
# 標準出力をリアルタイム確認
./scripts/run_axolotl_train.sh 2>&1 | tee training.log

# または学習中にログを監視
tail -f outputs/axolotl_qwen_finetune/training.log
```

---

## セットアップ手順

### 環境要件

```yaml
GPU: NVIDIA A100 80GB × 4
CUDA: 12.1以上
Python: 3.10以上
OS: Ubuntu 20.04/22.04（推奨）
ストレージ: 500GB以上（モデル + データ + チェックポイント）
```

### クラウド環境での初期セットアップ

#### 1. リポジトリクローン
```bash
git clone https://github.com/<your-org>/Qwen-finetune.git
cd Qwen-finetune
```

#### 2. Pythonパッケージインストール
```bash
# 仮想環境作成（オプション）
python3 -m venv .venv
source .venv/bin/activate

# 依存関係インストール
pip install --upgrade pip
pip install -r requirements.txt

# Flash Attention 2（A100最適化）
pip install flash-attn --no-build-isolation
```

#### 3. データファイルの配置
```bash
# クラウド環境にアップロード（以下3ファイル、合計468MB）
# - data/train/hawks_train.json (26,221サンプル)
# - data/train/hawks_val.json (27,798サンプル)
# - data/train/person_dummy_2025-12-23-1817.json (100,000サンプル)

# 確認
ls -lh data/train/*.json
```

#### 4. Accelerate設定（初回のみ）
```bash
accelerate config

# 以下を選択:
# - Compute environment: This machine
# - Distributed type: MULTI_GPU
# - Number of processes: 4
# - Mixed precision: bf16
# - Use DeepSpeed: No（YAML設定で指定）
```

#### 5. GPUと設定の確認
```bash
# GPU確認
nvidia-smi
# → A100 × 4が表示される

# Accelerate設定確認
accelerate env
# → num_processes=4, mixed_precision=bf16
```

---

## 学習実行

### オプション1: ラッパースクリプト（推奨）
```bash
chmod +x scripts/run_axolotl_train.sh
./scripts/run_axolotl_train.sh
```

**処理フロー**:
1. `scripts/prepare_curriculum_data.py` でデータ自動マージ
2. Axolotlで学習開始（4GPU、bf16、DeepSpeed ZeRO-2）
3. 200ステップごとに評価 + メトリクス保存

### オプション2: 直接実行
```bash
# データ準備
python scripts/prepare_curriculum_data.py

# 学習実行
accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    -m axolotl.cli.train \
    src/axolotl_configs/qwen_finetune.yml
```

### オプション3: 小規模テスト（デバッグ用）
```bash
# YAML設定を一時的に変更
# max_steps: 10
# eval_steps: 5

# 1GPUでテスト
accelerate launch --num_processes=1 \
    -m axolotl.cli.train \
    src/axolotl_configs/qwen_finetune.yml

# YAML設定を本番に戻す
# num_epochs: 1.5
# max_steps: -1
```

---

## 学習結果の確認

### チェックポイント
```bash
ls -la outputs/axolotl_qwen_finetune/
# → checkpoint-200/
# → checkpoint-400/
# → ... (save_total_limit: 3 で最新3つのみ保持)
```

### 生成評価メトリクス
```bash
ls outputs/axolotl_qwen_finetune/gen_metrics_step*.json

# 例: 最新メトリクス確認
cat outputs/axolotl_qwen_finetune/gen_metrics_step$(ls -t outputs/axolotl_qwen_finetune/gen_metrics_step*.json | head -1 | grep -oP 'step\K\d+').json | python -m json.tool
```

### MLflow実験トラッキング
```bash
# MLflow UIを起動
mlflow ui --host 0.0.0.0 --port 5000

# ブラウザで http://<IP>:5000 にアクセス
# → 学習グラフ、メトリクス推移、ハイパーパラメータを確認
```

### 学習ログ
```bash
# Axolotlのログを確認
tail -f outputs/axolotl_qwen_finetune/training.log
# または標準出力を監視
```

---

## パフォーマンス最適化ポイント

### なぜbf16か？（4bitではなく）
- **A100ネイティブ**: Tensor Coreが bf16 をネイティブサポート（超高速）
- **高精度**: 4bit量子化による精度劣化なし
- **数値安定性**: 勾配計算が安定（4bitより）
- **メモリ効率**: DeepSpeed ZeRO-2で対応（30GB/GPU）

### なぜMoE対応LoRAか？
- **Expert層学習**: `up_proj, down_proj, gate_proj` を含める
- **Attention層のみ不可**: MoE性能が死ぬ（Qwen3の強みは Expert多様性）
- **YAML記述**: `lora_target_modules` で明示的に指定

### なぜNegative Samplingか？
- **ハルシネーション防止**: 「抽出すべき」と「無視すべき」を混在学習
- **比率バランス**: Hawks ×3 (78K) : Dummy (96K) ≈ 1:1 で最大効果
- **ランダムシャッフル**: Axolotlのデフォルト動作に任せる（順序固定しない）

### なぜDeepSpeed ZeRO-2か？
- **メモリ削減**: Optimizer States 120GB → 30GB/GPU（4分の1）
- **OOM防止**: スパイク時のメモリエラーを完全防止
- **安定性**: CPU Offload対応で更に安全

---

## ローカル環境での検証（オプション）

データセットの検証と小規模デバッグ用：

```bash
# データ準備
python scripts/prepare_curriculum_data.py

# データサイズ確認
du -sh data/train/hawks_train_curriculum.json
# → 約468MB, 174,663サンプル

# YAMLテスト（構文チェック）
python -c "import yaml; yaml.safe_load(open('src/axolotl_configs/qwen_finetune.yml'))"

# 依存関係確認
pip check
```

---

## トラブルシューティング

### GPUメモリ不足（OOM）
**症状**: `RuntimeError: CUDA out of memory`

**対策**:
1. DeepSpeed ZeRO-2が正しく読み込まれているか確認
   ```bash
   grep "deepspeed" src/axolotl_configs/qwen_finetune.yml
   ```
2. バッチサイズを削減（YAML: `micro_batch_size: 1` が最小）
3. `sequence_len` を削減（デフォルト: 1536）

### 学習が進まない
**症状**: Loss が下がらない、評価メトリクスが改善しない

**対策**:
1. データが正しくマージされているか確認
   ```bash
   python -c "import json; d = json.load(open('data/train/hawks_train_curriculum.json')); print(f'Total: {len(d)}, Sample: {d[0].keys()}')"
   ```
2. 学習率調整（YAML: `learning_rate` を2e-4から変更）
3. 評価データの品質確認

### コールバック実行エラー
**症状**: `AttributeError: module 'callbacks' has no attribute ...`

**対策**:
1. `src/callbacks/__init__.py` が存在確認
2. Python Pathにsrcが含まれているか確認
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/Users/terusato/Dev/Qwen-finetune"
   ```
3. Axolotl plugins セクション構文をチェック

---

## 既存ファイル参照

### Unsloth版（旧実装）
- `qwen_finetune.py`: Unsloth + SFTTrainerベースの2ステージ学習
  - **参照用**: ロジック理解、コールバック実装確認に利用可
  - **非推奨**: Axolotl版が新しい標準

### 旧CLAUDE.md
タスク概要とCSV評価形式の詳細は [CLAUDE.md](./CLAUDE.md) を参照

---

## クラウドへのデプロイ

### GitHub へのプッシュ
```bash
git add .
git commit -m "feat: Axolotl migration with bf16 + Negative Sampling"
git push origin main
```

### 別のクラウド環境での実行
```bash
# 1. クローン + セットアップ
git clone https://github.com/<org>/Qwen-finetune.git
cd Qwen-finetune
pip install -r requirements.txt

# 2. データアップロード（AWS S3, GCS等）
# data/train/ に3ファイルを配置

# 3. 学習実行
./scripts/run_axolotl_train.sh
```

### スケーリング（GPU数変更）
```bash
# 8GPU で実行
./scripts/run_axolotl_train.sh src/axolotl_configs/qwen_finetune.yml 8

# またはYAML編集
# src/axolotl_configs/qwen_finetune.yml の deepspeed 設定を確認
```

---

## まとめ

| 項目 | 旧（Unsloth） | 新（Axolotl） |
|------|--------------|--------------|
| **設定管理** | Python CODE | YAML（git管理容易） |
| **実行方法** | `torchrun qwen_finetune.py` | `accelerate launch -m axolotl.cli.train` |
| **学習戦略** | 2ステージ（段階的） | 単一ステージ（Negative Sampling） |
| **精度** | 4bit（QLoRA） | **bf16（フル精度）** |
| **メモリ** | ~100GB/GPU | ~70-75GB/GPU（ZeRO-2） |
| **速度** | Unsloth最適化 | **FA2ネイティブ最適化** |
| **拡張性** | 低（カスタム実装必要） | **高（プラグイン可能）** |
| **再現性** | 中（seed固定のみ） | **高（YAML + seed）** |

---

## 質問・サポート

詳細は以下を参照：
- Axolotl公式: https://github.com/axolotl-ai-cloud/axolotl
- Qwen公式: https://github.com/QwenLM/Qwen
- Plan: [floofy-popping-meteor.md](.claude/plans/floofy-popping-meteor.md)
