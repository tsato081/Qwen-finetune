# 変更内容・エラー・現状の問題 まとめ

## 1. ✅ 修正済みの変更内容

### 1.1 設定ファイル修正（YAML）

#### [src/axolotl_configs/qwen_finetune.yml](src/axolotl_configs/qwen_finetune.yml)

- **Line 46:** `chat_template: qwen3_instruct` → `chat_template: qwen3`
  - 理由: Axolotlの ChatTemplate enumが`qwen3`のみをサポート（Unslothサンプルは`qwen3_instruct`だが、Axolotlでは`qwen3`が正）

- **Lines 95-97:** プラグインパス修正
  ```yaml
  # 修正前（間違い）
  plugins:
    - src.callbacks.GenerationEvalCallback
    - src.callbacks.MLflowLoggerCallback

  # 修正後（正）
  plugins:
    - src.callbacks.generation_eval.GenerationEvalPlugin
    - src.callbacks.mlflow_logger.MLflowLoggerPlugin
  ```
  - 理由: Axolotlのプラグインローダーはフルモジュールパスが必要

#### [src/axolotl_configs/qwen_finetune_test.yml](src/axolotl_configs/qwen_finetune_test.yml) (新規作成)

- 2ステップの最小限テスト設定
- `max_steps: 2`
- `sequence_len: 512`（本番は1536）
- `plugins: []`（デバッグ用に無効化）
- DeepSpeed無効化（単GPU検証用）
- 最適化フラグ無効化：
  ```yaml
  gradient_checkpointing: true
  flash_attention: false
  unsloth: false
  torch_compile: false
  lora_qkv_kernel: false
  lora_o_kernel: false
  lora_mlp_kernel: false
  ```

### 1.2 Pythonコールバック修正

#### [src/callbacks/generation_eval.py](src/callbacks/generation_eval.py)

- Axolotlプラグインインターフェース完全実装
- `load_datasets()` メソッド追加（戻り値: None）
- 20+個のプラグインライフサイクルメソッド実装
- `GenerationEvalPlugin` クラス追加

#### [src/callbacks/mlflow_logger.py](src/callbacks/mlflow_logger.py)

- Axolotlプラグインインターフェース完全実装
- `load_datasets()` メソッド追加（戻り値: None）
- 20+個のプラグインライフサイクルメソッド実装
- `MLflowLoggerPlugin` クラス追加

#### [src/callbacks/__init__.py](src/callbacks/__init__.py)

- 新プラグインクラスをexportに追加

### 1.3 シェルスクリプト修正

#### [scripts/run_axolotl_train.sh](scripts/run_axolotl_train.sh)

- Line 43: PYTHONPATH設定追加
  ```bash
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  ```
- Lines 57-71: ファイル検証チェック追加
  - `hawks_val.json` 存在チェック
  - `zero2.json` 存在チェック

### 1.4 パッケージ設定修正

#### [pyproject.toml](pyproject.toml)

- `[build-system]` セクション追加
  ```toml
  [build-system]
  requires = ["setuptools>=61.0"]
  build-backend = "setuptools.build_meta"

  [tool.setuptools.packages.find]
  where = ["."]
  include = ["src*"]
  ```

---

## 2. ✅ 修正済みエラー

| # | エラー | 原因 | 修正 | 状態 |
|---|--------|------|------|------|
| 1 | `ValidationError: chat_template 'qwen3_instruct' is invalid enum` | chat_templateがAxolotlのenum値ではない | `qwen3` に変更 | ✅ 完了 |
| 2 | `AttributeError: module 'src.callbacks' has no attribute 'GenerationEvalCallback'` | プラグインパスが不完全（モジュール名欠落） | フルパス指定に修正 | ✅ 完了 |
| 3 | `ImportError: undefined symbol in flash-attn CUDA binary` | flash_attn 2.8.3 ← PyTorch 2.6 CUDA 12.4だが実行環境CUDA 12.8 | `flash_attention: false` | ✅ 完了 |
| 4 | `AttributeError: get_callbacks, load_datasets, post_model_load...` (20+個) | Axolotlプラグインインターフェースメソッド未実装 | 全ライフサイクルメソッド実装 | ✅ 完了 |
| 5 | `ValueError: DeepSpeed train_batch_size=64 vs HF calculated=16` | DeepSpeedとHFの計算が矛盾 | テスト設定からDeepSpeed削除 | ✅ 完了 |

---

## 3. 🚧 進行中/未解決の問題

### 3.1 主問題: トレーニング初期化ハング（CRITICAL）

#### 症状

```
0%|          | 0/3896 [00:00<?, ?it/s]
```

- プログレスバーが0%で**9分以上凍結**
- モデルロードとLoRA適用までは完了
- 最後のログ: `"After initializing ZeRO optimizer"`

#### テスト結果

| テスト | GPU数 | 最適化 | DeepSpeed | 結果 | 結論 |
|--------|-------|--------|-----------|------|------|
| v1 本番 | 4 | あり | Zero2 | ハング | NCCL疑い |
| v2 テスト | 1 | 削減 | なし | ハング | NCCL否定 |
| v3 テスト | 1 | さらに削減 | なし | **timeout** | 単GPU も ハング |

#### 根本原因

不明（絞り込み中）

#### 可能性（優先度順）

1. LoRAカーネル自動コンパイル（`lora_qkv_kernel: true`デフォルト時）
2. 最初のバッチ処理の遅延
3. Torch Inductor JITコンパイル
4. Gradient checkpointingの再計算オーバーヘッド
5. 大規模モデル（30B）のメモリ初期化

#### 判明している事実

- ✅ NCCL通信でない（単一GPU でもハング）
- ✅ プラグインローディングでない（plugin: []でもハング）
- ✅ DeepSpeedでない（単GPU テストで無効化）
- ❓ LoRA最適化フラグ（`lora_*_kernel: false` しても未改善）
- ❓ Gradient Checkpointing（有効なまま、未検証）

---

## 4. 現在の環境状態

### ハードウェア

- GPU: 4x（クラウド環境）
- メモリ: 各GPU ~68GB確保

### 実行構成

```bash
# 本番テスト（4GPU）
accelerate launch \
  --num_processes=4 \
  --multi_gpu \
  --mixed_precision=bf16 \
  -m axolotl.cli.train \
  src/axolotl_configs/qwen_finetune.yml

# 最新テスト（1GPU、300秒timeout）
export CUDA_VISIBLE_DEVICES=0
timeout 300 bash -c '
accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  -m axolotl.cli.train \
  src/axolotl_configs/qwen_finetune_test.yml
'
```

### データ

- ✅ `data/train/hawks_train_curriculum.json` 存在
- ✅ `data/train/hawks_val.json` 存在（19MB）
- ✅ `src/deepspeed_configs/zero2.json` 存在

---

## 5. 次のステップ（提案）

### 診断順序

1. **Gradient Checkpointingを無効化してテスト**
   ```yaml
   gradient_checkpointing: false  # 再計算オーバーヘッド確認
   ```

2. **LoRA対象を削減してテスト**
   ```yaml
   lora_target_modules:
     - q_proj
     - v_proj
     - o_proj
     # Expert層を一時的に外す
   ```

3. **Timeout延長テスト**
   ```bash
   timeout 600 bash -c '...'  # 10分に延長
   ```

4. **詳細タイミング計測**
   - Axolotlのログレベル上げて、各ステップの実行時間を記録

---

## 6. サマリ

| カテゴリ | 完了 | 進行中 | 状態 |
|---------|------|--------|------|
| 設定ファイル修正 | ✅ 5個 | - | 完了 |
| Pythonコード修正 | ✅ 2個 | - | 完了 |
| シェルスクリプト修正 | ✅ 1個 | - | 完了 |
| ✅ 解決済みエラー | ✅ 5個 | - | 完了 |
| 🚧 未解決の問題 | - | 🚧 1個（主問題） | **進行中** |

**現状:** トレーニング初期化ハング原因特定のため、段階的に最適化を無効化してどこで加速するかを検証中。
