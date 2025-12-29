#!/usr/bin/env python3
"""
Upload merged model with auto-generated Model Card to HF Hub.

Usage:
    python3 upload_merged_model.py

This script should be run after `axolotl train` completes.
It will:
1. Generate a Model Card (README.md) with usage instructions
2. Upload the merged model to HuggingFace Hub
3. Display the model URL
"""

import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = (
    os.getenv("HF_AUTH_TOKEN")
    or os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
)

if not HF_TOKEN:
    print("✗ Error: Hugging Face token not found")
    print("  Set one of the following in .env or environment variables:")
    print("  - HF_AUTH_TOKEN=hf_...")
    print("  - HF_TOKEN=hf_...")
    print("  - HUGGINGFACE_HUB_TOKEN=hf_...")
    exit(1)

# Configuration
MERGED_DIR = Path("./outputs/gpt-oss-out/merged")
REPO_ID = "teru00801/gpt-oss-20b-person-v1"

# Verify merged model exists
if not MERGED_DIR.exists():
    print(f"✗ Error: Merged model directory not found: {MERGED_DIR}")
    print("  Please run training first: axolotl train src/axolotl_configs/gpt-oss_20b.yml")
    exit(1)

print(f"\n{'='*80}")
print("Model Upload & Model Card Generation")
print(f"{'='*80}")
print(f"Source directory: {MERGED_DIR}")
print(f"Target repository: {REPO_ID}")
print(f"Token: ✓ Configured\n")

# Generate Model Card (README.md)
MODEL_CARD = '''---
license: apache-2.0
language:
  - ja
tags:
  - gpt-oss-20b
  - lora
  - finetuning
  - japanese
  - criminal-information-extraction
  - person-extraction
---

# GPT-OSS-20B (Person Extraction)

日本語のニュース/事件記事から**犯罪者情報**を構造化抽出するタスク用にLoRAファインチューニングしたモデルです。

## 概要

- **ベースモデル**: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- **手法**: LoRA (Rank=64, Alpha=128)
- **訓練データ**: Hawks犯罪記事 + person_dummy（再生重み0.2）
- **生成日**: {DATE}
- **形式**: マージ版（スタンドアロン、ベースモデル不要）

## 使用方法

### インストール

```bash
pip install transformers torch
```

### 基本的な推論

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルの読み込み
model = AutoModelForCausalLM.from_pretrained(
    "teru00801/gpt-oss-20b-person-v1",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "teru00801/gpt-oss-20b-person-v1",
    trust_remote_code=True
)

# プロンプト構築
system_instruction = """あなたはニュース記事から犯罪者情報を抽出するアシスタントです。
出力はJSONのみ。推測禁止。犯罪者が明確でない場合も必ずスキーマどおりのオブジェクトを返す。
{{
  "persons": [
    {{
      "name": "人物の姓名",
      "is_real": true or false,
      "is_offender": true,
      "age": "年齢（数字のみ）",
      "position": "役職",
      "affiliation": "所属組織",
      "address": "住所",
      "category": "法令違反 | PEPS | 破産 | その他",
      "act": "違反行為（加害者のみ）"
    }}
  ],
  "occured_locations": ["..."],
  "police_departments": ["..."]
}}"""

article_text = """
[ニュース記事の本文をここに挿入]
"""

messages = [
    {{"role": "system", "content": system_instruction}},
    {{"role": "user", "content": f"【記事】\\n{article_text}"}}
]

# チャットテンプレートを適用
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 推論実行
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False
    )

# 結果の取得
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
```

### 出力フォーマット

```json
{{
  "persons": [
    {{
      "name": "山田太郎",
      "is_real": true,
      "is_offender": true,
      "age": "45",
      "position": "社長",
      "affiliation": "ABC株式会社",
      "address": "東京都千代田区",
      "category": "法令違反",
      "act": "詐欺"
    }}
  ],
  "occured_locations": ["東京都千代田区"],
  "police_departments": ["丸の内署"]
}}
```

## モデル詳細

### アーキテクチャ
- アーキテクチャ: GPT-OSS
- パラメータ数: 20B
- シーケンス長: 4096トークン

### LoRA設定
- LoRA Rank: 64
- LoRA Alpha: 128
- LoRA Dropout: 0
- ターゲットモジュール: gate_proj, down_proj, up_proj, q_proj, v_proj, k_proj, o_proj

### 訓練設定
- オプティマイザ: AdamW fused (torch)
- 学習率: 2e-4
- LR Scheduler: Cosine
- バッチサイズ: 8 (micro) × 2 (accumulation) = 16
- ウォームアップ: 3%
- DeepSpeed: ZeRO-2

## 評価

このモデルは Hawks犯罪記事データセット上で以下のメトリクスで評価されています：

```bash
python3 evaluate_model.py
```

詳細は `evaluate_model.py` を参照。

## ライセンス

Apache License 2.0

このモデルは以下のライセンス下で提供されます:
- ベースモデル (GPT-OSS): Apache 2.0
- LoRA適応: Apache 2.0

## 引用

このモデルを使用する場合:

```bibtex
@misc{{gpt_oss_person_extraction,
  author = {{Your Name}},
  title = {{GPT-OSS-20B (Person Extraction)}},
  year = {{{YEAR}}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{REPO_ID}}}}}
}}
```

## サポート

問題が発生した場合:
1. [GitHubのIssueページ](https://github.com/your-org/repo/issues)を確認
2. Hugging Face Discussions で質問
3. Axolotl公式ドキュメント: https://docs.axolotl.ai

## 関連リソース

- [Axolotl ファインチューニング](https://docs.axolotl.ai)
- [Transformers 推論ガイド](https://huggingface.co/docs/transformers/generation)
- [GPT-OSS 公式](https://huggingface.co/openai/gpt-oss-20b)
'''

# Replace placeholders
MODEL_CARD = MODEL_CARD.replace("{DATE}", datetime.now().strftime("%Y-%m-%d"))
MODEL_CARD = MODEL_CARD.replace("{REPO_ID}", REPO_ID)
MODEL_CARD = MODEL_CARD.replace("{YEAR}", datetime.now().strftime("%Y"))

# Write Model Card
print("Step 1: Generating Model Card...")
readme_path = MERGED_DIR / "README.md"
readme_path.write_text(MODEL_CARD, encoding="utf-8")
print(f"  ✓ Model Card generated: {readme_path}")

# Upload to HuggingFace Hub
print("\nStep 2: Uploading to Hugging Face Hub...")
try:
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    # Verify repo exists or create it
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="model")
        print(f"  ✓ Repository exists: {REPO_ID}")
    except Exception:
        print(f"  ℹ Creating new repository: {REPO_ID}")
        api.create_repo(repo_id=REPO_ID, repo_type="model", private=False)

    # Upload folder
    print(f"  Uploading {MERGED_DIR}...")
    api.upload_folder(
        folder_path=str(MERGED_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload Phase 2 merged model with Model Card"
    )
    print(f"  ✓ Upload complete")

except ImportError:
    print("  ✗ Error: huggingface_hub not installed")
    print("  Install it with: pip install huggingface-hub")
    exit(1)
except Exception as e:
    print(f"  ✗ Upload failed: {e}")
    exit(1)

# Display success
print(f"\n{'='*80}")
print("✓ SUCCESS")
print(f"{'='*80}")
print(f"Model URL: https://huggingface.co/{REPO_ID}")
print(f"Model Card: https://huggingface.co/{REPO_ID}/blob/main/README.md")
print(f"\nYou can now use it with:")
print(f"  from_pretrained('{REPO_ID}')")
print(f"{'='*80}\n")
