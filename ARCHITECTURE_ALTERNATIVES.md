# アーキテクチャ選択肢の検討：タスク分析と最適化

## 1. タスク再分析：NER vs LLM

### 現在のタスク定義（CLAUDE.md より）

```
タスク 1: 人物抽出・分類
  入力: 記事テキスト + 人物候補ヒント
  出力: 以下の構造化情報を持つ人物配列
    - name (必須): フルネーム
    - is_real (必須): 実在人物か
    - is_offender (必須): 加害者/容疑者か
    - age: 年齢（数値）
    - position: 役職名
    - affiliation: 所属組織
    - category: 法令違反/PEPS/破産/その他
    - act: 加害者の違法行為名
    - reason (必須): 判定根拠の記事抽出

タスク 2: 住所抽出
  入力: 記事テキスト + 企業/人物名ターゲット
  出力:
    - headquarters_locations: 本社住所
    - residences: 居住地住所
    - physical_crime_occured_locations: 犯行地点
  ルール: 記事に明示された情報のみ（推測禁止）
```

---

## 2. タスクの性質分類

### 2.1 構造化抽出問題

| 側面 | 説明 | 難度 |
|------|------|------|
| **Entity Extraction** | 名前・年齢・組織 etc の抽出 | ⭐ 低 (NER で対応可) |
| **Entity Linking** | 同一人物の表記ゆれ統一 | ⭐⭐ 中等 (ルール + 簡易マッチング) |
| **Attribute Classification** | 各 Entity の属性分類（実在か/加害者か等） | ⭐⭐⭐ 高（文脈理解必須） |
| **Address Extraction** | 住所の正規化・抽出 | ⭐⭐ 中等 (パターン + 辞書) |

### 2.2 各段階で必要な技術

```
人物抽出
  ├─ Named Entity Recognition (NER)
  │   └─ BioBERT / RoBERTa-JP など軽量モデルで十分
  │
  └─ 属性分類 (is_offender, is_real, category, act)
      └─ テキスト分類 or 軽量 LLM の推論

住所抽出
  ├─ 正規表現 / パターンマッチング （都道府県等）
  ├─ 簡易 NER （建物名・施設名）
  └─ 明示性の判定（オプション：軽量 LLM）
```

### 2.3 LLM が活躍する部分

| 処理 | LLM が必要か | 代替案 |
|------|------------|--------|
| **テキスト理解** | △ 実在性・加害者性判定 | 文脈ルール + キーワード辞書 |
| **複雑推論** | △ 判定根拠の生成 | テンプレート化 + 抽出箇所の直接参照 |
| **文脈処理** | ○ 記事全体の理解 | BioBERT + 決定木分類 で対応可能性あり |

---

## 3. 現在のセットアップの問題

### Qwen3-30B を使う理由？

```
メリット:
  ✓ 高精度な推論
  ✓ 複雑な指示理解
  ✓ 日本語処理が得意

デメリット:
  ✗ 30B パラメータ = 遅い
  ✗ GPU メモリ大量消費 (68GB 必須)
  ✗ クラウド費用高額
  ✗ 現タスクの複雑度に見合わない
```

### タスク難度 vs モデルサイズ

```
タスク難度の客観評価:
  - NER: 中等 (BERT 級で十分)
  - 属性分類: 中等〜高 (小〜中規模 LLM)

現モデル: Qwen3-30B（過度）
推奨: 7B 級 or 従来型 NER ベース
```

---

## 4. 代替アーキテクチャの提案

### **案 A: モデルダウンサイズ（推奨: 高速化 + 保守性）**

#### A-1: Qwen2-7B-Instruct に切り替え

```yaml
base_model: Qwen/Qwen2-7B-Instruct  # 30B → 7B
model_type: qwen2
adapter: lora
lora_r: 16
lora_alpha: 32
micro_batch_size: 4  # メモリに余裕 → 上げられる
gradient_accumulation_steps: 4  # 削減可能
deepspeed: none  # 単一 GPU で動作
```

**期待効果:**
```
メモリ: 30B (68GB) → 7B (16GB)
速度: 1 step → 2.5-3 倍高速化見込み
学習時間: 100+ 時間 → 40-50 時間
```

**トレードオフ:**
- 精度若干低下の可能性
- テスト必須

#### A-2: Qwen2.5-3B (最軽量)

```yaml
base_model: Qwen/Qwen2.5-3B-Instruct
micro_batch_size: 8
gradient_accumulation_steps: 2
deepspeed: none  # 単 GPU で十分
```

**期待効果:**
```
速度: 5-7 倍高速化
学習時間: 100+ → 15-20 時間
メモリ: 5-8GB のみ
費用: 大幅削減
```

**懸念:**
- 複雑推論（実在性判定等）で精度低下の可能性

---

### **案 B: ハイブリッド NER + 軽量分類モデル（最適効率）**

#### B-1: パイプライン構成

```
記事テキスト
  ↓
[Stage 1] NER (RoBERTa-JP-Large or SpaCy NER)
  → 人物名、組織名、住所候補を抽出
  ↓
[Stage 2] Post-processing (ルール)
  → 表記ゆれ統一、重複排除
  ↓
[Stage 3] 属性分類 (Qwen2-7B-Instruct or BertForSequenceClassification)
  → is_offender, is_real, category, act の判定
  ↓
[Stage 4] 住所抽出 (正規表現 + 簡易 NER)
  → 緯度経度/都道府県の抽出
  ↓
構造化出力 JSON
```

**実装形態:**
```python
# pseudo code
def extract_persons(article: str, person_hints: list) -> dict:
    # Stage 1: NER で基本的な entity 抽出
    entities = ner_model.predict(article)  # spaCy 等

    # Stage 2: キーワード/ルール マッチング
    persons = postprocess_entities(entities, person_hints)

    # Stage 3: 属性分類 (軽量分類器)
    for person in persons:
        context = extract_context(article, person['name'])
        person['is_offender'] = classifier(context)  # 7B LLM か小規模分類器
        person['is_real'] = real_person_check(person['name'])  # 辞書 + ルール
        # ...

    return persons
```

**期待効果:**
```
速度: 案 A-1 並み（7B ベース）or 更に高速（NER ベース）
精度: NER は得意分野 → 高精度
保守性: 各 stage を独立改善可能
```

**懸念:**
- パイプライン実装の手間

---

### **案 C: 従来型 NER（最高速、精度トレードオフ）**

#### C-1: spaCy + カスタムルール

```python
import spacy
from spacy.training import Example

nlp = spacy.load("ja_core_news_lg")

def extract_persons_rule_based(article: str) -> dict:
    doc = nlp(article)

    persons = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            person = {
                "name": ent.text,
                "is_offender": "容疑者" in doc[ent.start_char-20:ent.start_char+20].text
                             or "逮捕" in context,
                "is_real": check_real_person(ent.text),  # 辞書查詢
                ...
            }
            persons.append(person)

    return {"persons": persons}
```

**期待効果:**
```
速度: 秒単位 (GPU 不要)
費用: ゼロ
メモリ: 数 GB のみ
```

**懸念:**
- 複雑な判定精度が落ちる
- 推論より「ルール整備」に時間

---

## 5. 各案の比較表

| 項目 | 案 A-1 (7B) | 案 A-2 (3B) | 案 B (NER + 分類) | 案 C (spaCy) |
|------|-----------|-----------|-----------------|-----------|
| **モデルサイズ** | 7B | 3B | NER (中) + 分類 (7B) | なし |
| **メモリ必要** | 16GB | 5-8GB | 20-25GB | <2GB |
| **1 ステップ時間** | ~15 秒 | ~5 秒 | ~20 秒 | <1 秒 |
| **学習時間** (10k steps) | 40-50 時間 | 15-20 時間 | 50-60 時間 | N/A |
| **GPU 費用** | 中 | 低 | 中 | なし |
| **精度 (人物抽出)** | 高 | 中 | 高〜中等 | 中〜低 |
| **精度 (属性分類)** | 高 | 中 | 高 | 低 |
| **保守性** | 高 | 高 | 最高 | 低 |
| **開発工数** | 低 | 低 | 中 | 高 |

---

## 6. 推奨ロードマップ

### **Phase 1: 即座の高速化（次クラウド実行）**

#### 1.1 Qwen2-7B-Instruct テスト

```bash
# 現在の qwen_finetune.yml をコピー
cp src/axolotl_configs/qwen_finetune.yml \
   src/axolotl_configs/qwen_finetune_7b.yml

# 修正内容：
#   base_model: Qwen/Qwen2-7B-Instruct
#   micro_batch_size: 4
#   gradient_accumulation_steps: 4
#   deepspeed: none (または zero0.json に変更)
#   max_steps: 2 (テスト用)
```

**検証項目:**
- ✓ メモリ不足なし？
- ✓ ステップ時間が 4-5 倍高速化?
- ✓ 精度維持されている？（評価セットで確認）

**予測結果:**
```
成功 → 本番は 7B で運用
失敗 (精度大幅低下) → 案 B 検討
```

---

### **Phase 2: 中期的な最適化（1-2 週間）**

#### 2.1 案 B: NER ベース パイプライン試行

**実装順:**
1. 既存のカスタムコールバック（GenerationEval 等）を改造
2. Stage 1-2 を実装（NER + Post-processing）
3. Stage 3 を軽量分類器に変更（Qwen2-7B で問題ないことが前提）
4. Stage 4 を実装（住所抽出）

**コスト:**
- 開発時間: 3-5 日
- テスト: 2-3 日

**利益:**
- 速度: 30-50% 更に改善
- 精度: 構造化部分（NER）は高精度
- 保守性: 各 stage 独立改善可能

---

### **Phase 3: 実務最適化（必要に応じて）**

#### 3.1 従来型 NER への完全移行検討

**判定条件:**
- 案 B で精度十分？ → Phase 3 不要（案 B 運用続行）
- 案 B でも精度不足？ → 案 C 検討

---

## 7. 次クラウド実行時のアクションプラン

### **最初のステップ（決定）**

```bash
# ステップ 1: Qwen2-7B 設定ファイル作成
cat > src/axolotl_configs/qwen_finetune_7b.yml << 'EOF'
base_model: Qwen/Qwen2-7B-Instruct
model_type: qwen2
# ... (他は qwen_finetune.yml と同等、以下を変更)
micro_batch_size: 4
gradient_accumulation_steps: 4
# deepspeed 行を削除 or コメント化
max_steps: 2  # テスト用
EOF

# ステップ 2: 環境セットアップ
cd /Users/terusato/Dev/Qwen-finetune

# pyproject.toml から Unsloth 削除
# (既に計画済みの改善)

uv pip install -e . --upgrade

# ステップ 3: 1-step テスト
export CUDA_VISIBLE_DEVICES=0
time accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m axolotl.cli.train src/axolotl_configs/qwen_finetune_7b.yml \
  2>&1 | tee test_7b_baseline.log

# ステップ 4: ログ分析
# - ステップ時間を記録
# - メモリピーク確認
# - エラー有無確認
```

### **検証フロー**

```
Qwen2-7B テスト実行
  ↓
ステップ時間 < 30 秒?
  ├─ Yes → 承認 (7B で本訓練)
  └─ No → 原因調査 (DeepSpeed 設定? batch size?)

精度検証 (評価セットで)
  ├─ 許容範囲 → 採用
  └─ 精度大幅低下 → Qwen2.5-3B テスト
```

---

## 8. CLAUDE.md（タスク定義）への影響

### 変更不要項目

```yaml
出力形式は変わらない:
  persons: [...]  # 同じ JSON スキーマ
  residences: [...]
  headquarters_locations: [...]
  physical_crime_occured_locations: [...]
```

### 評価方法の注意

```
CSV 正解データマッピング時も変わらない
（内部実装が Qwen3 → Qwen2 に変わるだけ）

ただし、精度の変動を監視:
  - 評価セット (Hawks_正解データマスター.csv) で実測
  - F1 スコア、正解率を記録
  - 閾値判定が変わる可能性あり
```

---

## 9. 最終推奨

### **今すぐ実施**

1. ✅ Unsloth 削除（pyproject.toml）
2. ✅ Flash-attn バージョン修正
3. ⭐ **Qwen2-7B テスト** ← 最優先

### **テスト結果次第**

- **7B で十分** → そのまま本運用、完了
- **精度落ちる** → 案 B (NER パイプライン) 検討
- **更に高速必須** → 案 A-2 (3B) or 案 C (spaCy)

### **短期的な期待値**

```
現状: 1 ステップ = 2-3 分（未検証）
目標:
  - 7B: 1 ステップ = 30-45 秒 (4-5 倍高速)
  - 3B: 1 ステップ = 10-15 秒 (10+ 倍高速)

学習時間:
  - Qwen3-30B: 100+ 時間 (3896 steps × 2-3 分)
  - Qwen2-7B: 40-50 時間 (3896 steps × 30-45 秒)
  - Qwen2.5-3B: 15-20 時間
```

---

## 10. サマリ

| 項目 | 現状 | 改善案 | 期待値 |
|------|------|--------|--------|
| **モデルサイズ** | 30B（過度） | 7B → 3B | 4-10 倍高速化 |
| **メモリ** | 68GB | 16GB → 5GB | 削減 |
| **費用** | 高額 | 中 → 低 | 大幅削減 |
| **精度** | 高い | 若干低下 | テストで確認 |
| **学習時間** | 100+ 時間 | 40-50 時間 | 実現可能に |

**結論:** Qwen3-30B は過度。Qwen2-7B テストで 4-5 倍高速化が期待できる。テスト優先。
