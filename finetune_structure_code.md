了解。**「新しく一から（= ベースモデルから）学習」**で、いまの前提（125k＝正例12.5k + 空N0 112.5k、N2なし、統合タスク、運用は空90%）に最適化した“事故りにくい設計”をそのまま置きます。
結論は **2段階（Stage1→Stage2）を最初からやる**のが一番安定です。125kをいきなり一発で回すと「全部空」寄りに崩れやすいので。

---

## 0) データ分割（ここだけは固定）

### Devは必ず2つ作る（90%空の罠回避）

* **Dev-All（本番分布）**：例）5,000件（空4,500 / 正例500）

  * 見る指標：**false positive率（空なのに出した割合）** と JSONパース成功率
* **Dev-Positive-only**：例）2,000件（正例のみ、25k側から抽出）

  * 見る指標：**recall（personsが空になってないか）**、act/住所/署名の取りこぼし

※N0しかない空だと境界は弱いので、Positive-onlyの監視が超重要です。

---

## 1) 学習は最初から2段階（ベースから連続で実行）

### Stage1（抽出力づくり）

* 使うデータ：**25kだけ（空50/正例50の混在）**
* 目的：正例の抽出・JSON安定・フィールド埋めの癖付け
* 推奨学習量：**3 epoch**

  * global batch=64想定 → 25,000/64 ≒ 391 step/epoch → **約1,170 step**

### Stage2（本番分布にキャリブレーション）

* 使うデータ：**125k（空90%）**
* 目的：N0で余計に出さない（FPを落とす）
* 推奨学習量：**0.25〜0.5 epoch**（短く！）

  * 125,000/64 ≒ 1,953 step/epoch
  * 0.25 epoch ≒ 490 step / 0.5 epoch ≒ 975 step

> N2なし＆空がN0のみなので、Stage2を長くやると「全部空」へ寄りやすいです。短く締めるのがコツ。

---

## 2) モデル/学習方式（A100×4ならこれが無難）

* **学習は BF16 LoRA（4bit学習しない）**
* 配布はあなたの案通り **LoRAマージ→GGUF→4bit量子化（Ollama）**でOK

---

## 3) Unsloth + LoRA の“基準点”設定（まずこれで回す）

### 重要：seq_len と packing

* `max_seq_length = 4096`（まず固定。足りなければ次に8192）
* **packing = False**（JSON抽出は混入が怖いので）

### LoRA

* target_modules：**Attention + MLP** を推奨
  `q_proj,k_proj,v_proj,o_proj, gate_proj,up_proj,down_proj`
* `r=64`, `lora_alpha=128`, `lora_dropout=0.05`

### 最重要：lossは“assistant出力だけ”

system/user（記事本文）にlossを掛けると、復唱癖やノイズ学習が増えます。
Unslothの「回答のみ学習」機能か、completion-onlyのcollatorで必ずマスクしてください。

---

## 4) バッチ・step設計（前に失敗しがちな所）

4GPUで **global batchを64前後**に固定すると扱いやすいです。

* 例A（堅い）：`per_device_batch=1`, `grad_accum=16`, GPU=4 → global=64
* 例B（速い）：`per_device_batch=2`, `grad_accum=8`, GPU=4 → global=64

学習率（LoRAの基準点）：

* Stage1：`lr=2e-4`
* Stage2：`lr=1e-4`（or 5e-5）

---

## 5) 実行プラン（そのまま運用できる）

### 手順

1. **Sanity overfit**（超重要）：正例100〜200件だけで200 step回して、lossが落ちて出力が近づくことを確認
   → ここで動かないなら、マスク/テンプレ/target_modulesが壊れてます
2. Stage1：25kを3 epoch
3. Stage2：125kを0.25〜0.5 epoch（短く）
4. LoRA保存 → マージ → 4bit量子化 → **Ollamaで Dev-All / Dev-Positive-only 評価**

---

## 6) “このまま行く”ときの注意（N0しかない空の弱点への対策）

N2なし・空がN0だけだと、Stage2後に **Positive-onlyのrecallが落ちやすい**です。落ちたら次の調整で戻せます：

* Stage2の **stepを減らす**（0.5→0.25 epoch）
* Stage2の **lrを下げる**（1e-4→5e-5）
* それでも落ちるなら、Stage2の学習データ内で **正例比率を10%→15%**に上げる
  （本番分布はDev-Allで担保できるので、trainは少しズラしてOK）

---

## 7) 配布（4bit量子化）で見るべきこと

* 評価は必ず **量子化後（Ollama）**でやる（BF16で良くても4bitで崩れます）
* まずは Q4_K_M 系の“無難な4bit”で1回見る → 速度/精度を見て調整

---

もし次に、あなたが使う **Unslothノートのセル構成（どこからどこまで触れるか）**に合わせて、
「Stage1→Stage2を同じスクリプトで回す」形の **具体パラメータ値（batch/accum/epochs/steps）**をそのまま貼れるように書き起こします。



はい、**Unsloth側の“基本推奨は lora_dropout=0”で合ってます**。理由は大きく2つで、

* **Unslothは lora_dropout=0 のとき内部最適化が効いて少し速い**
* さらに、短めのSFTでは **LoRA dropout が「安定した正則化」にならないことがある**（効いたり効かなかったり）ので、まず0で回すのが無難、ただし **過学習が見えるなら非0を推奨**…という立て付けです。 ([docs.unsloth.ai][1])

### あなたの条件だとどうするのが良い？

**結論：最初は 0 でOK**（むしろおすすめ）。理由は、いまの学習は「空90%にキャリブレーション」色が強く、dropoutでノイズを入れるより **出力の一貫性（JSON厳密・空の安定）**が重要だからです。

* **Stage1（抽出力づくり）**：`lora_dropout=0` を基準
* **Stage2（空90%で締め）**：`lora_dropout=0` 固定推奨（ここで0.05にすると“空に倒す/倒さない”が揺れやすい）

### じゃあいつ 0 以外を試す？

Unslothも「過学習が疑わしいなら non-zero」と言ってる通り、次の症状が出たら **Stage1だけ** `0.05` を試す価値があります。 ([docs.unsloth.ai][1])

* Dev-Positive-only で **学習が進むほど悪化**（典型的な過学習）
* 正例で “特定の言い回しだけ” に引っ張られる／同じ住所を貼りがち、など **ラベルノイズへの過適合**が強い

研究的にも「LoRAでも過学習しうる」「dropout系で一般化を改善しうる」という方向性はあります。 ([arXiv][2])

**実務のおすすめ試行順（最短）**

1. `lora_dropout=0` で Stage1→Stage2 を通す
2. もし過学習症状が出たら、**Stage1だけ** `0.05` にして再学習（Stage2は0のまま）
3. それでもダメなら dropoutより先に **Stage2のstep/LRを下げる**（空90%はここが効きやすい）

この順で行くと、ムダな探索が減ります。

[1]: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide?utm_source=chatgpt.com "LoRA fine-tuning Hyperparameters Guide - Unsloth"
[2]: https://arxiv.org/abs/2403.00812?utm_source=chatgpt.com "[2403.00812] LoRA Meets Dropout under a Unified Framework"

