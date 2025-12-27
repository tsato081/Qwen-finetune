# リポジトリの説明
本リポジトリは、日本語のニュース/事件記事から「人物」と「住所」を構造化抽出するためのプロンプト（および将来的なLoRA/QLoRA/蒸留等の学習）を管理します。

## タスク概要（運用の前提）
本システムでは1回の抽出で完結せず、以下の **2タスク** に分割して実行します。

1. 人物抽出・分類（記事内の全人物を列挙し、加害者かどうか等を判定）: `structure_extract_person_prompt.py`
2. 住所抽出（会社本社/人物居住地/物理的な犯行地点の住所のみを抽出）: `structure_extract_address_prompt.py`

## 1) 人物抽出・分類（`structure_extract_person_prompt.py`）
### 目的
記事に登場する **全人物** を抽出し、実在性・加害者性・属性（年齢/役職/所属）・分類・行為（加害者のみ）を付与します。

### 入力（想定）
- `posted_at`: 投稿日時（年齢が生年表記の場合の計算に利用）
- `article`: 記事本文
- `persons`: 事前に得られた人物候補（ヒント、名前の表記ゆれ補助）

### 出力（JSON）
`{"persons": [...]}` 形式で、`persons` は以下のオブジェクト配列です。
- `name`（必須）: 人物のフルネーム（姓名両方が揃うもの）
- `is_real`（必須）: 実在人物なら `true`、架空・偽名なら `false`
- `is_offender`（必須）: 加害者/容疑者なら `true`、被害者等は `false`
- `age`: 記事掲載時の年齢（数字のみ、例 `"45"`）。不明時は `null`（または空相当）
- `position`: 役職名（例 `"社長"`）。不明は `null`
- `affiliation`: 所属組織名。詐称は空相当。不明は `null`
- `category`: `"法令違反" | "PEPS" | "破産" | "その他"` のいずれか。不明は `null`
- `act`: 加害者のみ（記事記載どおりの法的名称、複数はカンマ区切り）。非加害者は `null`
- `reason`（必須）: 判定根拠となる記事内の記述

### 期待される出力例
入力（要旨）: `東京のIT企業ABC社の山田太郎社長（45）が、詐欺容疑で逮捕された。`
出力（JSON）:
```json
{
  "persons": [
    {
      "name": "山田太郎",
      "is_real": true,
      "is_offender": true,
      "age": "45",
      "position": "社長",
      "affiliation": "ABC社",
      "category": "法令違反",
      "act": "詐欺",
      "reason": "「山田太郎社長（45）が、詐欺容疑で逮捕」から容疑者として特定、詐欺容疑での逮捕により法令違反に分類"
    }
  ]
}
```

## 2) 住所抽出（`structure_extract_address_prompt.py`）
### 目的
記事に **明示された住所のみ** を抽出します（推測禁止）。

### 入力（想定）
- `article`: 記事本文
- `company_names`: 会社名ターゲット（owner名としてそのまま採用）
- `person_names`: 人名ターゲット（owner名としてそのまま採用）

### 出力（JSON）
以下3つを同時に返します。
- `headquarters_locations`: `[{ "address": <本社住所>, "owner": <会社名> }]`
- `residences`: `[{ "address": <居住地住所>, "owner": <人名> }]`
- `physical_crime_occured_locations`: `["<対面で直接行われた犯行地点の住所>", ...]`

### 重要ルール（抜粋）
- 記事に書かれていない住所は **必ず空文字 `""`**（補完・推測しない）
- 住所は「イベント場所」「行政管轄」「会社所在地（本社以外）」「勤務先」を混ぜない
- 人物住所は原則 `"氏名(年齢)=住所"` / `"氏名(年齢)(住所)"` のような **明示パターンのみ** を採用（曖昧なら `""`）
- `physical_crime_occured_locations` は「対面で直接の犯行」があった場合のみ（電話詐欺の被害者宅は含めない）

### 期待される出力例
入力（要旨）:
`岐阜県に住む８０代の女性の自宅に息子を名乗ってうその電話をかけ... 逮捕されたのは、住所不定、職業不詳の伊藤光 容疑者（２４）です。`
出力（JSON）:
```json
{
  "headquarters_locations": [],
  "residences": [
    { "address": "", "owner": "伊藤光" },
    { "address": "岐阜県", "owner": "８０代の女性" }
  ],
  "physical_crime_occured_locations": []
}
```

## パイプライン（例）
1. 人物抽出を実行し、`persons[].name` を `person_names` として渡せる形に整形
2. 住所抽出を実行し、人物/会社の住所と物理犯行地点を回収
3. 下流で必要に応じて「加害者のみの一覧」等へ整形（このリポジトリのコアは 1) と 2) の抽出）

## 評価で再現すべき「後処理」（学習後に必要）
学習（プロンプトSFT等）では必須ではない場合が多い一方、**最終評価（CSV正解データとの一致判定）では運用時と同等の後処理を再現する必要があります**。

- 人物抽出結果の整形: 記事内出現確認・実在性チェック、年齢/役職/所属のノイズ除去、住所は空で初期化
- 加害者の確定: `is_offender==true` かつ `act` ありのみを「加害者」として採用（それ以外は非加害者）
- 企業トピックの棄却: 記事が企業トピック色の場合、記事ごと除外（=予測を空扱い）されるケースがある
- 住所抽出の検証・補完: 形式チェック → 都道府県等の補完 → 本文一致確認 → owner一致時のみ反映（曖昧なら `""`）
- 犯行場所（`occured_locations`）: 形式/本文一致した住所のみ採用。**加害者が1人もいない場合は空にリセット**
- 警察署名（`police_departments`）: 2プロンプト外のNER等から抽出・補完される前提（評価でも同等の入力/処理が必要）

## 旧仕様（8フィールド犯罪者JSON）との違い
- 旧仕様は「犯罪者がいる場合のみ、8フィールドの配列」を直接出力していました。
- 現仕様は「人物抽出」と「住所抽出」に分割し、人物側は **加害者以外も含めて列挙** します。
- `police_department` の抽出は、現行の2プロンプトには含まれていません（必要なら別タスク追加/拡張が必要です）。

## CSV正解データ（Hawks_...csv）評価への最小マッピング（端的）
正解データは `data/test/Hawks_正解データマスター - ver 5.0 csv出力用.csv`（1518件）です。
本リポジトリの評価は **このCSVの8フィールドに必ず合わせます**。そのため、2タスク出力から「加害者のみ」を取り出して1つの配列に整形し、未記載値は `""` に正規化します（`null` は `""` に変換）。

- 犯罪者の有無: `persons[].is_offender == true` が0件なら `[]`
- 複数犯罪者: `is_offender==true` の人物を記事出現順（または抽出順）で並べる
- CSVカラム対応（例）:
  - `person_name` → `persons[i].name`
  - `person_age_at_published` → `persons[i].age`（無ければ `""` 扱いに寄せるなど、評価側で統一）
  - `person_position_name` → `persons[i].position`
  - `person_belongs_company` → `persons[i].affiliation`
  - `person_address` → 住所抽出の `residences` から `owner==persons[i].name` を引いた `address`（無ければ `""`）
  - `occured_locations` → `physical_crime_occured_locations`（該当無ければ `""`）
  - `person_violated_law` → `persons[i].act`
  - `police_departments` →（現2タスクでは未抽出。必要なら別抽出タスクを追加）




## axolotlのconfigを書く際
最新版の公式referenceはdocs/axolotl_config_reference.ymlに記載されているので参考にすること。
