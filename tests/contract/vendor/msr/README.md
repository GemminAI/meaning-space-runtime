# msr/ — IC-02（MSR の計測）と calibration の契約

**IC-02 は 0.10.0**（D-132：窓の位置と長さを契約の項目にした。0.9.0＝D-129 は先頭 128 行の固定窓）。IC-02 の版は下の表の組で、各 payload の版は `$id` の版（IC-00 の `contract.version` とは別）。

| ファイル | 版 | 中身 |
|---|---|---|
| `geometry-measurement.schema.json` | **0.10.0** | 計測の payload（`mode: single` / `pair`）。`series` は系列の名前をキーにした整数の Blob（§3.3.2）。0.9.0 からの変更：`input_scope` の `window` が `rule`（`prefix_rows` / `rows`）・`start`（0 以上）・`length`（1 以上）を宣言する。`input_ref.length` は 128 固定でなく `scope.length`（Gate が確かめる）。`pair` のスキーマは変えていない。下の「窓」の節 |
| `trajectory-window.schema.json` | **0.10.0** | 窓の成果物の payload（`msr.trajectory-window`）。元の軌跡の行 `[start, start + length)` を別の成果物として持つ（D-132）。下の「窓」の節 |
| `calibration.schema.json` | **0.8.0**（変更なし） | calibration の payload。`exponent`（整数、必須）。下の「calibration の量子化の指数」の節 |
| `archive/0.6.1/`・`archive/0.6.2/`・`archive/0.7.0/`・`archive/0.7.1/`・`archive/0.8.0/`・`archive/0.9.0/` | — | 旧スキーマと記録の写し。**そのまま残す**（書き換えない。スキーマは `contract.lock` の `archived` に sha256）。`archive/0.8.0/` は IC-02 0.8.0 の計測のスキーマ（`$id` は 0.7.1、`input_scope` が無い）と、0.8.0 の README。0.7.1 の計測は旧版の規則のまま Gate が読む（`input_scope` を要求せず、足すと拒否。0.9.0 としては読まない）。`archive/0.9.0/` は IC-02 0.9.0 の計測・窓のスキーマ（`$id` は 0.9.0、先頭 128 行の固定）、0.9.0 の README、窓の見本 `vectors/window.json`（47 件、変更なし）。0.9.0 の成果物は 0.9.0 の規則のまま Gate が読む。0.7.0 の calibration も旧版の規則のまま（指数の欄が無く −30 固定。0.8.0 としては読まない） |

Blob の参照の型（`intBlob`・`floatBlob`）は `../common/blob.schema.json`（0.7.0）。`series` の各系列は `intBlob` への `$ref`。
`payload_schema` の `{id, version}` は `msr.geometry-measurement` `0.10.0`（旧 `0.9.0`・`0.7.1` も各版の規則で受理）、`msr.trajectory-window` `0.10.0`（旧 `0.9.0` も受理）、`msr.calibration` `0.8.0`（`$id` の版。IC-00 の `contract.version` とは別。設計書 §2.6、S39）。

定義の出典：meaning-space-runtime の `feature/msr-ic02`（`13d0a51`）の `src/msr/geometry/metrics.py` の冒頭。この README が**規範**で、コードの冒頭は正本にしない（設計書 §3.3.2）。

## 系列と指標

軌跡は `T` 点（`token_index` が増える順、S1）の `x_0 … x_{T-1}`（`D` 次元）。**`series[name][t]` は点 `0..t` から計算した値**（`t = 0..T-1`）。
最後の要素（`t = T-1`）は `payload.metrics` の同じ値と同じ整数（S37）。`ε`・`δ`・`w_min` は payload の `thresholds`（封筒の整数。距離は 2⁻³⁰ の刻みを実数に戻したもの）。
**値なし**は Blob の要素が −2^63。**下の表の 10 系列はすべて値なしを持ちうるので、`undefined: "int64_min"` を常に宣言する**（実際に値なしが無い計測でも宣言する。表に無い系列に宣言があれば拒否する。S36）。値なしは表の条件に当たる位置にだけ現れ、それ以外の位置が値なしなら誤り（S36）。逆に、条件に当たる位置は必ず値なし。

**変位の区間が無いとき（`T = 1`）**：`displacement.count` は 0、`displacement.mean`・`median`・`longest_sub_delta_run` は null（値なし）、変位の 3 系列の `t = 0`（= 最後の要素）は値なし。S37 は「最後の要素が値なし ⇔ `metrics` の値が null」でそのまま成り立つ。

| 系列の名前（キー） | `metrics` の場所 | 種類・単位（`unit`） | 定義 | 値なしになる位置 | `undefined` の宣言 |
|---|---|---|---|---|---|
| `displacement_mean` | `displacement.mean` | 実数・`distance` | 変位 `d_s = ‖x_s − x_{s−1}‖`（`s = 1..t`）の平均 | `t = 0`（変位が無い） | 常に |
| `displacement_median` | `displacement.median` | 実数・`distance` | 同じ `d_s` の中央値 | `t = 0` | 常に |
| `displacement_longest_sub_delta_run` | `displacement.longest_sub_delta_run` | 個数・`steps` | `d_s < δ` が続く最長の長さ（`s = 1..t`） | `δ` が null なら全位置。そうでなければ `t = 0` | 常に |
| `recurrence_pair_fraction` | `recurrence.pair_fraction` | 実数・`fraction` | 対 `i < j ≤ t`（`j − i > w_min`）のうち `‖x_i − x_j‖ < ε` の割合 | `ε` か `w_min` が null なら全位置。そうでなければ対が 1 つも無い位置、すなわち `t ≤ w_min` | 常に |
| `recurrence_return_count` | `recurrence.return_count` | 個数・`returns` | 「戻り」の数。`j ≤ t` で、ある過去の点 `i`（`j − i > w_min`）が `‖x_i − x_j‖ < ε` かつ `‖x_i − x_{j−1}‖ ≥ ε`（ε の球を出て戻った）となる `j` の数 | `ε` か `w_min` が null なら全位置。そうでなければ無し（0 から始まる） | 常に |
| `recurrence_return_rate` | `recurrence.return_rate` | 実数・`returns_per_point` | 戻りの数 ÷ `(t + 1)`（0〜1。1 点につき戻りは高々 1 回） | `recurrence_return_count` と同じ | 常に |
| `diversity_cell_entropy` | `diversity.cell_entropy` | 実数・`nats` | 点 `0..t` の符号帳のセル（最も近い中心）の占有のシャノン・エントロピー（自然対数） | 符号帳を使っていない（封筒の `parents` に役割 `codebook` が無い）なら全位置。そうでなければ無し | 常に |
| `diversity_participation_ratio` | `diversity.participation_ratio` | 実数・`dimensionless` | 点 `0..t` の共分散 `C`（点の数で割る）について `tr(C)² / tr(C²)`（`tr(C²)` が 0、つまり全点が一致なら 0） | `t = 0` | 常に |
| `persistence_longest_stay` | `persistence.longest_stay` | 個数・`points` | 点のセル番号が同じまま続く最長の長さ。セル番号は、符号帳があれば最も近い中心、なければ `ε` を半径とする先頭からの逐次のクラスタリング（点が既存のどの代表点とも `ε` 以上離れれば新しい代表点） | 符号帳が無く、かつ `ε` が null なら全位置。そうでなければ無し | 常に |
| `persistence_dominant_cell_share` | `persistence.dominant_cell_share` | 実数・`fraction` | 最も多くの点を持つセルの点数 ÷ `(t + 1)` | `persistence_longest_stay` と同じ | 常に |

- 実数の系列は Blob の `quantization` が `{quantum: {base: 2, exponent: −30}, rounding: ties-to-even, tolerance_quanta: 1}`。**個数の系列は `quantization: null`** で、値は 0 以上の整数そのもの（許容は 0）。
- 実数の系列の要素は `round_ties_to_even(値 × 2³⁰)`。範囲外・NaN・無限大は丸めずに計測を拒否する（§7.1）。
- 各 Blob は `shape: [T]`、`T = input_ref.length`（S35）。`uri` は `blob:sha256:<sha256>`（S34）、バイト列は int64・リトルエンディアン・C の順・ヘッダなし（S24）。
- 系列の `unit` と種類（実数・個数）は**スキーマが正本**で、上の表はその写し（`tests/contracts` が一致を確かめる）。
- 系列の名前の集合は 10 個で固定（スキーマの `additionalProperties: false`・すべて必須）。名前を足す・変えるのはスキーマの版を上げること。

## ペアの判定の式（設計書 §3.3.3、S38）

入力はすべて整数で、同じ単位（軌跡の空間の L2 距離）・同じ刻み（2⁻³⁰）：`d` ＝ `payload.distance.value`、`lo`・`hi` ＝ calibration の `values.noise_floor_lower`・`noise_floor_upper`。

| 条件 | `verdict.state` |
|---|---|
| `payload.calibration` が null | `UNCALIBRATED` |
| `d < lo` | `STAGNANT` |
| `lo ≤ d ≤ hi` | `NOISE_FLOOR_INTERNAL` |
| `d > hi` | `STRUCTURAL_SHIFT_DETECTED` |

- 量子化した後の整数で比べる（浮動小数で比べてから量子化しない）。境界（`d = lo`、`d = hi`）は `NOISE_FLOOR_INTERNAL`。
- calibration は `lo`・`hi` を必ず持つ（`lo ≤ hi` は S19）。noise floor を作れないときは calibration を作らず、比較は `UNCALIBRATED`。
- `x-common_prefix` の比較も同じ式（検査器は「拡張による判定」として区別して記録する）。
- 1 本の計測（`single`）の `verdict.state` は null（S3）。

## calibration の検証（0.7.1 の決定、Tomo 11-06 05:10）

- calibration の `content_id`・`artifact_id` を計算し直すには、親（`null_set` の各成果物、`codebook`）の `content_id` が要る。**親の `content_id` は HEKB から取得する**（取得の主体は nvs-runtime の Gate）。見つからない、または保存されている `content_id` と食い違うなら、検証は失敗する。
- 呼び出し側が任意に渡した親の情報（ファイルの写しに添えたもの、引数など）を、**無条件に信頼しない**。取得した親で `content_id`・`artifact_id` を計算し直し、封筒の値と一致することを確かめる。
- **ファイルだけで自己完結して検証できることは契約しない**。ファイルは写しで、正本は HEKB（設計書 §3.3.1）。
- 量子化の指数は calibration の payload の `exponent` に宣言する（次の節）。計測のスキーマの指数 −30 は定数のまま。

## calibration の量子化の指数（0.8.0 の仮決定、Tomo 11-07 15:29・15:30）

- payload に `exponent`（整数）を足した。**必須**。`values` の距離の欄（`noise_floor_lower`・`noise_floor_upper`・`background_mean`・`epsilon`・`delta`）の刻みは `2^exponent`。実数の値は `q × 2^exponent`（`math.ldexp`）。
- **−30 に固定しない**。許容範囲はここでは決めない（契約設計で後日）。スキーマにも Gate にも範囲の検査は無い。
- **Gate が検証するのは指数の有無と形式だけ**：欠落、文字列・小数（`-20.0` のような整数値の小数も）・真偽値・null は拒否（符号 `schema`）。
- **読む側は宣言された `exponent` で解釈する**（`epsilon` の復元など）。−30 を前提にしない。
- 分位点の水準（`method.quantiles`、確率）は 2⁻³⁰ の格子のままで、`exponent` の対象外。
- 計測（`geometry-measurement` 0.7.1 と 0.9.0）の `thresholds`・`distance` は 2⁻³⁰ のまま。刻みの違う calibration を黙って換算しない（MSR は拒否する）。Gate の `S38` は、整数どうしの比較が同じ刻みであることを前提にしており、刻みの照合は 0.8.0 では Gate に入れていない（指数の有無と形式だけ、という仮決定のため）。
- **0.7.x（指数の欄が無い）を 0.8.0 として暗黙に解釈しない**。`payload_schema` が 0.8.0 で `exponent` が無ければ拒否。0.7.0 は旧版の規則のまま（欄を足すと拒否）。旧データを 0.8.0 へ移す方法は別の決定で、ここでは実装しない。
- D-97 (5) の原文は変えていない。置き換える決定は Tomo が別に記録する。
- 見本は `vectors/calibration-exponent.json`（受理 3・拒否 8）。

## 窓（IC-02 0.10.0、D-132。0.9.0 は D-129）

決定（D-129）：窓は**別の成果物**で、元の軌跡との対応は `parents` の親参照。`common_prefix` で全長を計測する案は採らない（D-120）。
決定（D-132）：**窓の位置と長さを契約の項目にする**（先頭 128 行の固定をやめる）。0.9.0（先頭 128 行の固定窓）の文章・スキーマ・見本は `archive/0.9.0/` に書き換えずに移した。Gate は **0.9.0 と 0.10.0 の両方を受け付ける**（各版は自分の規則で読み、互いに読み替えない）。

### 窓の規則（0.10.0）

| 項目 | 内容 |
|---|---|
| `window.rule` | `rows`（`start` 行目から `length` 行、連続）、または `prefix_rows`（`start = 0` の特別な場合。0.9.0 の窓の表現を残したもの） |
| `window.start` | 整数、0 以上（`prefix_rows` は 0 のみ） |
| `window.length` | 整数、1 以上。**128 に固定しない**(`prefix_rows` も任意の長さ。D-133 (2))。位置ごとに変えてもよい |
| 位置の候補 | 契約は**狭めない**（先頭・中ほど・末尾など、`start`・`length` が範囲内なら何でもよい。D-132 (5)） |
| `policy.name` | 空でない文字列（`policy.sha256` は名前の UTF-8 の sha256）。作り方の名前の例：`first-128-rows/1`、`rows-86+128/1`。形式は自由な文字列のまま(D-133 (3)) |

- `prefix_rows`（`start = 0`）と `rows`（`start = 0`）は**同じ行を指す**。`window.rule` の宣言だけが違い、Gate の判定は同じ（見本 `accept-equivalence-*`）。
- 0.9.0 の窓の成果物（`rule: prefix_rows`、`start: 0`、`length: 128`）は 0.9.0 の版のまま読める。0.10.0 の窓として読み替えない。
- `origin.length` は 1 以上（0.9.0 は 128 以上）。窓が元より長い（`start + length > origin.length`）ことは Gate が拒否する。

### 窓の成果物（`msr.trajectory-window` 0.10.0）

0.9.0 と同じ表のうち、次を変える。他（`kind`・`producer.name`・`parents`・`origin` の読み合わせ・`content_sha256` と Blob の sha256・有限性）は 0.9.0 のまま。

| 項目 | 内容 |
|---|---|
| 封筒の `space_id` | null でない。**窓の空間は位置で分ける**（D-132 (4)）。校正の空間は `row_rule` に位置（`start`・`length`）を含めて分けるので、位置ごとの校正は**別の空間**。同じ `space_id` を、位置（`start`・`length`）の違う窓が使うと拒否する（`window_space_position`。D-133 (4)：`(start, length)` を比べて拒否する）。計測の `space_id` は窓のものと同じ |
| `payload.window` | `rule`・`start`・`length`・`content_sha256`（窓の中身の sha256）。**元の行 `[start, start + length)` を指す** |
| `payload.points` | 窓の `length` 行の Blob（binary32・LE・行優先・量子化なし、`shape: [length, dimension]`）。`content_sha256` ＝ `points.sha256`。行数が `window.length` と違えば `window_shape` |
| `payload.token_index` | 窓の `length` 行のトークン番号（int64、`shape: [length]`、厳密に増える） |

- **Gate が確かめるもの（0.10.0 で足したもの）**：`start + length <= origin.length`（`window_range`）、`token_index` の各値が `[0, origin.length)` に入ること（`window_token_range`。D-133 (1)：`[0, origin.length)` の外は拒否。トークン番号と行番号の対応は決めていない）、空間と位置の対応（`window_space_position`）。ほかは 0.9.0 の検査（行数は固定の 128 でなく `window.length`）。
- **Gate が確かめられないもの**：窓が元の RAW の行 `[start, start + length)` そのものであること（RAW は参照で持つため）。RAW の行を持つ読み手は `rows_match`（`tools/contracts/ic02_window.py`。先頭 128 行の 0.9.0 は `prefix_matches`）で確かめる。
- 符号の一覧は `tools/contracts/ic02_window.py` の冒頭。

### 計測（`msr.geometry-measurement` 0.10.0、`mode: single`）

`input_scope`（必須）：

| `input_scope` | 意味 | 親の役割 `trajectory` |
|---|---|---|
| `{"kind": "window", "rule", "start", "length", "source_trajectory_id", "content_sha256"}` | 窓の成果物を入力にした計測。`rule`・`start`・`length` は窓の成果物と同じ値（整数、`start >= 0`、`length >= 1`、`prefix_rows` は `start = 0`）。`input_ref.length == scope.length` | 窓の成果物 1 件（0.9.0 の窓、つまり `start = 0` の `prefix_rows` でもよい。D-133 (5)） |
| `{"kind": "full"}` | 窓なしの全長の入力（0.9.0 と同じ） | 窓でない軌跡の成果物 1 件 |

`check_scope` は 0.9.0 の検査に加えて、`rule` の一致（`window_mismatch`）と、`input_ref.length == scope.length`（0.9.0 はスキーマの定数 128 が保証していたもの。0.10.0 は Gate が `window_length` で言う）を確かめる。

### ペア（`mode: pair`、0.10.0）

- **長さの異なる窓どうしは `equal_only` で比べられない**（`common_prefix` は採らない。D-120）。`left`・`right` の親が保存された窓で、`window.length` が違えば `pair_window_length` で拒否する。位置（`start`）の違う窓の比較（`equal_only` は長さだけを見る）は**未規定のまま**（D-133 (6)）。窓位置の実験では同じ位置どうしだけを比べる。
- 親が窓でない、または保存されていない場合は 0.9.0 と同じく、ここでは判定しない。
- `pair` に `input_scope` を足すことは今回もしない。

### 確定（D-133、11-08 20:16）と未規定

D-133 で次の 5 点を、実装の挙動のとおり確定した（見本は `vectors/window-rows.json`、各件の `derives_from` に D-133 の番号）。
1. `token_index` の範囲は `[0, origin.length)` の外を拒否（`window_token_range`）。窓が元より長い場合は `window_range` と `window_token_range` の両方が出る。
2. `prefix_rows` の長さは 128 に固定せず、任意の `length`（1 以上）を許す。
3. `policy.name` の形式は自由な文字列のまま。
4. 同じ `space_id` を位置の違う窓が使うと、`(start, length)` を比べて拒否（`window_space_position`）。
5. 0.10.0 の計測は 0.9.0 の窓（`start = 0` の `prefix_rows`）を親にできる。

**未規定のまま残すもの**（`vectors/window-rows-undetermined.json`、lock・凍結の対象外）：
- (6) 位置の違う窓どうしの比較（`equal_only` は長さだけを見る）。窓位置の実験では同じ位置どうしだけを比べる。
- トークン番号と行番号の対応（窓の行が元の何番目のトークンか）。範囲の内側なら拒否しない、までが決まっている。

### 今回の変更ではないもの

- `pair` への `input_scope`、窓の Blob の `common/float32-blob`（草案）への寄せ。
- MSR・nvs-runtime の実装（別の PR）。
- 窓の位置が生成全体のループを反映するかは未確認（D-129）。位置の比較は実験で確かめる。

## 検査の責任

- **完全なスキーマ検査は Gate が保存前に行う**（設計書 §5）。
- 成果物を**読む側**は、(1) `content_id`・`artifact_id` の再計算、(2) 使う欄の型と範囲、(3) 契約の版（知らない版は拒否、S39）、(4) 親の参照の整合、を検証する。Gate を経ていない成果物（ファイルの写しなど）を無条件に信頼しない。

## 親の参照整合（calibration の検証の中身、D-97 (2)）

calibration を**読む側**が、取得した親（`parent_contents`：親の `artifact_id` → 保存されている `content_id`）に対して確かめる 5 点。**1 つでも違えば検証失敗**で、その calibration は使わない。

| # | 規範 | 見本の符号 |
|---|---|---|
| 1 | 封筒の `parents` の各親（`artifact_ids` のすべて）が、取得・確認済みの親（`parent_contents`）の中にある。さらに、`parent_contents` の各 `content_id` は `cnt:` ＋ 64 文字の小文字の 16 進の形式である（形式違反はハッシュの不一致としてではなく、専用の符号で拒否する） | `parent_unknown`、`parent_content_id_form` |
| 2 | codebook の `content_id`（取得した親の）が、payload の `values.codebook_ref.content_id` と一致する。`codebook_ref` が null なら対象外 | `codebook_content_mismatch` |
| 3 | payload が参照する役割の集合（`null_set` は常に、`codebook` は `values.codebook_ref` が null でないときだけ）と、封筒の `parents` の役割の集合が一致する。役割が足りなくても余っても不一致。**`parents` の中に同じ役割が 2 つあるものは拒否する** | `role_set_mismatch`、`role_duplicate` |
| 4 | 役割 `null_set` と役割 `codebook` の `ordered` が、どちらも `false` である | `null_set_ordered`、`codebook_ordered` |
| 5 | payload の `null_set`（整列済みの `artifact_id` の列）が、役割 `null_set` の `artifact_ids` と一致する（順序も含めて同じ列） | `null_set_mismatch` |

- 違反は検証失敗である。補って読み進めない（推測で親を足さない、役割を直さない）。
- **取得と確認の主体は nvs-runtime（Gate）**：`parent_contents` は、Gate が HEKB から取得し、保存されている成果物と照合して確かめたものだけを渡す。**MSR は HEKB を呼ばない**。呼び出し側が任意に渡した親の情報は、上の 5 点を満たしても無条件には信頼しない（`content_id`・`artifact_id` の計算し直しが通ることが、さらに要る）。
- 見本は `vectors/calibration-parents.json`（0.8.0 の payload。受理 2・拒否 9。5 点それぞれの拒否と、`cnt:` 形式・役割の重複・codebook の `ordered` を含む）。検査器は `tools/contracts/ic02.py` の `check_calibration_parents`。

## 検査器（`tools/contracts`）での扱い

`blob.py`（S34・S24/S25・S36 の宣言・個数の負の値）、`ic02.py`（S35・S36 の位置・S37・S38、calibration の親の参照整合）、`ic02_window.py`（窓の成果物と `input_scope`、0.9.0 と 0.10.0）、`gate.py`（S39 と payload のスキーマ）。
Blob の見本は `../common/vectors/blob/blob-refs.json`、系列（T=1・T=2、δ なし・あり）の見本は `vectors/series.json`。
