# NuMojo

![logo](../../assets/numojo_logo_360x360.png)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)
[![Mojo](https://img.shields.io/badge/mojo-%3E%3D1.0.0-orange.svg)](https://www.modular.com/mojo)

NuMojoは、Python の NumPy と同様の数値計算機能を Mojo 🔥 で提供するライブラリです。

**[サンプルを見る»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo-Examples-and-Benchmarks/blob/main/docs/README.md)**  |  **[変更履歴»](../user-guide/changelog.md)**  |  **[Discordに参加»](https://discord.gg/NcnSH5n26F)**

**[中文·简»](./readme_zhs.md)**  |  **[中文·繁»](./readme_zht.md)**  |  **[English»](../../README.MD)**  |  **[한국어»](./readme_kr.md)**

**目次**

1. [プロジェクトについて](#プロジェクトについて)
2. [なぜNuMojoか](#なぜnumojoか)
3. [機能と目標](#機能と目標)
4. [使用方法](#使用方法)
5. [インストール方法](#インストール方法)
6. [貢献について](#貢献について)
7. [注意事項](#注意事項)
8. [ライセンス](#ライセンス)
9. [謝辞](#謝辞)
10. [貢献者](#貢献者)

## プロジェクトについて

NuMojoは、NumPyにある幅広い数値計算機能を実現することを目指しています。

私たちは、ベクトル化、並列化、GPUアクセラレーションを含む、Mojoの潜在能力を最大限に活用することを目指しています。現在、NuMojoは、標準ライブラリの数学関数のほぼすべてを配列入力に対応するように拡張しています。

NuMojoのビジョンは、機械学習の逆伝播システムの追加的な負荷なしに、高速な数学演算を必要とする他のMojoライブラリにとって不可欠な構成要素として機能することです。

## なぜNuMojoか

- **Mojoネイティブ。** NuMojoの`NDArray`は、NumPyやMAXのテンソル型をラップしたものではなく、Mojoネイティブな SIMD ベースの型です。そのため、Pythonとの相互運用によるオーバーヘッドなしにプログラムへコンパイルされます。
- **NumPyに親しんだAPI。** スライス、ブロードキャスト、行列積のための`@`演算子、関数名などは、理にかなう範囲でNumPyに合わせているため、これまでの感覚をそのまま活かせます。
- **Mojoの強みを活かす設計。** 各ルーチンではベクトル化と並列化が全面的に活用されており、Mojo自体のデバイスサポートが成熟するのに合わせて、GPUなどのアクセラレータ対応（`AcceleratorNDArray`）も実装されていく予定です。

## 機能と目標

私たちの主な目的は、Mojoで高速で包括的な数値計算ライブラリを開発することです。以下に、いくつかの機能と長期的な目標を示します。一部はすでに（完全または部分的に）実装されています。

コアデータ型：

- ネイティブn次元配列（`numojo.NDArray`）
- ネイティブn次元複素数配列（`numojo.ComplexNDArray`）
- ネイティブ固定次元配列（トレイトパラメータ化が利用可能になったときに実装予定）

ルーチンとオブジェクト：

- 配列作成ルーチン（`numojo.creation`）
- 配列操作ルーチン（`numojo.manipulation`）
- 入力と出力（`numojo.io`）
- 線形代数（`numojo.linalg`）
- 論理関数（`numojo.logic`）
- 数学関数（`numojo.math`）
- 指数と対数（`numojo.exponents`）
- 極値の発見（`numojo.extrema`）
- 丸め（`numojo.rounding`）
- 三角関数（`numojo.trig`）
- ランダムサンプリング（`numojo.random`）
- ソートと検索（`numojo.sorting`、`numojo.searching`）
- 統計（`numojo.statistics`）
- その他...

利用可能なすべての関数とオブジェクトは[こちら](../user-guide/features.md)でご確認ください。最新のロードマップは[docs/user-guide/roadmap.md](../user-guide/roadmap.md)で管理されています。

## 使用方法

実行可能なサンプルは`examples/`ディレクトリにあります（例：`examples/quickstart.mojo`）。

n次元配列（`NDArray`型）の例は以下の通りです。

```mojo
import numojo as nm
from numojo.prelude import *


def main() raises:
    # ランダムなfloat64値で2つの1000x1000行列を生成
    var A = nm.random.randn(Shape(1000, 1000)) # Shapeはnumojoにおいて形状に関するすべての操作に使用されます。
    var B = nm.random.randn(Shape(1000, 1000))

    # 文字列表現から3x2行列を生成
    var X = nm.fromstring[f32]("[[1.1, -0.32, 1], [0.1, -3, 2.124]]")

    # 配列を出力
    print(A)

    # 配列の乗算
    var C = A @ B

    # 配列の逆行列
    var I = nm.inv(A)

    # 配列のスライス
    var A_slice = A[1:3, 4:19]

    # 配列からスカラーを取得
    var A_item = A[Item(291, 141)] # Item()はnumojoにおいてndarrayの座標を定義するために使用されます。
    var A_item_2 = A.item(291, 141)

    # 軸に沿ってソートとargsort
    print(nm.sort(A, axis=1))
    print(nm.argsort(A, axis=0))

    # 軸に沿って合計
    print(nm.sum(A))
    print(nm.sum(A, axis=1))

    # 線形システムを解く
    print(nm.solve(A, B))
```

`ComplexNDArray`の例は以下の通りです：

```mojo
import numojo as nm
from numojo.prelude import *


def main() raises:
    # 複素数スカラー 5 + 5j を作成
    # cf32はnumojoにおいて複素数型を識別するために使用される、f32（DType.float32）の複素数版です。
    var complexscalar = CScalar[cf32](5) # ComplexSIMD[cf32](5, 5) と同等
    # 単純に 5 + 5*`1j` と定義することもできます！
  
    # 複素数配列を作成
    var A = nm.full[cf32](Shape(1000, 1000), fill_value=complexscalar)  # (5+5j)で埋める
    var B = nm.ones[cf32](Shape(1000, 1000))                            # (1+1j)で埋める

    # 配列を出力
    print(A)

    # 配列のスライス
    var A_slice = A[1:3, 4:19]

    # 配列の乗算
    var C = A * B

    # 配列からスカラーを取得
    var A_item = A[Item(291, 141)]
    # 配列の要素を設定
    A[item(291, 141)] = complexscalar
```

## インストール方法

NuMojoは、さまざまな開発ニーズに対応できるよう複数のインストール方法を提供しています。ワークフローに最も適した方法をお選びください。

### 方法1：pixi-build-mojoを用いたGitインストール（推奨）

GitHubリポジトリから直接NuMojoをインストールすることで、安定版と最新機能の両方にアクセスできます。この方法は、最新の機能を使いたい開発者や、最新の安定版を利用したい開発者に最適です。

既存の`pixi.toml`に以下を追加してください：

```toml
[workspace]
preview = ["pixi-build"]

[package]
name = "your_project_name"
version = "0.1.0"

[package.build]
backend = {name = "pixi-build-mojo", version = "0.*"}

[package.build.config.pkg]
name = "your_package_name"

[package.host-dependencies]
mojo = "==1.0.0"
max-core = "==26.5.0"

[package.build-dependencies]
mojo = "==1.0.0"
max-core = "==26.5.0"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}

[package.run-dependencies]
mojo = "==1.0.0"
max-core = "==26.5.0"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}

[dependencies]
mojo = ">=1.0.0, <1.1.0"
max-core = ">=26.5.0,<27"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}
```

次に、以下を実行します：
```bash
pixi install
```

**ブランチの選択：**
- **`main`ブランチ**：最新の安定版を提供します。現在はNuMojo v0.10.0で、Mojo >=1.0.0, <1.1.0 に対応しています。それ以前のNuMojoバージョンを使用する場合は、方法2をご利用ください。
- **`pre-x.y`ブランチ**：次のリリースに向けた開発中のブランチです。このブランチは頻繁に更新され、機能や構文に破壊的変更が生じる場合があることにご注意ください。

パッケージはPixi環境内で自動的に利用可能になり、VSCode LSPがインテリジェントなコードヒントを提供します。

### 方法2：Pixi（prefix.dev）経由での安定版インストール

ほとんどのユーザーには、互換性と再現性を保証するため、Pixi経由で安定版をインストールすることをお勧めします。

`pixi.toml`ファイルに以下を追加してください：

```toml
[workspace]
channels = ["https://repo.prefix.dev/modular-community"]

[dependencies]
numojo = "=0.10.0"
```

次に、以下を実行します：
```bash
pixi install
```

**バージョン互換性：**

| NuMojoバージョン | 必要なMojoバージョン |
| --------------- | -------------------- |
| v0.10.0         | ==1.0.0               |
| v0.9.0          | ==26.2                |
| v0.8.0          | ==25.7                |
| v0.7.0          | ==25.3                |
| v0.6.1          | ==25.2                |
| v0.6.0          | ==25.2                |

### 方法3：スタンドアロンパッケージのビルド

この方法では、複数のプロジェクトで使い回せる可搬性の高い`numojo.mojopkg`ファイルを作成します。オフライン開発やhermetic buildに最適です。

1. リポジトリをクローンします：
   ```bash
   git clone https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo.git
   cd NuMojo
   ```

2. パッケージをビルドします：
   ```bash
   pixi run package
   ```

3. `numojo.mojopkg`をプロジェクトディレクトリにコピーするか、その親ディレクトリをインクルードパスに追加します。

### 方法4：ソースコードを直接統合する

開発中にNuMojoのソースコードを変更できるようにするなど、最大限の柔軟性を求める場合：

1. リポジトリを任意の場所にクローンします：
   ```bash
   git clone https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo.git
   ```

2. コードをコンパイルする際に、NuMojoのソースパスを含めます：
   ```bash
   mojo run -I "/path/to/NuMojo" your_program.mojo
   ```

3. **VSCode LSPの設定**（コードヒントと自動補完のため）：
   - VSCodeの環境設定を開きます
   - `Mojo › Lsp: Include Dirs`に移動します
   - `Add Item`をクリックし、NuMojoディレクトリへのフルパスを入力します（例：`/Users/YourName/Projects/NuMojo`）
   - Mojo LSPサーバーを再起動します

設定後、VSCodeはNuMojoの関数に対してインテリジェントなコード補完とヒントを提供します！

## 貢献について

どのような貢献でも**大変感謝いたします**。NuMojoはまだ発展途上のプロジェクトであり、貢献できる余地がたくさんあります。ルーチンの実装、テストの作成、ドキュメントの改善はもちろん、気になった点があれば issue を立てるだけでも大歓迎です。

貢献者向けのクイックスタート：

```bash
git clone https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo.git
cd NuMojo
pixi install
pixi run final   # フォーマット + テスト
```

コーディングスタイルやディレクトリ構成、PRの進め方などの詳細なガイドラインは[docs/developer-guide/contributing.md](../developer-guide/contributing.md)を、docstring やフォーマットの規約については[docs/developer-guide/style-guide.md](../developer-guide/style-guide.md)を、PRを開く前に実行すべき内容については[docs/developer-guide/pre-pr-checks.md](../developer-guide/pre-pr-checks.md)をご覧ください。

## 注意事項

このライブラリはまだ初期段階にあり、マイナーバージョン間で破壊的変更が導入される可能性があります。本番環境や研究コードではバージョンを固定してください。

## ライセンス

LLVM例外付きApache 2.0ライセンスの下で配布されています。詳細については、[LICENSE](../../LICENSE)およびLLVM [License](https://llvm.org/LICENSE.txt)をご覧ください。

このプロジェクトには、Apache License v2.0 with LLVM Exceptions（LLVM [License](https://llvm.org/LICENSE.txt)を参照）でライセンスされた[Mojo Standard Library](https://github.com/modularml/mojo)からのコードが含まれています。MAXとMojoの使用と配布は、[MAX & Mojo Community License](https://www.modular.com/legal/max-mojo-license)の下でライセンスされています。

## 謝辞

[Modular](https://github.com/modularml)によって作成されたネイティブ[Mojo](https://github.com/modularml/mojo)で構築されています。

## 貢献者

<a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Mojo-Numerics-and-Algorithms-group/NuMojo" />
</a>
