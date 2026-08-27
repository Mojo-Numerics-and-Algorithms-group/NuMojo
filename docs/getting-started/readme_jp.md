# NuMojo

![logo](../../assets/numojo_logo_360x360.png)

NuMojoは、Python の NumPy、SciPy と同様の数値計算機能を Mojo 🔥 で提供するライブラリです。

**[ドキュメントを見る»](https://numojo.readthedocs.io)**  |  **[サンプルを見る»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo-Examples-and-Benchmarks/blob/main/docs/README.md)**  |  **[変更履歴»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/user-guide/changelog.md)**  |  **[Discordに参加»](https://discord.gg/NcnSH5n26F)**

**[中文·简»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/getting-started/readme_zhs.md)**  |  **[中文·繁»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/getting-started/readme_zht.md)**  |  **[English»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.md)** |  **[한국어»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/getting-started/readme_kr.md)**

**目次**

1. [プロジェクトについて](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#about-the-project)
2. [目標](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#goals)
3. [使用方法](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#usage)
4. [インストール方法](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#how-to-install)
5. [貢献について](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#contributing)
6. [注意事項](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#warnings)
7. [ライセンス](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#license)
8. [謝辞](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#acknowledgments)
9. [貢献者](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#Contributors)

## プロジェクトについて

NuMojoは、NumPy、SciPy、Scikit-learnなどのPythonパッケージにある幅広い数値計算機能の実現を目指しています。

***NuMojoとは***

私たちは、ベクトル化、並列化、GPUアクセラレーションを含む、Mojoの潜在能力を最大限に活用することを目指しています。現在、NuMojoは、標準ライブラリの数学関数のほぼすべてを配列入力に対応するように拡張しています。

NuMojoのビジョンは、機械学習の逆伝播システムの追加的な負荷なしに、高速な数学演算を必要とする他のMojoパッケージにとって不可欠な構成要素として機能することです。

***NuMojoでないもの***

NuMojoは機械学習ライブラリではなく、ベースライブラリの一部として逆伝播を含むことはありません。

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

利用可能なすべての関数とオブジェクトは[こちら](../user-guide/features.md)でご確認ください。最新のロードマップは[../user-guide/roadmap.md](../user-guide/roadmap.md)で管理されています。

詳細なロードマップについては、[../user-guide/roadmap.md](../user-guide/roadmap.md)ファイルを参照してください。

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
- **`main`ブランチ**：安定版を提供します。現在、Mojo 26.2に対応するNuMojo v0.9.0をサポートしています。それ以前のNuMojoバージョンを使用する場合は、方法2をご利用ください。
- **`pre-x.y`ブランチ**：最新のMojoバージョンをサポートする開発中のブランチです（現在はNuMojo v0.10.0、Mojo >=1.0.0, <1.1.0 が必要）。このブランチは頻繁に更新され、機能や構文に破壊的変更が生じる場合があることにご注意ください。

パッケージはPixi環境内で自動的に利用可能になり、VSCode LSPがインテリジェントなコードヒントを提供します。

### 方法2：Pixi（prefix.dev）経由での安定版インストール

ほとんどのユーザーには、互換性と再現性を保証するため、Pixi経由で安定版をインストールすることをお勧めします。

`pixi.toml`ファイルに以下を追加してください：

```toml
[workspace]
channels = ["https://repo.prefix.dev/modular-community"]

[dependencies]
numojo = "=0.9.0"
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

どのような貢献でも**大変感謝いたします**。ガイドライン（コーディングスタイル、テスト、ドキュメント、リリースサイクル）については、[CONTRIBUTING.md](CONTRIBUTING.md)をご覧ください。

## 注意事項

このライブラリはまだ初期段階にあり、マイナーバージョン間で破壊的変更が導入される可能性があります。本番環境や研究コードではバージョンを固定してください。

## ライセンス

LLVM例外付きApache 2.0ライセンスの下で配布されています。詳細については、[LICENSE](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE)およびLLVM [License](https://llvm.org/LICENSE.txt)をご覧ください。

このプロジェクトには、Apache License v2.0 with LLVM Exceptions（LLVM [License](https://llvm.org/LICENSE.txt)を参照）でライセンスされた[Mojo Standard Library](https://github.com/modularml/mojo)からのコードが含まれています。MAXとMojoの使用と配布は、[MAX & Mojo Community License](https://www.modular.com/legal/max-mojo-license)の下でライセンスされています。

## 謝辞

[Modular](https://github.com/modularml)によって作成されたネイティブ[Mojo](https://github.com/modularml/mojo)で構築されています。

## 貢献者

<a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Mojo-Numerics-and-Algorithms-group/NuMojo" />
</a>
