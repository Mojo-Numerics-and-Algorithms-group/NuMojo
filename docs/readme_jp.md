<a name="readme-top"></a>
<!-- add these later -->
<!-- [![MIT License][license-shield]][] -->

<div align="center">
  <a href="">
    <img src="../assets/numojo_logo.png" alt="Logo" width="350" height="350">
  </a>

  <h1 align="center" style="font-size: 3em; color: white; font-family: 'Avenir'; text-shadow: 1px 1px orange;">NuMojo</h1>

  <p align="center">
    NuMojoは、PythonのNumPyやSciPyに似たMojo🔥で数値計算を行うためのライブラリです.
    <br />
    <!-- when we create docs -->
    <a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo-Examples-and-Benchmarks/blob/main/docs/README.md"><strong>ドキュメントを読む» </strong></a>
    <br>
    <a href="https://discord.com/channels/1149778565756366939/1149778566603620455"><strong>
    Discord チャンネルに参加する» </strong></a>
    <br />
    <!-- <br /> -->
    <!-- <a href="">View Demo</a>
    ·
    <a href="">Report Bug</a>
    ·
    <a href="">Request Feature</a> -->
  </p>
</div>

<!-- <details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#what-numojo-is"> What NuMojo is </a></li>
        <li><a href="#what-numojo-is-not">What NuMojo is not</a></li>
      </ul>
    </li>
    <a href="#goals-roadmap">Goals/Roadmap</a>
      <ul>
        <li><a href="#long-term-goals">Long term goals</a></li>
      </ul>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#how-to-install">How to install</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#warnings">Warnings</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details> -->

## プロジェクトについて

### NuMojoとは

NuMojoは、PythonのNumPy、SciPyとScikit に存在する幅広い数値機能を取り込むことを目的としています。

ベクトル化、並列化、GPUアクセラレーション（利用可能になった場合）など、Mojoの機能を最大限に活用することを試みています。現在、NuMojoは、配列入力で動作するようにスタンダードライブラリの数学関数を（ほとんど）拡張しています。

NuMojoは、MLのバックとフォワード伝搬システムの負荷なしに高速な計算を必要とする他のMojoパッケージのためのビルディングブロックになることを意図している

注意：NuMojoは機械学習ライブラリではなく、コアライブラリに機械学習アルゴリズムが含まれることはありません。

## 目標

詳細なロードマップについては、[roadmap.md](roadmap.md)(英語)ファイルを参照してください。

私たちの主な目標は、Mojoに高速で包括的な数値計算ライブラリを実装することです。以下はNuMojoの長期目標です、

###　長期目標

* 線形代数
  * ネイティブの n 次元配列
  * ベクトル化、並列化された数学演算
  * 配列操作 - vstack、スライス、連結など
* 微積分
  * 積分と微分など
* オプティマイザ
* 関数近似
* 並べ替え

## 使い方

以下にコード例を示します、

```mojo
import numojo as nm

fn main() raises:
    # ランダムな float64 値を使用して 2 つの 1000x1000 行列を生成する。
    var A = nm.NDArray[nm.f64](shape=List[Int](1000,1000), random=True)
    var B = nm.NDArray[nm.f64](1000,1000, random=True)

    # A*B
    print(nm.linalg.matmul_parallelized(A, B))
```

利用可能なすべての機能は[ここ](features.md)で見つけてください 

## インストール方法

NuMojoパッケージをインストールして利用するには２つの方法があります。

### パッケージのビルド方法

このアプローチでは、スタンドアロンパッケージファイル `mojopkg` をビルドする。

1. リポジトリをクローンする。
2. `mojo pacakge numojo` を使用してパッケージをビルドする。
3. numojo.mojopkg をあなたのコードを含むディレクトリに移動する。

### コンパイラとLSPにNuMojoのパスを含める。

この方法では、パッケージファイルを作成する必要はありません。コードをコンパイルするときに、以下のコマンドでNuMojoリポジトリのパスをインクルードできます：

```console
mojo run -I "../NuMojo" example.mojo
```

これは、コードをテストするときにNuMojoソースファイルを編集できるので、より柔軟です。

VSCode LSPがインポートされた `numojo` パッケージを解決できるようにするには、次のようにします：

1. VSCodeの環境設定ページを開きます。
2. Mojo ' Lsp: Include Dirs` に移動します。
3. add item` をクリックし、Numojo リポジトリがあるパスを追加します。例えば `/Users/Name/Programs/NuMojo` です。
。
4. Mojo LSPサーバーを再起動します。

これでVSCodeがNumojoパッケージの関数ヒントを表示できるようになります！

## 貢献

どのような貢献でも大歓迎です**。コントリビュートに関する詳細やガイドラインは、[こちら](CONTRIBUTING.md)を参照してください。

## 警告

このライブラリはまだ非常に未完成であり、いつでも変更される可能性があります。

## ライセンス

LLVM例外を含むApache 2.0ライセンスの下で配布されています。詳細は[LICENSE](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE)とLLVM [License](https://llvm.org/LICENSE.txt)を参照してください。

## 謝辞

* Modular](https://github.com/modularml)によって作成されたネイティブの[Mojo](https://github.com/modularml/mojo)で構築されています。