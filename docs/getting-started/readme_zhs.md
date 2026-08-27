<a name="readme-top"></a>
<!-- add these later -->
<!-- [![MIT License][license-shield]][] -->

<div align="center">
  <a href="">
    <img src="../../assets/numojo_logo.png" alt="Logo" width="350" height="350">
  </a>

  <h1 align="center" style="font-size: 3em; color: white; font-family: 'Avenir'; text-shadow: 1px 1px orange;">NuMojo</h1>

  <p align="center">
    NuMojo 是为 Mojo 🔥 设计的多维数组运算库，类似 Python 中的 NumPy, SciPy。
    <br />
    <div style="font-family: 'Arial'; border: 1px solid black; padding: 5px;">
        <a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo-Examples-and-Benchmarks/blob/main/docs/README.md"><strong>阅读文档» </strong></a> &nbsp; &nbsp; 
        <a href="../user-guide/changelog.md"><strong>更新日志» </strong></a> &nbsp; &nbsp;
        <a href="https://discord.gg/NcnSH5n26F" ><strong>加入 Discord 讨论频道» </strong></a>
    </div>
    <br />
    <div style="font-family: 'Arial'; border: 1px solid black; padding: 5px;">
        <a href="./readme_zht.md"><strong>中文·繁» </strong></a> &nbsp;
        <a href="./readme_jp.md"><strong>日本語» </strong></a>
        <a href="../../README.MD"><strong>English» </strong></a> &nbsp;
    </div>
  </p>
</div>

## 关于本项目

NuMojo，旨在为 Mojo 语言生态系统提供数值计算和多维数组运算功能，类似于 NumPy、SciPy 和 Scikit 等数学库在 Python 语言生态系统中所扮演的角色。

***NuMojo 是什么***

我们致力于充分发挥 Mojo 的潜力，包括向量化、并行化和 GPU 加速。目前，NuMojo 已经为大部分（乃至全部）Mojo 标准库中的数学函数扩展了对数组输入的支持。

我们希望 NuMojo 能够成为其他需要高速数值运算的 Mojo 第三方库的基础构建模块，同时不必附带机器学习反向传播系统所带来的额外负担。

***NuMojo 不是什么***

NuMojo 不是一个机器学习库，它永远不会在核心库中包含反向传播功能。

## 功能与目标

我们的核心目标，是使用 Mojo 打造一个快速、全面的数值计算库。以下是部分功能与长期目标，其中一些已经完整或部分实现。

核心数据类型：

- 原生 N 维数组（`numojo.NDArray`）。
- 原生 N 维复数数组（`numojo.ComplexNDArray`）。
- 原生固定维度数组（待 Mojo 支持 trait 参数化后实现）。

例程与对象：

- 数组创建例程（`numojo.creation`）
- 数组操作例程（`numojo.manipulation`）
- 输入与输出（`numojo.io`）
- 线性代数（`numojo.linalg`）
- 逻辑函数（`numojo.logic`）
- 数学函数（`numojo.math`）
- 指数与对数（`numojo.exponents`）
- 极值查找（`numojo.extrema`）
- 取整（`numojo.rounding`）
- 三角函数（`numojo.trig`）
- 随机采样（`numojo.random`）
- 排序与搜索（`numojo.sorting`、`numojo.searching`）
- 统计（`numojo.statistics`）
- 等等……

完整的函数与对象列表请参阅[此文档](../user-guide/features.md)。项目的实时路线图维护在 [docs/user-guide/roadmap.md](../user-guide/roadmap.md) 中。

有关本项目详细的路线图，请参阅 [docs/user-guide/roadmap.md](../user-guide/roadmap.md) 文件。

## 使用方法

`examples/` 目录下提供了可直接运行的示例（例如 `examples/quickstart.mojo`）。

N 维数组（`NDArray` 类型）的示例如下：

```mojo
import numojo as nm
from numojo.prelude import *


def main() raises:
    # Generate two 1000x1000 matrices with random float64 values
    var A = nm.random.randn(Shape(1000, 1000)) # Shape is used for all shape related operations in numojo. 
    var B = nm.random.randn(Shape(1000, 1000))

    # Generate a 3x2 matrix from string representation
    var X = nm.fromstring[f32]("[[1.1, -0.32, 1], [0.1, -3, 2.124]]")

    # Print array
    print(A)

    # Array multiplication
    var C = A @ B

    # Array inversion
    var I = nm.inv(A)

    # Array slicing
    var A_slice = A[1:3, 4:19]

    # Get scalar from array
    var A_item = A[Item(291, 141)] # Item() is used to define coordinates of an ndarray in numojo. 
    var A_item_2 = A.item(291, 141)

    # Sort and argsort along axis
    print(nm.sort(A, axis=1))
    print(nm.argsort(A, axis=0))

    # Sum along axis
    print(nm.sum(A))
    print(nm.sum(A, axis=1))

    # Solve a linear system
    print(nm.solve(A, B))
```

`ComplexNDArray` 的示例如下：

```mojo
import numojo as nm
from numojo.prelude import *


def main() raises:
    # Create a complex scalar 5 + 5j
    # cf32 is the complex version of f32 (DType.float32) used to identify complex types in numojo.
    var complexscalar = CScalar[cf32](5) # Equivalently ComplexSIMD[cf32](5, 5)
    # Also can be define as simple as  5 + 5*`1j`!
  
    # Create complex arrays
    var A = nm.full[cf32](Shape(1000, 1000), fill_value=complexscalar)  # filled with (5+5j)
    var B = nm.ones[cf32](Shape(1000, 1000))                            # filled with (1+1j)

    # Print array
    print(A)

    # Array slicing
    var A_slice = A[1:3, 4:19]

    # Array multiplication
    var C = A * B

    # Get scalar from array
    var A_item = A[Item(291, 141)]
    # Set an element of the array
    A[item(291, 141)] = complexscalar
```

## 安装方法

NuMojo 提供多种安装方式，以适配不同的开发需求。请根据你的工作流选择最合适的方法：

### 方法一：使用 pixi-build-mojo 通过 Git 安装（推荐）

直接从 GitHub 仓库安装 NuMojo，既可以获取稳定版本，也可以体验最新功能。这种方式非常适合希望使用最新特性、或需要使用最新稳定版本的开发者。

在你现有的 `pixi.toml` 中添加以下内容：

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
mojo = ">=1.0.0, <1.1.0"
max-core = ">=26.5.0,<27"

[package.build-dependencies]
mojo = ">=1.0.0, <1.1.0"
max-core = ">=26.5.0,<27"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}

[package.run-dependencies]
mojo = ">=1.0.0, <1.1.0"
max-core = ">=26.5.0,<27"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}

[dependencies]
mojo = ">=1.0.0, <1.1.0"
max-core = ">=26.5.0,<27"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}
```

然后运行：
```bash
pixi install
```

**分支选择：**
- **`main` 分支**：提供稳定版本。目前支持 NuMojo v0.9.0，兼容 Mojo 26.2。如需使用更早期的 NuMojo 版本，请使用方法二。
- **`pre-x.y` 分支**：活跃开发分支，支持最新的 Mojo 版本（当前为 NuMojo v0.10.0，需要 Mojo >=1.0.0, <1.1.0）。请注意，该分支更新频繁，功能和语法可能发生破坏性变更。

该软件包会自动出现在你的 Pixi 环境中，VSCode LSP 也会提供智能代码提示。

### 方法二：通过 Pixi（prefix.dev）安装稳定版本

对大多数用户来说，我们推荐通过 Pixi 安装稳定版本，以获得可靠的兼容性和可复现性。

在你的 `pixi.toml` 文件中添加以下内容：

```toml
[workspace]
channels = ["https://repo.prefix.dev/modular-community"]

[dependencies]
numojo = "=0.9.0"
```

然后运行：
```bash
pixi install
```

**版本兼容性：**

| NuMojo 版本 | 所需 Mojo 版本 |
| -------------- | --------------------- |
| v0.9.0         | ==26.2                |
| v0.8.0         | ==25.7                |
| v0.7.0         | ==25.3                |
| v0.6.1         | ==25.2                |
| v0.6.0         | ==25.2                |

### 方法三：构建独立文件包

这种方法会构建一个可移植的 `numojo.mojopkg` 文件，可在多个项目中复用，非常适合离线开发或封闭式构建。

1. 克隆本仓库：
   ```bash
   git clone https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo.git
   cd NuMojo
   ```

2. 构建文件包：
   ```bash
   pixi run package
   ```

3. 将 `numojo.mojopkg` 复制到你的项目目录，或将其所在目录添加到你的 include 路径中。

### 方法四：直接集成源码

如果你需要最大的灵活性，并希望在开发过程中修改 NuMojo 源码：

1. 将本仓库克隆到你想要的位置：
   ```bash
   git clone https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo.git
   ```

2. 编译你的代码时，指定 NuMojo 的源码路径：
   ```bash
   mojo run -I "/path/to/NuMojo" your_program.mojo
   ```

3. **配置 VSCode LSP**（用于代码提示和自动补全）：
   - 打开 VSCode 的偏好设置
   - 进入 `Mojo › Lsp: Include Dirs`
   - 点击 `Add Item`，并输入 NuMojo 目录的完整路径（例如 `/Users/YourName/Projects/NuMojo`）
   - 重启 Mojo LSP server

完成设置后，VSCode 便会为 NuMojo 的函数提供智能代码补全和提示！

## 贡献

我们**非常欢迎**任何形式的贡献。请参阅 [CONTRIBUTING.md](../../CONTRIBUTING.md) 了解贡献指南（代码风格、测试、文档、发布节奏）。

## 警告

本库仍处于早期阶段，次版本之间可能引入破坏性变更。在生产环境或研究代码中使用时，请固定版本号。

## 许可证

基于 Apache 2.0 协议（附加 LLVM 附加条款）分发。详情请参阅 [LICENSE](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE) 以及 LLVM 的[许可证](https://llvm.org/LICENSE.txt)。

本项目包含来自 [Mojo 标准库](https://github.com/modularml/mojo) 的代码，该代码基于 Apache License v2.0（附加 LLVM 附加条款）授权（详见 LLVM [许可证](https://llvm.org/LICENSE.txt)）。MAX 与 Mojo 的使用及分发遵循 [MAX & Mojo 社区许可证](https://www.modular.com/legal/max-mojo-license)。

## 致谢

基于原生 [Mojo](https://github.com/modularml/mojo) 构建，Mojo 由 [Modular](https://github.com/modularml) 创建。

## 贡献者

<a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Mojo-Numerics-and-Algorithms-group/NuMojo" />
</a>
