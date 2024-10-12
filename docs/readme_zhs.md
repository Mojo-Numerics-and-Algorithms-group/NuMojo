<a name="readme-top"></a>
<!-- add these later -->
<!-- [![MIT License][license-shield]][] -->

<div align="center">
  <a href="">
    <img src="../assets/numojo_logo.png" alt="Logo" width="350" height="350">
  </a>

  <h1 align="center" style="font-size: 3em; color: white; font-family: 'Avenir'; text-shadow: 1px 1px orange;">NuMojo</h1>

  <p align="center">
    NuMojo 是为 Mojo 🔥 设计的多维数组运算库，类似 Python 中的 NumPy, SciPy。
    <br />
    <div style="font-family: 'Arial'; border: 1px solid black; padding: 5px;">
        <a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo-Examples-and-Benchmarks/blob/main/docs/README.md"><strong>阅读文档» </strong></a> &nbsp; &nbsp; <a href="https://discord.com/channels/1149778565756366939/1149778566603620455" ><strong>加入 Discord 讨论频道» </strong></a>
    </div>
    <br />
    <div style="font-family: 'Arial'; border: 1px solid black; padding: 5px;">
        <a href="./docs/readme_zht.md"><strong>中文·傳統漢字» </strong></a>
    </div>
  </p>
</div>

## 关于本项目

NuMojo，旨在为 Mojo 语言生态系统提供数值计算和多维数组运算功能，类似于 NumPy、SciPy 和 Scikit 等数学库在 Python 语言生态系统中所扮演的角色。

NuMojo 充分利用 Mojo 的潜力，包括向量化、并行化和 GPU 加速。目前，NuMojo 扩展了大部分 Mojo 标准库中的数学函数，用以处理多维数组。

NuMojo 也可为其他需要高速数值计算、多维数组运算等功能的 Mojo 第三方库提供基础类型和函数。

注意：NuMojo 不是一个机器学习库，它永远不会在核心库中包含机器学习算法。

## 目标及路线图

有关本项目详细的路线图，请参阅 [Roadmap.md](../Roadmap.md) 文件（英文）。

我们的核心目标，是使用 Mojo 实现一个快速、全面的数值计算库。以下是部分长期目标：

- 原生 N 维数组类型
- 向量化、并行化的数值运算
- 线性代数
- 数组操作：叠加、切片、拼合等
- 微积分
- 优化
- 函数逼近和估值
- 排序

## 使用方法

以下为部分代码实例：

```mojo
import numojo as nm
from numojo.prelude import *

fn main() raises:
    # 生成两个 1000x1000 矩阵，数值随机且为 64 位浮点数
    var A = nm.random.randn[f64](shape=List[Int](1000, 1000))
    var B = nm.random.randn[f64](shape=List[Int](1000, 1000))

    # 根据字符串生成 3x2 矩阵，数据类型为 32 位浮点数
    var X = nm.fromstring[f32]("[[1.1, -0.32, 1], [0.1, -3, 2.124]]")

    # 打印矩阵
    print(A)

    # 矩阵相乘
    var C = A @ B

    # 矩阵求逆
    var I = nm.inv(A)

    # 矩阵切片
    var A_slice = A[1:3, 4:19]

    # 提取矩阵元素
    var A_item = A.item(291, 141)
```

请在 [此文档](../features.md) 中查询所有可用的函数。

## 安装方法

Numojo 库可通过两种方法安装并使用。

### 构建文件包

这种方法会构建一个独立文件包 `numojo.mojopkg`。步骤为：

1. 克隆本仓库。
1. 在控制台使用 `mojo package numojo` 命令构建文件包。
1. 将 `numojo.mojopkg` 移动到包含代码的目录中，即可使用。

### 将 NuMojo 路径添加至编译器和 LSP

这种方法不需要生成文件包，仅需在编译时，通过以下命令指明 `Numojo` 的文件路径：

```console
mojo run -I "../NuMojo" example.mojo
```

这种方法自由度更高，允许你在调试你代码的过程中修改 `NuMojo` 源码。它适合想要为本库贡献代码的用户。

为了使 VSCode 的 LSP （语言服务引擎）解析 `numojo` 库，你可以：

1. 进入 VSCode 的 Preference 页面（偏好设置）。
1. 选择 `Mojo › Lsp: Include Dirs`
1. 点击 `add item`，并添加 NuMojo 源码所在的路径，例如 `/Users/Name/Programs/NuMojo`
1. 重启 Mojo LSP server

如此，VSCode 便可以提供 NuMojo 包的函数提示。
