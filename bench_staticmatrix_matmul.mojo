from sys import argv, simd_width_of, size_of
from sys import has_apple_gpu_accelerator

from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    keep,
)
from testing import assert_equal
from math import sqrt
from pathlib import Path

from numojo.core.staticmatrix import StaticMatrix
from numojo.core.gpu.device import Device, check_accelerator
from gpu.host import DeviceContext

alias dtype = DType.float32

alias SIZE_64 = 64
alias SIZE_128 = 128
alias SIZE_256 = 256
alias SIZE_512 = 512
alias SIZE_1024 = 1024
alias SIZE_2048 = 2048
alias SIZE_4096 = 4096
alias SIZE_8192 = 8192


@always_inline
fn random_fill_cpu_tensor[
    dtype: DType
](mut m: StaticMatrix[dtype, Device.CPU]) raises:
    var ptr = m._buf.get_host_ptr()
    for i in range(m.size):
        ptr[i] = Scalar[dtype]((i % 100) / 100.0)


@always_inline
fn random_fill_gpu_tensor[
    dtype: DType
](mut m: StaticMatrix[dtype, Device.GPU]) raises:
    with m._buf.get_device_buffer().map_to_host() as host_ptr:
        for i in range(m.size):
            host_ptr[i] = Scalar[dtype]((i % 100) / 100.0)


# ===----------------------------------------------------------------------=== #
# CPU Matrix Multiplication Benchmark
# ===----------------------------------------------------------------------=== #


@parameter
@always_inline
fn benchmark_tensor_matmul_cpu[
    dtype: DType, size: Int
](mut bencher: Bencher) raises:
    var a = StaticMatrix[dtype, Device.CPU]((size, size))
    var b = StaticMatrix[dtype, Device.CPU]((size, size))

    random_fill_cpu_tensor[dtype](a)
    random_fill_cpu_tensor[dtype](b)

    @parameter
    @always_inline
    fn matmul_iteration(ctx: DeviceContext) raises:
        var result = a @ b
        keep(result._buf.get_host_ptr())

    bencher.iter_custom[matmul_iteration](DeviceContext())
    keep(a._buf.get_host_ptr())
    keep(b._buf.get_host_ptr())


# ===----------------------------------------------------------------------=== #
# GPU Matrix Multiplication Benchmark
# ===----------------------------------------------------------------------=== #


@parameter
fn benchmark_tensor_matmul_gpu[
    dtype: DType, size: Int
](mut bencher: Bencher) raises:
    var bench_ctx = DeviceContext()

    var a = StaticMatrix[dtype, Device.GPU]((size, size))
    var b = StaticMatrix[dtype, Device.GPU]((size, size))

    random_fill_gpu_tensor[dtype](a)
    random_fill_gpu_tensor[dtype](b)

    @parameter
    @always_inline
    fn matmul_iteration(ctx: DeviceContext) raises:
        var result = a @ b
        keep(result._buf.get_device_ptr())

    bencher.iter_custom[matmul_iteration](bench_ctx)
    bench_ctx.synchronize()
    keep(a._buf.get_device_ptr())
    keep(b._buf.get_device_ptr())


@parameter
@always_inline
fn benchmark_tensor_matmul_cpu_64[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_cpu[dtype, SIZE_64](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_gpu_64[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_gpu[dtype, SIZE_64](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_cpu_128[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_cpu[dtype, SIZE_128](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_gpu_128[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_gpu[dtype, SIZE_128](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_cpu_256[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_cpu[dtype, SIZE_256](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_gpu_256[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_gpu[dtype, SIZE_256](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_cpu_512[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_cpu[dtype, SIZE_512](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_gpu_512[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_gpu[dtype, SIZE_512](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_cpu_1024[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_cpu[dtype, SIZE_1024](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_gpu_1024[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_gpu[dtype, SIZE_1024](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_cpu_2048[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_cpu[dtype, SIZE_2048](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_gpu_2048[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_gpu[dtype, SIZE_2048](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_cpu_4096[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_cpu[dtype, SIZE_4096](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_gpu_4096[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_gpu[dtype, SIZE_4096](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_cpu_8192[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_cpu[dtype, SIZE_8192](bencher)


@parameter
@always_inline
fn benchmark_tensor_matmul_gpu_8192[dtype: DType](mut bencher: Bencher) raises:
    benchmark_tensor_matmul_gpu[dtype, SIZE_8192](bencher)


from python import Python, PythonObject


def plot_results_from_csv(csv_file: String):
    """Plot CPU vs GPU performance from CSV file."""
    var pd = Python.import_module("pandas")
    var mpl = Python.import_module("matplotlib.pyplot")
    var np = Python.import_module("numpy")
    var df = pd.read_csv(csv_file)

    var sizes = Python.tuple(64, 128, 256, 512, 1024, 2048)  #
    var cpu_data = df[df["name"].str.contains("cpu")]
    var gpu_data = df[df["name"].str.contains("gpu")]

    # Extract timing data in the correct order
    var cpu_times = np.zeros(len(sizes))
    var gpu_times = np.zeros(len(sizes))

    for i in range(len(sizes)):
        var size = sizes[i]
        var size_str = String(size) + "x" + String(size)
        var cpu_row = cpu_data[cpu_data["name"].str.contains(size_str)]
        if len(cpu_row) > 0:
            cpu_times[i] = cpu_row["met (ms)"].iloc[0]

        var gpu_row = gpu_data[gpu_data["name"].str.contains(size_str)]
        if len(gpu_row) > 0:
            gpu_times[i] = gpu_row["met (ms)"].iloc[0]

    var fig = mpl.figure()
    fig.set_size_inches(12, 8)

    # Plot 1: Time comparison
    mpl.subplot(111)
    mpl.plot(
        sizes,
        cpu_times,
        marker="o",
        linewidth=2,
        markersize=8,
        label="CPU Time",
        color="#1f77b4",
    )
    mpl.plot(
        sizes,
        gpu_times,
        marker="s",
        linewidth=2,
        markersize=8,
        label="GPU Time",
        color="#ff7f0e",
    )
    mpl.xlabel("Matrix Size (N×N)")
    mpl.ylabel("Time (ms)")
    mpl.title("Matrix Multiplication Performance: CPU vs GPU")
    mpl.legend()
    mpl.grid(True, alpha=0.3)
    mpl.yscale("log")

    # # Plot 2: Speedup
    # mpl.subplot(2, 2, 2)
    # var speedups = Python.list()
    # for i in range(len(sizes)):
    #     if gpu_times[i] > 0:
    #         speedups.append(cpu_times[i] / gpu_times[i])
    #     else:
    #         speedups.append(0)

    # var colors = Python.list()
    # for s in speedups:
    #     if s < 1:
    #         colors.append('red')
    #     else:
    #         colors.append('green')

    # var bars = np.arange(len(sizes))
    # mpl.bar(bars, speedups, color=colors, alpha=0.7)
    # mpl.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='CPU = GPU')
    # mpl.xlabel('Matrix Size')
    # mpl.ylabel('GPU Speedup (x)')
    # mpl.title('GPU Speedup over CPU')

    # var labels = Python.list()
    # for s in sizes:
    #     labels.append(String(s) + "×" + String(s))
    # mpl.xticks(bars, labels)
    # mpl.legend()
    # mpl.grid(True, alpha=0.3)

    # # Plot 3: GFLOPS comparison
    # mpl.subplot(2, 2, 3)
    # gflops_cpu = Python.list()
    # gflops_gpu = Python.list()
    # for i in range(len(sizes)):
    #     ops = Python.float(2 * sizes[i] * sizes[i] * sizes[i])
    #     if cpu_times[i] > 0:
    #         cpu_gflops = Python.float(ops / (cpu_times[i] * 1e6))
    #         gflops_cpu.append(cpu_gflops)
    #     else:
    #         gflops_cpu.append(0)

    #     if gpu_times[i] > 0:
    #         gpu_gflops = Python.float(ops / (gpu_times[i] * 1e6))
    #         gflops_gpu.append(gpu_gflops)
    #     else:
    #         gflops_gpu.append(0)

    # mpl.plot(sizes, gflops_cpu, marker='o', linewidth=2, markersize=8, label='CPU GFLOPS', color='#1f77b4')
    # mpl.plot(sizes, gflops_gpu, marker='s', linewidth=2, markersize=8, label='GPU GFLOPS', color='#ff7f0e')
    # mpl.xlabel('Matrix Size (N×N)')
    # mpl.ylabel('Performance (GFLOPS)')
    # mpl.title('Computational Performance')
    # mpl.legend()
    # mpl.grid(True, alpha=0.3)

    # Plot 4: Performance scaling
    # mpl.subplot(2, 2, 4)
    # var cpu_relative = Python.list()
    # var gpu_relative = Python.list()
    # var theoretical = Python.list()

    # for i in range(len(sizes)):
    #     if cpu_times[0] > 0:
    #         cpu_relative.append(cpu_times[i] / cpu_times[0])
    #     else:
    #         cpu_relative.append(0)

    #     if gpu_times[0] > 0:
    #         gpu_relative.append(gpu_times[i] / gpu_times[0])
    #     else:
    #         gpu_relative.append(0)

    #     theoretical.append((sizes[i] / sizes[0]) ** 3)  # O(n^3) scaling

    # mpl.plot(sizes, cpu_relative, marker='o', linewidth=2, markersize=8, label='CPU Scaling', color='#1f77b4')
    # mpl.plot(sizes, gpu_relative, marker='s', linewidth=2, markersize=8, label='GPU Scaling', color='#ff7f0e')
    # mpl.plot(sizes, theoretical, linestyle='--', color='gray', label='O(n³) Theoretical')
    # mpl.xlabel('Matrix Size (N×N)')
    # mpl.ylabel('Relative Time')
    # mpl.title('Performance Scaling')
    # mpl.legend()
    # mpl.grid(True, alpha=0.3)
    # mpl.yscale('log')

    mpl.tight_layout()
    mpl.savefig("matmul_benchmark_plots_matrix.png", dpi=300)
    mpl.show()


fn main() raises:
    @parameter
    if not has_apple_gpu_accelerator():
        print("No Apple GPU found")
    else:
        ctx = DeviceContext()
        print("Found Apple GPU:", ctx.name())

    var bench_config = BenchConfig(max_iters=5, num_warmup_iters=2)
    var bench = Bench(bench_config.copy())

    bench.bench_function[benchmark_tensor_matmul_cpu_64[dtype]](
        BenchId("tensor_matmul_cpu_64x64")
    )

    @parameter
    if has_apple_gpu_accelerator():
        bench.bench_function[benchmark_tensor_matmul_gpu_64[dtype]](
            BenchId("tensor_matmul_gpu_64x64")
        )
    bench.bench_function[benchmark_tensor_matmul_cpu_128[dtype]](
        BenchId("tensor_matmul_cpu_128x128")
    )

    @parameter
    if has_apple_gpu_accelerator():
        bench.bench_function[benchmark_tensor_matmul_gpu_128[dtype]](
            BenchId("tensor_matmul_gpu_128x128")
        )
    bench.bench_function[benchmark_tensor_matmul_cpu_256[dtype]](
        BenchId("tensor_matmul_cpu_256x256")
    )

    @parameter
    if has_apple_gpu_accelerator():
        bench.bench_function[benchmark_tensor_matmul_gpu_256[dtype]](
            BenchId("tensor_matmul_gpu_256x256")
        )
    bench.bench_function[benchmark_tensor_matmul_cpu_512[dtype]](
        BenchId("tensor_matmul_cpu_512x512")
    )

    @parameter
    if has_apple_gpu_accelerator():
        bench.bench_function[benchmark_tensor_matmul_gpu_512[dtype]](
            BenchId("tensor_matmul_gpu_512x512")
        )
    bench.bench_function[benchmark_tensor_matmul_cpu_1024[dtype]](
        BenchId("tensor_matmul_cpu_1024x1024")
    )

    @parameter
    if has_apple_gpu_accelerator():
        bench.bench_function[benchmark_tensor_matmul_gpu_1024[dtype]](
            BenchId("tensor_matmul_gpu_1024x1024")
        )
    bench.bench_function[benchmark_tensor_matmul_cpu_2048[dtype]](
        BenchId("tensor_matmul_cpu_2048x2048")
    )

    @parameter
    if has_apple_gpu_accelerator():
        bench.bench_function[benchmark_tensor_matmul_gpu_2048[dtype]](
            BenchId("tensor_matmul_gpu_2048x2048")
        )
    bench.config.out_file = Path("matmul_benchmark_results.csv")
    print(bench)
    bench.dump_report()
    # plot_results_from_csv("matmul_benchmark_results.csv")
