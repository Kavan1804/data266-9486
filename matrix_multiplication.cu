#include <cblas.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int TILE_SIZE = 16;
constexpr float ABS_TOLERANCE = 1.0e-2f;
constexpr float REL_TOLERANCE = 1.0e-3f;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error__ = (call);                                           \
        if (error__ != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(error__) << std::endl;     \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                       \
    } while (false)

// Each thread computes one C[row, col] element; blockIdx and threadIdx map
// the two-dimensional grid and block to row-major matrix coordinates.
__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int n) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;

    // The grid dimensions cover ceil(n / TILE_SIZE) tiles in each direction.
    const int tile_count = (n + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < tile_count; ++tile) {
        const int a_col = tile * TILE_SIZE + threadIdx.x;
        const int b_row = tile * TILE_SIZE + threadIdx.y;

        // Boundary checks zero-pad partial tiles at matrix edges.
        tileA[threadIdx.y][threadIdx.x] =
            (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = value;
    }
}

struct Options {
    std::vector<int> sizes{256, 1024, 4096};
    int gpu_repeats = 5;
    int cpu_repeats = 3;
    unsigned int seed = 9486;
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " [--size N] [--gpu-repeats N] [--cpu-repeats N] [--seed N]"
              << std::endl;
}

int parse_positive(const std::string& value, const char* option) {
    try {
        std::size_t position = 0;
        const int parsed = std::stoi(value, &position);
        if (position != value.size() || parsed <= 0) {
            throw std::invalid_argument("not positive");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " requires a positive integer");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument(argument + " requires a value");
        }
        const std::string value(argv[++index]);
        if (argument == "--size") {
            options.sizes = {parse_positive(value, "--size")};
        } else if (argument == "--gpu-repeats") {
            options.gpu_repeats = parse_positive(value, "--gpu-repeats");
        } else if (argument == "--cpu-repeats") {
            options.cpu_repeats = parse_positive(value, "--cpu-repeats");
        } else if (argument == "--seed") {
            try {
                std::size_t position = 0;
                options.seed = static_cast<unsigned int>(std::stoul(value, &position));
                if (position != value.size()) {
                    throw std::invalid_argument("invalid seed");
                }
            } catch (const std::exception&) {
                throw std::invalid_argument("--seed requires a nonnegative integer");
            }
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return options;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) / 2.0;
    }
    return values[middle];
}

void fill_inputs(std::vector<float>& matrix, std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (float& value : matrix) {
        value = distribution(generator);
    }
}

struct Errors {
    float maximum_absolute = 0.0f;
    float maximum_relative = 0.0f;
    bool valid = true;
};

Errors compare_results(const std::vector<float>& reference, const std::vector<float>& result) {
    Errors errors;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const float absolute = std::fabs(reference[index] - result[index]);
        const float denominator = std::max(std::fabs(reference[index]), 1.0e-6f);
        const float relative = absolute / denominator;
        errors.maximum_absolute = std::max(errors.maximum_absolute, absolute);
        errors.maximum_relative = std::max(errors.maximum_relative, relative);
    }
    errors.valid = errors.maximum_absolute <= ABS_TOLERANCE &&
                   errors.maximum_relative <= REL_TOLERANCE;
    return errors;
}

double benchmark_cpu(const std::vector<float>& A, const std::vector<float>& B,
                     std::vector<float>& C, int n, int repeats) {
    std::vector<double> measurements;
    measurements.reserve(repeats);
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f,
                    A.data(), n, B.data(), n, 0.0f, C.data(), n);
        const auto stop = std::chrono::steady_clock::now();
        measurements.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    return median(measurements);
}

void launch_kernel(const float* device_A, const float* device_B, float* device_C, int n) {
    const dim3 threads(TILE_SIZE, TILE_SIZE);
    const dim3 blocks((n + TILE_SIZE - 1) / TILE_SIZE,
                      (n + TILE_SIZE - 1) / TILE_SIZE);
    tiled_matmul_kernel<<<blocks, threads>>>(device_A, device_B, device_C, n);
    CUDA_CHECK(cudaGetLastError());
}

struct GpuMeasurements {
    double kernel_ms;
    double transfer_ms;
};

GpuMeasurements benchmark_gpu(const std::vector<float>& A, const std::vector<float>& B,
                              std::vector<float>& C, int n, int repeats) {
    const std::size_t bytes = static_cast<std::size_t>(n) * n * sizeof(float);
    float* device_A = nullptr;
    float* device_B = nullptr;
    float* device_C = nullptr;
    CUDA_CHECK(cudaMalloc(&device_A, bytes));
    CUDA_CHECK(cudaMalloc(&device_B, bytes));
    CUDA_CHECK(cudaMalloc(&device_C, bytes));

    CUDA_CHECK(cudaMemcpy(device_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_B, B.data(), bytes, cudaMemcpyHostToDevice));
    launch_kernel(device_A, device_B, device_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), device_C, bytes, cudaMemcpyDeviceToHost));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> kernel_measurements;
    std::vector<double> transfer_measurements;
    kernel_measurements.reserve(repeats);
    transfer_measurements.reserve(repeats);

    for (int repeat = 0; repeat < repeats; ++repeat) {
        // Transfer timing contains exactly two H2D copies and one D2H copy.
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(device_A, A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_B, B.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(C.data(), device_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float transfer_elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&transfer_elapsed, start, stop));
        transfer_measurements.push_back(transfer_elapsed);

        CUDA_CHECK(cudaEventRecord(start));
        launch_kernel(device_A, device_B, device_C, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float kernel_elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_elapsed, start, stop));
        kernel_measurements.push_back(kernel_elapsed);
    }

    CUDA_CHECK(cudaMemcpy(C.data(), device_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_A));
    CUDA_CHECK(cudaFree(device_B));
    CUDA_CHECK(cudaFree(device_C));
    return {median(kernel_measurements), median(transfer_measurements)};
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        int runtime_version = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

        std::cout << "GPU: " << properties.name << std::endl;
        std::cout << "CUDA runtime version: " << runtime_version / 1000 << "."
                  << (runtime_version % 1000) / 10 << std::endl;
        std::cout << "TILE_SIZE: " << TILE_SIZE << std::endl;
        std::cout << "Seed: " << options.seed << std::endl;
        std::cout << "GPU repetitions: " << options.gpu_repeats << std::endl;
        std::cout << "CPU repetitions: " << options.cpu_repeats << std::endl;
        std::cout << "\nHuman-readable results (median milliseconds)\n";
        std::cout << "size\tCPU\tGPU kernel\tH2D+D2H\tGPU total\tspeedup\tvalid\n";

        std::mt19937 generator(options.seed);
        bool all_valid = true;
        for (const int n : options.sizes) {
            const std::size_t elements = static_cast<std::size_t>(n) * n;
            std::vector<float> A(elements);
            std::vector<float> B(elements);
            std::vector<float> cpu_result(elements);
            std::vector<float> gpu_result(elements);
            fill_inputs(A, generator);
            fill_inputs(B, generator);

            const double cpu_ms = benchmark_cpu(A, B, cpu_result, n, options.cpu_repeats);
            const GpuMeasurements gpu = benchmark_gpu(A, B, gpu_result, n, options.gpu_repeats);
            const Errors errors = compare_results(cpu_result, gpu_result);
            const double gpu_total_ms = gpu.kernel_ms + gpu.transfer_ms;
            const double speedup = cpu_ms / gpu_total_ms;
            all_valid = all_valid && errors.valid;

            std::cout << n << "\t" << cpu_ms << "\t" << gpu.kernel_ms << "\t"
                      << gpu.transfer_ms << "\t" << gpu_total_ms << "\t"
                      << speedup << "\t" << (errors.valid ? "PASS" : "FAIL") << std::endl;
            std::cout << std::setprecision(10)
                      << "RESULT,size=" << n << ",cpu_ms=" << cpu_ms
                      << ",gpu_kernel_ms=" << gpu.kernel_ms
                      << ",transfer_ms=" << gpu.transfer_ms
                      << ",gpu_total_ms=" << gpu_total_ms << ",speedup=" << speedup
                      << ",max_abs_error=" << errors.maximum_absolute
                      << ",max_relative_error=" << errors.maximum_relative
                      << ",valid=" << (errors.valid ? 1 : 0) << std::endl;
        }
        return all_valid ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "Argument error: " << error.what() << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
}
