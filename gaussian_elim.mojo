# ============================================
# This is Mojo's original matric mult. code
# Leaving it in just for referencing
# ============================================

# import benchmark
# from memory import memset_zero
# from random import rand, random_float64

# alias type = DType.float32

# struct Matrix[rows: Int, cols: Int]:
#     var data: UnsafePointer[Scalar[type]]

#     # Initialize zeroeing all values
#     fn __init__(inout self):
#         self.data = UnsafePointer[Scalar[type]].alloc(rows * cols)
#         memset_zero(self.data.address, rows * cols)

#     # Initialize taking a pointer, don't set any elements
#     fn __init__(inout self, data: UnsafePointer[Scalar[type]]):
#         self.data = data

#     # Initialize with random values
#     @staticmethod
#     fn rand() -> Self:
#         var data = UnsafePointer[Scalar[type]].alloc(rows * cols)
#         rand(data.address, rows * cols)
#         return Self(data)

#     fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
#         return self.load[1](y, x)

#     fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):
#         self.store[1](y, x, val)

#     fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
#         return self.data.load[width=nelts](y * self.cols + x)

#     fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
#         return self.data.store[width=nelts](y * self.cols + x, val)

# # Note that C, A, and B have types.
# fn matmul_naive(C: Matrix, A: Matrix, B: Matrix):
#     for m in range(C.rows):
#         for k in range(A.cols):
#             for n in range(C.cols):
#                 C[m, n] += A[m, k] * B[k, n]

# alias M = 1024
# alias N = 1024
# alias K = 1024

# @always_inline
# fn bench[func: fn (Matrix, Matrix, Matrix) -> None](base_gflops: Float64):
#     var C = Matrix[M, N]()
#     var A = Matrix[M, K].rand()
#     var B = Matrix[K, N].rand()

#     @always_inline
#     @parameter
#     fn test_fn():
#         _ = func(C, A, B)

#     var secs = benchmark.run[test_fn](max_runtime_secs=1).mean()

#     A.data.free()
#     B.data.free()
#     C.data.free()

#     var gflops = ((2 * M * N * K) / secs) / 1e9
#     var speedup: Float64 = gflops / base_gflops

#     print(gflops, "GFLOP/s, a", speedup, "x speedup over Python")

# def main():
#     bench[matmul_naive](0.0021764307339275034)

# ============================================
# This is the working Gauss. elim code
# rand() and matmul() commented out just for reference
# ============================================

from memory import memset_zero
from random import rand

alias type = DType.float32

struct Matrix[rows: Int, cols: Int]:
    var data: UnsafePointer[Scalar[type]]

    # Initialize zeroing all values
    fn __init__(inout self):
        self.data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        memset_zero(self.data.address, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, data: UnsafePointer[Scalar[type]]):
        self.data = data

    # Initialize with random values
    # @staticmethod
    # fn rand() -> Self:
    #     var data = UnsafePointer[Scalar[type]].alloc(rows * cols)
    #     rand(data.address, rows * cols)
    #     return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.store[width=nelts](y * self.cols + x, val)

# Naive matrix multiplication
# fn matmul_naive(C: Matrix, A: Matrix, B: Matrix):
#     for m in range(C.rows):
#         for k in range(A.cols):
#             for n in range(C.cols):
#                 C[m, n] += A[m, k] * B[k, n]

alias M = 1024
alias N = 1024
alias K = 1024

fn gaussian_elimination(A: Matrix):
    for k in range(A.rows):  # Pivoting over rows
        # Step 1: Find the pivot element
        pivot = A[k, k]
        if pivot == 0:
            print("Matrix is singular, no unique solution.")
            return
        
        # Step 2: Normalize the pivot row
        for j in range(k, A.cols):
            A[k, j] /= pivot

        # Step 3: Eliminate below the pivot
        for i in range(k + 1, A.rows):
            factor = A[i, k]
            for j in range(k, A.cols):
                A[i, j] -= factor * A[k, j]

    # (Optional) Back substitution for RREF
    for k in range(A.rows - 1, -1, -1):  # Start from the bottom row
        for i in range(k - 1, -1, -1):  # Eliminate above the pivot
            factor = A[i, k]
            for j in range(k, A.cols):
                A[i, j] -= factor * A[k, j]

# ============================================
# This is the driver code for matric multiplication
# Leaving it in just for referencing
# ============================================

# def main():
#     # Create matrices
#     var C = Matrix[M, N]()
#     var A = Matrix[M, K].rand()
#     var B = Matrix[K, N].rand()

#     # Perform naive matrix multiplication
#     matmul_naive(C, A, B)

#     # Free memory
#     A.data.free()
#     B.data.free()
#     C.data.free()

#     print("Matrix multiplication completed.")

# def main():
#     # Create an empty matrix with dimensions M x K
#     var A = Matrix[2, 3]()
#     var B = Matrix[2, 3]()
#     var C = Matrix[2, 3]()  # Resultant matrix

#     # Set custom values for matrix A
#     A[0, 0] = 1.0
#     A[0, 1] = 2.0
#     A[0, 2] = 3.0
#     A[1, 0] = 4.0
#     A[1, 1] = 5.0
#     A[1, 2] = 6.0


#     # Set custom values for matrix B
#     B[0, 0] = 9.0
#     B[0, 1] = 8.0
#     B[0, 2] = 7.0
#     B[1, 0] = 6.0
#     B[1, 1] = 5.0
#     B[1, 2] = 4.0


#     # Perform naive matrix multiplication
#     matmul_naive(C, A, B)

#     # Print the resultant matrix C
#     print("Matrix C (Result):")
#     for i in range(2):
#         for j in range(3):
#             print(C[i, j], end=" ")
#         print()

#     # Free memory
#     A.data.free()
#     B.data.free()
#     C.data.free()

# ============================================
# This is the working Gauss. elim driver code
# Currently you input the values and adjust the matrix size
# ============================================

def main():
    # Define matrix here
    var A = Matrix[4, 5]()  # adjust size here
    A[0, 0] = 3.0
    A[0, 1] = -13.0
    A[0, 2] = 9.0
    A[0, 3] = 3.0
    A[0, 4] = -19.0

    A[1, 0] = -6.0
    A[1, 1] = 4.0
    A[1, 2] = 1.0
    A[1, 3] = -18.0
    A[1, 4] = -34.0

    A[2, 0] = 6.0
    A[2, 1] = -2.0
    A[2, 2] = 2.0
    A[2, 3] = 4.0
    A[2, 4] = 16.0

    A[3, 0] = 12.0
    A[3, 1] = -8.0
    A[3, 2] = 6.0
    A[3, 3] = 10.0
    A[3, 4] = 26.0


    # Perform Gaussian elimination
    gaussian_elimination(A)

    # Print the result matrix
    print("Resultant Matrix:")
    for i in range(4):  # adjust rows
        for j in range(5):  # adjust cols
            print(A[i, j], end=" ")
        print()

    # Free memory
    A.data.free()