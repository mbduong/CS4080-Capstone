from memory import memset_zero
from random import rand
from python import Python

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

    fn __copyinit__(inout self, existing: Self):
        self.data = existing.data

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.store[width=nelts](y * self.cols + x, val)

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


def main():
    # Manually input matrix details
    var A = Matrix[4, 5]()  # Define matrix size
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