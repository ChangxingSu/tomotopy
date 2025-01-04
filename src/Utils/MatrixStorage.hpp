#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace tomoto
{
    // Basic interface for matrix storage types
    template<typename _Scalar, MatrixStorageType _storage>
    struct MatrixStorage
    {
        // Dense matrix type
        using DenseMatrix = Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        // Sparse matrix type
        using SparseMatrix = Eigen::SparseMatrix<_Scalar>;
        
        // Select the actual matrix type based on storage type
        using Matrix = typename std::conditional<
            _storage == MatrixStorageType::Dense,
            DenseMatrix,
            SparseMatrix
        >::type;

        // Static method to create a matrix
        static Matrix createMatrix(size_t rows, size_t cols)
        {
            if constexpr (_storage == MatrixStorageType::Dense) {
                return Matrix::Zero(rows, cols);
            } else {
                Matrix mat(rows, cols);
                // Preallocate space, assuming about 10% of elements per column are non-zero
                mat.reserve(Eigen::VectorXi::Constant(cols, rows/10));
                return mat;
            }
        }

        // Method to update matrix elements
        static void addToElement(Matrix& mat, size_t row, size_t col, _Scalar val)
        {
            if constexpr (_storage == MatrixStorageType::Dense) {
                mat(row, col) += val;
            } else {
                mat.coeffRef(row, col) += val;
            }
        }

        // Method to get matrix elements
        static _Scalar getElement(const Matrix& mat, size_t row, size_t col)
        {
            if constexpr (_storage == MatrixStorageType::Dense) {
                return mat(row, col);
            } else {
                return mat.coeff(row, col);
            }
        }
    };
}