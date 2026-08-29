#pragma once

#include <vector>
#include <memory>
#include <iomanip>
#include <functional>

/**
 * @brief Namespace containing general mathematical utility functions.
 */
namespace CommonUtils {
    /**
     * @brief Efficiently calculates the square of a value.
     * @tparam T Numeric type (e.g., double, int).
     * @param x The value to square.
     * @return The squared value ($x^2$).
     */
    template <typename T>
    constexpr T sq(T x) {
        return x * x;
    }

    /**
     * @brief Efficiently calculates the fifth power of a value.
     * @tparam T Numeric type (e.g., double, int).
     * @param x The value.
     * @return The fifth power ($x^5$).
     */
    template <typename T>
    constexpr T pow5(T x) {
        return x * x * x * x * x;
    }
}

/**
 * @brief Reads a 1D vector of data from a text file.
 * @param filename The path to the file to read.
 * @return A vector of doubles containing the read values.
 */
std::vector<double> readVector(const std::string& filename);

/**
 * @brief Reads a 2D matrix of data from a structured file (e.g., CSV).
 * @param filename The path to the file to read.
 * @return A vector of vectors representing the matrix.
 */
std::vector<std::vector<double>> readMatrix(const std::string& filename);

/**
 * @brief Calculates the mean and standard error of a data sample.
 * @param vec Reference to the raw data vector to analyze.
 * @return A pair where `first` is the mean and `second` is the standard error.
 */
std::pair<double, double> mean_err(std::vector<double>& vec);

double norm(const std::vector<double>& vec);

double sqNorm(const std::vector<double>& vec);

double distance(const std::vector<double>& v, const std::vector<double>& w);

void inline swapVar(double& a, double& b) {
    double temp = a;
    a = b;
    b = temp;
}

std::vector<std::vector<double>> generate_mesh(
    std::vector<double>& lb, std::vector<double>& ub, std::vector<unsigned int>& nPoints);


template <typename Derived, typename Base>
std::unique_ptr<Derived> dynamic_unique_cast(std::unique_ptr<Base>&& p) {
    // Try to cast the raw pointer
    if (Derived* result = dynamic_cast<Derived*>(p.get())) {
        // Only if successful, release the original and wrap the new one
        p.release(); 
        return std::unique_ptr<Derived>(result);
    }
    // If it fails, the original unique_ptr 'p' safely retains ownership
    return std::unique_ptr<Derived>(nullptr); 
}

inline void print_colTitle(std::ostream& outs, const std::string& str, bool is_first = false, bool is_last = false) {
    const unsigned int width = 19;
    if (is_first)
        outs << '#';
    outs << std::setw(width - is_first) << str << (is_last ? "\n" : ",");
}

template <typename Type>
inline void print_colVal(std::ostream& outs, Type val, bool is_first = false, bool is_last = false) {
    const unsigned int prec = 12, width = 19;
    outs << std::setprecision(prec) << std::setw(width) << val << (is_last ? "\n" : ",");
}