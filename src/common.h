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

/**
 * @brief Calculates the ||vec||_2 norm of a vector vec.
 * @param vec Vector.
 * @return The norm.
 */
double norm(const std::vector<double>& vec);

/**
 * @brief Calculates the squared ||vec||_2 norm of a vector vec.
 * @param vec Vector.
 * @return The squared norm.
 */
double sqNorm(const std::vector<double>& vec);

/**
 * @brief Calculates the distance-2 between two vectors.
 * @param v First vector.
 * @param w Second vector.
 * @return The distance.
 */
double distance(const std::vector<double>& v, const std::vector<double>& w);

/**
 * @brief Swaps the values of two double variables.
 * @param a First variable.
 * @param b Second variable.
 */
void inline swapVar(double& a, double& b) {
    double temp = a;
    a = b;
    b = temp;
}

/**
 * @brief Generates a mesh grid of parameters with bounds lb and ub,
 * * and nPoints in between the bounds ( semi-open intervals [lb, ub) ).
 * @param lb Lower bounds.
 * @param ub Upper bounds.
 * @param nPoints Number of points between bounds.
 * @result The mesh grid.
 */
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

/**
 * @brief Utility function to print a column's title in the header of a log file.
 * @param outs log file.
 * @param str Column's title.
 * @param is_first Prints a '#' in front of str.
 * @param is_last Follows the title with '\n' instead of ','.
 */
inline void print_colTitle(std::ostream& outs, const std::string& str, bool is_first = false, bool is_last = false) {
    const unsigned int width = 21;
    if (is_first)
        outs << '#';
    outs << std::setw(width - is_first) << str << (is_last ? "\n" : ",");
}

/**
 * @brief Utility function to print a value in a column of a log file.
 * @param outs log file.
 * @param val Value.
 * @param is_first Placeholder for future necessity.
 * @param is_last Follows the value with '\n' instead of ','.
 */
template <typename Type>
inline void print_colVal(std::ostream& outs, Type val, bool is_first = false, bool is_last = false) {
    const unsigned int prec = 13, width = 21;
    outs << std::setprecision(prec) << std::setw(width) << val << (is_last ? "\n" : ",");
}