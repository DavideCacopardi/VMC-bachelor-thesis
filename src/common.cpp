#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <functional>

#include "common.h"

using namespace CommonUtils;

std::vector<double> readVector(const std::string & filename) {
    std::ifstream file(filename);
    std::vector<double> vec;
    double temp;
    while (file >> temp) {
        vec.push_back(temp);
        while (file.peek() == ',' || file.peek() == ' ' || file.peek() == '\t') {
            file.ignore(); 
        }
    }
    file.close();
    return vec;
}

std::vector<std::vector<double>> readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> mat;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;  // skip empty lines and comments
        std::vector<double> row;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try { row.push_back(std::stod(token)); }
            catch (...) { continue; }  // skip malformed tokens
        }
        if (!row.empty()) mat.push_back(row);
    }
    file.close();
    return mat;
}

std::pair<double, double> mean_err(std::vector<double>& vec) {
    double sum = 0;
    for (double val : vec) {
        sum += val;
    }
    double mean = sum / (double)vec.size();
    sum = 0;
    for (double val : vec) {
        sum += sq(val - mean);
    }
    double var = sum / (double)(vec.size() - 1);
    double err = sqrt(var) / sqrt((double)vec.size());
    return std::make_pair(mean, err);
}

double norm(const std::vector<double>& vec) {
    return sqrt(sqNorm(vec));
}

double sqNorm(const std::vector<double>& vec) {
    double temp = 0;
    for (double val : vec) {
        temp += sq(val);
    }
    return temp;
}

double distance(const std::vector<double>& v, const std::vector<double>& w) {
    if (v.size() != w.size()) return -1;

    double temp = 0;
    for (unsigned int i = 0; i < v.size(); i++) {
        temp += sq(v[i] - w[i]);
    }
    return sqrt(temp);
}

