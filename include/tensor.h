#pragma once

#include <vector>
#include <string>
#include <cassert>
#include <numeric>
#include <functional>
#include <iostream>
#include <algorithm>

namespace gpt2 {

class Tensor {
public:
    Tensor() = default;

    explicit Tensor(std::vector<int> shape)
        : shape_(std::move(shape))
        , data_(numel_from_shape(shape_))
    {}

    Tensor(std::vector<int> shape, std::vector<float> data)
        : shape_(std::move(shape))
        , data_(std::move(data))
    {
        assert(static_cast<int>(data_.size()) == numel());
    }

    // Move and copy
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    // Data access
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    int numel() const { return static_cast<int>(data_.size()); }
    int ndim() const { return static_cast<int>(shape_.size()); }
    const std::vector<int>& shape() const { return shape_; }

    int shape(int dim) const {
        if (dim < 0) dim += ndim();
        return shape_[dim];
    }

    // Flat indexing
    float& operator[](int i) { return data_[i]; }
    float operator[](int i) const { return data_[i]; }

    // 2D indexing (row-major)
    float& at(int i, int j) { return data_[i * shape_[1] + j]; }
    float at(int i, int j) const { return data_[i * shape_[1] + j]; }

    void zero() { std::fill(data_.begin(), data_.end(), 0.0f); }

    void print(const std::string& name = "", int max = 8) const {
        if (!name.empty()) std::cout << name << " ";
        std::cout << "[";
        for (int i = 0; i < ndim(); i++) {
            if (i) std::cout << ",";
            std::cout << shape_[i];
        }
        std::cout << "] = {";
        int n = std::min(max, numel());
        for (int i = 0; i < n; i++) {
            if (i) std::cout << ", ";
            std::cout << data_[i];
        }
        if (numel() > max) std::cout << ", ...";
        std::cout << "}\n";
    }

private:
    static int numel_from_shape(const std::vector<int>& s) {
        if (s.empty()) return 0;
        return std::accumulate(s.begin(), s.end(), 1, std::multiplies<int>());
    }

    std::vector<int> shape_;
    std::vector<float> data_;
};

} // namespace gpt2