#pragma once

#include <openvino/openvino.hpp>

#include <cstddef>
#include <memory>
#include <string>

class OpenVinoEngine {
public:
    OpenVinoEngine(
        const std::string& model_path,
        const std::string& device,
        int num_streams,
        int num_threads
    );

    std::size_t input_size() const;
    std::size_t output_size() const;

    void predict(const float* input, float* output);

private:
    static std::size_t shape_size(const ov::Shape& shape);

private:
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;

    ov::Tensor input_tensor_;
    ov::Tensor output_tensor_;

    std::size_t input_size_ = 0;
    std::size_t output_size_ = 0;
};
