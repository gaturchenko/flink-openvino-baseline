#include "OpenVinoEngine.hpp"

#include <openvino/openvino.hpp>

#include <chrono>
#include <cstring>
#include <iostream>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

OpenVinoEngine::OpenVinoEngine(
    const std::string& model_path,
    const std::string& device,
    int num_streams,
    int num_threads
) {
    ov::Core core;

    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    // Pin a dynamic batch dimension to 1: the job feeds one frame per predict() call, and
    // get_shape()/ov::Tensor below need fully static shapes. The SqueezeDet export carries a
    // dynamic leading dim ([?, H, W, 3]); NanoDet is already static.
    ov::PartialShape input_pshape = model->input().get_partial_shape();
    if (input_pshape.rank().is_static() && input_pshape[0].is_dynamic()) {
        input_pshape[0] = 1;
        model->reshape(input_pshape);
    }

    // Mirror the XXXX-2 OpenVINO backend's compile options (OpenVinoRuntimeBackend.cpp): ACCURACY +
    // LATENCY hints with EXPLICIT num_streams / inference_num_threads values. XXXX-2 compiles with
    // num_streams(0) and inference_num_threads(1) by default, so 0 must reach the plugin as an
    // explicit value, not be dropped. Pass a negative value to leave a property unset.
    ov::AnyMap compile_options{
        ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY),
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
    };
    if (num_streams >= 0) {
        compile_options.insert(ov::num_streams(num_streams));
    }
    if (num_threads >= 0) {
        compile_options.insert(ov::inference_num_threads(num_threads));
    }
    compiled_model_ = core.compile_model(model, device, compile_options);

    infer_request_ = compiled_model_.create_infer_request();

    ov::Output<const ov::Node> input_port = compiled_model_.input();
    ov::Output<const ov::Node> output_port = compiled_model_.output();

    ov::Shape input_shape = input_port.get_shape();
    ov::Shape output_shape = output_port.get_shape();

    ov::element::Type input_type = input_port.get_element_type();
    ov::element::Type output_type = output_port.get_element_type();

    if (input_type != ov::element::f32) {
        throw std::runtime_error(
            "This baseline expects f32 input. Actual input type: " +
            input_type.to_string()
        );
    }

    if (output_type != ov::element::f32) {
        throw std::runtime_error(
            "This baseline expects f32 output. Actual output type: " +
            output_type.to_string()
        );
    }

    input_size_ = shape_size(input_shape);
    output_size_ = shape_size(output_shape);

    input_tensor_ = ov::Tensor(input_type, input_shape);
    output_tensor_ = ov::Tensor(output_type, output_shape);

    infer_request_.set_input_tensor(input_tensor_);
    infer_request_.set_output_tensor(output_tensor_);
}

std::size_t OpenVinoEngine::shape_size(const ov::Shape& shape) {
    return std::accumulate(
        shape.begin(),
        shape.end(),
        static_cast<std::size_t>(1),
        std::multiplies<>()
    );
}

std::size_t OpenVinoEngine::input_size() const {
    return input_size_;
}

std::size_t OpenVinoEngine::output_size() const {
    return output_size_;
}

void OpenVinoEngine::predict(const float* input, float* output) {
    auto* input_data = input_tensor_.data<float>();

    std::memcpy(
        input_data,
        input,
        input_size_ * sizeof(float)
    );

    infer_request_.infer();

    const auto* output_data = output_tensor_.data<const float>();

    std::memcpy(
        output,
        output_data,
        output_size_ * sizeof(float)
    );
}
