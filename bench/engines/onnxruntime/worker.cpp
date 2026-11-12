#define _GNU_SOURCE
#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sched.h>
#include <string>
#include <thread>
#include <vector>

namespace {

void fail(const char * message) {
    std::fprintf(stderr, "onnx-worker: %s\n", message);
    std::exit(1);
}

int default_threads() {
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int count = CPU_COUNT(&set);
        if (count > 0) {
            return count;
        }
    }
    const unsigned hardware = std::thread::hardware_concurrency();
    return hardware > 0 ? static_cast<int>(hardware) : 1;
}

size_t value_at(const std::string & line, const char * key) {
    size_t at = line.find(key);
    if (at == std::string::npos) {
        fail("request is missing a field");
    }
    at = line.find(':', at + std::strlen(key));
    if (at == std::string::npos) {
        fail("request field has no value");
    }
    return at + 1;
}

std::vector<int64_t> parse_ids(const std::string & line) {
    size_t at = line.find('[', value_at(line, "\"ids\""));
    if (at == std::string::npos) {
        fail("ids is not an array");
    }
    std::vector<int64_t> ids;
    const char * cursor = line.c_str() + at + 1;
    while (true) {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') {
            ++cursor;
        }
        if (*cursor == ']') {
            break;
        }
        char * end = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if (end == cursor) {
            fail("ids has a malformed entry");
        }
        ids.push_back(static_cast<int64_t>(value));
        cursor = end;
    }
    if (ids.empty()) {
        fail("request has no ids");
    }
    return ids;
}

long parse_int(const std::string & line, const char * key) {
    const char * cursor = line.c_str() + value_at(line, key);
    char * end = nullptr;
    const long value = std::strtol(cursor, &end, 10);
    if (end == cursor) {
        fail("request field is not an integer");
    }
    return value;
}

bool parse_bool(const std::string & line, const char * key) {
    const char * cursor = line.c_str() + value_at(line, key);
    while (*cursor == ' ') {
        ++cursor;
    }
    return std::strncmp(cursor, "true", 4) == 0;
}

void write_logits(const float * logits, int64_t count) {
    std::string out = "{\"logits\": [";
    out.reserve(static_cast<size_t>(count) * 12 + 16);
    char number[32];
    for (int64_t i = 0; i < count; ++i) {
        const int length = std::snprintf(number, sizeof(number), i == 0 ? "%.9g" : ",%.9g", logits[i]);
        out.append(number, static_cast<size_t>(length));
    }
    out += "]}\n";
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fflush(stdout);
}

const float * last_row(const Ort::Value & value, int64_t vocab) {
    const auto info = value.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (shape.size() != 3 || shape[0] != 1) {
        fail("logits has an unexpected shape");
    }
    return value.GetTensorData<float>() + (shape[1] - 1) * vocab;
}

}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    int threads = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else {
            fail("usage: onnx-worker --model <model.onnx> [--threads N]");
        }
    }
    if (model_path == nullptr) {
        fail("usage: onnx-worker --model <model.onnx> [--threads N]");
    }
    if (threads <= 0) {
        threads = default_threads();
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-worker");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(threads);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    const auto load_started = std::chrono::steady_clock::now();
    Ort::Session session(env, model_path, session_options);
    const std::string version = Ort::GetVersionString();
    Ort::AllocatorWithDefaultOptions allocator;

    const size_t input_count = session.GetInputCount();
    const size_t output_count = session.GetOutputCount();
    if (input_count < 4 || (input_count - 2) % 2 != 0 || output_count != input_count - 1) {
        fail("the model does not have a past key/value cache");
    }
    const int64_t layers = static_cast<int64_t>((input_count - 2) / 2);

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    input_names.reserve(input_count);
    output_names.reserve(output_count);
    for (size_t i = 0; i < input_count; ++i) {
        input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
    }
    for (size_t i = 0; i < output_count; ++i) {
        output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
    }
    if (input_names[0] != "input_ids" || input_names[1] != "attention_mask") {
        fail("the model does not start with input_ids and attention_mask");
    }
    std::vector<const char *> input_names_c;
    std::vector<const char *> output_names_c;
    for (const auto & name : input_names) {
        input_names_c.push_back(name.c_str());
    }
    for (const auto & name : output_names) {
        output_names_c.push_back(name.c_str());
    }

    const auto past_shape_raw =
        session.GetInputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape();
    const auto logits_shape_raw =
        session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (past_shape_raw.size() != 4 || logits_shape_raw.size() != 3) {
        fail("unexpected past or logits rank");
    }
    const int64_t heads = past_shape_raw[1];
    const int64_t head_dim = past_shape_raw[3];
    const int64_t vocab = logits_shape_raw[2];
    if (heads <= 0 || head_dim <= 0 || vocab <= 0) {
        fail("the model has dynamic head or vocabulary dimensions");
    }

    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_started).count();
    std::fprintf(stderr, "onnxruntime worker: %d threads, %lld layers, %lld heads, %lld vocab, %s\n",
                 threads, static_cast<long long>(layers), static_cast<long long>(heads),
                 static_cast<long long>(vocab), version.c_str());

    std::printf("{\"ready\": true, \"load_ms\": %.3f}\n", load_ms);
    std::fflush(stdout);

    const std::vector<int64_t> past_shape = {1, heads, 0, head_dim};
    float empty_past = 0.0f;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }
        std::vector<int64_t> ids = parse_ids(line);
        const int64_t max_new_tokens = parse_int(line, "\"max_new_tokens\"");
        const bool return_logits = parse_bool(line, "\"return_logits\"");

        std::vector<int64_t> mask(ids.size(), 1);
        std::vector<Ort::Value> inputs;
        inputs.reserve(input_count);
        std::vector<int64_t> prompt_shape = {1, static_cast<int64_t>(ids.size())};
        std::vector<int64_t> mask_shape = {1, static_cast<int64_t>(mask.size())};
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, ids.data(), ids.size(), prompt_shape.data(), prompt_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, mask.data(), mask.size(), mask_shape.data(), mask_shape.size()));
        for (int64_t i = 0; i < layers * 2; ++i) {
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info, &empty_past, 0, past_shape.data(), past_shape.size()));
        }

        std::vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{nullptr}, input_names_c.data(), inputs.data(), inputs.size(),
            output_names_c.data(), output_count);

        if (return_logits) {
            write_logits(last_row(outputs[0], vocab), vocab);
        }

        for (int64_t step = 0; step < max_new_tokens; ++step) {
            const float * logits = last_row(outputs[0], vocab);
            int64_t best = 0;
            float best_logit = logits[0];
            for (int64_t id = 1; id < vocab; ++id) {
                if (logits[id] > best_logit) {
                    best_logit = logits[id];
                    best = id;
                }
            }
            std::printf("{\"token\": %lld}\n", static_cast<long long>(best));
            std::fflush(stdout);
            if (step + 1 >= max_new_tokens) {
                break;
            }

            for (int64_t i = 0; i < layers * 2; ++i) {
                inputs[2 + i] = std::move(outputs[1 + i]);
            }
            ids.assign(1, best);
            mask.push_back(1);
            const std::vector<int64_t> token_shape = {1, 1};
            const std::vector<int64_t> full_mask_shape = {1, static_cast<int64_t>(mask.size())};
            inputs[0] = Ort::Value::CreateTensor<int64_t>(
                memory_info, ids.data(), ids.size(), token_shape.data(), token_shape.size());
            inputs[1] = Ort::Value::CreateTensor<int64_t>(
                memory_info, mask.data(), mask.size(), full_mask_shape.data(),
                full_mask_shape.size());
            outputs = session.Run(
                Ort::RunOptions{nullptr}, input_names_c.data(), inputs.data(), inputs.size(),
                output_names_c.data(), output_count);
        }

        std::printf("{\"done\": true}\n");
        std::fflush(stdout);
    }

    return 0;
}
