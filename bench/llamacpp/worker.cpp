#define _GNU_SOURCE
#include "llama.h"

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
    std::fprintf(stderr, "llama-worker: %s\n", message);
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

std::vector<llama_token> parse_ids(const std::string & line) {
    size_t at = line.find('[', value_at(line, "\"ids\""));
    if (at == std::string::npos) {
        fail("ids is not an array");
    }
    std::vector<llama_token> ids;
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
        ids.push_back(static_cast<llama_token>(value));
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

void write_logits(const float * logits, int count) {
    std::string out = "{\"logits\": [";
    out.reserve(static_cast<size_t>(count) * 12 + 16);
    char number[32];
    for (int i = 0; i < count; ++i) {
        const int length = std::snprintf(number, sizeof(number), i == 0 ? "%.9g" : ",%.9g", logits[i]);
        out.append(number, static_cast<size_t>(length));
    }
    out += "]}\n";
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fflush(stdout);
}

bool decode(llama_context * ctx, llama_token * tokens, int count) {
    return llama_decode(ctx, llama_batch_get_one(tokens, count)) == 0;
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
            fail("usage: llama-worker --model <file.gguf> [--threads N]");
        }
    }
    if (model_path == nullptr) {
        fail("usage: llama-worker --model <file.gguf> [--threads N]");
    }
    if (threads <= 0) {
        threads = default_threads();
    }

    llama_backend_init();

    const auto load_started = std::chrono::steady_clock::now();
    const llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        fail("cannot load model");
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 1024;
    context_params.n_batch = 512;
    context_params.n_ubatch = 512;
    context_params.n_threads = threads;
    context_params.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (ctx == nullptr) {
        fail("cannot create context");
    }

    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_started).count();
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::fprintf(stderr, "llama.cpp worker: %d threads, %d vocab, %s\n", threads, n_vocab,
                 llama_version());

    std::printf("{\"ready\": true, \"load_ms\": %.3f}\n", load_ms);
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }
        std::vector<llama_token> ids = parse_ids(line);
        const int max_new_tokens = static_cast<int>(parse_int(line, "\"max_new_tokens\""));
        const bool return_logits = parse_bool(line, "\"return_logits\"");

        llama_memory_clear(llama_get_memory(ctx), true);
        if (!decode(ctx, ids.data(), static_cast<int>(ids.size()))) {
            fail("prompt decode failed");
        }
        if (return_logits) {
            write_logits(llama_get_logits_ith(ctx, -1), n_vocab);
        }

        for (int step = 0; step < max_new_tokens; ++step) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (logits == nullptr) {
                fail("no logits after decode");
            }
            llama_token best = 0;
            float best_logit = logits[0];
            for (int id = 1; id < n_vocab; ++id) {
                if (logits[id] > best_logit) {
                    best_logit = logits[id];
                    best = id;
                }
            }
            std::printf("{\"token\": %d}\n", static_cast<int>(best));
            std::fflush(stdout);
            if (step + 1 < max_new_tokens && !decode(ctx, &best, 1)) {
                fail("token decode failed");
            }
        }

        std::printf("{\"done\": true}\n");
        std::fflush(stdout);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
