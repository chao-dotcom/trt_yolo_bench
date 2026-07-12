#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kInputW = 640;
constexpr int kInputH = 640;
constexpr int kNumClasses = 80;
constexpr int kNumPreds = 8400;
constexpr int kOutputChannels = 4 + kNumClasses;

const std::vector<int> kCoco80To91 = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[trt] " << msg << "\n";
    }
  }
};

struct SliceRecord {
  int image_id = -1;
  std::string file_name;
};

struct Detection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

struct LetterboxInfo {
  float scale = 1.0f;
  int pad_x = 0;
  int pad_y = 0;
  int src_w = 0;
  int src_h = 0;
};

struct AppConfig {
  std::string engine_path;
  std::string slice_json;
  std::string out_json = "results.json";
  std::string data_root = "data/val2017";
  float conf_thresh = 0.25f;
  float nms_iou_thresh = 0.45f;
  int warmup_iters = 20;
  int bench_iters = 300;
  int max_images = -1;
};

void CheckCuda(cudaError_t code, const std::string& msg) {
  if (code != cudaSuccess) {
    throw std::runtime_error(msg + ": " + cudaGetErrorString(code));
  }
}

size_t NumElements(const nvinfer1::Dims& dims) {
  size_t elements = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      throw std::runtime_error("Dynamic dimension unresolved");
    }
    elements *= static_cast<size_t>(dims.d[i]);
  }
  return elements;
}

std::vector<char> ReadBinaryFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Unable to open engine file: " + path);
  }
  file.seekg(0, std::ios::end);
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<size_t>(size));
  if (!file.read(data.data(), size)) {
    throw std::runtime_error("Unable to read engine file: " + path);
  }
  return data;
}

std::string ReadTextFile(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Unable to open text file: " + path);
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

std::vector<SliceRecord> ParseSliceJson(const std::string& path) {
  const std::string text = ReadTextFile(path);
  std::regex object_pattern("\\{[^\\}]*\\\"id\\\"\\s*:\\s*(\\d+)[^\\}]*\\\"file_name\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"[^\\}]*\\}");

  std::vector<SliceRecord> records;
  auto begin = std::sregex_iterator(text.begin(), text.end(), object_pattern);
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it) {
    SliceRecord record;
    record.image_id = std::stoi((*it)[1].str());
    record.file_name = (*it)[2].str();
    records.push_back(record);
  }

  if (records.empty()) {
    throw std::runtime_error("Failed to parse slice JSON records from: " + path);
  }
  return records;
}

std::vector<float> PreprocessLetterbox(const cv::Mat& bgr, LetterboxInfo* info) {
  if (bgr.empty()) {
    throw std::runtime_error("Input image is empty");
  }

  const int src_w = bgr.cols;
  const int src_h = bgr.rows;
  const float scale = std::min(static_cast<float>(kInputW) / static_cast<float>(src_w),
                               static_cast<float>(kInputH) / static_cast<float>(src_h));
  const int resized_w = std::max(1, static_cast<int>(std::round(src_w * scale)));
  const int resized_h = std::max(1, static_cast<int>(std::round(src_h * scale)));

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

  cv::Mat letterboxed(kInputH, kInputW, CV_8UC3, cv::Scalar(114, 114, 114));
  const int pad_x = (kInputW - resized_w) / 2;
  const int pad_y = (kInputH - resized_h) / 2;
  resized.copyTo(letterboxed(cv::Rect(pad_x, pad_y, resized_w, resized_h)));

  cv::Mat rgb;
  cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
  cv::Mat rgb_f32;
  rgb.convertTo(rgb_f32, CV_32FC3, 1.0 / 255.0);

  std::vector<float> chw(3 * kInputH * kInputW);
  for (int y = 0; y < kInputH; ++y) {
    const cv::Vec3f* row = rgb_f32.ptr<cv::Vec3f>(y);
    for (int x = 0; x < kInputW; ++x) {
      const cv::Vec3f px = row[x];
      const int idx = y * kInputW + x;
      chw[idx] = px[0];
      chw[kInputH * kInputW + idx] = px[1];
      chw[2 * kInputH * kInputW + idx] = px[2];
    }
  }

  if (info != nullptr) {
    info->scale = scale;
    info->pad_x = pad_x;
    info->pad_y = pad_y;
    info->src_w = src_w;
    info->src_h = src_h;
  }

  return chw;
}

float IoU(const Detection& a, const Detection& b) {
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);

  const float iw = std::max(0.0f, x2 - x1);
  const float ih = std::max(0.0f, y2 - y1);
  const float inter = iw * ih;

  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float uni = area_a + area_b - inter;
  if (uni <= 0.0f) {
    return 0.0f;
  }
  return inter / uni;
}

std::vector<Detection> GreedyNmsPerClass(const std::vector<Detection>& detections, float iou_thresh) {
  std::map<int, std::vector<Detection>> grouped;
  for (const Detection& det : detections) {
    grouped[det.class_id].push_back(det);
  }

  std::vector<Detection> kept;
  for (auto& kv : grouped) {
    auto& class_dets = kv.second;
    std::sort(class_dets.begin(), class_dets.end(), [](const Detection& lhs, const Detection& rhs) {
      return lhs.score > rhs.score;
    });

    std::vector<char> suppressed(class_dets.size(), 0);
    for (size_t i = 0; i < class_dets.size(); ++i) {
      if (suppressed[i]) {
        continue;
      }
      kept.push_back(class_dets[i]);
      for (size_t j = i + 1; j < class_dets.size(); ++j) {
        if (!suppressed[j] && IoU(class_dets[i], class_dets[j]) > iou_thresh) {
          suppressed[j] = 1;
        }
      }
    }
  }

  return kept;
}

std::vector<Detection> DecodeYoloOutput(const std::vector<float>& output, const LetterboxInfo& lb,
                                        float conf_thresh, float nms_iou_thresh) {
  if (output.size() < static_cast<size_t>(kOutputChannels * kNumPreds)) {
    throw std::runtime_error("Unexpected output size for YOLO decode");
  }

  std::vector<Detection> candidates;
  candidates.reserve(kNumPreds / 2);

  for (int i = 0; i < kNumPreds; ++i) {
    const float cx = output[0 * kNumPreds + i];
    const float cy = output[1 * kNumPreds + i];
    const float w = output[2 * kNumPreds + i];
    const float h = output[3 * kNumPreds + i];

    float best = -std::numeric_limits<float>::infinity();
    int cls = -1;
    for (int c = 0; c < kNumClasses; ++c) {
      const float score = output[(4 + c) * kNumPreds + i];
      if (score > best) {
        best = score;
        cls = c;
      }
    }

    if (best < conf_thresh || cls < 0) {
      continue;
    }

    float x1 = cx - 0.5f * w;
    float y1 = cy - 0.5f * h;
    float x2 = cx + 0.5f * w;
    float y2 = cy + 0.5f * h;

    x1 = (x1 - static_cast<float>(lb.pad_x)) / lb.scale;
    y1 = (y1 - static_cast<float>(lb.pad_y)) / lb.scale;
    x2 = (x2 - static_cast<float>(lb.pad_x)) / lb.scale;
    y2 = (y2 - static_cast<float>(lb.pad_y)) / lb.scale;

    x1 = std::clamp(x1, 0.0f, static_cast<float>(lb.src_w - 1));
    y1 = std::clamp(y1, 0.0f, static_cast<float>(lb.src_h - 1));
    x2 = std::clamp(x2, 0.0f, static_cast<float>(lb.src_w - 1));
    y2 = std::clamp(y2, 0.0f, static_cast<float>(lb.src_h - 1));

    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    Detection d;
    d.x1 = x1;
    d.y1 = y1;
    d.x2 = x2;
    d.y2 = y2;
    d.score = best;
    d.class_id = cls;
    candidates.push_back(d);
  }

  return GreedyNmsPerClass(candidates, nms_iou_thresh);
}

std::string EscapeJson(const std::string& s) {
  std::ostringstream out;
  for (char ch : s) {
    if (ch == '"') {
      out << "\\\"";
    } else if (ch == '\\') {
      out << "\\\\";
    } else {
      out << ch;
    }
  }
  return out.str();
}

void WriteCocoResultsJson(const std::string& output_path,
                          const std::vector<std::pair<int, Detection>>& rows) {
  std::ofstream out(output_path);
  if (!out) {
    throw std::runtime_error("Unable to open output JSON for writing: " + output_path);
  }

  out << "[\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const int image_id = rows[i].first;
    const Detection& d = rows[i].second;
    const float w = d.x2 - d.x1;
    const float h = d.y2 - d.y1;

    out << "  {\"image_id\": " << image_id
        << ", \"category_id\": " << kCoco80To91[d.class_id]
        << ", \"bbox\": [" << std::fixed << std::setprecision(3) << d.x1 << ", " << d.y1
        << ", " << w << ", " << h << "]"
        << ", \"score\": " << std::setprecision(6) << d.score << "}";
    if (i + 1 != rows.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "]\n";
}

AppConfig ParseArgs(int argc, char** argv) {
  if (argc < 3) {
    throw std::runtime_error(
        "Usage: trt_yolo_bench <engine_path> <slice_json> [--out results.json] "
        "[--data-root data/val2017] [--conf 0.25] [--nms 0.45] [--warmup 20] "
        "[--iters 300] [--max-images -1]");
  }

  AppConfig cfg;
  cfg.engine_path = argv[1];
  cfg.slice_json = argv[2];

  for (int i = 3; i < argc; ++i) {
    const std::string key = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for argument: " + key);
      }
      ++i;
      return argv[i];
    };

    if (key == "--out") {
      cfg.out_json = next();
    } else if (key == "--data-root") {
      cfg.data_root = next();
    } else if (key == "--conf") {
      cfg.conf_thresh = std::stof(next());
    } else if (key == "--nms") {
      cfg.nms_iou_thresh = std::stof(next());
    } else if (key == "--warmup") {
      cfg.warmup_iters = std::stoi(next());
    } else if (key == "--iters") {
      cfg.bench_iters = std::stoi(next());
    } else if (key == "--max-images") {
      cfg.max_images = std::stoi(next());
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }

  return cfg;
}

double PercentileMs(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double idx = p * static_cast<double>(values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  const double frac = idx - static_cast<double>(lo);
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const AppConfig cfg = ParseArgs(argc, argv);

    Logger logger;
    const std::vector<char> engine_data = ReadBinaryFile(cfg.engine_path);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (runtime == nullptr) {
      throw std::runtime_error("Failed to create TensorRT runtime");
    }

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (engine == nullptr) {
      runtime->destroy();
      throw std::runtime_error("Failed to deserialize TensorRT engine");
    }

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (context == nullptr) {
      engine->destroy();
      runtime->destroy();
      throw std::runtime_error("Failed to create execution context");
    }

    if (engine->getNbBindings() != 2) {
      std::cerr << "[warn] expected 2 bindings (input/output), got " << engine->getNbBindings() << "\n";
    }

    int input_idx = -1;
    int output_idx = -1;
    for (int i = 0; i < engine->getNbBindings(); ++i) {
      if (engine->bindingIsInput(i)) {
        input_idx = i;
      } else {
        output_idx = i;
      }
    }
    if (input_idx < 0 || output_idx < 0) {
      throw std::runtime_error("Failed to identify input/output bindings");
    }

    nvinfer1::Dims input_dims = engine->getBindingDimensions(input_idx);
    if (input_dims.nbDims == 4 && input_dims.d[0] == -1) {
      input_dims.d[0] = 1;
      if (!context->setBindingDimensions(input_idx, input_dims)) {
        throw std::runtime_error("Failed to set dynamic input dimensions");
      }
    }

    const nvinfer1::Dims resolved_input_dims = context->getBindingDimensions(input_idx);
    const nvinfer1::Dims resolved_output_dims = context->getBindingDimensions(output_idx);
    const size_t input_elems = NumElements(resolved_input_dims);
    const size_t output_elems = NumElements(resolved_output_dims);

    std::cout << "[info] input dims:";
    for (int i = 0; i < resolved_input_dims.nbDims; ++i) {
      std::cout << " " << resolved_input_dims.d[i];
    }
    std::cout << "\n";

    std::cout << "[info] output dims:";
    for (int i = 0; i < resolved_output_dims.nbDims; ++i) {
      std::cout << " " << resolved_output_dims.d[i];
    }
    std::cout << "\n";

    if (output_elems < static_cast<size_t>(kOutputChannels * kNumPreds)) {
      throw std::runtime_error("Output tensor smaller than expected YOLO shape");
    }

    void* device_input = nullptr;
    void* device_output = nullptr;
    CheckCuda(cudaMalloc(&device_input, input_elems * sizeof(float)), "cudaMalloc input failed");
    CheckCuda(cudaMalloc(&device_output, output_elems * sizeof(float)), "cudaMalloc output failed");

    cudaStream_t stream = nullptr;
    CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate failed");

    std::vector<void*> bindings(engine->getNbBindings(), nullptr);
    bindings[input_idx] = device_input;
    bindings[output_idx] = device_output;

    std::vector<SliceRecord> records = ParseSliceJson(cfg.slice_json);
    if (cfg.max_images > 0 && static_cast<int>(records.size()) > cfg.max_images) {
      records.resize(static_cast<size_t>(cfg.max_images));
    }

    if (records.empty()) {
      throw std::runtime_error("No records selected for benchmarking");
    }

    // Cache a small pool of preprocessed tensors so timing excludes disk I/O.
    const size_t cache_count = std::min<size_t>(records.size(), 64);
    std::vector<std::vector<float>> cached_inputs;
    cached_inputs.reserve(cache_count);
    std::vector<LetterboxInfo> cached_lb;
    cached_lb.reserve(cache_count);

    for (size_t i = 0; i < cache_count; ++i) {
      const std::string image_path = cfg.data_root + "/" + records[i].file_name;
      cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
      if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
      }
      LetterboxInfo lb;
      cached_inputs.push_back(PreprocessLetterbox(image, &lb));
      cached_lb.push_back(lb);
    }

    for (int i = 0; i < cfg.warmup_iters; ++i) {
      const size_t idx = static_cast<size_t>(i) % cache_count;
      CheckCuda(cudaMemcpyAsync(device_input, cached_inputs[idx].data(), input_elems * sizeof(float),
                                cudaMemcpyHostToDevice, stream),
                "Warmup H2D failed");
      if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
        throw std::runtime_error("enqueueV2 failed during warmup");
      }
      CheckCuda(cudaStreamSynchronize(stream), "Warmup synchronize failed");
    }

    std::vector<double> inference_ms;
    inference_ms.reserve(static_cast<size_t>(cfg.bench_iters));
    for (int i = 0; i < cfg.bench_iters; ++i) {
      const size_t idx = static_cast<size_t>(i) % cache_count;
      CheckCuda(cudaMemcpyAsync(device_input, cached_inputs[idx].data(), input_elems * sizeof(float),
                                cudaMemcpyHostToDevice, stream),
                "Benchmark H2D failed");

      const auto start = std::chrono::high_resolution_clock::now();
      if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
        throw std::runtime_error("enqueueV2 failed during benchmark");
      }
      CheckCuda(cudaStreamSynchronize(stream), "Benchmark synchronize failed");
      const auto end = std::chrono::high_resolution_clock::now();

      const double ms = std::chrono::duration<double, std::milli>(end - start).count();
      inference_ms.push_back(ms);
    }

    const double mean_ms = std::accumulate(inference_ms.begin(), inference_ms.end(), 0.0) /
                           static_cast<double>(inference_ms.size());
    const double p50_ms = PercentileMs(inference_ms, 0.50);
    const double p99_ms = PercentileMs(inference_ms, 0.99);
    const double fps = 1000.0 / mean_ms;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[bench] mean_ms=" << mean_ms << " p50_ms=" << p50_ms << " p99_ms=" << p99_ms
              << " fps=" << fps << "\n";

    std::vector<float> host_output(output_elems, 0.0f);
    std::vector<std::pair<int, Detection>> coco_rows;

    for (const SliceRecord& record : records) {
      const std::string image_path = cfg.data_root + "/" + record.file_name;
      cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
      if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
      }

      LetterboxInfo lb;
      std::vector<float> input = PreprocessLetterbox(image, &lb);

      CheckCuda(cudaMemcpyAsync(device_input, input.data(), input_elems * sizeof(float),
                                cudaMemcpyHostToDevice, stream),
                "Eval H2D failed");
      if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
        throw std::runtime_error("enqueueV2 failed during eval");
      }
      CheckCuda(cudaMemcpyAsync(host_output.data(), device_output, output_elems * sizeof(float),
                                cudaMemcpyDeviceToHost, stream),
                "Eval D2H failed");
      CheckCuda(cudaStreamSynchronize(stream), "Eval synchronize failed");

      const std::vector<Detection> kept = DecodeYoloOutput(host_output, lb, cfg.conf_thresh, cfg.nms_iou_thresh);
      for (const Detection& d : kept) {
        coco_rows.emplace_back(record.image_id, d);
      }
    }

    WriteCocoResultsJson(cfg.out_json, coco_rows);
    std::cout << "[write] " << cfg.out_json << " detections=" << coco_rows.size() << "\n";

    cudaStreamDestroy(stream);
    cudaFree(device_output);
    cudaFree(device_input);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[error] " << ex.what() << "\n";
    return 1;
  }
}
