cmake_minimum_required(VERSION 3.16)
project(llama_inference VERSION 0.2.0 LANGUAGES CXX)
# Layout: common/ (model-agnostic) + models/<name>/ (per-model engine,
# plugins, weights). See SPEC/09_MODEL_LAYOUT.md.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Plugins link the static common lib: dylibs (esp. arm64) require PIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra)
endif()

# ---- shared, model-agnostic library ----
add_library(inference_common STATIC
  common/src/core/plugin_loader.cc
  common/src/core/reference_backend.cc
  common/src/tokenizer/tokenizer.cc
  common/src/sampling/sampler.cc
  common/src/app/repl.cc
)
target_include_directories(inference_common PUBLIC common/include)

# ---- per-model subprojects ----
add_subdirectory(models/llama2)
add_subdirectory(models/gemma2)
# Next model: add_subdirectory(models/llama3)

# ---- benchmarking tools ----
add_subdirectory(perf)
