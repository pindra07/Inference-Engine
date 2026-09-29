# Benchmark tools: shared harness + one binary per model.
# Commands: perf/README.md. Methodology: SPEC/11_BENCHMARKING.md.

add_library(bench_common STATIC src/harness.cc)
target_include_directories(bench_common PUBLIC include)
if(WIN32)
  target_link_libraries(bench_common PRIVATE psapi)  # GetProcessMemoryInfo
endif()

function(model_bench target srcfile model_lib)
  add_executable(${target} ${srcfile})
  target_link_libraries(${target} PRIVATE ${model_lib} bench_common)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/perf)
endfunction()

model_bench(bench_llama llama2/bench_llama.cc inference_llama2)
model_bench(bench_gemma gemma2/bench_gemma.cc inference_gemma2)
# Next model: model_bench(bench_<m> <m>/bench_<m>.cc inference_<m>)
