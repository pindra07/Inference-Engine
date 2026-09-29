# Llama-2 model subproject: engine lib + hardware plugins + CLI.
# See SPEC/09_MODEL_LAYOUT.md. New models copy this file's shape.

# ---- llama2 engine (static lib) ----
add_library(inference_llama2 STATIC
  src/weights.cc
  src/transformer.cc
  src/engine.cc
)
target_include_directories(inference_llama2 PUBLIC include)
target_link_libraries(inference_llama2 PUBLIC inference_common)

# ---- hardware plugins (one shared lib per target) ----
function(llama2_plugin target srcfile)
  add_library(${target} SHARED ${srcfile})
  target_link_libraries(${target} PRIVATE inference_common)
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME ${target}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/models/llama2/plugins
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/models/llama2/plugins
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/models/llama2/plugins)
endfunction()

llama2_plugin(plugin_llama2_cpu_baseline plugins/cpu_baseline/plugin_cpu.cc)
llama2_plugin(plugin_llama2_mac_mseries plugins/mac_mseries/plugin_mac.cc)
llama2_plugin(plugin_llama2_windows_intel plugins/windows_intel/plugin_intel.cc)

if(APPLE)
  target_link_libraries(plugin_llama2_mac_mseries PRIVATE "-framework Accelerate")
  # Old CBLAS entry points are deprecated since macOS 13.3 but stable.
  target_compile_options(plugin_llama2_mac_mseries PRIVATE -Wno-deprecated-declarations)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  if(NOT MSVC)
    target_compile_options(plugin_llama2_windows_intel PRIVATE -mavx2 -mfma)
  else()
    target_compile_options(plugin_llama2_windows_intel PRIVATE /arch:AVX2)
  endif()
endif()

# ---- example CLI ----
add_executable(run_llama examples/run_llama.cc)
target_link_libraries(run_llama PRIVATE inference_llama2)
set_target_properties(run_llama PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/models/llama2/examples)
