# Gemma2 model subproject: engine lib + hardware plugins + CLI.
# Same shape as models/llama2 (see SPEC/09_MODEL_LAYOUT.md).

# ---- gemma2 engine (static lib) ----
add_library(inference_gemma2 STATIC
  src/weights.cc
  src/transformer.cc
  src/engine.cc
)
target_include_directories(inference_gemma2 PUBLIC include)
target_link_libraries(inference_gemma2 PUBLIC inference_common)

# ---- hardware plugins (one shared lib per target) ----
function(gemma2_plugin target srcfile)
  add_library(${target} SHARED ${srcfile})
  target_link_libraries(${target} PRIVATE inference_common)
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME ${target}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/models/gemma2/plugins
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/models/gemma2/plugins
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/models/gemma2/plugins)
endfunction()

gemma2_plugin(plugin_gemma2_cpu_baseline plugins/cpu_baseline/plugin_cpu.cc)
gemma2_plugin(plugin_gemma2_mac_mseries plugins/mac_mseries/plugin_mac.cc)
gemma2_plugin(plugin_gemma2_windows_intel plugins/windows_intel/plugin_intel.cc)

if(APPLE)
  target_link_libraries(plugin_gemma2_mac_mseries PRIVATE "-framework Accelerate")
  target_compile_options(plugin_gemma2_mac_mseries PRIVATE -Wno-deprecated-declarations)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  if(NOT MSVC)
    target_compile_options(plugin_gemma2_windows_intel PRIVATE -mavx2 -mfma)
  else()
    target_compile_options(plugin_gemma2_windows_intel PRIVATE /arch:AVX2)
  endif()
endif()

# ---- example CLI ----
add_executable(run_gemma examples/run_gemma.cc)
target_link_libraries(run_gemma PRIVATE inference_gemma2)
set_target_properties(run_gemma PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/models/gemma2/examples)
