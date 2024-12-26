# Graphs showing performance

Perplexity, Model size/bits-per-weight, Token generation speed


## Bit Packing

It would be nice to know the exact serialization format on disk.

* https://github.com/ggerganov/llama.cpp/pull/8151#issuecomment-2256706172
* https://github.com/ggerganov/llama.cpp/wiki/Tensor-Encoding-Schemes


## Importance Matrix Calculation

* [llama.cpp#4861](https://github.com/ggerganov/llama.cpp/pull/4861), ikawrakow's PR for adding importance matrix calculation to llama.cpp
* It'd be great to have a more thorough explanation of what this is and how it helps reduce error when quantizing weights.

* (legacy quants) https://github.com/ggerganov/llama.cpp/pull/4969
* (k-quants) https://github.com/ggerganov/llama.cpp/pull/4930

## Precision types by Model Part

Ex. the different precision types used for each part of a model in GGUF:
 * [Q4_K_S](https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF?show_file_info=Llama-3.3-70B-Instruct-Q4_K_S.gguf) vs [Q4_K_M](https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF?show_file_info=Llama-3.3-70B-Instruct-Q4_K_M.gguf) vs [Q4_K_L](https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF?show_file_info=Llama-3.3-70B-Instruct-Q4_K_L.gguf)

    * These all use Q4_K, Q5_K, or Q6_K for different parts of the attention layers, and all have different token embedding weight and output weight quantization types.