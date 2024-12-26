# Model Quantization Documentation

Quantization is among the more useful recent innovations in machine learning model inference, allowing large models to be compressed and run on common consumer hardware. I could not find a comprehensive explanation of what it is, how it works, its limitations, or a comparison of different types of quantization, so I started this repository so that others may find it useful.

Maarten Grootendorst wrote a great but general overview with plenty of visual aids to describe quantization, the problems it solves, and some of the problems that occur with quantization: [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)

## Overview

### What is Quantization?

Quantization is the lossy compression of the floating-point values that make up a machine learning model.

### Why Quantize?

Generally, to lower the amount of memory needed to run a model.

#### To Overcome Memory Limitations

A typical open-weights model like the Llama 3.1 8b-instruct BF16 model is 16 gigabytes, which only [7%](https://store.steampowered.com/hwsurvey/Steam-Hardware-Software-Survey-Welcome-to-Steam) of users can run. However, by quantizing the weights to the Q6_K format, the model can be reduced to 6.6 gigabytes, which is small enough for 64% of users to run. With quantization, the median graphics card owner with at least 8 GB of graphics memory can now run this model on their own hardware.

#### To Increase Processing and Generation Speed

Memory bandwidth is most often the bottleneck when running models, as the entire model is used for processing input. By reducing model size, it takes less time for layers to load.

By reducing the amount of memory needed to run a model, overall throughput can be increased by allowing a larger batch size, and latency (time to first token) can be reduced.

#### To Increase Context Length

Memory usage scales linearly with the number of tokens and with the amount of memory needed for the attention layer. By quantizing the attention layer, a larger context can fit in memory.

## Quantization Formats

* [GGUF](./GGUF/README.md)- GPT-Generated Unified Format
  - This format is supported by `llama.cpp`, `ollama`, `vLLM`, `aphrodite-engine` and `transformers`.
  - GGUF models can easily be quantized using the [tutorial](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md) from the main [`llama.cpp`](https://github.com/ggerganov/llama.cpp) repository.
* [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) - BitsandBytes
  - Supported by PyTorch, `bitsandbytes`, `aphrodite-engine`, and `transformers`.
* [AWQ](https://github.com/mit-han-lab/llm-awq) - Activation-aware Weight Quantization, which quantizes all but 1% of weights.
  - This format is supported by Huggingface's `transformers` library if `AutoAWQ` is installed, `vLLM`, `TensorRT-LLM`, and `aphrodite-engine`.
  - This quantization format works best when you have a calibration dataset, though it is not required.
* [GPTQ](https://huggingface.co/blog/gptq-integration) - Generative Post-Training Quantization
  - This format is supported by `AutoGPTQ`, `ExLlamaV2`, and `TensorRT-LLM`
  - Huggingface models can be converted to GPTQ format using the [`AutoGPTQ`](https://github.com/AutoGPTQ/AutoGPTQ) Python library.
  - `AutoGPTQ` quantization may require a calibration dataset. It's unclear from the docs, though all examples show one being used.
* [FBGEMM](https://huggingface.co/docs/transformers/en/quantization/fbgemm_fp8) - Facebook General Matrix Multiplication
  - The Pytorch-native quantization format, supporting 4-bit and 8-bit weights.
* [EXL2](https://github.com/turboderp/exllamav2#exl2-quantization)- ExLlamaV2 quantization format
  - This is GPTQ but with support for mixed-precision and different weight sizes. It's supported by `ExLlamaV2`.
  - A model can be converted to the `EXL2` format using the official `ExLlamaV2` [conversion script](https://github.com/turboderp/exllamav2/blob/master/doc/convert.md).
  - `EXL2` requires a calibration dataset for quantization, though if one is not provided, a default dataset will be used..
* [SmoothQuant](https://github.com/mit-han-lab/smoothquant) - SmoothQuant is an INT8 Quantization format
  - Supported by `TensorRT-LLM`, `aphrodite-engine`, and `onnxruntime`.
  - A model can be converted to the `SmoothQuant` format using the official [conversion script](https://github.com/mit-han-lab/smoothquant/blob/main/examples/smoothquant_opt_demo.ipynb).
  - `SmoothQuant` requires a calibration dataset to generate activation channel scales before quantizing, though many are provided for some model architectures.
* [AQLM](https://github.com/Vahe1994/AQLM) - Additive Quantization of Language Models
  - A member of the Multi-Codebook Quantization family.
  - Supported by `AQLM`, `aphrodite-engine`, and usable with `transformers` with the `AQLM` library.
  - The main page of the [`AQLM`](https://github.com/Vahe1994/AQLM) repository has a guide for quantizing models to the `AQLM` format.
  - This quantization format works best when you have some of the model's original training data for calibration, though it is not required.
  * [HQQ](https://github.com/mobiusml/hqq) - Half-Quadratic Quantization
  - Supported by the `hqq` library, `transformers`, and `oobabooga`.
  - The main page of the [`HQQ`](https://github.com/mobiusml/hqq) repository has a guide for quantizing to this format.
  - No need for calibration data.
* [SpPR](https://github.com/Vahe1994/SpQR) - Sparse-Quantized Representation
  - Supported by the `SpQR` library.
  - The main page of the [`SpQR`](https://github.com/Vahe1994/SpQR) repository has a guide for quantizing to this format.
  - This quantization format works best when you have some of the model's original training data for calibration, though it is not required.