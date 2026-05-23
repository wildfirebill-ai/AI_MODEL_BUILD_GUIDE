# WebLLM (MLC)

**Version:** 0.2+ (npm package `@mlc-ai/web-llm`)

**Purpose:** Run LLMs entirely in the browser using WebGPU, with no server-side inference required. WebLLM compiles models via MLC's TVM-based framework into WebGPU shaders, enabling client-side chat, completion, and embedding. Useful for demos, edge-deployed models, and privacy-sensitive applications where data never leaves the client.

## Installation

```bash
npm install @mlc-ai/web-llm
```

Or via CDN:
```html
<script type="module" src="https://esm.run/@mlc-ai/web-llm"></script>
```

**Requirements:**
- Chrome 113+ with WebGPU enabled
- ~4 GB GPU memory minimum
- HTTPS or localhost (WebGPU requires secure context)

## Basic Usage

```javascript
import { CreateMLCEngine } from "@mlc-ai/web-llm";

// Create engine — downloads model on first run
const engine = await CreateMLCEngine("Llama-3.2-1B-q4f32_1-MLC");

// Chat completion
const reply = await engine.chat.completions.create({
    messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "What is machine learning?" },
    ],
    max_tokens: 256,
    temperature: 0.7,
    top_p: 0.95,
});
console.log(reply.choices[0].message.content);

// Streaming
const stream = await engine.chat.completions.create({
    messages: [{ role: "user", content: "Tell me a story" }],
    stream: true,
});
for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || "");
}
```

## Advanced Usage / Configuration

### Model selection and performance
| Model | Size | RAM | Speed (RTX 4090) |
|-------|------|-----|------------------|
| `Llama-3.2-1B-q4f32_1-MLC` | ~700 MB | 4 GB | 30-50 tok/s |
| `Llama-3.2-3B-q4f32_1-MLC` | ~2 GB | 8 GB | 15-25 tok/s |
| `Qwen2-0.5B-Instruct-q4f16_1-MLC` | ~400 MB | 4 GB | 40-60 tok/s |
| `Llama-3.1-8B-q4f16_1-MLC` | ~4.5 GB | 16 GB | 5-10 tok/s |

### Custom model compilation
```bash
# Compile your own model for WebLLM
pip install mlc-ai-nightly
mlc_chat gen_config ./model --quantization q4f16_1
mlc_chat compile --device webgpu
```

### Progress tracking
```javascript
const engine = await CreateMLCEngine("Llama-3.2-1B-q4f32_1-MLC", {
    initProgressCallback: (progress) => {
        console.log(`Loading: ${progress.text} (${(progress.progress * 100).toFixed(0)}%)`);
    },
});
```

## Integration with WFB Model
WebLLM enables browser-based demos of the WFB model via:
- `demos/webllm/` — web demo showing inference with WFB model compiled to WebGPU
- Model weights are converted via MLC's compilation pipeline using the `scripts/convert_to_webllm.py` script
- The compiled web weight files are hosted on HuggingFace or a CDN for client-side loading

## Common Pitfalls / Troubleshooting
- **WebGPU not available:** Check `navigator.gpu` in console; requires Chrome 113+, Edge 113+; not available in Firefox or Safari as of 2026
- **Model download fails:** CORS issue — serve the model weights from a CORS-enabled CDN; HuggingFace's CDN works by default
- **Out of memory on low-end GPUs:** Use smaller models (Qwen2-0.5B) or increase quantization (q4f16_1 uses less memory than q4f32_1)
- **Slow first load:** WebGPU shader compilation is cached after first run — subsequent loads are faster; consider pre-warming during app startup
- **Browser tab crash:** GPU process crashed — reduce `max_tokens`, reduce model size, or ensure the page is focused (some browsers throttle background tabs)
- **`CreateMLCEngine` hangs:** Check the browser's WebGPU limit — some browsers cap GPU memory to 1-2 GB; try a smaller model

## Documentation
- https://github.com/mlc-ai/web-llm
- https://webllm.mlc.ai/
- https://mlc.ai/
