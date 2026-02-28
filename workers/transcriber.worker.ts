import {
  pipeline,
  type AutomaticSpeechRecognitionPipeline,
  env,
} from "@huggingface/transformers";

type WorkerRequest =
  | { type: "LOAD_MODEL" }
  | { type: "TRANSCRIBE"; payload: Float32Array };

type WorkerResponse =
  | { status: "loading" }
  | { status: "ready"; device: "webgpu" | "wasm" }
  | { status: "progress"; progress: number; file?: string }
  | { status: "transcribing" }
  | { status: "complete"; text: string }
  | { status: "error"; error: string };

const MODEL_ID = "onnx-community/whisper-tiny.en";
const ctx: Worker = self as unknown as Worker;

let transcriber: AutomaticSpeechRecognitionPipeline | null = null;
let loadPromise: Promise<void> | null = null;
const createAsrPipeline = pipeline as unknown as (
  task: "automatic-speech-recognition",
  model: string,
  options: {
    device: "webgpu" | "wasm";
    progress_callback: (progressItem: { progress?: number; file?: string }) => void;
  },
) => Promise<AutomaticSpeechRecognitionPipeline>;

env.allowLocalModels = false;

async function initTranscriber() {
  if (transcriber) return;
  if (loadPromise) return loadPromise;

  ctx.postMessage({ status: "loading" } satisfies WorkerResponse);

  loadPromise = (async () => {
    const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;
    const tryDevice = hasWebGPU ? (["webgpu", "wasm"] as const) : (["wasm"] as const);
    let lastError: unknown = null;

    for (const device of tryDevice) {
      try {
        transcriber = await createAsrPipeline("automatic-speech-recognition", MODEL_ID, {
          device,
          progress_callback: (progressItem) => {
            const progress =
              typeof progressItem.progress === "number" ? progressItem.progress : 0;
            ctx.postMessage({
              status: "progress",
              progress,
              file: progressItem.file,
            } satisfies WorkerResponse);
          },
        });
        ctx.postMessage({ status: "ready", device } satisfies WorkerResponse);
        return;
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError instanceof Error ? lastError : new Error(String(lastError));
  })();

  try {
    await loadPromise;
  } finally {
    loadPromise = null;
  }
}

ctx.addEventListener("message", async (event: MessageEvent<WorkerRequest>) => {
  const { type } = event.data;

  try {
    if (type === "LOAD_MODEL") {
      await initTranscriber();
      return;
    }

    if (type === "TRANSCRIBE") {
      if (!transcriber) {
        await initTranscriber();
      }
      if (!transcriber) {
        throw new Error("Transcriber failed to initialize.");
      }

      ctx.postMessage({ status: "transcribing" } satisfies WorkerResponse);

      const result = await transcriber(event.data.payload, {
        chunk_length_s: 30,
        stride_length_s: 5,
        return_timestamps: false,
      });
      const text = Array.isArray(result)
        ? result.map((item) => item.text).join(" ")
        : result.text;

      ctx.postMessage({
        status: "complete",
        text: text.trim(),
      } satisfies WorkerResponse);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    ctx.postMessage({ status: "error", error: message } satisfies WorkerResponse);
  }
});
