"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type WorkerMessage =
  | { status: "loading" }
  | { status: "ready"; device: "webgpu" | "wasm" }
  | { status: "progress"; progress: number; file?: string }
  | { status: "transcribing" }
  | { status: "complete"; text: string }
  | { status: "error"; error: string };

const SAMPLE_RATE = 16_000;
const MAX_BUFFER_SECONDS = 28;
const MAX_BUFFER_SAMPLES = SAMPLE_RATE * MAX_BUFFER_SECONDS;
const MIN_COMMIT_SECONDS = 10;
const MIN_COMMIT_SAMPLES = SAMPLE_RATE * MIN_COMMIT_SECONDS;
const MAX_COMMIT_SECONDS = 25;
const MAX_COMMIT_SAMPLES = SAMPLE_RATE * MAX_COMMIT_SECONDS;
const INFERENCE_INTERVAL_MS = 2500;
const MIN_REMAINING_CHARS = 15;

/**
 * Find the last position where a sentence ends *internally* — i.e. a period,
 * exclamation, or question mark followed by enough remaining text that we're
 * confident it's a real boundary (not just Whisper's trailing punctuation).
 * Returns the character index to split at, or -1 if none found.
 */
function findInternalSentenceSplit(text: string): number {
  const re = /[.!?]["'']?\s+/g;
  let lastSplitPos = -1;
  let match;
  while ((match = re.exec(text)) !== null) {
    const candidatePos = match.index + match[0].length;
    if (text.slice(candidatePos).trim().length >= MIN_REMAINING_CHARS) {
      lastSplitPos = candidatePos;
    }
  }
  return lastSplitPos;
}

function appendToBuffer(existing: Float32Array, incoming: Float32Array, maxSamples: number) {
  const totalLength = existing.length + incoming.length;
  const merged = new Float32Array(totalLength);
  merged.set(existing, 0);
  merged.set(incoming, existing.length);

  if (merged.length <= maxSamples) {
    return merged;
  }

  return merged.slice(merged.length - maxSamples);
}

function downsampleTo16k(input: Float32Array, inputRate: number) {
  if (inputRate === SAMPLE_RATE) return input;

  const ratio = inputRate / SAMPLE_RATE;
  const outputLength = Math.round(input.length / ratio);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i += 1) {
    const sourceIndex = Math.floor(i * ratio);
    output[i] = input[sourceIndex] ?? 0;
  }

  return output;
}

export function TranscriptionApp() {
  const [modelState, setModelState] = useState<"loading" | "ready" | "error">("loading");
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [device, setDevice] = useState<"webgpu" | "wasm" | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [committedText, setCommittedText] = useState("");
  const [liveText, setLiveText] = useState("");

  const workerRef = useRef<Worker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const workletLoadedRef = useRef(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const rollingBufferRef = useRef<Float32Array>(new Float32Array(0));
  const queuedInferenceRef = useRef(false);
  const workerBusyRef = useRef(false);
  const shouldFinalizeRef = useRef(false);
  const totalSamplesRef = useRef(0);
  const lastCommitSamplesRef = useRef(0);
  const isCapturingRef = useRef(false);
  const lastInferenceAtRef = useRef(0);
  const liveTextRef = useRef("");
  const modelStateRef = useRef<"loading" | "ready" | "error">("loading");
  const flushInferenceQueue = useCallback(() => {
    const worker = workerRef.current;
    if (!worker || modelStateRef.current !== "ready") return;
    if (workerBusyRef.current || !queuedInferenceRef.current) return;

    queuedInferenceRef.current = false;
    if (rollingBufferRef.current.length === 0) return;

    workerBusyRef.current = true;
    setIsTranscribing(true);
    worker.postMessage({
      type: "TRANSCRIBE",
      payload: rollingBufferRef.current.slice(),
    });
  }, []);

  const queueInference = useCallback(() => {
    queuedInferenceRef.current = true;
    flushInferenceQueue();
  }, [flushInferenceQueue]);

  const stopMicrophoneTracks = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  const stopAudioProcessing = useCallback(() => {
    const workletNode = workletNodeRef.current;
    if (workletNode) {
      workletNode.port.onmessage = null;
      workletNode.port.close();
      workletNode.disconnect();
      workletNodeRef.current = null;
    }

    const sourceNode = sourceNodeRef.current;
    if (sourceNode) {
      sourceNode.disconnect();
      sourceNodeRef.current = null;
    }

    isCapturingRef.current = false;
    stopMicrophoneTracks();
  }, [stopMicrophoneTracks]);

  const maybeQueueInference = useCallback(
    (force = false) => {
      if (force) {
        queueInference();
        return;
      }

      const now = performance.now();
      if (now - lastInferenceAtRef.current < INFERENCE_INTERVAL_MS) return;
      lastInferenceAtRef.current = now;
      queueInference();
    },
    [queueInference],
  );

  const finalizeRecording = useCallback(() => {
    const finalLiveText = liveTextRef.current.trim();
    if (finalLiveText) {
      setCommittedText((previous) =>
        previous ? `${previous}\n\n${finalLiveText}` : finalLiveText,
      );
    }
    setLiveText("");
    liveTextRef.current = "";
    rollingBufferRef.current = new Float32Array(0);
    shouldFinalizeRef.current = false;
    setIsRecording(false);
  }, []);

  useEffect(() => {
    liveTextRef.current = liveText;
  }, [liveText]);

  useEffect(() => {
    modelStateRef.current = modelState;
  }, [modelState]);

  useEffect(() => {
    const worker = new Worker(
      new URL("../workers/transcriber.worker.ts", import.meta.url),
      { type: "module" },
    );
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const message = event.data;

      if (message.status === "loading") {
        setModelState("loading");
        return;
      }

      if (message.status === "progress") {
        const normalized = message.progress > 1 ? message.progress : message.progress * 100;
        setDownloadProgress(Math.max(0, Math.min(100, normalized)));
        return;
      }

      if (message.status === "ready") {
        setModelState("ready");
        setDevice(message.device);
        setError(null);
        setDownloadProgress(100);
        return;
      }

      if (message.status === "transcribing") {
        setIsTranscribing(true);
        return;
      }

      if (message.status === "complete") {
        workerBusyRef.current = false;
        setIsTranscribing(false);

        if (!shouldFinalizeRef.current) {
          const samplesSinceCommit = totalSamplesRef.current - lastCommitSamplesRef.current;
          const enoughAudio = samplesSinceCommit >= MIN_COMMIT_SAMPLES;
          const overdue = samplesSinceCommit >= MAX_COMMIT_SAMPLES;
          const text = message.text.trim();
          const splitPos = findInternalSentenceSplit(text);

          if (enoughAudio && splitPos > 0) {
            const committable = text.slice(0, splitPos).trim();
            const remaining = text.slice(splitPos).trim();

            if (committable) {
              setCommittedText((prev) => (prev ? `${prev}\n\n${committable}` : committable));
            }

            const ratio = splitPos / text.length;
            const samplesToKeep = Math.ceil(
              rollingBufferRef.current.length * (1 - ratio),
            );
            rollingBufferRef.current = rollingBufferRef.current.slice(-samplesToKeep);
            lastCommitSamplesRef.current = totalSamplesRef.current;
            queuedInferenceRef.current = false;
            lastInferenceAtRef.current = performance.now();
            setLiveText(remaining);
            liveTextRef.current = remaining;
            return;
          }

          if (overdue && text) {
            setCommittedText((prev) => (prev ? `${prev}\n\n${text}` : text));
            rollingBufferRef.current = new Float32Array(0);
            lastCommitSamplesRef.current = totalSamplesRef.current;
            queuedInferenceRef.current = false;
            lastInferenceAtRef.current = performance.now();
            setLiveText("");
            liveTextRef.current = "";
            return;
          }
        }

        setLiveText(message.text);
        liveTextRef.current = message.text;
        flushInferenceQueue();

        if (shouldFinalizeRef.current && !workerBusyRef.current && !queuedInferenceRef.current) {
          finalizeRecording();
        }
        return;
      }

      if (message.status === "error") {
        setError(message.error);
        setModelState("error");
        workerBusyRef.current = false;
        setIsTranscribing(false);
      }
    };

    worker.postMessage({ type: "LOAD_MODEL" });

    return () => {
      worker.terminate();
      stopAudioProcessing();
      audioContextRef.current?.close();
    };
  }, [finalizeRecording, flushInferenceQueue, stopAudioProcessing]);

  const getAudioContext = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }
    return audioContextRef.current;
  };

  const startRecording = async () => {
    if (modelState !== "ready") return;

    try {
      setError(null);
      rollingBufferRef.current = new Float32Array(0);
      setLiveText("");
      liveTextRef.current = "";
      shouldFinalizeRef.current = false;
      totalSamplesRef.current = 0;
      lastCommitSamplesRef.current = 0;
      queuedInferenceRef.current = false;
      workerBusyRef.current = false;
      isCapturingRef.current = false;
      lastInferenceAtRef.current = 0;

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const audioContext = getAudioContext();
      if (audioContext.state === "suspended") {
        await audioContext.resume();
      }
      if (!workletLoadedRef.current) {
        await audioContext.audioWorklet.addModule("/audio-capture-worklet.js");
        workletLoadedRef.current = true;
      }

      const sourceNode = audioContext.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioContext, "audio-capture-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        channelCount: 1,
      });

      workletNode.port.onmessage = (
        event: MessageEvent<{ type: "pcm"; data: Float32Array } | { type: "silence" }>,
      ) => {
        if (!isCapturingRef.current) return;

        if (event.data.type === "silence") {
          // Natural speech pause — trigger inference immediately so the sentence-end
          // check fires while the text is still fresh at a sentence boundary.
          queueInference();
          return;
        }

        const sampled = downsampleTo16k(event.data.data, audioContext.sampleRate);
        rollingBufferRef.current = appendToBuffer(rollingBufferRef.current, sampled, MAX_BUFFER_SAMPLES);
        totalSamplesRef.current += sampled.length;
        maybeQueueInference();
      };

      sourceNode.connect(workletNode);

      sourceNodeRef.current = sourceNode;
      workletNodeRef.current = workletNode;
      isCapturingRef.current = true;
      lastInferenceAtRef.current = performance.now();
      setIsRecording(true);
    } catch (recordError) {
      const message = recordError instanceof Error ? recordError.message : String(recordError);
      setError(message);
    }
  };

  const stopRecording = () => {
    if (!isCapturingRef.current) return;
    shouldFinalizeRef.current = true;
    stopAudioProcessing();
    maybeQueueInference(true);
  };

  const clearTranscript = () => {
    setCommittedText("");
    setLiveText("");
    liveTextRef.current = "";
    rollingBufferRef.current = new Float32Array(0);
    totalSamplesRef.current = 0;
    lastCommitSamplesRef.current = 0;
  };

  const fullTranscript = useMemo(() => {
    return [committedText.trim(), liveText.trim()].filter(Boolean).join("\n\n");
  }, [committedText, liveText]);

  const copyTranscript = async () => {
    if (!fullTranscript) return;
    try {
      await navigator.clipboard.writeText(fullTranscript);
    } catch {
      setError("Clipboard copy failed. Try selecting text manually.");
    }
  };

  const statusLabel = useMemo(() => {
    if (error) return "Error";
    if (modelState === "loading") return `Loading model (${Math.round(downloadProgress)}%)`;
    if (isRecording && isTranscribing) return "Recording + Transcribing";
    if (isRecording) return "Recording";
    if (isTranscribing) return "Transcribing";
    if (modelState === "ready") return "Ready";
    return "Idle";
  }, [downloadProgress, error, isRecording, isTranscribing, modelState]);

  const statusTone = error
    ? "border-red-400/40 bg-red-500/20 text-red-100"
    : modelState === "ready"
      ? "border-emerald-400/30 bg-emerald-500/20 text-emerald-100"
      : "border-sky-400/30 bg-sky-500/20 text-sky-100";

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-4xl flex-col gap-6 px-6 py-10">
      <header className="space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight text-white">
          In-Browser Voice Transcription
        </h1>
        <p className="text-sm text-zinc-300">
          Private, local speech-to-text powered by Whisper Tiny. Audio never leaves your device.
        </p>
      </header>

      <section className="rounded-2xl border border-white/10 bg-zinc-900/80 p-5 shadow-xl backdrop-blur">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <span className={`rounded-full border px-3 py-1 text-xs font-medium ${statusTone}`}>
            {statusLabel}
            {device ? ` · ${device.toUpperCase()}` : ""}
          </span>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={copyTranscript}
              disabled={!fullTranscript}
              className="rounded-lg border border-white/15 px-3 py-1.5 text-sm text-zinc-100 transition hover:border-white/30 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Copy
            </button>
            <button
              type="button"
              onClick={clearTranscript}
              disabled={!fullTranscript}
              className="rounded-lg border border-white/15 px-3 py-1.5 text-sm text-zinc-100 transition hover:border-white/30 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Clear
            </button>
          </div>
        </div>

        {modelState === "loading" && (
          <div className="mb-5">
            <div className="mb-1 flex items-center justify-between text-xs text-zinc-300">
              <span>Downloading model artifacts</span>
              <span>{Math.round(downloadProgress)}%</span>
            </div>
            <div className="h-2 w-full rounded-full bg-zinc-700">
              <div
                className="h-2 rounded-full bg-sky-400 transition-all"
                style={{ width: `${Math.max(3, downloadProgress)}%` }}
              />
            </div>
          </div>
        )}

        <div className="mb-5 flex items-center gap-3">
          <button
            type="button"
            onClick={isRecording ? stopRecording : startRecording}
            disabled={modelState !== "ready"}
            className={`relative inline-flex h-14 w-14 items-center justify-center rounded-full border text-2xl transition ${isRecording
              ? "border-red-300 bg-red-500/80 text-white shadow-lg shadow-red-500/30"
              : "border-white/20 bg-zinc-800 text-zinc-100 hover:border-white/40"
              } disabled:cursor-not-allowed disabled:opacity-50`}
          >
            {isRecording && <span className="absolute inset-0 rounded-full animate-ping bg-red-400/40" />}
            <span className="relative">{isRecording ? "■" : "🎙"}</span>
          </button>
          <div className="text-sm text-zinc-300">
            {isRecording
              ? "Listening live. Press stop to finalize this segment."
              : "Press the microphone to begin real-time transcription."}
          </div>
        </div>

        <div className="h-80 overflow-y-auto rounded-xl border border-white/10 bg-black/40 p-4">
          {!committedText && !liveText ? (
            <p className="text-sm text-zinc-500">Your transcription will appear here...</p>
          ) : (
            <>
              <p className="text-sm leading-6 text-zinc-100">
                {committedText || ''}
                {liveText && (
                  <span className="text-zinc-400 ml-1">
                    {liveText}
                  </span>
                )}
              </p>
            </>
          )}
        </div>

        {error && <p className="mt-4 text-sm text-red-300">{error}</p>}
      </section>
    </div>
  );
}
