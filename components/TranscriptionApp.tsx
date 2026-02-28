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
const MAX_BUFFER_SECONDS = 30;
const MAX_BUFFER_SAMPLES = SAMPLE_RATE * MAX_BUFFER_SECONDS;

function appendToRollingBuffer(existing: Float32Array, incoming: Float32Array) {
  const totalLength = existing.length + incoming.length;
  const merged = new Float32Array(totalLength);
  merged.set(existing, 0);
  merged.set(incoming, existing.length);

  if (merged.length <= MAX_BUFFER_SAMPLES) {
    return merged;
  }

  return merged.slice(merged.length - MAX_BUFFER_SAMPLES);
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

function getRecorderMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];

  return candidates.find((mimeType) => MediaRecorder.isTypeSupported(mimeType));
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
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const rollingBufferRef = useRef<Float32Array>(new Float32Array(0));
  const queuedInferenceRef = useRef(false);
  const workerBusyRef = useRef(false);
  const shouldFinalizeRef = useRef(false);
  const pendingDecodeRef = useRef(0);
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
        setLiveText(message.text);
        liveTextRef.current = message.text;
        workerBusyRef.current = false;
        setIsTranscribing(false);
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
      stopMicrophoneTracks();
      audioContextRef.current?.close();
    };
  }, [finalizeRecording, flushInferenceQueue, stopMicrophoneTracks]);

  const getAudioContext = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }
    return audioContextRef.current;
  };

  const decodeChunk = async (blob: Blob) => {
    pendingDecodeRef.current += 1;
    try {
      const audioContext = getAudioContext();
      const arrayBuffer = await blob.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      const channelData = audioBuffer.getChannelData(0);
      const sampled = downsampleTo16k(channelData, audioBuffer.sampleRate);

      rollingBufferRef.current = appendToRollingBuffer(rollingBufferRef.current, sampled);
      queueInference();
    } finally {
      pendingDecodeRef.current -= 1;
      if (shouldFinalizeRef.current && pendingDecodeRef.current === 0) {
        queueInference();
      }
    }
  };

  const startRecording = async () => {
    if (modelState !== "ready") return;

    try {
      setError(null);
      rollingBufferRef.current = new Float32Array(0);
      setLiveText("");
      liveTextRef.current = "";
      shouldFinalizeRef.current = false;
      queuedInferenceRef.current = false;
      workerBusyRef.current = false;
      pendingDecodeRef.current = 0;

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const mimeType = getRecorderMimeType();
      const mediaRecorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          void decodeChunk(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        stopMicrophoneTracks();
        queueInference();
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(3000);
      setIsRecording(true);
    } catch (recordError) {
      const message = recordError instanceof Error ? recordError.message : String(recordError);
      setError(message);
    }
  };

  const stopRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === "inactive") return;
    shouldFinalizeRef.current = true;
    recorder.stop();
    queueInference();
  };

  const clearTranscript = () => {
    setCommittedText("");
    setLiveText("");
    liveTextRef.current = "";
    rollingBufferRef.current = new Float32Array(0);
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
            className={`relative inline-flex h-14 w-14 items-center justify-center rounded-full border text-2xl transition ${
              isRecording
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
              {committedText && (
                <p className="whitespace-pre-wrap text-sm leading-6 text-zinc-100">
                  {committedText}
                </p>
              )}
              {liveText && (
                <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-zinc-400">
                  {liveText}
                </p>
              )}
            </>
          )}
        </div>

        {error && <p className="mt-4 text-sm text-red-300">{error}</p>}
      </section>
    </div>
  );
}
