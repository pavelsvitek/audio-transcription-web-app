class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._silenceThreshold = 0.015; // RMS amplitude considered silence
    // How many 128-sample frames of silence == a pause (~0.5 s)
    this._silenceFramesRequired = Math.ceil((sampleRate * 0.5) / 128);
    // Minimum speaking frames before a silence event fires (~0.3 s of speech first)
    this._speechFramesRequired = Math.ceil((sampleRate * 0.3) / 128);
    // Cooldown after firing: don't fire again for ~3 s
    this._cooldownFramesRequired = Math.ceil((sampleRate * 3) / 128);

    this._silenceFrames = 0;
    this._speechFrames = 0;
    this._cooldownFrames = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel || channel.length === 0) return true;

    this.port.postMessage({ type: "pcm", data: channel.slice(0) });

    let sum = 0;
    for (let i = 0; i < channel.length; i++) {
      sum += channel[i] * channel[i];
    }
    const rms = Math.sqrt(sum / channel.length);

    if (this._cooldownFrames > 0) {
      this._cooldownFrames--;
    }

    if (rms > this._silenceThreshold) {
      this._speechFrames++;
      this._silenceFrames = 0;
    } else {
      this._silenceFrames++;
      if (
        this._speechFrames >= this._speechFramesRequired &&
        this._silenceFrames >= this._silenceFramesRequired &&
        this._cooldownFrames === 0
      ) {
        this._speechFrames = 0;
        this._silenceFrames = 0;
        this._cooldownFrames = this._cooldownFramesRequired;
        this.port.postMessage({ type: "silence" });
      }
    }

    return true;
  }
}

registerProcessor("audio-capture-processor", AudioCaptureProcessor);
