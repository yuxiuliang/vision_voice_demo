use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Context, Result};
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, Sample, SampleFormat, SizedSample, Stream, StreamConfig,
};
use crossbeam_channel::{unbounded, Receiver, Sender};
use dasp_sample::conv::ToSample;
use parking_lot::Mutex;
use tracing::{info, warn};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// 识别完成后的事件
#[derive(Debug, Clone)]
pub struct VoiceEvent {
    pub transcript: String,
    pub duration_ms: u128,
}

enum VoiceRequest {
    CaptureOnce,
    Stop,
}

/// 控制录音和离线识别的后台线程
pub struct VoiceCommandEngine {
    tx: Sender<VoiceRequest>,
    rx: Receiver<VoiceEvent>,
}

impl VoiceCommandEngine {
    pub fn try_new(model_path: PathBuf, sample_rate: u32, capture_ms: u64) -> Result<Self> {
        let (req_tx, req_rx) = unbounded();
        let (evt_tx, evt_rx) = unbounded();
        thread::Builder::new()
            .name("voice_engine".into())
            .spawn(move || {
                if let Err(err) = voice_worker(req_rx, evt_tx, model_path, sample_rate, capture_ms)
                {
                    warn!(?err, "语音线程异常退出");
                }
            })?;
        Ok(Self {
            tx: req_tx,
            rx: evt_rx,
        })
    }

    pub fn trigger(&self) {
        let _ = self.tx.send(VoiceRequest::CaptureOnce);
    }

    pub fn try_fetch(&self) -> Option<VoiceEvent> {
        self.rx.try_recv().ok()
    }
}

impl Drop for VoiceCommandEngine {
    fn drop(&mut self) {
        let _ = self.tx.send(VoiceRequest::Stop);
    }
}

fn voice_worker(
    rx: Receiver<VoiceRequest>,
    tx: Sender<VoiceEvent>,
    model_path: PathBuf,
    target_sample_rate: u32,
    capture_ms: u64,
) -> Result<()> {
    info!("加载 Whisper 模型: {:?}", model_path);
    let model_str = model_path
        .to_str()
        .ok_or_else(|| anyhow!("模型路径包含非法字符"))?;
    let ctx_params = WhisperContextParameters::default();
    let ctx = WhisperContext::new_with_params(model_str, ctx_params)
        .map_err(|e| anyhow!("加载 Whisper 模型失败: {e}"))?;

    while let Ok(msg) = rx.recv() {
        match msg {
            VoiceRequest::CaptureOnce => {
                let start = Instant::now();
                let audio = match capture_audio_blocking(target_sample_rate, capture_ms) {
                    Ok(data) => data,
                    Err(err) => {
                        warn!(?err, "录音失败");
                        continue;
                    }
                };
                match transcribe(&ctx, &audio, target_sample_rate) {
                    Ok(text) => {
                        let duration_ms = start.elapsed().as_millis();
                        let _ = tx.send(VoiceEvent {
                            transcript: text,
                            duration_ms,
                        });
                    }
                    Err(err) => warn!(?err, "语音识别失败"),
                }
            }
            VoiceRequest::Stop => break,
        }
    }
    Ok(())
}

fn capture_audio_blocking(target_sample_rate: u32, capture_ms: u64) -> Result<Vec<f32>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("未检测到可用的麦克风输入")?;
    let config = device.default_input_config().context("查询输入配置失败")?;
    let sample_format = config.sample_format();
    let config: StreamConfig = config.into();
    let actual_rate = config.sample_rate.0;
    let channels = config.channels as usize;
    let target_samples = (actual_rate as u64 * capture_ms / 1000) as usize * channels;

    let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let finished = Arc::new(AtomicBool::new(false));

    let err_fn = |err| warn!(?err, "音频输入流出现问题");

    let stream = build_stream(
        &device,
        &config,
        sample_format,
        buffer.clone(),
        finished.clone(),
        target_samples,
        err_fn,
    )
    .map_err(|e| anyhow!("构建音频输入流失败: {e}"))?;

    stream.play()?;
    while buffer.lock().len() < target_samples && !finished.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(20));
    }
    finished.store(true, Ordering::SeqCst);
    thread::sleep(Duration::from_millis(50));
    drop(stream);
    let raw = buffer.lock().clone();

    let mono = downmix_to_mono(&raw, channels);
    Ok(resample_linear(&mono, actual_rate, target_sample_rate))
}

fn build_stream<F>(
    device: &Device,
    config: &StreamConfig,
    format: SampleFormat,
    buffer: Arc<Mutex<Vec<f32>>>,
    finished: Arc<AtomicBool>,
    target_samples: usize,
    err_fn: F,
) -> Result<Stream, cpal::BuildStreamError>
where
    F: FnMut(cpal::StreamError) + Send + 'static,
{
    match format {
        SampleFormat::F32 => {
            build_typed_stream::<f32, _>(device, config, buffer, finished, target_samples, err_fn)
        }
        SampleFormat::I16 => {
            build_typed_stream::<i16, _>(device, config, buffer, finished, target_samples, err_fn)
        }
        SampleFormat::U16 => {
            build_typed_stream::<u16, _>(device, config, buffer, finished, target_samples, err_fn)
        }
        _ => unimplemented!("暂不支持的采样格式"),
    }
}

fn build_typed_stream<T, F>(
    device: &Device,
    config: &StreamConfig,
    buffer: Arc<Mutex<Vec<f32>>>,
    finished: Arc<AtomicBool>,
    target_samples: usize,
    err_fn: F,
) -> Result<Stream, cpal::BuildStreamError>
where
    T: Sample + SizedSample + Send + 'static + ToSample<f32>,
    F: FnMut(cpal::StreamError) + Send + 'static,
{
    let mut handler = err_fn;
    device.build_input_stream(
        config,
        move |data: &[T], _| {
            if finished.load(Ordering::SeqCst) {
                return;
            }
            let mut guard = buffer.lock();
            guard.extend(data.iter().map(|sample| (*sample).to_sample::<f32>()));
            if guard.len() >= target_samples {
                finished.store(true, Ordering::SeqCst);
            }
        },
        move |err| handler(err),
        None,
    )
}

fn downmix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    data.chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn resample_linear(samples: &[f32], input_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || input_rate == target_rate {
        return samples.to_vec();
    }
    let ratio = target_rate as f32 / input_rate as f32;
    let target_len = (samples.len() as f32 * ratio) as usize;
    let mut output = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let src_pos = i as f32 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f32;
        let next_idx = (idx + 1).min(samples.len() - 1);
        let value = samples[idx] * (1.0 - frac) + samples[next_idx] * frac;
        output.push(value);
    }
    output
}

fn transcribe(ctx: &WhisperContext, audio: &[f32], _sample_rate: u32) -> Result<String> {
    let mut state = ctx
        .create_state()
        .map_err(|e| anyhow::anyhow!("创建 Whisper 状态失败: {e}"))?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(num_cpus::get() as i32);
    params.set_language(Some("zh"));
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_single_segment(true);

    state
        .full(params, audio)
        .map_err(|e| anyhow::anyhow!("Whisper 推理失败: {e}"))?;

    let num_segments = state.full_n_segments()?;
    let mut transcript = String::new();
    for idx in 0..num_segments {
        let segment = state.full_get_segment_text(idx).unwrap_or_default();
        transcript.push_str(segment.trim());
    }
    Ok(transcript.trim().to_string())
}
