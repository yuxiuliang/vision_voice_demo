mod tracking;
mod types;
mod vision;
mod voice;

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::Parser;
use nokhwa::utils::FrameFormat;
use tracing_subscriber::{fmt, EnvFilter};

use vision::{CameraBackendKind, PixelFormatKind, VisionConfig, VisionPipeline};
use voice::VoiceCommandEngine;

/// 命令行参数
#[derive(Debug, Parser)]
#[command(author, version, about = "语音指令 + YOLO + 卡尔曼追踪 Demo", long_about = None)]
struct Cli {
    /// ONNX 模型路径（建议使用 YOLOv5n/v8n）
    #[arg(long, default_value = "models/yolov5n.onnx")]
    model: PathBuf,

    /// 标签文件路径（按行列出 COCO 类别名）
    #[arg(long, default_value = "models/coco.labels")]
    labels: PathBuf,

    /// Whisper GGML 模型路径
    #[arg(long, default_value = "models/ggml-base.bin")]
    whisper_model: PathBuf,

    /// 摄像头索引
    #[arg(long, default_value_t = 0)]
    camera_index: i32,

    /// YOLO 输入尺寸（必须符合模型导出尺寸，如 640）
    #[arg(long, default_value_t = 640)]
    input_size: i32,

    /// 置信度阈值
    #[arg(long, default_value_t = 0.35)]
    score: f32,

    /// NMS 阈值
    #[arg(long, default_value_t = 0.45)]
    nms: f32,

    /// 禁用语音指令
    #[arg(long)]
    no_voice: bool,

    /// 单次录音时长（毫秒）
    #[arg(long, default_value_t = 4000)]
    voice_duration_ms: u64,

    /// Whisper 目标采样率
    #[arg(long, default_value_t = 16000)]
    voice_sample_rate: u32,

    /// 摄像头输出宽度（像素，0 表示沿用默认）
    #[arg(long, default_value_t = 0)]
    camera_width: i32,

    /// 摄像头输出高度（像素，0 表示沿用默认）
    #[arg(long, default_value_t = 0)]
    camera_height: i32,

    /// 摄像头 FOURCC（设置为 auto 跳过配置）
    #[arg(long, default_value = "auto")]
    camera_fourcc: String,

    /// 摄像头后端（auto/v4l2/gstreamer/any）
    #[arg(long, default_value = "auto")]
    camera_backend: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    fmt().with_env_filter(EnvFilter::from_default_env()).init();

    let camera_fourcc = parse_fourcc(&cli.camera_fourcc)?;
    let camera_backend = parse_backend(&cli.camera_backend)?;

    let voice_engine = if cli.no_voice {
        None
    } else {
        Some(VoiceCommandEngine::try_new(
            cli.whisper_model.clone(),
            cli.voice_sample_rate,
            cli.voice_duration_ms,
        )?)
    };

    let cfg = VisionConfig {
        model_path: cli.model,
        labels_path: cli.labels,
        input_size: cli.input_size,
        score_threshold: cli.score,
        nms_threshold: cli.nms,
        camera_index: cli.camera_index,
        frame_width: cli.camera_width,
        frame_height: cli.camera_height,
        frame_format: camera_fourcc,
        camera_backend,
    };

    let pipeline = VisionPipeline::new(cfg, voice_engine)?;
    pipeline.run()?;
    Ok(())
}

fn parse_fourcc(tag: &str) -> Result<PixelFormatKind> {
    if tag.eq_ignore_ascii_case("auto") {
        return Ok(PixelFormatKind::Auto);
    }
    let upper = tag.trim().to_ascii_uppercase();
    let fmt = match upper.as_str() {
        "MJPG" | "MJPEG" => FrameFormat::MJPEG,
        "YUYV" => FrameFormat::YUYV,
        "NV12" => FrameFormat::NV12,
        "RGB3" | "RAWRGB" => FrameFormat::RAWRGB,
        "BGR3" | "RAWBGR" => FrameFormat::RAWBGR,
        "GRAY" => FrameFormat::GRAY,
        other => return Err(anyhow!("暂不支持的 FOURCC: {other}")),
    };
    Ok(PixelFormatKind::Format(fmt))
}

fn parse_backend(name: &str) -> Result<CameraBackendKind> {
    let norm = name.trim().to_ascii_lowercase();
    let backend = match norm.as_str() {
        "auto" | "any" | "gstreamer" => CameraBackendKind::Auto,
        "v4l2" => CameraBackendKind::V4l2,
        "msmf" | "mediafoundation" => CameraBackendKind::MediaFoundation,
        other => {
            return Err(anyhow!(
                "未知摄像头后端 {other}，可选值: auto/v4l2/msmf/any"
            ))
        }
    };
    Ok(backend)
}
