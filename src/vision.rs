use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
    sync::{Arc, Mutex},
    sync::atomic::{AtomicBool, Ordering as AtomicOrdering},
    time::Instant,
};

use anyhow::{anyhow, bail, Context, Result};
use embedded_graphics::{
    mono_font::{ascii::FONT_6X10, MonoTextStyle},
    pixelcolor::Rgb888,
    prelude::*,
    text::Text,
};
use image::{imageops::FilterType, Rgb, RgbImage};
use nokhwa::{
    pixel_format::RgbFormat,
    threaded::CallbackCamera,
    utils::{
        ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
        Resolution,
    },
    Camera,
};
use pixels::{Pixels, SurfaceTexture};
use tract_onnx::prelude::{tract_ndarray::Array4, TypedModel, *};
use tracing::{debug, error, info, warn};
use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use crate::{
    tracking::{MultiObjectTracker, TrackedObject},
    types::{Detection, Rect},
    voice::VoiceCommandEngine,
};

/// 视频推理相关参数
pub struct VisionConfig {
    pub model_path: PathBuf,
    pub labels_path: PathBuf,
    pub input_size: i32,
    pub score_threshold: f32,
    pub nms_threshold: f32,
    pub camera_index: i32,
    pub frame_width: i32,
    pub frame_height: i32,
    pub frame_format: PixelFormatKind,
    pub camera_backend: CameraBackendKind,
}

/// 摄像头后端枚举，由 CLI 层传递
#[derive(Clone, Copy)]
pub enum CameraBackendKind {
    Auto,
    V4l2,
    MediaFoundation,
}

/// FOURCC 提示，目前仅映射常见格式
#[derive(Clone, Copy)]
pub enum PixelFormatKind {
    Auto,
    Format(FrameFormat),
}

impl Default for PixelFormatKind {
    fn default() -> Self {
        Self::Auto
    }
}

pub struct VisionPipeline {
    detector: YoloDetector,
    tracker: MultiObjectTracker,
    _camera: CallbackCamera,
    _camera_settings: CameraSettings,
    voice: Option<VoiceCommandEngine>,
    last_command: Option<String>,
    last_command_time: Instant,
    latest_frame: Arc<Mutex<Option<RgbImage>>>,
    frame_width: u32,
    frame_height: u32,
}

#[derive(Clone, Copy)]
struct CameraSettings {
    index: i32,
    backend: CameraBackendKind,
    width: i32,
    height: i32,
    format: PixelFormatKind,
}

impl CameraSettings {
    fn backend(&self) -> ApiBackend {
        match self.backend {
            CameraBackendKind::Auto => ApiBackend::Auto,
            CameraBackendKind::V4l2 => ApiBackend::Video4Linux,
            CameraBackendKind::MediaFoundation => ApiBackend::MediaFoundation,
        }
    }

    fn requested_format(&self) -> RequestedFormat<'static> {
        if self.width > 0 && self.height > 0 {
            let frame_format = match self.format {
                PixelFormatKind::Auto => FrameFormat::MJPEG,
                PixelFormatKind::Format(fmt) => fmt,
            };
            let fmt = CameraFormat::new(
                Resolution::new(self.width as u32, self.height as u32),
                frame_format,
                30,
            );
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(fmt))
        } else {
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate)
        }
    }
}

impl VisionPipeline {
    pub fn new(cfg: VisionConfig, voice: Option<VoiceCommandEngine>) -> Result<Self> {
        let VisionConfig {
            model_path,
            labels_path,
            input_size,
            score_threshold,
            nms_threshold,
            camera_index,
            frame_width,
            frame_height,
            frame_format,
            camera_backend,
        } = cfg;

        let labels = load_labels(&labels_path)?;
        let detector = YoloDetector::new(
            model_path,
            labels,
            input_size,
            score_threshold,
            nms_threshold,
        )?;
        let settings = CameraSettings {
            index: camera_index,
            backend: camera_backend,
            width: frame_width,
            height: frame_height,
            format: frame_format,
        };
        let (camera, latest_frame, width, height) = Self::init_camera(&settings)?;

        Ok(Self {
            detector,
            tracker: MultiObjectTracker::new(8, 0.3),
            _camera: camera,
            _camera_settings: settings,
            voice,
            last_command: None,
            last_command_time: Instant::now(),
            latest_frame,
            frame_width: width as u32,
            frame_height: height as u32,
        })
    }

    pub fn run(mut self) -> Result<()> {
        info!("进入实时推理循环，按 Q 退出，按 V 触发语音");
        let event_loop = EventLoop::new();
        let initial_size = LogicalSize::new(self.frame_width as f64, self.frame_height as f64);
        let window = WindowBuilder::new()
            .with_title("AIGlasses Preview")
            .with_inner_size(initial_size)
            .build(&event_loop)
            .map_err(|e| anyhow!("创建窗口失败: {e}"))?;

        let size = window.inner_size();
        let surface_texture = SurfaceTexture::new(size.width, size.height, &window);
        let mut pixels = Pixels::new(self.frame_width, self.frame_height, surface_texture)
            .map_err(|e| anyhow!("初始化像素缓冲失败: {e}"))?;

        event_loop
            .run(move |event, _, control_flow| {
                *control_flow = ControlFlow::Poll;
                match event {
                    Event::WindowEvent { event, .. } => match event {
                        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                        WindowEvent::KeyboardInput { input, .. } => {
                            if let Some(key) = input.virtual_keycode {
                                match (key, input.state) {
                                    (VirtualKeyCode::Q, ElementState::Pressed) => {
                                        *control_flow = ControlFlow::Exit;
                                    }
                                    (VirtualKeyCode::V, ElementState::Pressed) => {
                                        if let Some(voice) = &self.voice {
                                            voice.trigger();
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        WindowEvent::Resized(size) => {
                            if let Err(err) = pixels.resize_surface(size.width, size.height) {
                                error!(?err, "调整窗口表面失败");
                                *control_flow = ControlFlow::Exit;
                            }
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            if let Err(err) =
                                pixels.resize_surface(new_inner_size.width, new_inner_size.height)
                            {
                                error!(?err, "缩放窗口失败");
                                *control_flow = ControlFlow::Exit;
                            }
                        }
                        _ => {}
                    },
                    Event::MainEventsCleared => {
                        if let Some(frame) = match self.try_process_frame() {
                            Ok(frame) => frame,
                            Err(err) => {
                                warn!(?err, "处理帧失败");
                                None
                            }
                        } {
                            self.copy_to_pixels(&frame, pixels.frame_mut());
                            window.request_redraw();
                        }
                    }
                    Event::RedrawRequested(_) => {
                        if let Err(err) = pixels.render() {
                            error!(?err, "绘制失败");
                            *control_flow = ControlFlow::Exit;
                        }
                    }
                    _ => {}
                }
            });
        #[allow(unreachable_code)]
        Ok(())
    }

    fn try_process_frame(&mut self) -> Result<Option<RgbImage>> {
        if let Some(voice) = &self.voice {
            if let Some(event) = voice.try_fetch() {
                let duration_sec = event.duration_ms as f32 / 1000.0;
                self.last_command = if event.transcript.is_empty() {
                    Some(format!("(未识别出内容，耗时{duration_sec:.2}s)"))
                } else {
                    Some(format!("{} ({duration_sec:.2}s)", event.transcript))
                };
                self.last_command_time = Instant::now();
            }
        }

        let mut frame = match self.latest_frame.lock() {
            Ok(mut slot) => match slot.take() {
                Some(frame) => frame,
                None => return Ok(None),
            },
            Err(err) => return Err(anyhow!("获取摄像头帧锁失败: {err}")),
        };

        let sample = frame.get_pixel(0, 0);
        debug!(
            "frame sample rgb=({}, {}, {}) size={}x{}",
            sample[0],
            sample[1],
            sample[2],
            frame.width(),
            frame.height()
        );

        let detections = match self.detector.detect(&frame) {
            Ok(det) => det,
            Err(err) => {
                warn!(?err, "YOLO 推理失败，跳过当前帧");
                return Ok(None);
            }
        };
        let tracked = self.tracker.update(&detections);
        let last_command = self
            .last_command
            .as_deref()
            .filter(|_| self.last_command_time.elapsed().as_secs() <= 5);
        draw_overlays(&mut frame, &tracked, last_command);
        Ok(Some(frame))
    }

    fn copy_to_pixels(&mut self, frame: &RgbImage, target: &mut [u8]) {
        if target.len() != (self.frame_width * self.frame_height * 4) as usize {
            warn!(
                "像素缓冲大小不匹配: expected {} got {}",
                self.frame_width * self.frame_height * 4,
                target.len()
            );
            return;
        }
        for (src, dst) in frame
            .pixels()
            .zip(target.chunks_exact_mut(4))
        {
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = 0xFF;
        }
    }

    fn init_camera(
        settings: &CameraSettings,
    ) -> Result<(CallbackCamera, Arc<Mutex<Option<RgbImage>>>, usize, usize)> {
        let base = Camera::with_backend(
            CameraIndex::Index(settings.index.max(0) as u32),
            settings.requested_format(),
            settings.backend(),
        )
        .map_err(|e| anyhow!("打开摄像头失败: {e}"))?;
        let camera_format = base.camera_format();
        let width = camera_format.width() as usize;
        let height = camera_format.height() as usize;
        let latest = Arc::new(Mutex::new(None));
        let latest_clone = latest.clone();
        let dumped = Arc::new(AtomicBool::new(false));
        let dumped_clone = dumped.clone();
        let mut threaded = CallbackCamera::with_custom(base, move |buffer| {
            match buffer.decode_image::<RgbFormat>() {
                Ok(image) => {
                    if let Ok(mut slot) = latest_clone.lock() {
                        let first = slot.is_none();
                        *slot = Some(image);
                        if first {
                            debug!("首次收到摄像头帧");
                            if !dumped_clone.swap(true, AtomicOrdering::SeqCst) {
                                if let Some(frame) = slot.as_ref() {
                                    if let Err(err) = frame.save("first_frame_debug.png") {
                                        error!(?err, "保存调试帧失败");
                                    } else {
                                        info!("已保存 first_frame_debug.png，方便排查");
                                    }
                                }
                            }
                        }
                    }
                }
                Err(err) => error!(?err, "解码摄像头帧失败"),
            }
        });
        threaded
            .open_stream()
            .map_err(|e| anyhow!("启动摄像头数据流失败: {e}"))?;
        Ok((threaded, latest, width, height))
    }

}

struct YoloDetector {
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, TypedModel>,
    labels: Vec<String>,
    input_size: usize,
    score_threshold: f32,
    nms_threshold: f32,
}

impl YoloDetector {
    fn new(
        model_path: PathBuf,
        labels: Vec<String>,
        input_size: i32,
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Self> {
        let runnable = tract_onnx::onnx()
            .model_for_path(&model_path)?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec!(1, 3, input_size as usize, input_size as usize),
                ),
            )?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self {
            model: runnable,
            labels,
            input_size: input_size as usize,
            score_threshold,
            nms_threshold,
        })
    }

    fn detect(&self, frame: &RgbImage) -> Result<Vec<Detection>> {
        let resized = image::imageops::resize(
            frame,
            self.input_size as u32,
            self.input_size as u32,
            FilterType::Triangle,
        );
        let mut input = Array4::<f32>::zeros((
            1,
            3,
            self.input_size,
            self.input_size,
        ));
        let raw = resized.as_raw();
        for y in 0..self.input_size {
            for x in 0..self.input_size {
                let idx = (y * self.input_size + x) * 3;
                input[(0, 0, y, x)] = raw[idx] as f32 / 255.0;
                input[(0, 1, y, x)] = raw[idx + 1] as f32 / 255.0;
                input[(0, 2, y, x)] = raw[idx + 2] as f32 / 255.0;
            }
        }
        let tensor: Tensor = input.into();
        let outputs = self.model.run(tvec!(tensor.into_tvalue()))?;
        let output = outputs
            .get(0)
            .ok_or_else(|| anyhow!("模型未返回输出"))?
            .to_array_view::<f32>()?;
        let dims = output.shape();
        let (rows, cols) = match dims.len() {
            3 => (dims[1], dims[2]),
            2 => (dims[0], dims[1]),
            _ => bail!("模型输出维度异常: {:?}", dims),
        };
        let data = output
            .as_slice()
            .ok_or_else(|| anyhow!("模型输出不可作为连续切片"))?;
        let frame_size = frame.dimensions();
        parse_yolo(
            data,
            rows,
            cols,
            frame_size,
            self.input_size as i32,
            &self.labels,
            self.score_threshold,
            self.nms_threshold,
        )
    }
}

fn parse_yolo(
    data: &[f32],
    rows: usize,
    cols: usize,
    frame_size: (u32, u32),
    input_size: i32,
    labels: &[String],
    score_threshold: f32,
    nms_threshold: f32,
) -> Result<Vec<Detection>> {
    let mut detections = Vec::new();
    for row in 0..rows {
        let base = row * cols;
        if base + 5 >= data.len() {
            break;
        }
        let objectness = data[base + 4];
        if objectness < score_threshold {
            continue;
        }
        let class_slice = &data[base + 5..base + cols];
        if class_slice.is_empty() {
            continue;
        }
        let (class_id, class_score) = class_slice
            .iter()
            .enumerate()
            .fold((0usize, 0f32), |acc, (idx, val)| {
                if *val > acc.1 {
                    (idx, *val)
                } else {
                    acc
                }
            });
        let score = objectness * class_score;
        if score < score_threshold {
            continue;
        }
        let label = labels
            .get(class_id)
            .cloned()
            .unwrap_or_else(|| format!("class_{class_id}"));
        let cx = data[base];
        let cy = data[base + 1];
        let w = data[base + 2];
        let h = data[base + 3];
        let scale_x = frame_size.0 as f32 / input_size as f32;
        let scale_y = frame_size.1 as f32 / input_size as f32;
        let left = (cx - w / 2.0) * scale_x;
        let top = (cy - h / 2.0) * scale_y;
        let rect = Rect::new(
            left as f64,
            top as f64,
            (w * scale_x) as f64,
            (h * scale_y) as f64,
        );
        detections.push(Detection::new(rect, score, class_id as i32, label));
    }
    Ok(apply_nms(detections, nms_threshold))
}

fn apply_nms(mut detections: Vec<Detection>, threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    let mut result = Vec::new();
    for det in detections.into_iter() {
        let keep = result
            .iter()
            .all(|picked: &Detection| rect_iou(&picked.rect, &det.rect) < threshold as f64);
        if keep {
            result.push(det);
        }
    }
    result
}

fn draw_overlays(frame: &mut RgbImage, tracked: &[TrackedObject], last_command: Option<&str>) {
    for obj in tracked {
        let color = if obj.stale {
            [0u8, 255, 255]
        } else {
            [0u8, 255, 0]
        };
        draw_rect(frame, &obj.rect, color);
        let label = format!("#{:02} {} {:.2}", obj.id, obj.label_name, obj.confidence);
        draw_text(
            frame,
            &label,
            (obj.rect.x.round() as i32, (obj.rect.y - 12.0).max(16.0) as i32),
            color,
        );
    }
    draw_text(
        frame,
        "Press V to record, Q to quit",
        (16, 24),
        [255, 255, 255],
    );
    if let Some(text) = last_command {
        draw_text(frame, &format!("Cmd: {text}"), (16, 44), [0, 215, 255]);
    }
}

fn draw_rect(frame: &mut RgbImage, rect: &Rect, color: [u8; 3]) {
    let x1 = rect.x.max(0.0) as i32;
    let y1 = rect.y.max(0.0) as i32;
    let x2 = (rect.x + rect.width).min(frame.width() as f64) as i32;
    let y2 = (rect.y + rect.height).min(frame.height() as f64) as i32;
    for x in x1..x2 {
        set_pixel_safe(frame, x, y1, color);
        set_pixel_safe(frame, x, y2.max(y1), color);
    }
    for y in y1..y2 {
        set_pixel_safe(frame, x1, y, color);
        set_pixel_safe(frame, x2.max(x1), y, color);
    }
}

fn set_pixel_safe(frame: &mut RgbImage, x: i32, y: i32, color: [u8; 3]) {
    if x >= 0 && y >= 0 && x < frame.width() as i32 && y < frame.height() as i32 {
        frame.put_pixel(x as u32, y as u32, Rgb(color));
    }
}

fn draw_text(frame: &mut RgbImage, text: &str, pos: (i32, i32), color: [u8; 3]) {
    let style = MonoTextStyle::new(&FONT_6X10, Rgb888::new(color[0], color[1], color[2]));
    let mut target = ImageDrawTarget::new(frame);
    let _ = Text::new(text, Point::new(pos.0, pos.1), style).draw(&mut target);
}

struct ImageDrawTarget<'a> {
    image: &'a mut RgbImage,
}

impl<'a> ImageDrawTarget<'a> {
    fn new(image: &'a mut RgbImage) -> Self {
        Self { image }
    }
}

impl OriginDimensions for ImageDrawTarget<'_> {
    fn size(&self) -> Size {
        Size::new(self.image.width(), self.image.height())
    }
}

impl DrawTarget for ImageDrawTarget<'_> {
    type Color = Rgb888;
    type Error = core::convert::Infallible;

    fn draw_iter<I>(&mut self, pixels: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = Pixel<Self::Color>>,
    {
        let width = self.image.width() as i32;
        let height = self.image.height() as i32;
        for Pixel(coord, color) in pixels {
            if coord.x < 0 || coord.y < 0 || coord.x >= width || coord.y >= height {
                continue;
            }
            let pixel = self
                .image
                .get_pixel_mut(coord.x as u32, coord.y as u32);
            *pixel = Rgb([color.r(), color.g(), color.b()]);
        }
        Ok(())
    }
}

fn load_labels(path: &PathBuf) -> Result<Vec<String>> {
    let file = File::open(path).context("无法读取标签文件")?;
    let reader = BufReader::new(file);
    Ok(reader
        .lines()
        .flatten()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect())
}

fn rect_iou(a: &Rect, b: &Rect) -> f64 {
    let ax2 = a.x + a.width;
    let ay2 = a.y + a.height;
    let bx2 = b.x + b.width;
    let by2 = b.y + b.height;
    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let intersection = iw * ih;
    if intersection <= 0.0 {
        return 0.0;
    }
    let union = a.width * a.height + b.width * b.height - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}
