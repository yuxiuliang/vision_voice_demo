/// 简单矩形，所有坐标均以像素为单位
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn center(&self) -> (f64, f64) {
        (
            self.x + self.width / 2.0,
            self.y + self.height / 2.0,
        )
    }
}

/// 表示一次 YOLO 推理输出的检测框
pub struct Detection {
    pub rect: Rect,
    pub confidence: f32,
    pub label_id: i32,
    pub label_name: String,
}

impl Detection {
    pub fn new(rect: Rect, confidence: f32, label_id: i32, label_name: String) -> Self {
        Self {
            rect,
            confidence,
            label_id,
            label_name,
        }
    }
}
