use crate::types::{Detection, Rect};

/// 维护单个目标状态，内部使用简单的匀速模型
struct TrackerEntry {
    id: u64,
    rect: Rect,
    velocity: (f64, f64),
    label_id: i32,
    label_name: String,
    confidence: f32,
    misses: u32,
}

/// 对外输出的追踪结果
#[derive(Clone)]
pub struct TrackedObject {
    pub id: u64,
    pub rect: Rect,
    pub label_name: String,
    pub confidence: f32,
    pub stale: bool,
}

impl TrackerEntry {
    fn new(id: u64, detection: &Detection) -> Self {
        Self {
            id,
            rect: detection.rect,
            velocity: (0.0, 0.0),
            label_id: detection.label_id,
            label_name: detection.label_name.clone(),
            confidence: detection.confidence,
            misses: 0,
        }
    }

    fn predict(&mut self) {
        self.rect.x += self.velocity.0;
        self.rect.y += self.velocity.1;
    }

    fn correct(&mut self, detection: &Detection) {
        let prev_center = self.rect.center();
        let new_center = detection.rect.center();
        self.velocity = (
            (new_center.0 - prev_center.0).clamp(-50.0, 50.0),
            (new_center.1 - prev_center.1).clamp(-50.0, 50.0),
        );
        self.rect = detection.rect;
        self.label_id = detection.label_id;
        self.label_name = detection.label_name.clone();
        self.confidence = detection.confidence;
        self.misses = 0;
    }
}

/// 多目标追踪器，内部用贪心方式将 detection 分配给 tracker
pub struct MultiObjectTracker {
    trackers: Vec<TrackerEntry>,
    next_id: u64,
    max_misses: u32,
    match_threshold: f64,
}

impl MultiObjectTracker {
    pub fn new(max_misses: u32, match_threshold: f64) -> Self {
        Self {
            trackers: Vec::new(),
            next_id: 1,
            max_misses,
            match_threshold,
        }
    }

    pub fn update(&mut self, detections: &[Detection]) -> Vec<TrackedObject> {
        for tracker in &mut self.trackers {
            tracker.predict();
            tracker.misses += 1;
        }

        let mut unmatched: Vec<bool> = detections.iter().map(|_| true).collect();

        for tracker in &mut self.trackers {
            if let Some((best_idx, _)) = detections
                .iter()
                .enumerate()
                .filter(|(idx, det)| unmatched[*idx] && det.label_id == tracker.label_id)
                .map(|(idx, det)| (idx, iou(&tracker.rect, &det.rect)))
                .filter(|(_, score)| *score >= self.match_threshold)
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            {
                tracker.correct(&detections[best_idx]);
                unmatched[best_idx] = false;
            }
        }

        for (idx, detection) in detections.iter().enumerate() {
            if unmatched[idx] {
                let entry = TrackerEntry::new(self.next_id, detection);
                self.next_id += 1;
                self.trackers.push(entry);
            }
        }

        self.trackers
            .retain(|tracker| tracker.misses <= self.max_misses);

        self.trackers
            .iter()
            .map(|tracker| TrackedObject {
                id: tracker.id,
                rect: tracker.rect,
                label_name: tracker.label_name.clone(),
                confidence: tracker.confidence,
                stale: tracker.misses > 0,
            })
            .collect()
    }
}

fn iou(a: &Rect, b: &Rect) -> f64 {
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
