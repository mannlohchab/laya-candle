//! Temperature scaling and answer decoding (mirrors Agent._decode_answers).

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::model::AgentConfig;
use crate::sequence::{InternalQuestion, QTYPE_CHOICE, QTYPE_NOUL, QTYPE_SCORE};

const TEMP_MIN: f64 = 0.5;
const TEMP_MAX: f64 = 5.0;

pub fn clamp_temperature(t: f64) -> f64 {
    if !t.is_finite() {
        return 1.0;
    }
    t.clamp(TEMP_MIN, TEMP_MAX)
}

fn temp_bucket(qtype: i64, k: usize) -> String {
    let name = match qtype {
        QTYPE_CHOICE => "choice",
        QTYPE_SCORE => "score",
        QTYPE_NOUL => "noul",
        _ => "choice",
    };
    let size = if k <= 2 {
        "2"
    } else if k <= 5 {
        "3-5"
    } else if k <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{name}:{size}")
}

pub struct Temperatures {
    pub by_type: Vec<f64>,
    pub by_options: HashMap<String, f64>,
}

impl Temperatures {
    pub fn from_config(cfg: &AgentConfig) -> Self {
        let by_type = if cfg.temperature.len() == 3 {
            cfg.temperature.iter().copied().map(clamp_temperature).collect()
        } else {
            vec![1.0, 1.0, 1.0]
        };
        let by_options = cfg
            .temperature_by_options
            .iter()
            .map(|(k, v)| (k.clone(), clamp_temperature(*v)))
            .collect();
        Self { by_type, by_options }
    }

    fn scale(&self, qtype: i64, k: usize) -> f64 {
        let bucket = temp_bucket(qtype, k);
        self.by_options
            .get(&bucket)
            .copied()
            .unwrap_or_else(|| self.by_type.get(qtype as usize).copied().unwrap_or(1.0))
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|z| (z - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn answer_confidence(p: &[f64]) -> f64 {
    p.iter().cloned().fold(0.0, f64::max).clamp(0.0, 1.0)
}

fn confidence_from_probs(p: &[f64]) -> f64 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let ent: f64 = -p
        .iter()
        .map(|&x| {
            let x = x.clamp(1e-12, 1.0);
            x * x.ln()
        })
        .sum::<f64>();
    (1.0 - ent / (k as f64).ln()).clamp(0.0, 1.0)
}

pub fn decode_answers(
    logits: &[Vec<f64>],
    act: &[Vec<f64>],
    ids: &[String],
    internal: &HashMap<String, InternalQuestion>,
    markers_len: &[usize],
    temps: &Temperatures,
) -> Result<Value, anyhow::Error> {
    let mut answers = serde_json::Map::new();
    for (j, qid) in ids.iter().enumerate() {
        let q = &internal[qid];
        let k = markers_len[j];
        let qt = match q.t.as_str() {
            "choice" => QTYPE_CHOICE,
            "score" => QTYPE_SCORE,
            "noul" => QTYPE_NOUL,
            _ => QTYPE_CHOICE,
        };
        let t_scale = temps.scale(qt, k);
        let z: Vec<f64> = logits[j][..k].iter().map(|v| v / t_scale).collect();
        let p = softmax(&z);
        let ans_conf = (answer_confidence(&p) * 10000.0).round() / 10000.0;
        let act_p = (act[j][0] * 10000.0).round() / 10000.0;
        let action = json!({ "act_probability": act_p });

        let entry = match q.t.as_str() {
            "choice" => {
                let keys: Vec<String> = q
                    .crit
                    .as_object()
                    .unwrap()
                    .keys()
                    .cloned()
                    .collect();
                let best = p
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let mut probs = serde_json::Map::new();
                for (kk, v) in keys.iter().zip(p.iter()) {
                    probs.insert(kk.clone(), json!(((v * 10000.0).round() / 10000.0)));
                }
                json!({
                    "type": "choice",
                    "choice": keys[best],
                    "probabilities": probs,
                    "confidence": ((confidence_from_probs(&p) * 10000.0).round() / 10000.0),
                    "answer_confidence": ans_conf,
                    "action": action,
                })
            }
            "score" => {
                let exp_score: f64 = p.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
                let legend: serde_json::Map<String, Value> = q
                    .crit
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i.to_string(), c.clone()))
                    .collect();
                let mut probs = serde_json::Map::new();
                for (i, v) in p.iter().enumerate() {
                    probs.insert(i.to_string(), json!(((v * 10000.0).round() / 10000.0)));
                }
                json!({
                    "type": "score",
                    "score": ((exp_score * 10000.0).round() / 10000.0),
                    "legend": legend,
                    "probabilities": probs,
                    "confidence": ((confidence_from_probs(&p) * 10000.0).round() / 10000.0),
                    "answer_confidence": ans_conf,
                    "action": action,
                })
            }
            "noul" => {
                let noul = p[1];
                json!({
                    "type": "noul",
                    "noul": ((noul * 10000.0).round() / 10000.0),
                    "confidence": ((noul.max(1.0 - noul) * 10000.0).round() / 10000.0),
                    "answer_confidence": ans_conf,
                    "action": action,
                })
            }
            other => anyhow::bail!("unknown type {other}"),
        };
        answers.insert(qid.clone(), entry);
    }
    Ok(Value::Object(answers))
}
