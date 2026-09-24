//! Port of Laya `build_sequence` / `render_options` / `serialize_state`.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

pub const QTYPE_CHOICE: i64 = 0;
pub const QTYPE_SCORE: i64 = 1;
pub const QTYPE_NOUL: i64 = 2;

#[derive(Debug, Clone)]
pub struct InternalQuestion {
    pub t: String,
    pub ins: String,
    pub crit: Value,
    pub labels: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct EncodedItem {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: i64,
}

pub struct SpecialIds {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub pad: u32,
    pub mask_token: String,
}

impl SpecialIds {
    pub fn from_tokenizer(tok: &Tokenizer) -> Result<Self> {
        let get = |name: &str| -> Result<u32> {
            tok.token_to_id(name)
                .with_context(|| format!("tokenizer missing special token {name}"))
        };
        Ok(Self {
            cls: get("[CLS]")?,
            sep: get("[SEP]")?,
            mask: get("[MASK]")?,
            pad: get("[PAD]")?,
            mask_token: "[MASK]".to_string(),
        })
    }
}

pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn render_criterion(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| other.to_string()),
    }
}

fn resolve_noul_labels(labels: Option<&Value>) -> Result<(String, String)> {
    let default = serde_json::json!({"false": "false", "true": "true"});
    let labels = labels.unwrap_or(&default);
    let obj = labels
        .as_object()
        .context("noul labels must be an object with false/true")?;
    if obj.len() != 2 || !obj.contains_key("false") || !obj.contains_key("true") {
        bail!("noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
    }
    let false_label = obj["false"]
        .as_str()
        .context("noul false label must be a string")?
        .trim();
    let true_label = obj["true"]
        .as_str()
        .context("noul true label must be a string")?
        .trim();
    if false_label.is_empty() || true_label.is_empty() || false_label == true_label {
        bail!("noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
    }
    Ok((false_label.to_string(), true_label.to_string()))
}

pub fn render_options(q: &InternalQuestion) -> Result<Vec<String>> {
    if q.t != "noul" && q.labels.is_some() {
        bail!("labels is only supported for noul questions");
    }
    match q.t.as_str() {
        "choice" => {
            let obj = q
                .crit
                .as_object()
                .context("choice criteria must be an object")?;
            Ok(obj
                .iter()
                .map(|(k, v)| {
                    if v.is_null() || v.as_str() == Some("") {
                        k.clone()
                    } else {
                        format!("{k}: {}", render_criterion(v))
                    }
                })
                .collect())
        }
        "score" => {
            let arr = q.crit.as_array().context("score criteria must be a list")?;
            Ok(arr
                .iter()
                .enumerate()
                .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
                .collect())
        }
        "noul" => {
            let (false_label, true_label) = resolve_noul_labels(q.labels.as_ref())?;
            let crit = q.crit.as_object();
            let false_crit = crit.and_then(|c| c.get("false"));
            let true_crit = crit.and_then(|c| c.get("true"));
            let false_text = match false_crit {
                None | Some(Value::Null) => "no, the statement does not hold".to_string(),
                Some(Value::String(s)) if s.is_empty() => {
                    "no, the statement does not hold".to_string()
                }
                Some(v) => render_criterion(v),
            };
            let true_text = match true_crit {
                None | Some(Value::Null) => "yes, the statement holds".to_string(),
                Some(Value::String(s)) if s.is_empty() => "yes, the statement holds".to_string(),
                Some(v) => render_criterion(v),
            };
            Ok(vec![
                format!("{false_label}: {false_text}"),
                format!("{true_label}: {true_text}"),
            ])
        }
        other => bail!("unknown question type {other}"),
    }
}

pub fn qtype_id(t: &str) -> Result<i64> {
    match t {
        "choice" => Ok(QTYPE_CHOICE),
        "score" => Ok(QTYPE_SCORE),
        "noul" => Ok(QTYPE_NOUL),
        other => bail!("unknown question type {other}"),
    }
}

fn encode_no_special(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
    Ok(encoding.get_ids().to_vec())
}

fn encode_truncated(tok: &Tokenizer, text: &str, max_length: usize) -> Result<Vec<u32>> {
    let mut ids = encode_no_special(tok, text)?;
    if ids.len() > max_length {
        ids.truncate(max_length);
    }
    Ok(ids)
}

/// Format: [CLS] <type> question: instructions [SEP] [MASK] opt0 ... [SEP] state [SEP]
pub fn build_sequence(
    tok: &Tokenizer,
    special: &SpecialIds,
    state: &Value,
    q: &InternalQuestion,
    max_len: usize,
    head_max_len: usize,
    truncate_left: bool,
    state_ids: Option<&[u32]>,
) -> Result<(Vec<u32>, Vec<usize>)> {
    let opts = render_options(q)?;
    let order: Vec<usize> = (0..opts.len()).collect();
    let ins = q.ins.replace(&special.mask_token, " ");
    let head_text = format!("{} question: {ins}", q.t);
    let mut head_ids = encode_no_special(tok, &head_text)?;

    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(order.len());
    for &i in &order {
        let text = format!(" {}", opts[i].replace(&special.mask_token, " "));
        let tokens = encode_truncated(tok, &text, 48)?;
        let mut row = Vec::with_capacity(1 + tokens.len());
        row.push(special.mask);
        row.extend(tokens);
        opt_ids.push(row);
    }

    let opt_sum: usize = opt_ids.iter().map(|o| o.len()).sum();
    let mut opt_budget = head_max_len as isize - opt_sum as isize;
    if opt_budget < 16 {
        let per = ((head_max_len as isize - 16).max(0) as usize / order.len().max(1)).max(4);
        for o in &mut opt_ids {
            o.truncate(per);
        }
        let opt_sum: usize = opt_ids.iter().map(|o| o.len()).sum();
        opt_budget = head_max_len as isize - opt_sum as isize;
    }
    let head_keep = (opt_budget as usize).max(8);
    head_ids.truncate(head_keep);

    let mut ids = Vec::with_capacity(max_len);
    ids.push(special.cls);
    ids.extend(head_ids);
    ids.push(special.sep);

    let mut markers = Vec::new();
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend(o);
    }
    ids.push(special.sep);

    let room = max_len.saturating_sub(ids.len() + 1);
    let owned_state;
    let state_ids = if let Some(s) = state_ids {
        s
    } else {
        let st = serialize_state(state).replace(&special.mask_token, " ");
        owned_state = encode_no_special(tok, &st)?;
        &owned_state
    };
    let st = if truncate_left {
        if room == 0 {
            &state_ids[state_ids.len()..state_ids.len()]
        } else if state_ids.len() > room {
            &state_ids[state_ids.len() - room..]
        } else {
            state_ids
        }
    } else {
        &state_ids[..state_ids.len().min(room)]
    };
    ids.extend(st);
    ids.push(special.sep);
    if ids.len() > max_len {
        ids.truncate(max_len);
    }
    let markers: Vec<usize> = markers.into_iter().filter(|&m| m < max_len).collect();
    Ok((ids, markers))
}

pub fn to_internal(qid: &str, qdef: &Value) -> Result<InternalQuestion> {
    let obj = qdef
        .as_object()
        .with_context(|| format!("question {qid}: definition must be an object"))?;
    let t = obj
        .get("type")
        .and_then(|v| v.as_str())
        .with_context(|| format!("question {qid}: missing type"))?
        .to_string();
    if !matches!(t.as_str(), "choice" | "score" | "noul") {
        bail!("question {qid}: unknown type {t}");
    }
    let ins = match obj.get("instructions") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => bail!("question {qid}: no 'instructions'"),
    };
    let mut crit = obj.get("criteria").cloned().unwrap_or(Value::Null);
    if t == "choice" {
        if let Value::Array(arr) = &crit {
            let mut map = serde_json::Map::new();
            for c in arr {
                let key = match c {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                map.insert(key, Value::Null);
            }
            crit = Value::Object(map);
        }
        if !crit.is_object() {
            bail!("question {qid}: choice criteria must be object or list");
        }
    } else if t == "score" {
        if !crit.is_array() {
            bail!("question {qid}: score criteria must be a list");
        }
    } else if crit.is_null() {
        crit = Value::Object(serde_json::Map::new());
    } else if let Value::Object(map) = &crit {
        let mut norm = serde_json::Map::new();
        for (k, v) in map {
            norm.insert(k.to_lowercase(), v.clone());
        }
        crit = Value::Object(norm);
    } else {
        bail!("question {qid}: noul criteria must be an object or omitted");
    }
    let labels = obj.get("labels").cloned();
    Ok(InternalQuestion {
        t,
        ins,
        crit,
        labels,
    })
}
