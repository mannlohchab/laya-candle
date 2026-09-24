//! High-level Agent: load checkpoint, encode questions, run DecisionModel.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::ops::softmax;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::decode::{decode_answers, Temperatures};
use crate::model::{
    load_agent_config, load_encoder_config, load_varbuilder, AgentConfig, DecisionModel,
};
use crate::sequence::{
    build_sequence, qtype_id, render_options, serialize_state, to_internal, EncodedItem,
    InternalQuestion, SpecialIds,
};

pub struct Agent {
    pub cfg: AgentConfig,
    pub model: DecisionModel,
    pub tok: Tokenizer,
    pub special: SpecialIds,
    pub temps: Temperatures,
    pub device: Device,
    pub model_dir: PathBuf,
}

impl Agent {
    pub fn load(model_id_or_path: &str, device: Device) -> Result<Self> {
        let model_dir = resolve_model_dir(model_id_or_path)?;
        let cfg = load_agent_config(&model_dir.join("rl_agent_config.json"))?;
        let enc_cfg = load_encoder_config(&model_dir.join("encoder/config.json"))?;
        let tok = Tokenizer::from_file(model_dir.join("tokenizer/tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        let special = SpecialIds::from_tokenizer(&tok)?;
        let vb = load_varbuilder(&model_dir.join("model.safetensors"), &device)?;
        let model = DecisionModel::load(vb, &enc_cfg, &cfg)?;
        let temps = Temperatures::from_config(&cfg);
        Ok(Self {
            cfg,
            model,
            tok,
            special,
            temps,
            device,
            model_dir,
        })
    }

    pub fn system_one(&self, state: &Value, questions: &Value) -> Result<Value> {
        let qobj = questions
            .as_object()
            .context("questions must be a JSON object")?;
        let ids: Vec<String> = qobj.keys().cloned().collect();
        let mut internal: HashMap<String, InternalQuestion> = HashMap::new();
        for qid in &ids {
            internal.insert(qid.clone(), to_internal(qid, &qobj[qid])?);
        }

        let max_len = self.cfg.max_len;
        let head_max_len = self.cfg.head_max_len;
        let truncate_left = state.is_array();
        let state_text = serialize_state(state).replace(&self.special.mask_token, " ");
        let state_ids = self
            .tok
            .encode(state_text, false)
            .map_err(|e| anyhow::anyhow!("tokenize state: {e}"))?
            .get_ids()
            .to_vec();

        let mut items: Vec<EncodedItem> = Vec::with_capacity(ids.len());
        for qid in &ids {
            let q = &internal[qid];
            let (seq, markers) = build_sequence(
                &self.tok,
                &self.special,
                state,
                q,
                max_len,
                head_max_len,
                truncate_left,
                Some(&state_ids),
            )?;
            let n_opts = render_options(q)?.len();
            if markers.len() != n_opts {
                bail!("question {qid} options exceed head_max_len={head_max_len}");
            }
            items.push(EncodedItem {
                ids: seq,
                markers,
                qtype: qtype_id(&q.t)?,
            });
        }

        let batch = collate(&items, self.special.pad, &self.device)?;
        let (logits_t, act_t) = self.model.forward(
            &batch.input_ids,
            &batch.attention_mask,
            &batch.marker_pos,
            &batch.marker_mask,
            &batch.qtype,
        )?;
        let act_prob = softmax(&act_t, candle_core::D::Minus1)?;

        let logits = tensor_to_vec2(&logits_t)?;
        let act = tensor_to_vec2(&act_prob)?;
        let markers_len: Vec<usize> = items.iter().map(|it| it.markers.len()).collect();
        let answers = decode_answers(&logits, &act, &ids, &internal, &markers_len, &self.temps)?;

        Ok(serde_json::json!({
            "answers": answers,
            "model": self.model_dir.display().to_string(),
        }))
    }
}

struct Batch {
    input_ids: Tensor,
    attention_mask: Tensor,
    marker_pos: Tensor,
    marker_mask: Tensor,
    qtype: Tensor,
}

fn collate(items: &[EncodedItem], pad_id: u32, device: &Device) -> Result<Batch> {
    let n = items.len();
    let l = items.iter().map(|it| it.ids.len()).max().unwrap_or(0);
    let kmax = items.iter().map(|it| it.markers.len()).max().unwrap_or(0);

    let mut ids = vec![pad_id; n * l];
    let mut att = vec![0u32; n * l];
    let mut mpos = vec![0i64; n * kmax];
    let mut mmask = vec![0u8; n * kmax];
    let mut qtypes = vec![0i64; n];

    for (i, it) in items.iter().enumerate() {
        for (j, &tok) in it.ids.iter().enumerate() {
            ids[i * l + j] = tok;
            att[i * l + j] = 1;
        }
        for (j, &m) in it.markers.iter().enumerate() {
            mpos[i * kmax + j] = m as i64;
            mmask[i * kmax + j] = 1;
        }
        qtypes[i] = it.qtype;
    }

    Ok(Batch {
        input_ids: Tensor::from_vec(ids, (n, l), device)?,
        attention_mask: Tensor::from_vec(att, (n, l), device)?,
        marker_pos: Tensor::from_vec(mpos, (n, kmax), device)?,
        marker_mask: Tensor::from_vec(mmask, (n, kmax), device)?,
        qtype: Tensor::from_vec(qtypes, n, device)?,
    })
}

fn tensor_to_vec2(t: &Tensor) -> Result<Vec<Vec<f64>>> {
    let t = t.to_dtype(candle_core::DType::F32)?;
    let data = t.to_vec2::<f32>()?;
    Ok(data
        .into_iter()
        .map(|row| row.into_iter().map(|v| v as f64).collect())
        .collect())
}

fn resolve_model_dir(model_id_or_path: &str) -> Result<PathBuf> {
    let p = Path::new(model_id_or_path);
    if p.join("rl_agent_config.json").is_file() {
        return Ok(p.to_path_buf());
    }
    if p.is_dir() {
        bail!(
            "directory {} exists but has no rl_agent_config.json",
            p.display()
        );
    }

    // Prefer local HF cache snapshot if present (english root of laya bundle).
    if let Some(cached) = find_cached_laya(model_id_or_path) {
        return Ok(cached);
    }

    bail!(
        "model {model_id_or_path:?} not found locally. Pass a directory with \
         rl_agent_config.json / model.safetensors / tokenizer / encoder, or download \
         convaiinnovations/laya into the Hugging Face hub cache first \
         (e.g. `huggingface-cli download convaiinnovations/laya`)."
    );
}

fn find_cached_laya(model_id: &str) -> Option<PathBuf> {
    // Known aliases for the english checkpoint root.
    let aliases = [
        model_id,
        "convaiinnovations/laya",
        "laya",
        "english",
    ];
    let hub = dirs_hub_cache()?;
    for name in aliases {
        let repo = name.replace('/', "--");
        let base = hub.join(format!("models--{repo}"));
        let snaps = base.join("snapshots");
        if let Ok(rd) = std::fs::read_dir(&snaps) {
            for ent in rd.flatten() {
                let dir = ent.path();
                if dir.join("rl_agent_config.json").is_file()
                    && dir.join("model.safetensors").is_file()
                {
                    return Some(dir);
                }
            }
        }
    }
    None
}

fn dirs_hub_cache() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        return Some(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".cache/huggingface/hub"))
}
