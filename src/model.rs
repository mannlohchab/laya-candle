//! Laya DecisionModel: ModernBERT encoder + typed decision head.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::modernbert::{self, ModernBert};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    #[allow(dead_code)]
    pub encoder: String,
    pub head_layers: usize,
    pub max_len: usize,
    pub head_max_len: usize,
    #[serde(default)]
    pub act_costs: HashMap<String, f64>,
    #[serde(default)]
    pub temperature: Vec<f64>,
    #[serde(default)]
    pub temperature_by_options: HashMap<String, f64>,
}

pub struct DecisionModel {
    pub encoder: ModernBert,
    pub type_emb: Embedding,
    pub head: Vec<TransformerEncoderLayer>,
    pub scorer_norm: LayerNorm,
    pub scorer_fc1: Linear,
    pub scorer_fc2: Linear,
    pub act_fc1: Linear,
    pub act_fc2: Linear,
    #[allow(dead_code)]
    pub hidden_size: usize,
}

/// PyTorch `nn.TransformerEncoderLayer` with `norm_first=True`, `batch_first=True`, ReLU.
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

struct MultiHeadAttention {
    in_proj: Linear,
    out_proj: Linear,
    nhead: usize,
    head_dim: usize,
    d_model: usize,
}

impl MultiHeadAttention {
    fn load(vb: VarBuilder, d_model: usize, nhead: usize) -> Result<Self> {
        let head_dim = d_model / nhead;
        let in_proj_weight = vb.get((3 * d_model, d_model), "in_proj_weight")?;
        let in_proj_bias = vb.get(3 * d_model, "in_proj_bias")?;
        let in_proj = Linear::new(in_proj_weight, Some(in_proj_bias));
        let out_proj = linear(d_model, d_model, vb.pp("out_proj"))?;
        Ok(Self {
            in_proj,
            out_proj,
            nhead,
            head_dim,
            d_model,
        })
    }

    /// `key_padding_mask`: true = pad (ignore), shape [b, seq]
    fn forward(&self, xs: &Tensor, key_padding_mask: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;
        let projected = xs.apply(&self.in_proj)?; // [b, seq, 3d]
        let chunks = projected.chunk(3, D::Minus1)?;
        let q = chunks[0]
            .reshape((b, seq, self.nhead, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?; // [b, nhead, seq, head_dim]
        let k = chunks[1]
            .reshape((b, seq, self.nhead, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = chunks[2]
            .reshape((b, seq, self.nhead, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let mut att = q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?; // [b,h,s,s]
        att = (att / scale)?;

        let mask = key_padding_mask
            .to_dtype(DType::F32)?
            .unsqueeze(1)?
            .unsqueeze(2)?; // [b,1,1,seq]
        let neg_inf = Tensor::full(f32::NEG_INFINITY, mask.shape(), xs.device())?;
        let zeros = Tensor::zeros(mask.shape(), DType::F32, xs.device())?;
        let additive = mask.gt(0f32)?.where_cond(&neg_inf, &zeros)?;
        att = att.broadcast_add(&additive)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let xs = att.matmul(&v)?; // [b,h,s,d]
        let xs = xs
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq, self.d_model))?;
        Ok(xs.apply(&self.out_proj)?)
    }
}

impl TransformerEncoderLayer {
    fn load(vb: VarBuilder, d_model: usize, nhead: usize) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::load(vb.pp("self_attn"), d_model, nhead)?,
            linear1: linear(d_model, 4 * d_model, vb.pp("linear1"))?,
            linear2: linear(4 * d_model, d_model, vb.pp("linear2"))?,
            norm1: layer_norm(d_model, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(d_model, 1e-5, vb.pp("norm2"))?,
        })
    }

    fn forward(&self, xs: &Tensor, key_padding_mask: &Tensor) -> Result<Tensor> {
        // norm_first: x = x + attn(norm1(x)); x = x + ffn(norm2(x))
        let x2 = xs.apply(&self.norm1)?;
        let xs = (xs + self.self_attn.forward(&x2, key_padding_mask)?)?;
        let x2 = xs.apply(&self.norm2)?;
        let ff = x2.apply(&self.linear1)?.relu()?.apply(&self.linear2)?;
        Ok((xs + ff)?)
    }
}

impl DecisionModel {
    pub fn load(
        vb: VarBuilder,
        encoder_cfg: &modernbert::Config,
        agent_cfg: &AgentConfig,
    ) -> Result<Self> {
        let d = encoder_cfg.hidden_size;
        let nhead = (d / 64).max(1);
        let encoder = ModernBert::load(vb.clone(), encoder_cfg)?;
        let type_emb = candle_nn::embedding(3, d, vb.pp("type_emb"))?;

        let mut head = Vec::new();
        for i in 0..agent_cfg.head_layers {
            head.push(TransformerEncoderLayer::load(
                vb.pp(format!("head.layers.{i}")),
                d,
                nhead,
            )?);
        }

        // scorer: LayerNorm -> Linear(d,d) -> GELU -> Linear(d,1)
        let scorer_norm = layer_norm(d, 1e-5, vb.pp("scorer.0"))?;
        let scorer_fc1 = linear(d, d, vb.pp("scorer.1"))?;
        let scorer_fc2 = linear(d, 1, vb.pp("scorer.3"))?;

        let n_act = agent_cfg.act_costs.len() + 1;
        let act_fc1 = linear(d + 4, 256, vb.pp("act_head.0"))?;
        let act_fc2 = linear(256, n_act, vb.pp("act_head.2"))?;

        Ok(Self {
            encoder,
            type_emb,
            head,
            scorer_norm,
            scorer_fc1,
            scorer_fc2,
            act_fc1,
            act_fc2,
            hidden_size: d,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        marker_pos: &Tensor,
        marker_mask: &Tensor,
        qtype: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mut h = self.encoder.forward(input_ids, attention_mask)?; // [b,s,d]
        let te = self.type_emb.forward(qtype)?; // [b,d]
        h = h.broadcast_add(&te.unsqueeze(1)?)?;

        // key_padding_mask: true where pad (attention_mask == 0)
        let key_padding_mask = attention_mask.eq(0u32)?;
        for layer in &self.head {
            h = layer.forward(&h, &key_padding_mask)?;
        }

        let (b, _s, d) = h.dims3()?;
        let kmax = marker_pos.dim(1)?;
        let idx = marker_pos
            .clamp(0i64, (h.dim(1)? as i64) - 1)?
            .unsqueeze(D::Minus1)?
            .expand((b, kmax, d))?
            .contiguous()?;
        let m = h.gather(&idx, 1)?; // [b,k,d]

        let logits = m
            .apply(&self.scorer_norm)?
            .apply(&self.scorer_fc1)?
            .gelu()?
            .apply(&self.scorer_fc2)?
            .squeeze(D::Minus1)?
            .to_dtype(DType::F32)?; // [b,k]

        let neg = Tensor::full(-1e4f32, logits.shape(), logits.device())?;
        let logits = marker_mask.to_dtype(DType::U8)?.where_cond(&logits, &neg)?;

        let (top1, gap, ent, k_feat) = act_feats(&logits, marker_mask)?;

        let pooled = h.i((.., 0, ..))?.to_dtype(DType::F32)?; // [b,d]
        let feats = Tensor::stack(&[top1, gap, ent, k_feat], D::Minus1)?; // [b,4]
        let act_in = Tensor::cat(&[&pooled, &feats], D::Minus1)?;
        let act_logits = act_in.apply(&self.act_fc1)?.gelu()?.apply(&self.act_fc2)?;
        Ok((logits, act_logits))
    }
}

fn act_feats(logits: &Tensor, marker_mask: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    // Match Python: p = softmax(logits.detach()); feats from p
    let p = candle_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
    let mask = marker_mask.to_dtype(DType::F32)?;
    let k_count = mask.sum_keepdim(D::Minus1)?.maximum(&Tensor::full(
        2f32,
        mask.sum_keepdim(D::Minus1)?.shape(),
        mask.device(),
    )?)?; // [b,1]
    let log_k = k_count.log()?;
    let log_p = p.clamp(1e-9, 1.0)?.log()?;
    let ent = ((p.clone() * log_p)?.sum_keepdim(D::Minus1)?.neg()? / &log_k)?; // [b,1]

    let neg = Tensor::full(f32::NEG_INFINITY, p.shape(), p.device())?;
    let masked = mask.gt(0f32)?.where_cond(&p, &neg)?;
    let top1 = masked.max_keepdim(D::Minus1)?; // [b,1]

    let eq = masked.broadcast_eq(&top1)?;
    let masked2 = eq.where_cond(&neg, &masked)?;
    let top2_raw = masked2.max_keepdim(D::Minus1)?;
    let top2 = top2_raw.maximum(&Tensor::zeros_like(&top2_raw)?)?;

    let gap = (top1.clone() - &top2)?;
    let k_feat = (&k_count / 255f64)?;
    Ok((
        top1.squeeze(D::Minus1)?,
        gap.squeeze(D::Minus1)?,
        ent.squeeze(D::Minus1)?,
        k_feat.squeeze(D::Minus1)?,
    ))
}

pub fn load_agent_config(path: &Path) -> Result<AgentConfig> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

/// Build Candle ModernBERT config from Laya encoder/config.json (inject RoPE thetas).
pub fn load_encoder_config(path: &Path) -> Result<modernbert::Config> {
    let raw: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
    )?;
    let rope = &raw["rope_parameters"];
    let global = rope
        .get("full_attention")
        .and_then(|v| v.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .or_else(|| raw.get("global_rope_theta").and_then(|v| v.as_f64()))
        .unwrap_or(160_000.0);
    let local = rope
        .get("sliding_attention")
        .and_then(|v| v.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .or_else(|| raw.get("local_rope_theta").and_then(|v| v.as_f64()))
        .unwrap_or(10_000.0);

    let cleaned = serde_json::json!({
        "vocab_size": raw["vocab_size"],
        "hidden_size": raw["hidden_size"],
        "num_hidden_layers": raw["num_hidden_layers"],
        "num_attention_heads": raw["num_attention_heads"],
        "intermediate_size": raw["intermediate_size"],
        "max_position_embeddings": raw["max_position_embeddings"],
        "layer_norm_eps": raw.get("layer_norm_eps").or(raw.get("norm_eps")).unwrap_or(&serde_json::json!(1e-5)),
        "pad_token_id": raw["pad_token_id"],
        "global_attn_every_n_layers": raw["global_attn_every_n_layers"],
        "global_rope_theta": global,
        "local_attention": raw["local_attention"],
        "local_rope_theta": local,
    });
    Ok(serde_json::from_value(cleaned)?)
}

/// Load safetensors, remap `encoder.` → `model.` for Candle ModernBERT.
pub fn load_varbuilder(weights: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let tensors = candle_core::safetensors::load(weights, device)
        .with_context(|| format!("load weights {}", weights.display()))?;
    let mut remapped = HashMap::new();
    for (k, v) in tensors {
        let key = if let Some(rest) = k.strip_prefix("encoder.") {
            format!("model.{rest}")
        } else {
            k
        };
        remapped.insert(key, v);
    }
    Ok(VarBuilder::from_tensors(remapped, DType::F32, device))
}
