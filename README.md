# laya-candle

English Laya decision-model inference with Candle.

## Build

```bash
cargo build --release
```

## Run

```bash
cargo run --release -- --demo
```

```bash
cargo run --release -- \
  --state "your text here" \
  --questions questions.json
```

Model defaults to `convaiinnovations/laya` (Hub id or local directory).
