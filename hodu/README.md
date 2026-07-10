# hodu

The user layer of [hodu](https://github.com/hodulabs/hodu): `nn` modules, optimizers, losses, a data loader, weight-only quantization, and the `.hodu` artifact -- composed over [`hodu_core`](../hodu_core)'s `Tensor`.

## What's here

- **`nn`** -- one `Module` trait (`forward` + `children`). Layers: `Linear`, `Conv2d`, `MaxPool2d`/`AvgPool2d`, `Flatten`, `Embedding`, `MultiHeadAttention`, `TransformerBlock`/`TransformerEncoder`, `Lstm`/`Gru`, `LayerNorm`/`RmsNorm`/`GroupNorm`/`InstanceNorm`/`BatchNorm1d`/`BatchNorm2d`, `Dropout`, `Sequential`, plus activation modules. A leaf reports its own params/buffers; a container implements only `children()`, so every flat and named walk derives from one method.
- **`optim`** -- `Sgd` (with momentum/weight decay), `Adam`, `AdamW`; `StepLR`/`CosineAnnealingLR`/`LambdaLR`; `clip_grad_norm` and `accumulate_grads`. Optimizers carry `OptState` for checkpointing.
- **`loss`** -- `mse_loss`, `cross_entropy`, `bce_loss`/`bce_with_logits`, `nll_loss`, `huber_loss`, `hinge_loss`, `kl_div`.
- **`data`** -- `Dataset`/`DataLoader` over f32 features or i64 tokens, class or regression targets, with a train/val `split`.
- **`serialize`** -- the `.hodu` v1 container: `save`/`load` (named params + buffers + quant byte-buffers), `save_checkpoint`/`load_checkpoint` (adds optimizer state), and `save_runnable`/`load_runnable` (adds the forward graph and runs it, so the model runs from the file alone).
- **`QuantLinear`** -- weight-only int8/int4 quantization for a smaller deploy artifact.

## Model contract

Layers hold their parameters as fed Input leaves in a `Ctx`. Build the forward graph once, feed batches each step, and run the optimizer -- the graph node stays fixed while the host value is re-fed. Naming is a stable FQN per architecture, so a `.hodu` file loads by name into an identically-built model.

## Example

```rust
use hodu::prelude::*;

let ctx = Ctx::cpu();
let net = Sequential::new(vec![
    Box::new(Linear::new(&ctx, 2, 32, 0)),
    Box::new(Relu),
    Box::new(Linear::new(&ctx, 32, 3, 1)),
]);
// forward returns a Tensor in `ctx`; train with the optim/loss surface, then `save(...)`.
```

See [`examples`](examples) for full training loops.
