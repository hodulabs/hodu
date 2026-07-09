# hodu_core

The `Tensor` layer of [hodu](https://github.com/hodulabs/hodu): a NumPy-ergonomic handle over the [kurumi](https://github.com/hodulabs/kurumi) engine. It owns the graph, backend, and feeds, and inserts broadcasting + dtype promotion before the engine's strict ops.

## Design

- **The frontend's job: broadcast and promote.** kurumi's builder ops require identical shape and dtype. `Tensor::bin` applies NumPy broadcasting and dtype promotion, then calls the strict op -- so `x + bias` and mixed-dtype arithmetic just work.
- **`Ctx` owns the graph.** One `Ctx` holds a kurumi `Graph`, a backend (CPU or Metal), and the feed map (Input node -> host value). `Ctx::cpu()` / `Ctx::metal()` pick the device; `Ctx::build` is the escape hatch to any engine op not yet a method.
- **Static, build-once.** Shapes and dtypes are the engine's; the graph is built once and fed per step. Errors are reported at record time, pointing at the user's line, not an eval stack.
- **A broad surface.** Operator overloading, matmul, reductions, activations, conv/pool, attention (sdpa/rope), shape ops, and `quant_matmul` -- thin wraps of the engine, split by domain across `tensor/*.rs`.

## Example

```rust
use hodu_core::Ctx;

let ctx = Ctx::cpu();
let a = ctx.constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let bias = ctx.constant(vec![10.0, 20.0], vec![2]);   // [2]
let y = (&a + &bias).unwrap().relu();                 // broadcast [2] over [2,2], then relu
let out = y.realize();                                 // eval on the ctx backend
```

## Layout

- `ctx.rs` (+ `ctx/rng.rs`) -- the `Ctx`: graph, backend, feeds, and the dropout/train-eval plumbing
- `tensor.rs` -- the `Tensor` handle + the broadcasting machine
- `tensor/` -- the op surface by domain: `operators`, `elementwise`, `activation`, `reduce`, `linalg`, `shape`, `conv`, `attention`, `norm`, `loss`
