# Contributing

Thanks for your interest in hodu.

## Development

A Cargo workspace driven by [`just`](https://github.com/casey/just):

```
just check     # format check, lint, and all tests -- the CI gate
just format    # format Rust and the justfile
just --list    # all recipes
```

Requirements: a recent stable Rust (edition 2024). hodu is the static frontend over the
[kurumi](https://github.com/hodulabs/kurumi) engine; kurumi is a path dependency. The Metal
tests are macOS-only, on Apple Silicon; elsewhere they skip and the CPU path still builds
and tests.

## Where the layers live

- `hodu_core` -- the `Tensor` handle: numpy broadcasting and dtype promotion inserted before
  the engine's strict ops, plus the `Ctx` that owns the graph, backend, and feeds.
- `hodu` -- the user layer: `nn` modules, `optim`, `loss`, `data`, the `.hodu` artifact, and
  weight-only quant, composed over `hodu_core`.

The engine is the reference: kurumi's CPU interpreter defines correctness, so a wrapped op is
correct iff it matches the engine on the same graph.

## Adding an op or a layer

hodu builds on kurumi's builder surface -- most additions are thin wraps, not new engine ops.

- A new `Tensor` method wraps a `Graph` builder via `Ctx::build`; broadcasting and promotion
  go through `bin()` before the strict op.
- A new `nn` layer is a `Module`: a leaf reports its own `parameters`/`buffers`, a container
  implements only `children()` (every named/flat walk derives from it).
- Cover it with a test; for a numeric result, assert against the engine or a hand value.
- If an op genuinely belongs in the engine, add it to kurumi first, then wrap it here.

## Conventions

- No `mod.rs`. Modules are `foo.rs` + `foo/*.rs` (a `foo.rs` may hold code and declare its
  submodules).
- Comments explain the logic and the reason, kernel-style: terse, ASCII, no ornament.
- Keep `just check` clean -- `clippy --all-targets -D warnings` is part of the gate.

## Pull requests

- `just check` passes (format, lint, tests).
- One focused change per pull request.

## License

By contributing, you agree that your contributions are dual licensed under
[MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE), matching the project.
