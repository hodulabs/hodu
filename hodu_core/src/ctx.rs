//! Device context: owns the record graph, a backend, and the feed map (Input
//! node -> current host value). Cheap to clone (`Rc`); every `Tensor` holds one.
//! single-threaded record (`Rc`/`RefCell`); swap to `Arc` + lock to share across
//! threads.
use kurumi::{Backend, CpuBackend, DType, Error, Feeds, Graph, NodeId, Storage, TensorVal};
use std::cell::RefCell;
use std::rc::Rc;

use crate::Tensor;

mod rng;

#[derive(Clone)]
pub struct Ctx(Rc<CtxInner>);

struct CtxInner {
    graph: RefCell<Graph>,
    backend: Box<dyn Backend>,
    feeds: RefCell<Feeds>,
    rng: RefCell<RngReg>,
}

// Dropout plumbing for the build-once graph: one shared `flag` Input (1.0 train /
// 0.0 eval) turns every dropout into identity in eval, and each dropout's `seed`
// Input is refreshed per step (`tick_rng`) so the mask is fresh -- all without
// rebuilding the graph.
#[derive(Default)]
struct RngReg {
    training: bool,
    counter: u64,
    flag: Option<NodeId>,
    seeds: Vec<NodeId>,
}

impl Ctx {
    /// Context over the CPU reference backend.
    pub fn cpu() -> Ctx {
        Ctx::with_backend(Box::new(CpuBackend))
    }

    /// Context over any backend (e.g. Metal, injected from the top crate).
    pub fn with_backend(backend: Box<dyn Backend>) -> Ctx {
        Ctx(Rc::new(CtxInner {
            graph: RefCell::new(Graph::new()),
            backend,
            feeds: RefCell::new(Feeds::new()),
            rng: RefCell::new(RngReg { training: true, ..Default::default() }),
        }))
    }

    /// Context over the Metal backend, or `None` if no Metal device is available
    /// (unsupported ops fall back to the CPU oracle inside the engine).
    pub fn metal() -> Option<Ctx> {
        kurumi::metal::MetalBackend::new().map(|b| Ctx::with_backend(Box::new(b)))
    }

    /// An f32 constant tensor (baked into the graph).
    pub fn constant(&self, data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        self.build_inf(|g| g.constant(data, shape))
    }

    /// An all-zeros f32 constant of `shape` (e.g. an RNN's initial hidden/cell state).
    pub fn zeros(&self, shape: Vec<usize>) -> Tensor {
        let n: usize = shape.iter().product();
        self.constant(vec![0.0; n], shape)
    }

    /// Stack `parts` (all the same shape) on a NEW axis `axis`: `n` tensors of
    /// `[..]` -> `[.., n, ..]`. Used to reassemble per-timestep RNN outputs `[B,H]`
    /// into a sequence `[B,T,H]`.
    pub fn stack(&self, parts: &[&Tensor], axis: usize) -> Result<Tensor, Error> {
        let ns: Vec<NodeId> = parts.iter().map(|t| t.node()).collect();
        self.build(|g| g.stack(&ns, axis))
    }

    /// Concatenate `parts` along an EXISTING axis `axis`.
    pub fn concat(&self, parts: &[&Tensor], axis: usize) -> Result<Tensor, Error> {
        let ns: Vec<NodeId> = parts.iter().map(|t| t.node()).collect();
        self.build(|g| g.concat(&ns, axis))
    }

    /// A fed f32 leaf: a parameter or a data slot. Grad treats it as a leaf.
    /// Set its value with [`Ctx::feed`]; it is supplied on every eval.
    pub fn input(&self, shape: Vec<usize>) -> Tensor {
        self.build_inf(|g| g.input(shape, DType::F32))
    }

    /// Set the f32 value fed to an Input `node` on subsequent evals.
    pub fn feed(&self, node: NodeId, data: Vec<f32>, shape: Vec<usize>) {
        self.0.feeds.borrow_mut().insert(node, TensorVal { shape, storage: Storage::F32(data) });
    }

    /// A fed integer leaf of `dtype` (must be an integer dtype). For gather/embedding
    /// indices; feed it with [`Ctx::feed_i64`]. See also [`Ctx::input_i64`].
    pub fn input_dtype(&self, shape: Vec<usize>, dtype: DType) -> Tensor {
        self.build_inf(|g| g.input(shape, dtype))
    }

    /// A fed I64 leaf (token ids / gather indices). Set it with [`Ctx::feed_i64`].
    pub fn input_i64(&self, shape: Vec<usize>) -> Tensor {
        self.input_dtype(shape, DType::I64)
    }

    /// Set the i64 value fed to an integer Input `node` (from [`Ctx::input_i64`]) on
    /// subsequent evals.
    pub fn feed_i64(&self, node: NodeId, data: Vec<i64>, shape: Vec<usize>) {
        self.0.feeds.borrow_mut().insert(node, TensorVal { shape, storage: Storage::I64(data) });
    }

    /// Set the u8 value fed to a U8 Input `node` (e.g. a packed quant weight) on
    /// subsequent evals. Pair with [`Ctx::input_dtype`]`(shape, DType::U8)`.
    pub fn feed_u8(&self, node: NodeId, data: Vec<u8>, shape: Vec<usize>) {
        self.0.feeds.borrow_mut().insert(node, TensorVal { shape, storage: Storage::U8(data) });
    }

    /// Realize `node` to host f32 on the backend, supplying the current feeds.
    pub fn eval_f32(&self, node: NodeId) -> Vec<f32> {
        self.0.backend.eval_with(&self.0.graph.borrow(), node, &self.0.feeds.borrow()).f32().to_vec()
    }

    /// Realize several `nodes` to host f32 in ONE shared backend pass (a subgraph
    /// common to the outputs -- e.g. the forward+backward trunk shared by many grads
    /// -- computes once). Order preserved, one `Vec<f32>` per node.
    pub fn eval_many_f32(&self, ids: &[NodeId]) -> Vec<Vec<f32>> {
        self.0
            .backend
            .eval_many_with(&self.0.graph.borrow(), ids, &self.0.feeds.borrow())
            .iter()
            .map(|t| t.f32().to_vec())
            .collect()
    }

    /// Borrow the record graph, e.g. to serialize it. The closure scopes the borrow so
    /// the graph never escapes the `RefCell`.
    pub fn with_graph<R>(&self, f: impl FnOnce(&Graph) -> R) -> R {
        f(&self.0.graph.borrow())
    }

    /// Reverse-mode grad of `output` w.r.t. each node in `wrt` (adds backward
    /// nodes to the graph); returns the grad node ids.
    pub(crate) fn grad(&self, output: NodeId, wrt: &[NodeId]) -> Result<Vec<NodeId>, Error> {
        kurumi::grad(&mut self.0.graph.borrow_mut(), output, wrt)
    }

    /// Escape hatch: record any kurumi op via the raw builder and wrap the
    /// result, so an op not surfaced as a `Tensor` method is still one call away:
    /// `ctx.build(|g| g.some_op(a.node(), ..))`. The borrow ends before `wrap`
    /// re-borrows (the `borrow_mut` temporary drops at statement end).
    pub fn build<F>(&self, f: F) -> Result<Tensor, Error>
    where
        F: FnOnce(&mut Graph) -> Result<NodeId, Error>,
    {
        let n = f(&mut self.0.graph.borrow_mut())?;
        Ok(self.wrap(n))
    }

    /// Same, for infallible builders (constant / unary ops).
    pub fn build_inf<F: FnOnce(&mut Graph) -> NodeId>(&self, f: F) -> Tensor {
        let n = f(&mut self.0.graph.borrow_mut());
        self.wrap(n)
    }

    /// Like [`Ctx::build`] for a builder that yields SEVERAL nodes (e.g. `g.split`);
    /// wraps each. The `borrow_mut` drops at statement end, before `wrap` re-borrows.
    pub fn build_many<F>(&self, f: F) -> Result<Vec<Tensor>, Error>
    where
        F: FnOnce(&mut Graph) -> Result<Vec<NodeId>, Error>,
    {
        let ns = f(&mut self.0.graph.borrow_mut())?;
        Ok(ns.into_iter().map(|n| self.wrap(n)).collect())
    }

    fn wrap(&self, node: NodeId) -> Tensor {
        let g = self.0.graph.borrow();
        Tensor::new(self.clone(), node, g.shape(node).to_vec(), g.dtype(node))
    }
}
