//! The runnable-graph artifact: `save_runnable` adds the serialized forward graph and its
//! output/input bindings to the weight rows; `load_runnable` reads them back into a
//! [`RunnableModel`] that runs the forward from the `.hodu` file alone.
mod load;
mod save;

pub use load::{RunnableModel, load_runnable};
pub use save::{save_multi, save_runnable};

// Reserved input-binding name prefix for the internal RNG Inputs. Starts with NUL so it can
// never collide with a module FQN (dot-joined identifiers), letting load tell an auto-fed
// eval default apart from a real weight row. Shared by the save and load paths.
const RNG_MARK: char = '\0';

#[cfg(test)]
mod tests;
