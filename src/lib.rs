pub mod color;
pub mod entity;
pub mod math;
pub mod model;
pub mod physics;
pub mod render;

/// Epsilon value used in certain situations when correcting for floating point arithmetic is
/// required.
pub(crate) const EPSILON: f32 = 0.01;
