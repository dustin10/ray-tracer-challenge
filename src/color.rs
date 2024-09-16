use std::ops::{Add, Mul, Sub};

pub const WHITE: Color = Color::from(1.0, 1.0, 1.0);
pub const BLACK: Color = Color::from(0.0, 0.0, 0.0);
pub const RED: Color = Color::from(1.0, 0.0, 0.0);
pub const GREEN: Color = Color::from(0.0, 1.0, 0.0);
pub const BLUE: Color = Color::from(0.0, 0.0, 1.0);
pub const CYAN: Color = Color::from(0.0, 1.0, 1.0);
pub const MAGENTA: Color = Color::from(1.0, 0.0, 1.0);
pub const YELLOW: Color = Color::from(1.0, 1.0, 0.0);
pub const ORANGE: Color = Color::from(1.0, 0.65, 0.0);

/// Represents a color in the scene by using red, green and blue values.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color {
    /// Creates a new [Color] with all components set to zero, i.e. black.
    pub fn new() -> Self {
        Self::default()
    }
    /// Creates a new [Color] from the given values for the red, green and blue components.
    pub const fn from(r: f32, g: f32, b: f32) -> Self {
        Color { r, g, b }
    }
    /// Creates a new [Color] from the Hadamard product of the given [Color]s.
    pub fn from_hadamard(a: &Self, b: &Self) -> Self {
        Color::from(a.r * b.r, a.g * b.g, a.b * b.b)
    }
    /// Creates a new [Color] from an existing [Color] scaled by the given value.
    pub fn from_scaled(c: &Self, s: f32) -> Self {
        Color::from(c.r * s, c.g * s, c.b * s)
    }
    /// Transforms the existing [Color] to the result of the Hadamard product with
    /// the given [Color].
    pub fn hadamard(&mut self, c: &Self) {
        self.r *= c.r;
        self.g *= c.g;
        self.b *= c.b;
    }
    /// Transforms the existing [Color] to the result of scaling the components by the
    /// given value.
    pub fn scale(&mut self, s: f32) {
        self.r *= s;
        self.g *= s;
        self.b *= s;
    }
}

impl Add for Color {
    type Output = Color;

    /// Adds the two [Color]s together returning the result.
    fn add(self, rhs: Self) -> Self::Output {
        Color::from(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b)
    }
}

impl Sub for Color {
    type Output = Color;

    /// Subtracts the two [Color]s returning the result.
    fn sub(self, rhs: Self) -> Self::Output {
        Color::from(self.r - rhs.r, self.g - rhs.g, self.b - rhs.b)
    }
}

impl Mul for Color {
    type Output = Color;

    /// Computes the Hadamard product of the two [Color]s returning the result.
    fn mul(self, rhs: Self) -> Self::Output {
        Color::from_hadamard(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::Color;

    #[test]
    fn test_color_new() {
        let c = Color::new();

        assert_eq!(0.0, c.r);
        assert_eq!(0.0, c.g);
        assert_eq!(0.0, c.b);
    }

    #[test]
    fn test_color_of() {
        let c = Color::from(1.0, 2.0, 3.0);

        assert_eq!(1.0, c.r);
        assert_eq!(2.0, c.g);
        assert_eq!(3.0, c.b);
    }

    #[test]
    fn test_color_from_hadamard() {
        let a = Color::from(1.0, 2.0, 3.0);
        let b = Color::from(1.0, 2.0, 3.0);
        let c = Color::from_hadamard(&a, &b);

        assert_eq!(1.0, c.r);
        assert_eq!(4.0, c.g);
        assert_eq!(9.0, c.b);
    }

    #[test]
    fn test_color_from_scaled() {
        let a = Color::from(1.0, 2.0, 3.0);
        let c = Color::from_scaled(&a, 2.0);

        assert_eq!(2.0, c.r);
        assert_eq!(4.0, c.g);
        assert_eq!(6.0, c.b);
    }

    #[test]
    fn test_color_hadamard() {
        let mut a = Color::from(1.0, 2.0, 3.0);
        let b = Color::from(1.0, 2.0, 3.0);

        a.hadamard(&b);

        assert_eq!(1.0, a.r);
        assert_eq!(4.0, a.g);
        assert_eq!(9.0, a.b);
    }

    #[test]
    fn test_color_scale() {
        let mut a = Color::from(1.0, 2.0, 3.0);

        a.scale(2.0);

        assert_eq!(2.0, a.r);
        assert_eq!(4.0, a.g);
        assert_eq!(6.0, a.b);
    }

    #[test]
    fn test_color_add() {
        let a = Color::from(1.0, 2.0, 3.0);
        let b = Color::from(3.0, 4.0, 5.0);

        let c = a + b;

        assert_eq!(4.0, c.r);
        assert_eq!(6.0, c.g);
        assert_eq!(8.0, c.b);
    }

    #[test]
    fn test_color_sub() {
        let a = Color::from(1.0, 2.0, 3.0);
        let b = Color::from(3.0, 4.0, 5.0);

        let c = a - b;

        assert_eq!(-2.0, c.r);
        assert_eq!(-2.0, c.g);
        assert_eq!(-2.0, c.b);
    }

    #[test]
    fn test_color_mul() {
        let a = Color::from(1.0, 2.0, 3.0);
        let b = Color::from(3.0, 4.0, 5.0);

        let c = a * b;

        assert_eq!(3.0, c.r);
        assert_eq!(8.0, c.g);
        assert_eq!(15.0, c.b);
    }
}
