use std::ops::{Add, Mul, Sub};

/// A two-dimensional vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    /// Creates a new default [Vec2].
    pub fn new() -> Self {
        Self::default()
    }
    /// Creates a new [Vec2] initialized to zero values.
    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
    /// Creates a new [Vec2] initialized from the specified values.
    pub const fn from(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    /// Creates a new [Vec2] at the origin.
    pub const fn origin() -> Self {
        Self::zero()
    }
    /// Creates a new [Vec2] that is directed along the x-axis.
    pub const fn x_axis() -> Self {
        Self { x: 1.0, y: 0.0 }
    }
    /// Creates a new [Vec2] that is directed along the negative x-axis.
    pub const fn neg_x_axis() -> Self {
        Self { x: -1.0, y: 0.0 }
    }
    /// Creates a new [Vec2] that is directed along the y-axis.
    pub const fn y_axis() -> Self {
        Self { x: 0.0, y: 1.0 }
    }
    /// Creates a new [Vec2] that is directed along the negative y-axis.
    pub const fn neg_y_axis() -> Self {
        Self { x: 0.0, y: -1.0 }
    }
    /// Creates a new [Vec2] that is initialized with the normalized values
    /// from the given [Vec2].
    pub fn from_normalized(v: &Self) -> Self {
        let mut n = *v;
        n.normalize();
        n
    }
    /// Creates a new [Vec2] that is initialized with the scaled values from
    /// the given [Vec2].
    pub fn from_scaled(v: &Self, s: f32) -> Self {
        let mut r = *v;
        r.scale(s);
        r
    }
    /// Calculates the dot product of the vector with the given [Vec2].
    pub fn dot(&self, v: &Self) -> f32 {
        (self.x * v.x) + (self.y * v.y)
    }
    /// Transforms the vector by the given scalar value.
    pub fn scale(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
    }
    /// Transforms the components of the vector to their normalized values.
    pub fn normalize(&mut self) {
        let mag = self.mag();

        self.x /= mag;
        self.y /= mag;
    }
    /// Calculates the magnitude of the vector.
    pub fn mag(&self) -> f32 {
        let n = (self.x * self.x) + (self.y * self.y);
        n.sqrt()
    }
}

impl Add for Vec2 {
    type Output = Vec2;

    /// Adds the two vectors together to produce a new one.
    fn add(self, rhs: Self) -> Self::Output {
        Vec2::from(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Vec2 {
    type Output = Vec2;

    /// Subtracts the two vectors to produce a new one.
    fn sub(self, rhs: Self) -> Self::Output {
        Vec2::from(self.x - rhs.x, self.y - rhs.y)
    }
}

/// A three-dimensional vector which represents either a point or a direction.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    /// Creates a new default [Vec3].
    pub fn new() -> Self {
        Self::default()
    }
    /// Creates a new [Vec3] initialized to zero values.
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
    /// Creates a new [Vec3] initialized with the specified values.
    pub const fn from(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    /// Creates a new [Vec3] at the origin.
    pub const fn origin() -> Self {
        Self::zero()
    }
    /// Creates a new [Vec3] that is directed along the x-axis.
    pub const fn x_axis() -> Self {
        Self {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    }
    /// Creates a new [Vec3] that is directed along the negative x-axis.
    pub const fn neg_x_axis() -> Self {
        Self {
            x: -1.0,
            y: 0.0,
            z: 0.0,
        }
    }
    /// Creates a new [Vec3] that is directed along the y-axis.
    pub const fn y_axis() -> Self {
        Self {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    }
    /// Creates a new [Vec3] that is directed along the negative y-axis.
    pub const fn neg_y_axis() -> Self {
        Self {
            x: 0.0,
            y: -1.0,
            z: 0.0,
        }
    }
    /// Creates a new [Vec3] that is directed along the z-axis.
    pub const fn z_axis() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    }
    /// Creates a new [Vec3] that is directed along the negative z-axis.
    pub const fn neg_z_axis() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: -1.0,
        }
    }
    /// Creates a new [Vec3] that is initialized with the normalized values
    /// from the given [Vec3].
    pub fn from_normalized(v: &Self) -> Self {
        let mut n = *v;
        n.normalize();
        n
    }
    /// Creates a new [Vec3] that is initialized with the scaled values from
    /// the given [Vec3].
    pub fn from_scaled(v: &Self, s: f32) -> Self {
        let mut r = *v;
        r.scale(s);
        r
    }
    /// Creates a new [Vec3] that is the cross product from the two given vectors.
    pub fn from_cross(a: &Self, b: &Self) -> Self {
        let x = a.y * b.z - a.z * b.y;
        let y = a.z * b.x - a.x * b.z;
        let z = a.x * b.y - a.y * b.x;

        Self::from(x, y, z)
    }
    /// Transfors the components of the vector to their normalized values.
    pub fn normalize(&mut self) {
        let mag = self.mag();

        self.x /= mag;
        self.y /= mag;
        self.z /= mag;
    }
    /// Transforms the vector by the given scalar value.
    pub fn scale(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
    /// Calculates the dot product of the vector with the given [Vec3].
    pub fn dot(&self, v: &Self) -> f32 {
        (self.x * v.x) + (self.y * v.y) + (self.z * v.z)
    }
    /// Transforms the vector to the result of the cross product with the given [Vec3].
    pub fn cross(&mut self, v: &Vec3) {
        let x = self.y * v.z - self.z * v.y;
        let y = self.z * v.x - self.x * v.z;
        let z = self.x * v.y - self.y * v.x;

        self.x = x;
        self.y = y;
        self.z = z;
    }
    /// Calculates the magnitude of the vector.
    pub fn mag(&self) -> f32 {
        let n = (self.x * self.x) + (self.y * self.y) + (self.z * self.z);
        n.sqrt()
    }
}

impl Add for Vec3 {
    type Output = Vec3;

    /// Adds the two vectors together to produce a new one.
    fn add(self, rhs: Self) -> Self::Output {
        Vec3::from(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    /// Subtracts the two vectors to produce a new one.
    fn sub(self, rhs: Self) -> Self::Output {
        Vec3::from(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul for Vec3 {
    type Output = Vec3;

    /// Creates a new [Vec3] that is the cross product from the two given vectors.
    fn mul(self, rhs: Self) -> Self::Output {
        Vec3::from_cross(&self, &rhs)
    }
}

/// A four-dimensional vector. When the `w` component is set to 1.0 then it represents a point.
/// When it is set to 0.0 then it represents a direction.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    /// Creates a new default [Vec4].
    pub fn new() -> Self {
        Self::default()
    }
    /// Creates a new [Vec4] initialized to zero values.
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        }
    }
    /// Creates a new [Vec4] that is initialized with the given values.
    pub const fn from(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
    /// Creates a new [Vec4] that is initialized with the normalized values
    /// from the given [Vec4].
    pub fn from_normalized(v: &Self) -> Self {
        let mut n = *v;
        n.normalize();
        n
    }
    /// Creates a new [Vec4] that is initialized with the scaled values from
    /// the given [Vec4].
    pub fn from_scaled(v: &Self, s: f32) -> Self {
        let mut r = *v;
        r.scale(s);
        r
    }
    /// Transforms the components of the vector to their normalized values.
    pub fn normalize(&mut self) {
        let mag = self.mag();

        self.x /= mag;
        self.y /= mag;
        self.z /= mag;
        self.w /= mag;
    }
    /// Transforms the vector with the given scalar.
    pub fn scale(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
        self.w *= s;
    }
    /// Calculates the dot product of the vector with the given [Vec4].
    pub fn dot(&self, v: &Self) -> f32 {
        (self.x * v.x) + (self.y * v.y) + (self.z * v.z) + (self.w * v.w)
    }
    /// Calculates the magnitude of the vector.
    pub fn mag(&self) -> f32 {
        let n = (self.x * self.x) + (self.y * self.y) + (self.z * self.z) + (self.w * self.w);
        n.sqrt()
    }
}

impl Add for Vec4 {
    type Output = Vec4;

    /// Adds the two vectors together to produce a new one.
    fn add(self, rhs: Self) -> Self::Output {
        Vec4::from(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}

impl Sub for Vec4 {
    type Output = Vec4;

    /// Subtracts the two vectors to produce a new one.
    fn sub(self, rhs: Self) -> Self::Output {
        Vec4::from(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
            self.w - rhs.w,
        )
    }
}

/// A matrix with two rows and two columns.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Matrix2 {
    data: [f32; 4],
}

impl Matrix2 {
    /// Creates a new default [Matrix2].
    pub fn new() -> Self {
        Self::default()
    }
    /// Creates a new [Matrix2] with zero values.
    pub fn zero() -> Self {
        Self::default()
    }
    /// Creates a new [Matrix2] with values set to the identity.
    pub fn identity() -> Self {
        let mut m = Self::zero();
        m.data[0] = 1.0;
        m.data[3] = 1.0;

        m
    }
    /// Creates a new [Matrix2] from the result of the addition of the two given matrices.
    pub fn add(a: &Self, b: &Self) -> Self {
        let mut m = Matrix2::zero();
        for i in 0..m.data.len() {
            m.data[i] = a.data[i] + b.data[i];
        }

        m
    }
    /// Creates a new [Matrix2] from the result of the difference of the two given matrices.
    pub fn sub(a: &Self, b: &Self) -> Self {
        let mut m = Matrix2::zero();
        for i in 0..m.data.len() {
            m.data[i] = a.data[i] - b.data[i];
        }

        m
    }
    /// Gets a value from the matrix by row and column. The indices passed must be zero-based.
    pub fn get(&self, r: usize, c: usize) -> f32 {
        assert!(r < 2);
        assert!(c < 2);

        self.data[(r * 2) + c]
    }
    /// Sets a value in the matrix by row and column. The indices passed must be zero-based.
    pub fn set(&mut self, r: usize, c: usize, val: f32) {
        assert!(r < 2);
        assert!(c < 2);

        self.data[(r * 2) + c] = val;
    }
    /// Calculates the determinant from the matrix.
    pub fn det(&self) -> f32 {
        self.data[0] * self.data[3] - self.data[1] * self.data[2]
    }
}

impl Add for Matrix2 {
    type Output = Matrix2;

    /// Adds the two matrices together to produce a new one.
    fn add(self, rhs: Self) -> Self::Output {
        Matrix2::add(&self, &rhs)
    }
}

impl Sub for Matrix2 {
    type Output = Matrix2;

    /// Subtracts the two matrices to produce a new one.
    fn sub(self, rhs: Self) -> Self::Output {
        Matrix2::sub(&self, &rhs)
    }
}

/// A matrix with three rows and three columns.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Matrix3 {
    data: [f32; 9],
}

impl Matrix3 {
    /// Creates a new [Matrix3] with zero values.
    pub fn new() -> Self {
        Self::default()
    }
    /// Creates a new [Matrix3] with zero values.
    pub fn zero() -> Self {
        Self::default()
    }
    /// Creates a new [Matrix3] with values set to the identity.
    pub fn identity() -> Self {
        let mut m = Self::zero();
        m.data[0] = 1.0;
        m.data[4] = 1.0;
        m.data[8] = 1.0;

        m
    }
    /// Creates a new [Matrix3] that is the transpose of the given matrix.
    pub fn from_transpose(m: &Self) -> Self {
        let mut t = Self::zero();

        t.data[0] = m.data[0];
        t.data[1] = m.data[3];
        t.data[2] = m.data[6];

        t.data[3] = m.data[1];
        t.data[4] = m.data[4];
        t.data[5] = m.data[7];

        t.data[6] = m.data[2];
        t.data[7] = m.data[5];
        t.data[8] = m.data[8];

        t
    }
    /// Returns the inverse of the given [Matrix3] if the matrix is invertible.
    pub fn from_inverse(m: &Self) -> Option<Self> {
        let d = m.det();
        if d == 0.0 {
            return None;
        }

        let mut inv = Self::zero();

        for i in 0..3 {
            for j in 0..3 {
                let c = m.cofactor(i, j);

                inv.set(j, i, c / d);
            }
        }

        Some(inv)
    }
    /// Creates a new [Matrix3] from the result of the addition of the two matrices.
    pub fn add(a: &Self, b: &Self) -> Self {
        let mut m = Matrix3::zero();
        for i in 0..m.data.len() {
            m.data[i] = a.data[i] + b.data[i];
        }

        m
    }
    /// Creates a new [Matrix3] from the result of the difference of the two matrices.
    pub fn sub(a: &Self, b: &Self) -> Self {
        let mut m = Matrix3::zero();
        for i in 0..m.data.len() {
            m.data[i] = a.data[i] - b.data[i];
        }

        m
    }
    /// Multiplies a [Matrix3] by the given [Vec3] and returns the result.
    pub fn mul_vec3(m: &Self, v: &Vec3) -> Vec3 {
        let x = (m.data[0] * v.x) + (m.data[1] * v.y) + (m.data[2] * v.z);
        let y = (m.data[3] * v.x) + (m.data[4] * v.y) + (m.data[5] * v.z);
        let z = (m.data[6] * v.x) + (m.data[7] * v.y) + (m.data[8] * v.z);

        Vec3::from(x, y, z)
    }
    /// Gets a value from the matrix by row and column. The indices passed must be zero-based.
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[(r * 3) + c]
    }
    /// Sets a value on the matrix by row and column. The indices passed must be zero-based.
    pub fn set(&mut self, r: usize, c: usize, val: f32) {
        self.data[(r * 3) + c] = val;
    }
    /// Returns the sub-matrix created from the [Matrix3] by removing the specified zero-based
    /// row and column.
    #[allow(clippy::collapsible_else_if)]
    pub fn submat(&self, xr: usize, xc: usize) -> Matrix2 {
        let mut sub = Matrix2::zero();

        // brute force for now but should look for better algorithm based on xr and xc
        if xr == 0 {
            if xc == 0 {
                sub.data[0] = self.data[4];
                sub.data[1] = self.data[5];

                sub.data[2] = self.data[7];
                sub.data[3] = self.data[8];
            } else if xc == 1 {
                sub.data[0] = self.data[3];
                sub.data[1] = self.data[5];

                sub.data[2] = self.data[6];
                sub.data[3] = self.data[8];
            } else {
                sub.data[0] = self.data[3];
                sub.data[1] = self.data[4];

                sub.data[2] = self.data[6];
                sub.data[3] = self.data[7];
            }
        } else if xr == 1 {
            if xc == 0 {
                sub.data[0] = self.data[1];
                sub.data[1] = self.data[2];

                sub.data[2] = self.data[7];
                sub.data[3] = self.data[8];
            } else if xc == 1 {
                sub.data[0] = self.data[0];
                sub.data[1] = self.data[2];

                sub.data[2] = self.data[6];
                sub.data[3] = self.data[8];
            } else {
                sub.data[0] = self.data[0];
                sub.data[1] = self.data[1];

                sub.data[2] = self.data[6];
                sub.data[3] = self.data[7];
            }
        } else {
            if xc == 0 {
                sub.data[0] = self.data[1];
                sub.data[1] = self.data[2];

                sub.data[2] = self.data[4];
                sub.data[3] = self.data[5];
            } else if xc == 1 {
                sub.data[0] = self.data[0];
                sub.data[1] = self.data[2];

                sub.data[2] = self.data[3];
                sub.data[3] = self.data[5];
            } else {
                sub.data[0] = self.data[0];
                sub.data[1] = self.data[1];

                sub.data[2] = self.data[3];
                sub.data[3] = self.data[4];
            }
        }

        sub
    }
    /// Calculates the minor for the matrix at the given zero-based row and column.
    pub fn minor(&self, r: usize, c: usize) -> f32 {
        self.submat(r, c).det()
    }
    /// Calculates the cofactor for the matrix at the given zero-based row and column.
    pub fn cofactor(&self, r: usize, c: usize) -> f32 {
        let mut minor = self.minor(r, c);
        if (r + c) % 2 != 0 {
            minor *= -1.0;
        }

        minor
    }
    /// Calculates the determinant of the matrix.
    pub fn det(&self) -> f32 {
        let mut d = 0.0;
        for i in 0..3 {
            d += self.data[i] * self.cofactor(0, i);
        }

        d
    }
}

impl Add for Matrix3 {
    type Output = Matrix3;

    /// Adds the two matrices together to produce a new one.
    fn add(self, rhs: Self) -> Self::Output {
        Matrix3::add(&self, &rhs)
    }
}

impl Sub for Matrix3 {
    type Output = Matrix3;

    /// Subtracts the two matrices to produce a new one.
    fn sub(self, rhs: Self) -> Self::Output {
        Matrix3::sub(&self, &rhs)
    }
}

/// A matrix with four rows and four columns.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Matrix4 {
    data: [f32; 16],
}

impl Matrix4 {
    /// Creates a new default [Matrix4].
    pub fn new() -> Self {
        Self::default()
    }
    /// Creates a new [Matrix4] with zero values.
    pub fn zero() -> Self {
        Self::default()
    }
    /// Creates a new [Matrix4] with values set to the identity.
    pub fn identity() -> Self {
        let mut m = Self::zero();

        m.data[0] = 1.0;
        m.data[5] = 1.0;
        m.data[10] = 1.0;
        m.data[15] = 1.0;

        m
    }
    /// Creates a new [Matrix4] that is the transpose of the given matrix.
    pub fn from_transpose(m: &Self) -> Self {
        let mut t = Self::zero();

        t.data[0] = m.data[0];
        t.data[1] = m.data[4];
        t.data[2] = m.data[8];
        t.data[3] = m.data[12];

        t.data[4] = m.data[1];
        t.data[5] = m.data[5];
        t.data[6] = m.data[9];
        t.data[7] = m.data[13];

        t.data[8] = m.data[2];
        t.data[9] = m.data[6];
        t.data[10] = m.data[10];
        t.data[11] = m.data[14];

        t.data[12] = m.data[3];
        t.data[13] = m.data[7];
        t.data[14] = m.data[11];
        t.data[15] = m.data[15];

        t
    }
    /// Creates a new [Matrix4] by scaling all of the elements from the existing matrix by the
    /// given scalar value.
    pub fn from_scaled(m: &Self, s: f32) -> Self {
        let mut r = *m;
        r.data.iter_mut().for_each(|v| *v *= s);
        r
    }
    /// Creates a new [Matrix4] that is the inverse of the given matrix if the matrix is
    /// invertible.
    pub fn from_inverse(m: &Self) -> Option<Self> {
        let d = Matrix4::det(m);
        if d == 0.0 {
            return None;
        }

        let mut inv = Self::zero();

        for i in 0..4 {
            for j in 0..4 {
                let c = Self::cofactor(m, i, j);

                inv.set(j, i, c / d);
            }
        }

        Some(inv)
    }
    /// Creates a new [Matrix4] that represents a view transformation.
    pub fn view_transform(from: &Vec3, to: &Vec3, up: &Vec3) -> Self {
        let mut forward = *to - *from;
        forward.normalize();

        let upn = Vec3::from_normalized(up);
        let left = Vec3::from_cross(&forward, &upn);
        let true_up = Vec3::from_cross(&left, &forward);

        let mut orientation = Self::identity();
        orientation.set(0, 0, left.x);
        orientation.set(0, 1, left.y);
        orientation.set(0, 2, left.z);

        orientation.set(1, 0, true_up.x);
        orientation.set(1, 1, true_up.y);
        orientation.set(1, 2, true_up.z);

        orientation.set(2, 0, -forward.x);
        orientation.set(2, 1, -forward.y);
        orientation.set(2, 2, -forward.z);

        let translate = Self::op_translate(-from.x, -from.y, -from.z);

        orientation * translate
    }
    /// Creates a new [Matrix4] that represents a translation transformation.
    pub fn op_translate(x: f32, y: f32, z: f32) -> Self {
        let mut m = Self::identity();
        m.data[3] = x;
        m.data[7] = y;
        m.data[11] = z;

        m
    }
    /// Creates a new [Matrix4] that represents a translation transformation.
    pub fn op_translate_vec(v: &Vec3) -> Self {
        Self::op_translate(v.x, v.y, v.z)
    }
    /// Creates a new [Matrix4] that represents a scaling transformation.
    pub fn op_scale(x: f32, y: f32, z: f32) -> Self {
        let mut m = Self::identity();
        m.data[0] = x;
        m.data[5] = y;
        m.data[10] = z;

        m
    }
    /// Creates a new [Matrix4] that represents a scaling transformation.
    pub fn op_scale_vec(v: &Vec3) -> Self {
        Self::op_scale(v.x, v.y, v.z)
    }
    /// Creates a new [Matrix4] that represents a rotation transformation around the x-axis.
    pub fn op_rotate_x(rad: f32) -> Self {
        let s = rad.sin();
        let c = rad.cos();

        let mut r = Self::identity();

        r.set(1, 1, c);
        r.set(1, 2, -s);
        r.set(2, 1, s);
        r.set(2, 2, c);

        r
    }
    /// Creates a new [Matrix4] that represents a rotation transformation around the y-axis.
    pub fn op_rotate_y(rad: f32) -> Self {
        let s = rad.sin();
        let c = rad.cos();

        let mut r = Self::identity();

        r.set(0, 0, c);
        r.set(0, 2, s);
        r.set(2, 0, -s);
        r.set(2, 2, c);

        r
    }
    /// Creates a new [Matrix4] that represents a rotation transformation around the z-axis.
    pub fn op_rotate_z(rad: f32) -> Self {
        let s = rad.sin();
        let c = rad.cos();

        let mut r = Self::identity();

        r.set(0, 0, c);
        r.set(0, 1, -s);
        r.set(1, 0, s);
        r.set(1, 1, c);

        r
    }
    /// Creates a new [Matrix4] that represents a shear transformation.
    pub fn op_shear(xy: f32, xz: f32, yx: f32, yz: f32, zx: f32, zy: f32) -> Self {
        let mut s = Self::identity();

        s.data[1] = xy;
        s.data[2] = xz;

        s.data[4] = yx;
        s.data[6] = yz;

        s.data[8] = zx;
        s.data[9] = zy;

        s
    }
    /// Gets a value from the matrix by row and column. The indices passed must be zero-based.
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[(r * 4) + c]
    }
    /// Sets a value in the matrix by row and column. The indices passed must be zero-based.
    pub fn set(&mut self, r: usize, c: usize, val: f32) {
        self.data[(r * 4) + c] = val;
    }
    /// Creates a new [Matrix4] that is the result of the addition of the matrices.
    pub fn add(a: &Self, b: &Self) -> Self {
        let mut c = Self::zero();

        for i in 0..a.data.len() {
            c.data[i] = a.data[i] + b.data[i];
        }

        c
    }
    /// Creates a new [Matrix4] that is the result of the subtraction of the matrices.
    pub fn sub(a: &Self, b: &Self) -> Self {
        let mut c = Self::zero();

        for i in 0..a.data.len() {
            c.data[i] = a.data[i] - b.data[i];
        }

        c
    }
    /// Creates a new [Matrix4] that is the result of the multiplication of the matrices.
    pub fn mul(a: &Self, b: &Self) -> Self {
        let mut c = Self::zero();

        let mut i = 0;
        while i < 16 {
            let mut j = 0;
            while j < 4 {
                c.data[i + j] = (a.data[i] * b.data[j])
                    + (a.data[i + 1] * b.data[j + 4])
                    + (a.data[i + 2] * b.data[j + 8])
                    + (a.data[i + 3] * b.data[j + 12]);

                j += 1;
            }

            i += 4;
        }

        c
    }
    /// Returns the result of the multiplication of the matrix by the given [Vec3] using the w
    /// value for the fourth component of the vector.
    pub fn mul_vec3(m: &Self, v: &Vec3, w: f32) -> Vec3 {
        let r = Self::mul_vec4(m, &Vec4::from(v.x, v.y, v.z, w));
        Vec3::from(r.x, r.y, r.z)
    }
    /// Returns the result of the multiplication of the matrix by the given point [Vec3].
    pub fn mul_vec3_point(m: &Self, v: &Vec3) -> Vec3 {
        Self::mul_vec3(m, v, 1.0)
    }
    /// Returns the result of the multiplication of the matrix by the given directional [Vec3].
    pub fn mul_vec3_dir(m: &Self, v: &Vec3) -> Vec3 {
        Self::mul_vec3(m, v, 0.0)
    }
    /// Returns the result of the multiplication of the matrix by the given [Vec4].
    pub fn mul_vec4(m: &Self, v: &Vec4) -> Vec4 {
        let x = (m.data[0] * v.x) + (m.data[1] * v.y) + (m.data[2] * v.z) + (m.data[3] * v.w);
        let y = (m.data[4] * v.x) + (m.data[5] * v.y) + (m.data[6] * v.z) + (m.data[7] * v.w);
        let z = (m.data[8] * v.x) + (m.data[9] * v.y) + (m.data[10] * v.z) + (m.data[11] * v.w);
        let w = (m.data[12] * v.x) + (m.data[13] * v.y) + (m.data[14] * v.z) + (m.data[15] * v.w);

        Vec4::from(x, y, z, w)
    }
    /// Returns the sub-matrix of the matrix by removing the specified zero-based column and row.
    #[allow(clippy::collapsible_else_if)]
    pub fn submat(&self, xr: usize, xc: usize) -> Matrix3 {
        let mut sub = Matrix3::zero();

        // brute force for now but should look for better algorithm based on xr and xc
        if xr == 0 {
            if xc == 0 {
                sub.data[0] = self.data[5];
                sub.data[1] = self.data[6];
                sub.data[2] = self.data[7];

                sub.data[3] = self.data[9];
                sub.data[4] = self.data[10];
                sub.data[5] = self.data[11];

                sub.data[6] = self.data[13];
                sub.data[7] = self.data[14];
                sub.data[8] = self.data[15];
            } else if xc == 1 {
                sub.data[0] = self.data[4];
                sub.data[1] = self.data[6];
                sub.data[2] = self.data[7];

                sub.data[3] = self.data[8];
                sub.data[4] = self.data[10];
                sub.data[5] = self.data[11];

                sub.data[6] = self.data[12];
                sub.data[7] = self.data[14];
                sub.data[8] = self.data[15];
            } else if xc == 2 {
                sub.data[0] = self.data[4];
                sub.data[1] = self.data[5];
                sub.data[2] = self.data[7];

                sub.data[3] = self.data[8];
                sub.data[4] = self.data[9];
                sub.data[5] = self.data[11];

                sub.data[6] = self.data[12];
                sub.data[7] = self.data[13];
                sub.data[8] = self.data[15];
            } else {
                sub.data[0] = self.data[4];
                sub.data[1] = self.data[5];
                sub.data[2] = self.data[6];

                sub.data[3] = self.data[8];
                sub.data[4] = self.data[9];
                sub.data[5] = self.data[10];

                sub.data[6] = self.data[12];
                sub.data[7] = self.data[13];
                sub.data[8] = self.data[14];
            }
        } else if xr == 1 {
            if xc == 0 {
                sub.data[0] = self.data[1];
                sub.data[1] = self.data[2];
                sub.data[2] = self.data[3];

                sub.data[3] = self.data[9];
                sub.data[4] = self.data[10];
                sub.data[5] = self.data[11];

                sub.data[6] = self.data[13];
                sub.data[7] = self.data[14];
                sub.data[8] = self.data[15];
            } else if xc == 1 {
                sub.data[0] = self.data[0];
                sub.data[1] = self.data[2];
                sub.data[2] = self.data[3];

                sub.data[3] = self.data[8];
                sub.data[4] = self.data[10];
                sub.data[5] = self.data[11];

                sub.data[6] = self.data[12];
                sub.data[7] = self.data[14];
                sub.data[8] = self.data[15];
            } else if xc == 2 {
                sub.data[0] = self.data[0];
                sub.data[1] = self.data[1];
                sub.data[2] = self.data[3];

                sub.data[3] = self.data[8];
                sub.data[4] = self.data[9];
                sub.data[5] = self.data[11];

                sub.data[6] = self.data[12];
                sub.data[7] = self.data[13];
                sub.data[8] = self.data[15];
            } else {
                sub.data[0] = self.data[0];
                sub.data[1] = self.data[1];
                sub.data[2] = self.data[2];

                sub.data[3] = self.data[8];
                sub.data[4] = self.data[9];
                sub.data[5] = self.data[10];

                sub.data[6] = self.data[12];
                sub.data[7] = self.data[13];
                sub.data[8] = self.data[14];
            }
        } else if xr == 2 {
            if xc == 0 {
                sub.data[0] = self.data[1];
                sub.data[1] = self.data[2];
                sub.data[2] = self.data[3];

                sub.data[3] = self.data[5];
                sub.data[4] = self.data[6];
                sub.data[5] = self.data[7];

                sub.data[6] = self.data[13];
                sub.data[7] = self.data[14];
                sub.data[8] = self.data[15];
            } else if xc == 1 {
                sub.data[0] = self.data[0];
                sub.data[1] = self.data[2];
                sub.data[2] = self.data[3];

                sub.data[3] = self.data[4];
                sub.data[4] = self.data[6];
                sub.data[5] = self.data[7];

                sub.data[6] = self.data[12];
                sub.data[7] = self.data[14];
                sub.data[8] = self.data[15];
            } else if xc == 2 {
                sub.data[0] = self.data[0];
                sub.data[1] = self.data[1];
                sub.data[2] = self.data[3];

                sub.data[3] = self.data[4];
                sub.data[4] = self.data[5];
                sub.data[5] = self.data[7];

                sub.data[6] = self.data[12];
                sub.data[7] = self.data[13];
                sub.data[8] = self.data[15];
            } else {
                sub.data[0] = self.data[0];
                sub.data[1] = self.data[1];
                sub.data[2] = self.data[2];

                sub.data[3] = self.data[4];
                sub.data[4] = self.data[5];
                sub.data[5] = self.data[6];

                sub.data[6] = self.data[12];
                sub.data[7] = self.data[13];
                sub.data[8] = self.data[14];
            }
        } else {
            if xc == 0 {
                sub.data[0] = self.data[1];
                sub.data[1] = self.data[2];
                sub.data[2] = self.data[3];

                sub.data[3] = self.data[5];
                sub.data[4] = self.data[6];
                sub.data[5] = self.data[7];

                sub.data[6] = self.data[9];
                sub.data[7] = self.data[10];
                sub.data[8] = self.data[11];
            } else if xc == 1 {
                sub.data[0] = self.data[0];
                sub.data[1] = self.data[2];
                sub.data[2] = self.data[3];

                sub.data[3] = self.data[4];
                sub.data[4] = self.data[6];
                sub.data[5] = self.data[7];

                sub.data[6] = self.data[8];
                sub.data[7] = self.data[10];
                sub.data[8] = self.data[11];
            } else if xc == 2 {
                sub.data[0] = self.data[0];
                sub.data[1] = self.data[1];
                sub.data[2] = self.data[3];

                sub.data[3] = self.data[4];
                sub.data[4] = self.data[5];
                sub.data[5] = self.data[7];

                sub.data[6] = self.data[8];
                sub.data[7] = self.data[9];
                sub.data[8] = self.data[11];
            } else {
                sub.data[0] = self.data[0];
                sub.data[1] = self.data[1];
                sub.data[2] = self.data[2];

                sub.data[3] = self.data[4];
                sub.data[4] = self.data[5];
                sub.data[5] = self.data[6];

                sub.data[6] = self.data[8];
                sub.data[7] = self.data[9];
                sub.data[8] = self.data[10];
            }
        }

        sub
    }
    /// Calculates the minor of the matrix at the given row and column.
    pub fn minor(&self, r: usize, c: usize) -> f32 {
        self.submat(r, c).det()
    }
    /// Calculates the cofactor of the matrix at the given row and column.
    pub fn cofactor(&self, r: usize, c: usize) -> f32 {
        let mut minor = self.minor(r, c);
        if (r + c) % 2 != 0 {
            minor *= -1.0;
        }

        minor
    }
    /// Calculates the determinant of the matrix.
    pub fn det(&self) -> f32 {
        let mut d = 0.0;
        for i in 0..4 {
            d += self.data[i] * self.cofactor(0, i);
        }

        d
    }
}

impl Add for Matrix4 {
    type Output = Matrix4;

    /// Adds the two matrices together to produce a new one.
    fn add(self, rhs: Self) -> Self::Output {
        Matrix4::add(&self, &rhs)
    }
}

impl Sub for Matrix4 {
    type Output = Matrix4;

    /// Subtracts the two matrices together to produce a new one.
    fn sub(self, rhs: Self) -> Self::Output {
        Matrix4::sub(&self, &rhs)
    }
}

impl Mul for Matrix4 {
    type Output = Matrix4;

    /// Multiplies the two matrices together to produce a new one.
    fn mul(self, rhs: Self) -> Self::Output {
        Matrix4::mul(&self, &rhs)
    }
}

/// Ensures that the given value x is clamped between a minimum and maximum value. If x is less
/// than min then the min is returned. If x is larger then max then max is returned. Otherwise x is
/// returned.
pub fn clamp<T: PartialOrd>(x: T, min: T, max: T) -> T {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

/// A ray with an origin and a direction.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    /// Creates a new default [Ray].
    pub fn new() -> Self {
        Self::default()
    }
    /// Creates a new [Ray] initialized to zero values.
    pub fn zero() -> Self {
        Self::default()
    }
    /// Creates a new [Ray] directed along the x-axis.
    pub fn x_axis() -> Self {
        Ray::from(Vec3::zero(), Vec3::x_axis())
    }
    /// Creates a new [Ray] directed along the negative x-axis.
    pub fn neg_x_axis() -> Self {
        Ray::from(Vec3::zero(), Vec3::neg_x_axis())
    }
    /// Creates a new [Ray] directed along the y-axis.
    pub fn y_axis() -> Self {
        Ray::from(Vec3::zero(), Vec3::y_axis())
    }
    /// Creates a new [Ray] directed along the negative y-axis.
    pub fn neg_y_axis() -> Self {
        Ray::from(Vec3::zero(), Vec3::neg_y_axis())
    }
    /// Creates a new [Ray] directed along the z-axis.
    pub fn z_axis() -> Self {
        Ray::from(Vec3::zero(), Vec3::z_axis())
    }
    /// Creates a new [Ray] directed along the negative z-axis.
    pub fn neg_z_axis() -> Self {
        Ray::from(Vec3::zero(), Vec3::neg_z_axis())
    }
    /// Creates a new [Ray] from the given the origin and direction.
    pub fn from(origin: Vec3, direction: Vec3) -> Self {
        Ray { origin, direction }
    }
    /// Returns a [Vec3] that represents a position on the given [Ray] at t.
    pub fn position(r: &Self, t: f32) -> Vec3 {
        let v = Vec3::from_scaled(&r.direction, t);
        r.origin + v
    }
    /// Creates a new [Ray] from the transformation of a given ray by a transformation matrix.
    pub fn from_transformed(r: &Self, m: &Matrix4) -> Self {
        let origin = Matrix4::mul_vec3_point(m, &r.origin);
        let direction = Matrix4::mul_vec3_dir(m, &r.direction);

        Ray { origin, direction }
    }
    /// Transforms the ray with the given transformation matrix.
    pub fn transform(&mut self, m: &Matrix4) {
        let origin = Matrix4::mul_vec3_point(m, &self.origin);
        let direction = Matrix4::mul_vec3_dir(m, &self.direction);

        self.origin = origin;
        self.direction = direction;
    }
}

#[cfg(test)]
mod tests {
    use crate::EPSILON;

    use super::{Matrix2, Matrix3, Matrix4, Ray, Vec3};

    use approx::assert_relative_eq;
    use std::f32::consts::PI;

    #[test]
    fn test_matrix2_det() {
        let mut m = Matrix2::zero();
        m.set(0, 0, 1.0);
        m.set(0, 1, 5.0);
        m.set(1, 0, -3.0);
        m.set(1, 1, 2.0);

        assert_eq!(17.0, m.det());
    }

    #[test]
    fn test_matrix3_submat() {
        let mut m = Matrix3::zero();
        m.set(0, 0, 1.0);
        m.set(0, 1, 5.0);
        m.set(0, 2, 0.0);

        m.set(1, 0, -3.0);
        m.set(1, 1, 2.0);
        m.set(1, 2, 7.0);

        m.set(2, 0, 0.0);
        m.set(2, 1, 6.0);
        m.set(2, 2, -3.0);

        let sub = Matrix3::submat(&m, 0, 2);

        assert_eq!(-3.0, sub.get(0, 0));
        assert_eq!(2.0, sub.get(0, 1));

        assert_eq!(0.0, sub.get(1, 0));
        assert_eq!(6.0, sub.get(1, 1));
    }

    #[test]
    fn test_matrix3_minor() {
        let mut m = Matrix3::zero();
        m.set(0, 0, 3.0);
        m.set(0, 1, 5.0);
        m.set(0, 2, 0.0);

        m.set(1, 0, 2.0);
        m.set(1, 1, -1.0);
        m.set(1, 2, -7.0);

        m.set(2, 0, 6.0);
        m.set(2, 1, -1.0);
        m.set(2, 2, 5.0);

        let sub = Matrix3::submat(&m, 1, 0);

        assert_eq!(25.0, sub.det());
        assert_eq!(25.0, Matrix3::minor(&m, 1, 0));
    }

    #[test]
    fn test_matrix3_cofactor() {
        let mut m = Matrix3::zero();
        m.set(0, 0, 3.0);
        m.set(0, 1, 5.0);
        m.set(0, 2, 0.0);

        m.set(1, 0, 2.0);
        m.set(1, 1, -1.0);
        m.set(1, 2, -7.0);

        m.set(2, 0, 6.0);
        m.set(2, 1, -1.0);
        m.set(2, 2, 5.0);

        assert_eq!(-12.0, Matrix3::minor(&m, 0, 0));
        assert_eq!(-12.0, Matrix3::cofactor(&m, 0, 0));

        assert_eq!(25.0, Matrix3::minor(&m, 1, 0));
        assert_eq!(-25.0, Matrix3::cofactor(&m, 1, 0));
    }

    #[test]
    fn test_matrix3_det() {
        let mut m = Matrix3::zero();
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(0, 2, 6.0);

        m.set(1, 0, -5.0);
        m.set(1, 1, 8.0);
        m.set(1, 2, -4.0);

        m.set(2, 0, 2.0);
        m.set(2, 1, 6.0);
        m.set(2, 2, 4.0);

        assert_eq!(56.0, Matrix3::cofactor(&m, 0, 0));
        assert_eq!(12.0, Matrix3::cofactor(&m, 0, 1));
        assert_eq!(-46.0, Matrix3::cofactor(&m, 0, 2));
        assert_eq!(-196.0, Matrix3::det(&m));
    }

    #[test]
    fn test_matrix4_det() {
        let mut m = Matrix4::zero();
        m.set(0, 0, -2.0);
        m.set(0, 1, -8.0);
        m.set(0, 2, 3.0);
        m.set(0, 3, 5.0);

        m.set(1, 0, -3.0);
        m.set(1, 1, 1.0);
        m.set(1, 2, 7.0);
        m.set(1, 3, 3.0);

        m.set(2, 0, 1.0);
        m.set(2, 1, 2.0);
        m.set(2, 2, -9.0);
        m.set(2, 3, 6.0);

        m.set(3, 0, -6.0);
        m.set(3, 1, 7.0);
        m.set(3, 2, 7.0);
        m.set(3, 3, -9.0);

        assert_eq!(690.0, Matrix4::cofactor(&m, 0, 0));
        assert_eq!(447.0, Matrix4::cofactor(&m, 0, 1));
        assert_eq!(210.0, Matrix4::cofactor(&m, 0, 2));
        assert_eq!(51.0, Matrix4::cofactor(&m, 0, 3));
        assert_eq!(-4071.0, Matrix4::det(&m));
    }

    #[test]
    fn test_matrix4_inverse() {
        {
            let mut m = Matrix4::zero();
            m.set(0, 0, 6.0);
            m.set(0, 1, 4.0);
            m.set(0, 2, 4.0);
            m.set(0, 3, 4.0);

            m.set(1, 0, 5.0);
            m.set(1, 1, 5.0);
            m.set(1, 2, 7.0);
            m.set(1, 3, 6.0);

            m.set(2, 0, 4.0);
            m.set(2, 1, -9.0);
            m.set(2, 2, 3.0);
            m.set(2, 3, -7.0);

            m.set(3, 0, 9.0);
            m.set(3, 1, 1.0);
            m.set(3, 2, 7.0);
            m.set(3, 3, -6.0);

            assert_eq!(-2120.0, Matrix4::det(&m));

            let i = Matrix4::from_inverse(&m);
            assert!(i.is_some());
        }

        {
            let mut m = Matrix4::zero();
            m.set(0, 0, -4.0);
            m.set(0, 1, 2.0);
            m.set(0, 2, -2.0);
            m.set(0, 3, -3.0);

            m.set(1, 0, 9.0);
            m.set(1, 1, 6.0);
            m.set(1, 2, 2.0);
            m.set(1, 3, 6.0);

            m.set(2, 0, 0.0);
            m.set(2, 1, -5.0);
            m.set(2, 2, 1.0);
            m.set(2, 3, -5.0);

            m.set(3, 0, 0.0);
            m.set(3, 1, 0.0);
            m.set(3, 2, 0.0);
            m.set(3, 3, 0.0);

            assert_eq!(0.0, Matrix4::det(&m));

            let i = Matrix4::from_inverse(&m);
            assert!(i.is_none());
        }

        {
            let mut m = Matrix4::zero();
            m.set(0, 0, -5.0);
            m.set(0, 1, 2.0);
            m.set(0, 2, 6.0);
            m.set(0, 3, -8.0);

            m.set(1, 0, 1.0);
            m.set(1, 1, -5.0);
            m.set(1, 2, 1.0);
            m.set(1, 3, 8.0);

            m.set(2, 0, 7.0);
            m.set(2, 1, 7.0);
            m.set(2, 2, -6.0);
            m.set(2, 3, -7.0);

            m.set(3, 0, 1.0);
            m.set(3, 1, -3.0);
            m.set(3, 2, 7.0);
            m.set(3, 3, 4.0);

            assert_eq!(532.0, Matrix4::det(&m));
            assert_eq!(-160.0, Matrix4::cofactor(&m, 2, 3));
            assert_eq!(105.0, Matrix4::cofactor(&m, 3, 2));

            let inv = Matrix4::from_inverse(&m).expect("det > 0");

            assert_relative_eq!(-160.0 / 532.0, inv.get(3, 2));
            assert_relative_eq!(105.0 / 532.0, inv.get(2, 3));

            assert_relative_eq!(0.21805, inv.get(0, 0), epsilon = EPSILON);
            assert_relative_eq!(0.45113, inv.get(0, 1), epsilon = EPSILON);
            assert_relative_eq!(0.24060, inv.get(0, 2), epsilon = EPSILON);
            assert_relative_eq!(-0.04511, inv.get(0, 3), epsilon = EPSILON);

            assert_relative_eq!(-0.80827, inv.get(1, 0), epsilon = EPSILON);
            assert_relative_eq!(-1.45677, inv.get(1, 1), epsilon = EPSILON);
            assert_relative_eq!(-0.44361, inv.get(1, 2), epsilon = EPSILON);
            assert_relative_eq!(0.52068, inv.get(1, 3), epsilon = EPSILON);

            assert_relative_eq!(-0.07895, inv.get(2, 0), epsilon = EPSILON);
            assert_relative_eq!(-0.22368, inv.get(2, 1), epsilon = EPSILON);
            assert_relative_eq!(-0.05263, inv.get(2, 2), epsilon = EPSILON);
            assert_relative_eq!(0.19737, inv.get(2, 3), epsilon = EPSILON);

            assert_relative_eq!(-0.52256, inv.get(3, 0), epsilon = EPSILON);
            assert_relative_eq!(-0.81391, inv.get(3, 1), epsilon = EPSILON);
            assert_relative_eq!(-0.30075, inv.get(3, 2), epsilon = EPSILON);
            assert_relative_eq!(0.30639, inv.get(3, 3), epsilon = EPSILON);
        }

        {
            let mut m = Matrix4::zero();
            m.set(0, 0, 8.0);
            m.set(0, 1, -5.0);
            m.set(0, 2, 9.0);
            m.set(0, 3, 2.0);

            m.set(1, 0, 7.0);
            m.set(1, 1, 5.0);
            m.set(1, 2, 6.0);
            m.set(1, 3, 1.0);

            m.set(2, 0, -6.0);
            m.set(2, 1, 0.0);
            m.set(2, 2, 9.0);
            m.set(2, 3, 6.0);

            m.set(3, 0, -3.0);
            m.set(3, 1, 0.0);
            m.set(3, 2, -9.0);
            m.set(3, 3, -4.0);

            let inv = Matrix4::from_inverse(&m).expect("det > 0");

            assert_relative_eq!(-0.15385, inv.get(0, 0), epsilon = EPSILON);
            assert_relative_eq!(-0.15385, inv.get(0, 1), epsilon = EPSILON);
            assert_relative_eq!(-0.28205, inv.get(0, 2), epsilon = EPSILON);
            assert_relative_eq!(-0.53846, inv.get(0, 3), epsilon = EPSILON);

            assert_relative_eq!(-0.07692, inv.get(1, 0), epsilon = EPSILON);
            assert_relative_eq!(0.12308, inv.get(1, 1), epsilon = EPSILON);
            assert_relative_eq!(0.02564, inv.get(1, 2), epsilon = EPSILON);
            assert_relative_eq!(0.03077, inv.get(1, 3), epsilon = EPSILON);

            assert_relative_eq!(0.35897, inv.get(2, 0), epsilon = EPSILON);
            assert_relative_eq!(0.35897, inv.get(2, 1), epsilon = EPSILON);
            assert_relative_eq!(0.43590, inv.get(2, 2), epsilon = EPSILON);
            assert_relative_eq!(0.92308, inv.get(2, 3), epsilon = EPSILON);

            assert_relative_eq!(-0.69231, inv.get(3, 0), epsilon = EPSILON);
            assert_relative_eq!(-0.69231, inv.get(3, 1), epsilon = EPSILON);
            assert_relative_eq!(-0.76923, inv.get(3, 2), epsilon = EPSILON);
            assert_relative_eq!(-1.92308, inv.get(3, 3), epsilon = EPSILON);
        }

        {
            let mut m = Matrix4::zero();
            m.set(0, 0, 9.0);
            m.set(0, 1, 3.0);
            m.set(0, 2, 0.0);
            m.set(0, 3, 9.0);

            m.set(1, 0, -5.0);
            m.set(1, 1, -2.0);
            m.set(1, 2, -6.0);
            m.set(1, 3, -3.0);

            m.set(2, 0, -4.0);
            m.set(2, 1, 9.0);
            m.set(2, 2, 6.0);
            m.set(2, 3, 4.0);

            m.set(3, 0, -7.0);
            m.set(3, 1, 6.0);
            m.set(3, 2, 6.0);
            m.set(3, 3, 2.0);

            let inv = Matrix4::from_inverse(&m).expect("det > 0");

            assert_relative_eq!(-0.04074, inv.get(0, 0), epsilon = EPSILON);
            assert_relative_eq!(-0.07778, inv.get(0, 1), epsilon = EPSILON);
            assert_relative_eq!(0.14444, inv.get(0, 2), epsilon = EPSILON);
            assert_relative_eq!(-0.22222, inv.get(0, 3), epsilon = EPSILON);

            assert_relative_eq!(-0.07778, inv.get(1, 0), epsilon = EPSILON);
            assert_relative_eq!(0.03333, inv.get(1, 1), epsilon = EPSILON);
            assert_relative_eq!(0.36667, inv.get(1, 2), epsilon = EPSILON);
            assert_relative_eq!(-0.33333, inv.get(1, 3), epsilon = EPSILON);

            assert_relative_eq!(-0.02901, inv.get(2, 0), epsilon = EPSILON);
            assert_relative_eq!(-0.14630, inv.get(2, 1), epsilon = EPSILON);
            assert_relative_eq!(-0.10926, inv.get(2, 2), epsilon = EPSILON);
            assert_relative_eq!(0.12963, inv.get(2, 3), epsilon = EPSILON);

            assert_relative_eq!(0.17778, inv.get(3, 0), epsilon = EPSILON);
            assert_relative_eq!(0.06667, inv.get(3, 1), epsilon = EPSILON);
            assert_relative_eq!(-0.26667, inv.get(3, 2), epsilon = EPSILON);
            assert_relative_eq!(0.33333, inv.get(3, 3), epsilon = EPSILON);
        }
    }

    #[test]
    fn test_matrix4_submat() {
        let mut m = Matrix4::zero();
        m.set(0, 0, -6.0);
        m.set(0, 1, 1.0);
        m.set(0, 2, 1.0);
        m.set(0, 3, 6.0);

        m.set(1, 0, -8.0);
        m.set(1, 1, 5.0);
        m.set(1, 2, 8.0);
        m.set(1, 3, 6.0);

        m.set(2, 0, -1.0);
        m.set(2, 1, 0.0);
        m.set(2, 2, 8.0);
        m.set(2, 3, 2.0);

        m.set(3, 0, -7.0);
        m.set(3, 1, 1.0);
        m.set(3, 2, -1.0);
        m.set(3, 3, 1.0);

        let sub = Matrix4::submat(&m, 2, 1);

        assert_eq!(-6.0, sub.get(0, 0));
        assert_eq!(1.0, sub.get(0, 1));
        assert_eq!(6.0, sub.get(0, 2));

        assert_eq!(-8.0, sub.get(1, 0));
        assert_eq!(8.0, sub.get(1, 1));
        assert_eq!(6.0, sub.get(1, 2));

        assert_eq!(-7.0, sub.get(2, 0));
        assert_eq!(-1.0, sub.get(2, 1));
        assert_eq!(1.0, sub.get(2, 2));
    }

    #[test]
    fn test_matrix4_mul() {
        {
            let mut m = Matrix4::zero();
            m.set(0, 0, 1.0);
            m.set(0, 1, 2.0);
            m.set(0, 2, 3.0);
            m.set(0, 3, 4.0);

            m.set(1, 0, 5.0);
            m.set(1, 1, 6.0);
            m.set(1, 2, 7.0);
            m.set(1, 3, 8.0);

            m.set(2, 0, 9.0);
            m.set(2, 1, 8.0);
            m.set(2, 2, 7.0);
            m.set(2, 3, 6.0);

            m.set(3, 0, 5.0);
            m.set(3, 1, 4.0);
            m.set(3, 2, 3.0);
            m.set(3, 3, 2.0);

            let mut n = Matrix4::zero();
            n.set(0, 0, -2.0);
            n.set(0, 1, 1.0);
            n.set(0, 2, 2.0);
            n.set(0, 3, 3.0);

            n.set(1, 0, 3.0);
            n.set(1, 1, 2.0);
            n.set(1, 2, 1.0);
            n.set(1, 3, -1.0);

            n.set(2, 0, 4.0);
            n.set(2, 1, 3.0);
            n.set(2, 2, 6.0);
            n.set(2, 3, 5.0);

            n.set(3, 0, 1.0);
            n.set(3, 1, 2.0);
            n.set(3, 2, 7.0);
            n.set(3, 3, 8.0);

            let r = m * n;

            assert_eq!(20.0, r.get(0, 0));
            assert_eq!(22.0, r.get(0, 1));
            assert_eq!(50.0, r.get(0, 2));
            assert_eq!(48.0, r.get(0, 3));

            assert_eq!(44.0, r.get(1, 0));
            assert_eq!(54.0, r.get(1, 1));
            assert_eq!(114.0, r.get(1, 2));
            assert_eq!(108.0, r.get(1, 3));

            assert_eq!(40.0, r.get(2, 0));
            assert_eq!(58.0, r.get(2, 1));
            assert_eq!(110.0, r.get(2, 2));
            assert_eq!(102.0, r.get(2, 3));

            assert_eq!(16.0, r.get(3, 0));
            assert_eq!(26.0, r.get(3, 1));
            assert_eq!(46.0, r.get(3, 2));
            assert_eq!(42.0, r.get(3, 3));
        }

        {
            let mut m = Matrix4::zero();
            m.set(0, 0, 1.0);
            m.set(0, 1, 2.0);
            m.set(0, 2, 3.0);
            m.set(0, 3, 4.0);

            m.set(1, 0, 5.0);
            m.set(1, 1, 6.0);
            m.set(1, 2, 7.0);
            m.set(1, 3, 8.0);

            m.set(2, 0, 9.0);
            m.set(2, 1, 8.0);
            m.set(2, 2, 7.0);
            m.set(2, 3, 6.0);

            m.set(3, 0, 5.0);
            m.set(3, 1, 4.0);
            m.set(3, 2, 3.0);
            m.set(3, 3, 2.0);

            let n = Matrix4::identity();

            let r = m * n;

            assert_eq!(m, r);
        }
    }

    #[test]
    fn test_matrix4_mul_vec3_point() {
        {
            let t = Matrix4::op_translate(5.0, -3.0, 2.0);
            let p = Vec3::from(-3.0, 4.0, 5.0);

            let r = Matrix4::mul_vec3_point(&t, &p);

            assert_eq!(2.0, r.x);
            assert_eq!(1.0, r.y);
            assert_eq!(7.0, r.z);
        }

        {
            let t = Matrix4::op_translate(5.0, -3.0, 2.0);
            let i = Matrix4::from_inverse(&t).expect("det > 0");
            let p = Vec3::from(-3.0, 4.0, 5.0);

            let r = Matrix4::mul_vec3_point(&i, &p);

            assert_eq!(-8.0, r.x);
            assert_eq!(7.0, r.y);
            assert_eq!(3.0, r.z);
        }

        {
            let t = Matrix4::op_scale(2.0, 3.0, 4.0);
            let p = Vec3::from(-4.0, 6.0, 8.0);

            let r = Matrix4::mul_vec3_point(&t, &p);

            assert_eq!(-8.0, r.x);
            assert_eq!(18.0, r.y);
            assert_eq!(32.0, r.z);
        }

        {
            let t = Matrix4::op_scale(-1.0, 1.0, 1.0);
            let p = Vec3::from(2.0, 3.0, 4.0);

            let r = Matrix4::mul_vec3_point(&t, &p);

            assert_eq!(-2.0, r.x);
            assert_eq!(3.0, r.y);
            assert_eq!(4.0, r.z);
        }

        {
            let p = Vec3::y_axis();
            let half_quarter = Matrix4::op_rotate_x(PI / 4.0);
            let r = Matrix4::mul_vec3_point(&half_quarter, &p);

            let sqrt_2_over_2 = 2.0_f32.sqrt() / 2.0;

            assert_relative_eq!(0.0, r.x);
            assert_relative_eq!(sqrt_2_over_2, r.y, epsilon = EPSILON);
            assert_relative_eq!(sqrt_2_over_2, r.z, epsilon = EPSILON);
        }

        {
            let p = Vec3::y_axis();
            let full_quarter = Matrix4::op_rotate_x(PI / 2.0);
            let r = Matrix4::mul_vec3_point(&full_quarter, &p);

            assert_relative_eq!(0.0, r.x, epsilon = EPSILON);
            assert_relative_eq!(0.0, r.y, epsilon = EPSILON);
            assert_relative_eq!(1.0, r.z, epsilon = EPSILON);
        }

        {
            let p = Vec3::z_axis();
            let half_quarter = Matrix4::op_rotate_y(PI / 4.0);
            let r = Matrix4::mul_vec3_point(&half_quarter, &p);

            let sqrt_2_over_2 = 2.0_f32.sqrt() / 2.0;

            assert_relative_eq!(sqrt_2_over_2, r.x, epsilon = EPSILON);
            assert_relative_eq!(0.0, r.y);
            assert_relative_eq!(sqrt_2_over_2, r.z, epsilon = EPSILON);
        }

        {
            let p = Vec3::z_axis();
            let full_quarter = Matrix4::op_rotate_y(PI / 2.0);
            let r = Matrix4::mul_vec3_point(&full_quarter, &p);

            assert_relative_eq!(1.0, r.x, epsilon = EPSILON);
            assert_relative_eq!(0.0, r.y, epsilon = EPSILON);
            assert_relative_eq!(0.0, r.z, epsilon = EPSILON);
        }

        {
            let p = Vec3::y_axis();
            let half_quarter = Matrix4::op_rotate_z(PI / 4.0);
            let r = Matrix4::mul_vec3_point(&half_quarter, &p);

            let sqrt_2_over_2 = 2.0_f32.sqrt() / 2.0;

            assert_relative_eq!(-sqrt_2_over_2, r.x, epsilon = EPSILON);
            assert_relative_eq!(sqrt_2_over_2, r.y, epsilon = EPSILON);
            assert_relative_eq!(0.0, r.z);
        }

        {
            let p = Vec3::y_axis();
            let full_quarter = Matrix4::op_rotate_z(PI / 2.0);
            let r = Matrix4::mul_vec3_point(&full_quarter, &p);

            assert_relative_eq!(-1.0, r.x, epsilon = EPSILON);
            assert_relative_eq!(0.0, r.y, epsilon = EPSILON);
            assert_relative_eq!(0.0, r.z, epsilon = EPSILON);
        }

        {
            let t = Matrix4::op_shear(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            let p = Vec3::from(2.0, 3.0, 4.0);
            let r = Matrix4::mul_vec3_point(&t, &p);

            assert_eq!(5.0, r.x);
            assert_eq!(3.0, r.y);
            assert_eq!(4.0, r.z);
        }

        {
            let t = Matrix4::op_shear(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
            let p = Vec3::from(2.0, 3.0, 4.0);
            let r = Matrix4::mul_vec3_point(&t, &p);

            assert_eq!(6.0, r.x);
            assert_eq!(3.0, r.y);
            assert_eq!(4.0, r.z);
        }

        {
            let t = Matrix4::op_shear(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
            let p = Vec3::from(2.0, 3.0, 4.0);
            let r = Matrix4::mul_vec3_point(&t, &p);

            assert_eq!(2.0, r.x);
            assert_eq!(5.0, r.y);
            assert_eq!(4.0, r.z);
        }

        {
            let t = Matrix4::op_shear(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
            let p = Vec3::from(2.0, 3.0, 4.0);
            let r = Matrix4::mul_vec3_point(&t, &p);

            assert_eq!(2.0, r.x);
            assert_eq!(7.0, r.y);
            assert_eq!(4.0, r.z);
        }

        {
            let t = Matrix4::op_shear(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
            let p = Vec3::from(2.0, 3.0, 4.0);
            let r = Matrix4::mul_vec3_point(&t, &p);

            assert_eq!(2.0, r.x);
            assert_eq!(3.0, r.y);
            assert_eq!(6.0, r.z);
        }

        {
            let t = Matrix4::op_shear(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            let p = Vec3::from(2.0, 3.0, 4.0);
            let r = Matrix4::mul_vec3_point(&t, &p);

            assert_eq!(2.0, r.x);
            assert_eq!(3.0, r.y);
            assert_eq!(7.0, r.z);
        }
    }

    #[test]
    fn test_matrix4_mul_vec3_dir() {
        {
            let t = Matrix4::op_translate(5.0, -3.0, 2.0);
            let v = Vec3::from(-3.0, 4.0, 5.0);

            let r = Matrix4::mul_vec3_dir(&t, &v);

            assert_eq!(v.x, r.x);
            assert_eq!(v.y, r.y);
            assert_eq!(v.z, r.z);
        }

        {
            let t = Matrix4::op_scale(2.0, 3.0, 4.0);
            let p = Vec3::from(-4.0, 6.0, 8.0);

            let r = Matrix4::mul_vec3_dir(&t, &p);

            assert_eq!(-8.0, r.x);
            assert_eq!(18.0, r.y);
            assert_eq!(32.0, r.z);
        }

        {
            let t = Matrix4::op_scale(2.0, 3.0, 4.0);
            let i = Matrix4::from_inverse(&t).expect("det > 0");
            let p = Vec3::from(-4.0, 6.0, 8.0);

            let r = Matrix4::mul_vec3_dir(&i, &p);

            assert_eq!(-2.0, r.x);
            assert_eq!(2.0, r.y);
            assert_eq!(2.0, r.z);
        }

        {
            let p = Vec3::from(1.0, 0.0, 1.0);
            let a = Matrix4::op_rotate_x(PI / 2.0);
            let b = Matrix4::op_scale(5.0, 5.0, 5.0);
            let c = Matrix4::op_translate(10.0, 5.0, 7.0);

            let p2 = Matrix4::mul_vec3_point(&a, &p);
            assert_relative_eq!(1.0, p2.x, epsilon = EPSILON);
            assert_relative_eq!(-1.0, p2.y, epsilon = EPSILON);
            assert_relative_eq!(0.0, p2.z, epsilon = EPSILON);

            let p3 = Matrix4::mul_vec3_point(&b, &p2);
            assert_relative_eq!(5.0, p3.x, epsilon = EPSILON);
            assert_relative_eq!(-5.0, p3.y, epsilon = EPSILON);
            assert_relative_eq!(0.0, p3.z, epsilon = EPSILON);

            let p4 = Matrix4::mul_vec3_point(&c, &p3);
            assert_relative_eq!(15.0, p4.x, epsilon = EPSILON);
            assert_relative_eq!(0.0, p4.y, epsilon = EPSILON);
            assert_relative_eq!(7.0, p4.z, epsilon = EPSILON);
        }

        {
            let p = Vec3::from(1.0, 0.0, 1.0);
            let a = Matrix4::op_rotate_x(PI / 2.0);
            let b = Matrix4::op_scale(5.0, 5.0, 5.0);
            let c = Matrix4::op_translate(10.0, 5.0, 7.0);
            let t = c * b * a;

            let r = Matrix4::mul_vec3_point(&t, &p);
            assert_relative_eq!(15.0, r.x, epsilon = EPSILON);
            assert_relative_eq!(0.0, r.y, epsilon = EPSILON);
            assert_relative_eq!(7.0, r.z, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_matrix4_transpose() {
        let mut m = Matrix4::zero();
        m.set(0, 0, 0.0);
        m.set(0, 1, 9.0);
        m.set(0, 2, 3.0);
        m.set(0, 3, 0.0);

        m.set(1, 0, 9.0);
        m.set(1, 1, 8.0);
        m.set(1, 2, 0.0);
        m.set(1, 3, 8.0);

        m.set(2, 0, 1.0);
        m.set(2, 1, 8.0);
        m.set(2, 2, 5.0);
        m.set(2, 3, 3.0);

        m.set(3, 0, 0.0);
        m.set(3, 1, 0.0);
        m.set(3, 2, 5.0);
        m.set(3, 3, 8.0);

        let r = Matrix4::from_transpose(&m);

        assert_eq!(0.0, r.get(0, 0));
        assert_eq!(9.0, r.get(0, 1));
        assert_eq!(1.0, r.get(0, 2));
        assert_eq!(0.0, r.get(0, 3));

        assert_eq!(9.0, r.get(1, 0));
        assert_eq!(8.0, r.get(1, 1));
        assert_eq!(8.0, r.get(1, 2));
        assert_eq!(0.0, r.get(1, 3));

        assert_eq!(3.0, r.get(2, 0));
        assert_eq!(0.0, r.get(2, 1));
        assert_eq!(5.0, r.get(2, 2));
        assert_eq!(5.0, r.get(2, 3));

        assert_eq!(0.0, r.get(3, 0));
        assert_eq!(8.0, r.get(3, 1));
        assert_eq!(3.0, r.get(3, 2));
        assert_eq!(8.0, r.get(3, 3));
    }

    #[test]
    fn test_ray_position() {
        let o = Vec3::from(2.0, 3.0, 4.0);
        let dir = Vec3::x_axis();

        let r = Ray::from(o, dir);

        let pos = Ray::position(&r, 0.0);
        assert_eq!(o, pos);

        let pos = Ray::position(&r, 1.0);
        assert_eq!(Vec3::from(3.0, 3.0, 4.0), pos);

        let pos = Ray::position(&r, -1.0);
        assert_eq!(Vec3::from(1.0, 3.0, 4.0), pos);

        let pos = Ray::position(&r, 2.5);
        assert_eq!(Vec3::from(4.5, 3.0, 4.0), pos);
    }

    #[test]
    fn test_matrix4_view_transform() {
        {
            let f = Vec3::zero();
            let t = Vec3::neg_z_axis();
            let u = Vec3::y_axis();

            let m = Matrix4::view_transform(&f, &t, &u);

            assert_eq!(Matrix4::identity(), m);
        }

        {
            let f = Vec3::zero();
            let t = Vec3::z_axis();
            let u = Vec3::y_axis();

            let m = Matrix4::view_transform(&f, &t, &u);

            assert_eq!(Matrix4::op_scale(-1.0, 1.0, -1.0), m);
        }

        {
            let f = Vec3::from(0.0, 0.0, 8.0);
            let t = Vec3::zero();
            let u = Vec3::y_axis();

            let m = Matrix4::view_transform(&f, &t, &u);

            assert_eq!(Matrix4::op_translate(0.0, 0.0, -8.0), m);
        }

        {
            let f = Vec3::from(1.0, 3.0, 2.0);
            let t = Vec3::from(4.0, -2.0, 8.0);
            let u = Vec3::from(1.0, 1.0, 0.0);

            let m = Matrix4::view_transform(&f, &t, &u);

            assert_relative_eq!(-0.50709, m.get(0, 0), epsilon = EPSILON);
            assert_relative_eq!(0.50709, m.get(0, 1), epsilon = EPSILON);
            assert_relative_eq!(0.67612, m.get(0, 2), epsilon = EPSILON);
            assert_relative_eq!(-2.36643, m.get(0, 3), epsilon = EPSILON);

            assert_relative_eq!(0.76772, m.get(1, 0), epsilon = EPSILON);
            assert_relative_eq!(0.60609, m.get(1, 1), epsilon = EPSILON);
            assert_relative_eq!(0.12122, m.get(1, 2), epsilon = EPSILON);
            assert_relative_eq!(-2.82843, m.get(1, 3), epsilon = EPSILON);

            assert_relative_eq!(-0.35857, m.get(2, 0), epsilon = EPSILON);
            assert_relative_eq!(0.59761, m.get(2, 1), epsilon = EPSILON);
            assert_relative_eq!(-0.71714, m.get(2, 2), epsilon = EPSILON);
            assert_relative_eq!(0.00000, m.get(2, 3), epsilon = EPSILON);

            assert_relative_eq!(0.00000, m.get(3, 0), epsilon = EPSILON);
            assert_relative_eq!(0.00000, m.get(3, 1), epsilon = EPSILON);
            assert_relative_eq!(0.00000, m.get(3, 2), epsilon = EPSILON);
            assert_relative_eq!(1.00000, m.get(3, 3), epsilon = EPSILON);
        }
    }
}
