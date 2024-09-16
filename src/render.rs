use crate::{
    color::{self, Color},
    entity::{Entity, World},
    math::{self, Matrix4, Ray, Vec3},
    physics::{self, intersect_at, prepare_computations, world_to_object, Intersection},
};

use std::{cell::RefCell, rc::Rc};

/// Defines the maximum number of times a ray is allowed to reflect between entities.
const REFLECT_LIMIT: u32 = 5;

/// A light emitting a color from a point in the world.
#[derive(Clone, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub intensity: Color,
}

impl PointLight {
    /// Creates a new default [PointLight].
    pub fn new() -> Self {
        PointLight::default()
    }
    /// Creates a new [PointLight] from the given values.
    pub fn from(position: Vec3, intensity: Color) -> Self {
        PointLight {
            position,
            intensity,
        }
    }
}

impl Default for PointLight {
    /// Creates a default [PointLight].
    fn default() -> Self {
        PointLight::from(Vec3::zero(), color::WHITE)
    }
}

/// Defines the behavior required of a pattern that can be applied to a [Material].
pub trait Pattern {
    /// Returns the transform matrix for the pattern.
    fn transform(&self) -> &Matrix4;
    /// Returns the [Color] for the given local point.
    fn pattern_at(&self, point: &Vec3) -> Color;
    /// Returns the [Color] for the pattern applied to the object at the given world point.
    fn pattern_at_object(&self, entity: Rc<RefCell<Entity>>, point: &Vec3) -> Color {
        let object_point = world_to_object(Rc::clone(&entity), point);

        let inv_pat_transform =
            Matrix4::from_inverse(self.transform()).expect("transform is invertible");
        let pattern_point = Matrix4::mul_vec3_point(&inv_pat_transform, &object_point);

        self.pattern_at(&pattern_point)
    }
}

/// Implementation of a [Pattern] that is striped with two colors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stripes {
    pub a: Color,
    pub b: Color,
    pub transform: Matrix4,
}

impl Stripes {
    /// Creates a new [Stripes] with the colors set to white and black.
    pub fn new() -> Self {
        Self::from(color::WHITE, color::BLACK)
    }
    /// Creates a new [Stripes] from the given colors.
    pub fn from(a: Color, b: Color) -> Self {
        Self {
            a,
            b,
            transform: Matrix4::identity(),
        }
    }
}

impl Pattern for Stripes {
    /// Returns the transform matrix for the pattern.
    fn transform(&self) -> &Matrix4 {
        &self.transform
    }
    /// Returns the [Color] for the given local point.
    fn pattern_at(&self, point: &Vec3) -> Color {
        if point.x.floor() as i32 % 2 == 0 {
            self.a
        } else {
            self.b
        }
    }
}

impl Default for Stripes {
    /// Creates a default [Stripes].
    fn default() -> Self {
        Stripes::new()
    }
}

/// Implementation of a [Pattern] that is a gradient with two colors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gradient {
    pub a: Color,
    pub b: Color,
    pub transform: Matrix4,
}

impl Gradient {
    /// Creates a new [Gradient] with the colors set to white and black.
    pub fn new() -> Self {
        Self::from(color::WHITE, color::BLACK)
    }
    /// Creates a new [Gradient] from the given colors.
    pub fn from(a: Color, b: Color) -> Self {
        Self {
            a,
            b,
            transform: Matrix4::identity(),
        }
    }
}

impl Pattern for Gradient {
    /// Returns the transform matrix for the pattern.
    fn transform(&self) -> &Matrix4 {
        &self.transform
    }
    /// Returns the [Color] for the given local point.
    fn pattern_at(&self, point: &Vec3) -> Color {
        let distance = self.b - self.a;
        let fraction = point.x - point.x.floor();

        self.a + Color::from_scaled(&distance, fraction)
    }
}

impl Default for Gradient {
    /// Creates a default [Gradient].
    fn default() -> Self {
        Gradient::new()
    }
}

/// Implementation of a [Pattern] that is concentric rings with two colors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rings {
    pub a: Color,
    pub b: Color,
    pub transform: Matrix4,
}

impl Rings {
    /// Creates a new [Rings] with the colors set to white and black.
    pub fn new() -> Self {
        Self::from(color::WHITE, color::BLACK)
    }
    /// Creates a new [Rings] from the given colors.
    pub fn from(a: Color, b: Color) -> Self {
        Self {
            a,
            b,
            transform: Matrix4::identity(),
        }
    }
}

impl Pattern for Rings {
    /// Returns the transform matrix for the pattern.
    fn transform(&self) -> &Matrix4 {
        &self.transform
    }
    /// Returns the [Color] for the given local point.
    fn pattern_at(&self, point: &Vec3) -> Color {
        let n = (point.x * point.x) + (point.z + point.z);
        let floor = n.sqrt().floor() as i32;

        if floor % 2 == 0 {
            self.a
        } else {
            self.b
        }
    }
}

impl Default for Rings {
    /// Creates a default [Rings].
    fn default() -> Self {
        Rings::new()
    }
}

/// Implementation of a [Pattern] that is checkered with two colors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Checkers {
    pub a: Color,
    pub b: Color,
    pub transform: Matrix4,
}

impl Checkers {
    /// Creates a new default [Checkers] pattern consisting of whtie and black.
    pub fn new() -> Self {
        Self::from(color::WHITE, color::BLACK)
    }
    /// Creates a new [Checkers] pattern from the given colors.
    pub fn from(a: Color, b: Color) -> Self {
        Self {
            a,
            b,
            transform: Matrix4::identity(),
        }
    }
}

impl Pattern for Checkers {
    /// Returns the transform matrix for the pattern.
    fn transform(&self) -> &Matrix4 {
        &self.transform
    }
    /// Returns the [Color] for the given local point.
    fn pattern_at(&self, point: &Vec3) -> Color {
        let floor_x = point.x.floor() as i32;
        let floor_y = point.y.floor() as i32;
        let floor_z = point.z.floor() as i32;
        let sum = floor_x + floor_y + floor_z;

        if sum % 2 == 0 {
            self.a
        } else {
            self.b
        }
    }
}

impl Default for Checkers {
    /// Creates a default [Checkers] pattern.
    fn default() -> Self {
        Checkers::new()
    }
}

/// Represents the material that an object in the world is composed of.
#[derive(Clone)]
pub struct Material {
    pub color: Color,
    pub ambient: f32,
    pub diffuse: f32,
    pub specular: f32,
    pub shininess: f32,
    pub reflective: f32,
    pub transparency: f32,
    pub refractive_index: f32,
    pub pattern: Option<Rc<dyn Pattern>>,
}

impl Material {
    /// Creates a new default [Material].
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for Material {
    /// Creates a new [Material] with default values.
    fn default() -> Self {
        Material {
            color: color::WHITE,
            ambient: 0.1,
            diffuse: 0.9,
            specular: 0.9,
            shininess: 200.0,
            reflective: 0.0,
            transparency: 0.0,
            refractive_index: 1.0,
            pattern: None,
        }
    }
}

/// Represents a camera looking into the world.
#[derive(Debug)]
pub struct Camera {
    pub hsize: u32,
    pub vsize: u32,
    pub field_of_view: f32,
    pub transform: Matrix4,
    pub pixel_size: f32,
    pub half_width: f32,
    pub half_height: f32,
}

impl Camera {
    /// Creates a new [Camera] with the given configuration values.
    pub fn from(hsize: u32, vsize: u32, field_of_view: f32) -> Self {
        let half_view = (field_of_view / 2.0).tan();
        let aspect = hsize as f32 / vsize as f32;

        let (half_width, half_height) = if aspect >= 1.0 {
            (half_view, half_view / aspect)
        } else {
            (half_view * aspect, half_view)
        };

        Self {
            hsize,
            vsize,
            field_of_view,
            transform: Matrix4::identity(),
            pixel_size: (half_width * 2.0) / hsize as f32,
            half_width,
            half_height,
        }
    }
}

/// Calculates the [Color] for a point on an object based on the lighting.
pub fn lighting(
    material: &Material,
    entity: Rc<RefCell<Entity>>,
    light: &PointLight,
    point: &Vec3,
    eyev: &Vec3,
    normalv: &Vec3,
    in_shadow: bool,
) -> Color {
    let color = match &material.pattern {
        Some(p) => p.pattern_at_object(entity, point),
        None => material.color,
    };

    let effective_color = color * light.intensity;

    let mut lightv = light.position - *point;
    lightv.normalize();

    let ambient = Color::from_scaled(&effective_color, material.ambient);

    if in_shadow {
        return ambient;
    }

    let mut diffuse = color::BLACK;
    let mut specular = color::BLACK;

    let light_dot_normal = lightv.dot(normalv);
    if light_dot_normal >= 0.0 {
        diffuse = Color::from_scaled(&effective_color, material.diffuse * light_dot_normal);

        let reflectv = physics::reflect(&Vec3::from_scaled(&lightv, -1.0), normalv);

        let reflect_dot_eye = reflectv.dot(eyev);
        if reflect_dot_eye > 0.0 {
            let factor = reflect_dot_eye.powf(material.shininess);
            specular = Color::from_scaled(&light.intensity, material.specular * factor);
        }
    }

    ambient + diffuse + specular
}

/// Determines if the given point is shadowed.
pub fn is_shadowed(w: &World, p: &Vec3) -> bool {
    let v = w.light().position - *p;
    let distance = v.mag();
    let direction = Vec3::from_normalized(&v);

    let r = Ray::from(*p, direction);

    let hit_test = intersect_at(w, &r);

    matches!(hit_test.hit(), Some(hit) if hit.t < distance)
}

/// Calculates the [Color] that should be emitted for the [Intersection] in the [World].
pub fn shade_hit(w: &World, i: &Intersection, remaining: u32) -> Color {
    let shadowed = is_shadowed(w, &i.over_point);

    let surface = lighting(
        &i.entity.borrow().material,
        Rc::clone(&i.entity),
        w.light(),
        &i.over_point,
        &i.eyev,
        &i.normalv,
        shadowed,
    );

    let reflected = reflected_color(w, i, remaining);
    let refracted = refracted_color(w, i, remaining);

    let material = &i.entity.borrow().material;
    if material.reflective > 0.0 && material.transparency > 0.0 {
        let reflectance = schlick(i);

        surface
            + Color::from_scaled(&reflected, reflectance)
            + Color::from_scaled(&refracted, 1.0 - reflectance)
    } else {
        surface + reflected + refracted
    }
}

/// Calculates the [Color] that should be emitted when the [Ray] is cast through the [World].
pub fn color_at(w: &World, r: &Ray, remaining: u32) -> Color {
    let hit_test = intersect_at(w, r);
    let hit = hit_test.hit();

    match hit {
        None => color::BLACK,
        Some(hit) => {
            let i = prepare_computations(hit, r, hit_test.hits());
            shade_hit(w, &i, remaining)
        }
    }
}

/// Calculates the [Color] of the reflection at the intersection.
pub fn reflected_color(w: &World, i: &Intersection, remaining: u32) -> Color {
    let reflective = i.entity.borrow().material.reflective;

    if reflective == 0.0 || remaining < 1 {
        return color::BLACK;
    }

    let reflect_ray = Ray::from(i.over_point, i.reflectv);
    let color = color_at(w, &reflect_ray, remaining - 1);

    Color::from_scaled(&color, reflective)
}

/// Calculates the refracted [Color] at the intersection.
pub fn refracted_color(w: &World, i: &Intersection, remaining: u32) -> Color {
    if remaining < 1 || i.entity.borrow().material.transparency == 0.0 {
        return color::BLACK;
    }

    let n_ratio = i.n1 / i.n2;
    let cos_i = i.eyev.dot(&i.normalv);
    let sin2_t = n_ratio.powi(2) * (1.0 - cos_i.powi(2));

    if sin2_t > 1.0 {
        return color::BLACK;
    }

    let cos_t = (1.0 - sin2_t).sqrt();

    let direction = Vec3::from_scaled(&i.normalv, n_ratio * cos_i - cos_t)
        - Vec3::from_scaled(&i.eyev, n_ratio);

    let refract_ray = Ray::from(i.under_point, direction);

    let c = color_at(w, &refract_ray, remaining - 1);

    Color::from_scaled(&c, i.entity.borrow().material.transparency)
}

/// Calculates an approximation of the Fresnel equation which reprsents the reflectance, or what
/// fraction of light is reflected given the information at the [Intersection].
pub fn schlick(i: &Intersection) -> f32 {
    let mut cos = i.eyev.dot(&i.normalv);

    if i.n1 > i.n2 {
        let n_ratio = i.n1 / i.n2;

        let sin2_t = n_ratio.powi(2) * (1.0 - cos.powi(2));
        if sin2_t > 1.0 {
            return 1.0;
        }

        let cos_t = (1.0 - sin2_t).sqrt();
        cos = cos_t;
    }

    let r0 = ((i.n1 - i.n2) / (i.n1 + i.n2)).powi(2);

    r0 + ((1.0 - r0) * (1.0 - cos).powi(5))
}

/// A [Canvas] is what the scene is rendered to.
#[derive(Debug)]
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pixels: Vec<color::Color>,
}

impl Canvas {
    /// Creates new [Canvas] of the given size whose pixels are initalized to black.
    pub fn new(width: usize, height: usize) -> Self {
        Canvas::with_fill(width, height, &color::BLACK)
    }
    /// Creates new [Canvas] of the given size whose pixels are initalized to the fill [Color].
    pub fn with_fill(width: usize, height: usize, c: &color::Color) -> Self {
        let cap = width * height;
        let mut pixels = Vec::with_capacity(cap);

        for _ in 0..cap {
            pixels.push(*c);
        }

        Canvas {
            width,
            height,
            pixels,
        }
    }
    /// Writes the [Color] to the specified pixel.
    pub fn write_pixel(&mut self, x: usize, y: usize, c: &color::Color) {
        let idx = (y * self.width) + x;
        if idx >= self.pixels.len() {
            tracing::warn!("attempted to write pixel out of bounds: ({},{})", x, y);
        } else {
            tracing::debug!("writing pixel at ({x},{y})");
            self.pixels[idx] = *c;
        }
    }
    /// Retrieves the [Color] at the given pixel coordinates if available. Returns white if the
    /// pixel requested is out of bounds.
    pub fn pixel_at(&self, x: usize, y: usize) -> Option<&color::Color> {
        // return white if outside bounds
        if x >= self.width || y >= self.height {
            tracing::warn!("attempted to access pixel out of bounds: ({},{})", x, y);
            Some(&color::WHITE)
        } else {
            let idx = (y * self.width) + x;
            self.pixels.get(idx)
        }
    }
    /// Exports the pixels in the canvas to a String in the Portable Pixmap (PPM) file format.
    pub fn export_to_ppm(&self) -> String {
        let mut contents = format!("P3\n{} {}\n255", self.width, self.height);

        for (i, p) in self.pixels.iter().enumerate() {
            if i % 10 == 0 {
                contents.push('\n');
            } else {
                contents.push(' ');
            }

            let r = math::clamp((p.r * 255.0).round(), 0.0, 255.0);
            let g = math::clamp((p.g * 255.0).round(), 0.0, 255.0);
            let b = math::clamp((p.b * 255.0).round(), 0.0, 255.0);

            let pixel = format!("{} {} {}", r, g, b);
            contents.push_str(&pixel);
        }

        contents.push('\n');

        contents
    }
}

/// Creates a [Ray] from the [Camera] to the pixel.
pub fn ray_for_pixel(c: &Camera, x: u32, y: u32) -> Ray {
    let xoffset = (x as f32 + 0.5) * c.pixel_size;
    let yoffset = (y as f32 + 0.5) * c.pixel_size;

    let world_x = c.half_width - xoffset;
    let world_y = c.half_height - yoffset;

    let inv_transform = Matrix4::from_inverse(&c.transform).expect("transform is invertible");

    let pixel = Matrix4::mul_vec3_point(&inv_transform, &Vec3::from(world_x, world_y, -1.0));
    let origin = Matrix4::mul_vec3_point(&inv_transform, &Vec3::zero());

    let mut direction = pixel - origin;
    direction.normalize();

    Ray::from(origin, direction)
}

/// Renders the [World] onto a [Canvas] as seen through the [Camera].
pub fn render(c: &Camera, w: &World) -> Canvas {
    let mut image = Canvas::new(c.hsize as usize, c.vsize as usize);

    for y in 0..c.vsize - 1 {
        for x in 0..c.hsize - 1 {
            let ray = ray_for_pixel(c, x, y);
            let color = color_at(w, &ray, REFLECT_LIMIT);

            image.write_pixel(x as usize, y as usize, &color);
        }
    }

    image
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, f32::consts::PI, rc::Rc};

    use approx::assert_relative_eq;

    use crate::{
        color::{self, Color},
        entity::{Entity, World},
        math::{Matrix4, Ray, Vec3},
        physics::{prepare_computations, Hit, HitTest},
        render::{color_at, is_shadowed, shade_hit, Pattern, REFLECT_LIMIT},
        EPSILON,
    };

    use super::{
        lighting, ray_for_pixel, reflected_color, refracted_color, render, schlick, Camera,
        Checkers, Gradient, Material, PointLight, Rings, Stripes,
    };

    struct TestPattern {
        transform: Matrix4,
    }

    impl TestPattern {
        fn new() -> Self {
            Self::default()
        }
    }

    impl Default for TestPattern {
        fn default() -> Self {
            Self {
                transform: Matrix4::identity(),
            }
        }
    }

    impl Pattern for TestPattern {
        fn transform(&self) -> &Matrix4 {
            &self.transform
        }
        fn pattern_at(&self, point: &Vec3) -> Color {
            Color::from(point.x, point.y, point.z)
        }
    }

    #[test]
    fn test_lighting() {
        let sqrt_2_over_2 = 2.0_f32.sqrt() / 2.0;

        let mat = Material::default();
        let obj = Rc::new(RefCell::new(Entity::sphere()));
        let pos = Vec3::zero();

        {
            let eyev = Vec3::neg_z_axis();
            let normalv = Vec3::neg_z_axis();
            let light = PointLight::from(Vec3::from(0.0, 0.0, -10.0), color::WHITE);

            let r = lighting(&mat, Rc::clone(&obj), &light, &pos, &eyev, &normalv, false);
            assert_eq!(1.9, r.r);
            assert_eq!(1.9, r.g);
            assert_eq!(1.9, r.b);
        }

        {
            let eyev = Vec3::from(0.0, sqrt_2_over_2, -sqrt_2_over_2);
            let normalv = Vec3::neg_z_axis();
            let light = PointLight::from(Vec3::from(0.0, 0.0, -10.0), color::WHITE);

            let r = lighting(&mat, Rc::clone(&obj), &light, &pos, &eyev, &normalv, false);
            assert_eq!(1.0, r.r);
            assert_eq!(1.0, r.g);
            assert_eq!(1.0, r.b);
        }

        {
            let eyev = Vec3::neg_z_axis();
            let normalv = Vec3::neg_z_axis();
            let light = PointLight::from(Vec3::from(0.0, 10.0, -10.0), color::WHITE);

            let r = lighting(&mat, Rc::clone(&obj), &light, &pos, &eyev, &normalv, false);
            assert_relative_eq!(0.7364, r.r, epsilon = EPSILON);
            assert_relative_eq!(0.7364, r.g, epsilon = EPSILON);
            assert_relative_eq!(0.7364, r.b, epsilon = EPSILON);
        }

        {
            let eyev = Vec3::from(0.0, -sqrt_2_over_2, -sqrt_2_over_2);
            let normalv = Vec3::neg_z_axis();
            let light = PointLight::from(Vec3::from(0.0, 10.0, -10.0), color::WHITE);

            let r = lighting(&mat, Rc::clone(&obj), &light, &pos, &eyev, &normalv, false);
            assert_relative_eq!(1.6364, r.r, epsilon = EPSILON);
            assert_relative_eq!(1.6364, r.g, epsilon = EPSILON);
            assert_relative_eq!(1.6364, r.b, epsilon = EPSILON);
        }

        {
            let eyev = Vec3::neg_z_axis();
            let normalv = Vec3::neg_z_axis();
            let light = PointLight::from(Vec3::from(0.0, 0.0, 10.0), color::WHITE);

            let r = lighting(&mat, Rc::clone(&obj), &light, &pos, &eyev, &normalv, false);
            assert_relative_eq!(0.1, r.r, epsilon = EPSILON);
            assert_relative_eq!(0.1, r.g, epsilon = EPSILON);
            assert_relative_eq!(0.1, r.b, epsilon = EPSILON);
        }

        {
            let eyev = Vec3::neg_z_axis();
            let normalv = Vec3::neg_z_axis();
            let light = PointLight::from(Vec3::from(0.0, 0.0, -10.0), color::WHITE);

            let r = lighting(&mat, Rc::clone(&obj), &light, &pos, &eyev, &normalv, true);
            assert_relative_eq!(0.1, r.r, epsilon = EPSILON);
            assert_relative_eq!(0.1, r.g, epsilon = EPSILON);
            assert_relative_eq!(0.1, r.b, epsilon = EPSILON);
        }

        {
            let stripes = Rc::new(Stripes::new());

            let mut mat = Material::new();
            mat.pattern = Some(stripes);
            mat.ambient = 1.0;
            mat.diffuse = 0.0;
            mat.specular = 0.0;

            let eyev = Vec3::neg_z_axis();
            let normalv = Vec3::neg_z_axis();
            let light = PointLight::from(Vec3::from(0.0, 0.0, -10.0), color::WHITE);

            let c1 = lighting(
                &mat,
                Rc::clone(&obj),
                &light,
                &Vec3::from(0.9, 0.0, 0.0),
                &eyev,
                &normalv,
                false,
            );
            assert_eq!(color::WHITE, c1);

            let c2 = lighting(
                &mat,
                Rc::clone(&obj),
                &light,
                &Vec3::from(1.1, 0.0, 0.0),
                &eyev,
                &normalv,
                false,
            );
            assert_eq!(color::BLACK, c2);
        }
    }

    #[test]
    fn test_camera_pixel_size() {
        {
            let c = Camera::from(200, 125, PI / 2.0);
            assert_relative_eq!(0.01, c.pixel_size);
        }

        {
            let c = Camera::from(125, 200, PI / 2.0);
            assert_relative_eq!(0.01, c.pixel_size);
        }
    }

    #[test]
    fn test_ray_for_pixel() {
        let pi_over_2 = PI / 2.0;
        let pi_over_4 = PI / 4.0;
        let sqrt_2_over_2 = 2.0_f32.sqrt() / 2.0;

        {
            let c = Camera::from(201, 101, pi_over_2);
            let r = ray_for_pixel(&c, 100, 50);

            assert_eq!(Vec3::zero(), r.origin);

            assert_relative_eq!(0.0, r.direction.x, epsilon = EPSILON);
            assert_relative_eq!(0.0, r.direction.y, epsilon = EPSILON);
            assert_relative_eq!(-1.0, r.direction.z, epsilon = EPSILON);
        }

        {
            let c = Camera::from(201, 101, pi_over_2);
            let r = ray_for_pixel(&c, 0, 0);

            assert_eq!(Vec3::zero(), r.origin);

            assert_relative_eq!(0.66519, r.direction.x, epsilon = EPSILON);
            assert_relative_eq!(0.33259, r.direction.y, epsilon = EPSILON);
            assert_relative_eq!(-0.66851, r.direction.z, epsilon = EPSILON);
        }

        {
            let mut c = Camera::from(201, 101, pi_over_2);
            c.transform = Matrix4::op_rotate_y(pi_over_4) * Matrix4::op_translate(0.0, -2.0, 5.0);

            let r = ray_for_pixel(&c, 100, 50);

            assert_relative_eq!(0.0, r.origin.x, epsilon = EPSILON);
            assert_relative_eq!(2.0, r.origin.y, epsilon = EPSILON);
            assert_relative_eq!(-5.0, r.origin.z, epsilon = EPSILON);

            assert_relative_eq!(sqrt_2_over_2, r.direction.x, epsilon = EPSILON);
            assert_relative_eq!(0.0, r.direction.y, epsilon = EPSILON);
            assert_relative_eq!(-sqrt_2_over_2, r.direction.z, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_render() {
        let mut w = World::new();
        w.set_light(PointLight::from(
            Vec3::from(-10.0, 10.0, -10.0),
            color::WHITE,
        ));

        let mut s1 = Entity::sphere();
        s1.material.color = Color::from(0.8, 1.0, 0.6);
        s1.material.diffuse = 0.7;
        s1.material.specular = 0.2;

        let mut s2 = Entity::sphere();
        s2.transform = Matrix4::op_scale(0.5, 0.5, 0.5);

        let _ = w.add_entity(s1);
        let _ = w.add_entity(s2);

        let pi_over_2 = PI / 2.0;

        let from = Vec3::from(0.0, 0.0, -5.0);
        let to = Vec3::zero();
        let up = Vec3::y_axis();

        let mut c = Camera::from(11, 11, pi_over_2);
        c.transform = Matrix4::view_transform(&from, &to, &up);

        let image = render(&c, &w);

        let p = image.pixel_at(5, 5).expect("pixel exists");

        assert_relative_eq!(0.38066, p.r, epsilon = EPSILON);
        assert_relative_eq!(0.47583, p.g, epsilon = EPSILON);
        assert_relative_eq!(0.2855, p.b, epsilon = EPSILON);
    }

    #[test]
    fn test_in_shadow() {
        let mut w = World::new();
        w.set_light(PointLight::from(
            Vec3::from(-10.0, 10.0, -10.0),
            color::WHITE,
        ));

        let mut s1 = Entity::sphere();
        s1.material.color = Color::from(0.8, 1.0, 0.6);
        s1.material.diffuse = 0.7;
        s1.material.specular = 0.2;

        let mut s2 = Entity::sphere();
        s2.transform = Matrix4::op_scale(0.5, 0.5, 0.5);

        let _ = w.add_entity(s1);
        let _ = w.add_entity(s2);

        {
            let p = Vec3::from(0.0, 10.0, 0.0);
            assert!(!is_shadowed(&w, &p));
        }

        {
            let p = Vec3::from(10.0, -10.0, 10.0);
            assert!(is_shadowed(&w, &p));
        }

        {
            let p = Vec3::from(-20.0, 20.0, -20.0);
            assert!(!is_shadowed(&w, &p));
        }

        {
            let p = Vec3::from(-2.0, 2.0, -2.0);
            assert!(!is_shadowed(&w, &p));
        }
    }

    #[test]
    fn test_stripes_pattern_at() {
        {
            let p = Stripes::new();

            assert_eq!(color::WHITE, p.pattern_at(&Vec3::zero()));
            assert_eq!(color::WHITE, p.pattern_at(&Vec3::y_axis()));
            assert_eq!(color::WHITE, p.pattern_at(&Vec3::from(0.0, 2.0, 0.0)));
        }

        {
            let p = Stripes::new();

            assert_eq!(color::WHITE, p.pattern_at(&Vec3::zero()));
            assert_eq!(color::WHITE, p.pattern_at(&Vec3::z_axis()));
            assert_eq!(color::WHITE, p.pattern_at(&Vec3::from(0.0, 0.0, 2.0)));
        }

        {
            let p = Stripes::new();

            assert_eq!(color::WHITE, p.pattern_at(&Vec3::zero()));
            assert_eq!(color::WHITE, p.pattern_at(&Vec3::from(0.9, 0.0, 0.0)));
            assert_eq!(color::BLACK, p.pattern_at(&Vec3::x_axis()));
            assert_eq!(color::BLACK, p.pattern_at(&Vec3::from(-0.1, 0.0, 0.0)));
            assert_eq!(color::BLACK, p.pattern_at(&Vec3::neg_x_axis()));
            assert_eq!(color::WHITE, p.pattern_at(&Vec3::from(-1.1, 0.0, 0.0)));
        }
    }

    #[test]
    fn test_pattern_at_object() {
        {
            let mut obj = Entity::sphere();
            obj.transform = Matrix4::op_scale(2.0, 2.0, 2.0);
            let obj = Rc::new(RefCell::new(obj));

            let p = Stripes::new();

            let c = p.pattern_at_object(obj, &Vec3::from(1.5, 0.0, 0.0));
            assert_eq!(color::WHITE, c);
        }

        {
            let obj = Rc::new(RefCell::new(Entity::sphere()));

            let mut p = Stripes::new();
            p.transform = Matrix4::op_scale(2.0, 2.0, 2.0);

            let c = p.pattern_at_object(obj, &Vec3::from(1.5, 0.0, 0.0));
            assert_eq!(color::WHITE, c);
        }

        {
            let mut obj = Entity::sphere();
            obj.transform = Matrix4::op_scale(2.0, 2.0, 2.0);
            let obj = Rc::new(RefCell::new(obj));

            let mut p = Stripes::new();
            p.transform = Matrix4::op_translate(0.5, 0.0, 0.0);

            let c = p.pattern_at_object(obj, &Vec3::from(2.5, 0.0, 0.0));
            assert_eq!(color::WHITE, c);
        }
    }

    #[test]
    fn test_gradient_pattern_at() {
        let g = Gradient::new();

        {
            let c = g.pattern_at(&Vec3::zero());
            assert_eq!(color::WHITE, c);
        }

        {
            let c = g.pattern_at(&Vec3::from(0.25, 0.0, 0.0));
            assert_eq!(Color::from(0.75, 0.75, 0.75), c);
        }

        {
            let c = g.pattern_at(&Vec3::from(0.5, 0.0, 0.0));
            assert_eq!(Color::from(0.5, 0.5, 0.5), c);
        }

        {
            let c = g.pattern_at(&Vec3::from(0.75, 0.0, 0.0));
            assert_eq!(Color::from(0.25, 0.25, 0.25), c);
        }
    }

    #[test]
    fn test_rings_pattern_at() {
        let r = Rings::new();

        {
            let c = r.pattern_at(&Vec3::zero());
            assert_eq!(color::WHITE, c);
        }

        {
            let c = r.pattern_at(&Vec3::x_axis());
            assert_eq!(color::BLACK, c);
        }

        {
            let c = r.pattern_at(&Vec3::z_axis());
            assert_eq!(color::BLACK, c);
        }

        {
            let c = r.pattern_at(&Vec3::from(0.708, 0.0, 0.708));
            assert_eq!(color::BLACK, c);
        }
    }

    #[test]
    fn test_checkers_pattern_at() {
        let p = Checkers::new();

        {
            let c = p.pattern_at(&Vec3::zero());
            assert_eq!(color::WHITE, c);

            let c = p.pattern_at(&Vec3::from(0.99, 0.0, 0.0));
            assert_eq!(color::WHITE, c);

            let c = p.pattern_at(&Vec3::from(1.01, 0.0, 0.0));
            assert_eq!(color::BLACK, c);
        }

        {
            let c = p.pattern_at(&Vec3::zero());
            assert_eq!(color::WHITE, c);

            let c = p.pattern_at(&Vec3::from(0.0, 0.99, 0.0));
            assert_eq!(color::WHITE, c);

            let c = p.pattern_at(&Vec3::from(0.0, 1.01, 0.0));
            assert_eq!(color::BLACK, c);
        }

        {
            let c = p.pattern_at(&Vec3::zero());
            assert_eq!(color::WHITE, c);

            let c = p.pattern_at(&Vec3::from(0.0, 0.0, 0.99));
            assert_eq!(color::WHITE, c);

            let c = p.pattern_at(&Vec3::from(0.0, 0.0, 1.01));
            assert_eq!(color::BLACK, c);
        }
    }

    #[test]
    fn test_shade_hit() {
        let sqrt_2 = 2.0_f32.sqrt();
        let sqrt_2_over_2 = sqrt_2 / 2.0;

        let mut w = World::new();
        w.set_light(PointLight::from(
            Vec3::from(-10.0, 10.0, -10.0),
            color::WHITE,
        ));

        let mut s1 = Entity::sphere();
        s1.material.color = Color::from(0.8, 1.0, 0.6);
        s1.material.diffuse = 0.7;
        s1.material.specular = 0.2;

        let mut s2 = Entity::sphere();
        s2.transform = Matrix4::op_scale(0.5, 0.5, 0.5);

        let s1 = w.add_entity(s1);
        let s2 = w.add_entity(s2);

        {
            let r = Ray::from(Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis());

            let hit = Hit::from(4.0, Rc::clone(&s1));
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let i = prepare_computations(hit, &r, hit_test.hits());

            let c = shade_hit(&w, &i, REFLECT_LIMIT);

            assert_relative_eq!(0.38066, c.r, epsilon = EPSILON);
            assert_relative_eq!(0.47583, c.g, epsilon = EPSILON);
            assert_relative_eq!(0.2855, c.b, epsilon = EPSILON);
        }

        {
            w.set_light(PointLight::from(Vec3::from(0.0, 0.25, 0.0), color::WHITE));
            let r = Ray::from(Vec3::zero(), Vec3::z_axis());

            let hit = Hit::from(0.5, Rc::clone(&s2));
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let i = prepare_computations(hit, &r, hit_test.hits());
            let c = shade_hit(&w, &i, REFLECT_LIMIT);

            assert_relative_eq!(0.90498, c.r, epsilon = EPSILON);
            assert_relative_eq!(0.90498, c.g, epsilon = EPSILON);
            assert_relative_eq!(0.90498, c.b, epsilon = EPSILON);
        }

        {
            let mut w = World::new();
            w.set_light(PointLight::from(Vec3::from(0.0, 0.0, -10.0), color::WHITE));

            let s1 = Entity::sphere();

            let mut s2 = Entity::sphere();
            s2.transform = Matrix4::op_translate(0.0, 0.0, 10.0);

            let _ = w.add_entity(s1);
            let s2 = w.add_entity(s2);

            let r = Ray::from(Vec3::from(0.0, 0.0, 5.0), Vec3::z_axis());

            let hit = Hit::from(4.0, Rc::clone(&s2));
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let i = prepare_computations(hit, &r, hit_test.hits());
            let c = shade_hit(&w, &i, REFLECT_LIMIT);

            assert_eq!(0.1, c.r);
            assert_eq!(0.1, c.g);
            assert_eq!(0.1, c.b);
        }

        {
            let mut w = World::new();
            w.set_light(PointLight::from(
                Vec3::from(-10.0, 10.0, -10.0),
                color::WHITE,
            ));

            let mut s1 = Entity::sphere();
            s1.material.color = Color::from(0.8, 1.0, 0.6);
            s1.material.diffuse = 0.7;
            s1.material.specular = 0.2;

            let mut s2 = Entity::sphere();
            s2.transform = Matrix4::op_scale(0.5, 0.5, 0.5);

            let _ = w.add_entity(s1);
            let _ = w.add_entity(s2);

            let mut p = Entity::plane();
            p.material.reflective = 0.5;
            p.transform = Matrix4::op_translate(0.0, -1.0, 0.0);

            let p = w.add_entity(p);

            let r = Ray::from(
                Vec3::from(0.0, 0.0, -3.0),
                Vec3::from(0.0, -sqrt_2_over_2, sqrt_2_over_2),
            );

            let hit = Hit::from(sqrt_2, p);
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let i = prepare_computations(hit, &r, hit_test.hits());

            let c = shade_hit(&w, &i, REFLECT_LIMIT);
            assert_relative_eq!(0.87677, c.r, epsilon = EPSILON);
            assert_relative_eq!(0.92436, c.g, epsilon = EPSILON);
            assert_relative_eq!(0.82918, c.b, epsilon = EPSILON);
        }

        {
            let mut w = World::new();
            w.set_light(PointLight::from(
                Vec3::from(-10.0, 10.0, -10.0),
                color::WHITE,
            ));

            let mut s1 = Entity::sphere();
            s1.material.color = Color::from(0.8, 1.0, 0.6);
            s1.material.diffuse = 0.7;
            s1.material.specular = 0.2;

            let mut s2 = Entity::sphere();
            s2.transform = Matrix4::op_scale(0.5, 0.5, 0.5);

            let _ = w.add_entity(s1);
            let _ = w.add_entity(s2);

            let mut floor = Entity::plane();
            floor.transform = Matrix4::op_translate(0.0, -1.0, 0.0);
            floor.material.transparency = 0.5;
            floor.material.refractive_index = 1.5;

            let floor = w.add_entity(floor);

            let mut ball = Entity::sphere();
            ball.material.color = color::RED;
            ball.material.ambient = 0.5;
            ball.transform = Matrix4::op_translate(0.0, -3.5, -0.5);

            let _ = w.add_entity(ball);

            let r = Ray::from(
                Vec3::from(0.0, 0.0, -3.0),
                Vec3::from(0.0, -sqrt_2_over_2, sqrt_2_over_2),
            );

            let hit = Hit::from(sqrt_2, floor);
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let i = prepare_computations(hit, &r, hit_test.hits());

            let c = shade_hit(&w, &i, REFLECT_LIMIT);
            assert_relative_eq!(0.93642, c.r, epsilon = EPSILON);
            assert_relative_eq!(0.68642, c.g, epsilon = EPSILON);
            assert_relative_eq!(0.68642, c.b, epsilon = EPSILON);
        }

        {
            let mut w = World::new();
            w.set_light(PointLight::from(
                Vec3::from(-10.0, 10.0, -10.0),
                color::WHITE,
            ));

            let mut s1 = Entity::sphere();
            s1.material.color = Color::from(0.8, 1.0, 0.6);
            s1.material.diffuse = 0.7;
            s1.material.specular = 0.2;

            let mut s2 = Entity::sphere();
            s2.transform = Matrix4::op_scale(0.5, 0.5, 0.5);

            let _ = w.add_entity(s1);
            let _ = w.add_entity(s2);

            let mut floor = Entity::plane();
            floor.transform = Matrix4::op_translate(0.0, -1.0, 0.0);
            floor.material.reflective = 0.5;
            floor.material.transparency = 0.5;
            floor.material.refractive_index = 1.5;

            let floor = w.add_entity(floor);

            let mut ball = Entity::sphere();
            ball.material.color = color::RED;
            ball.material.ambient = 0.5;
            ball.transform = Matrix4::op_translate(0.0, -3.5, -0.5);

            let _ = w.add_entity(ball);

            let r = Ray::from(
                Vec3::from(0.0, 0.0, -3.0),
                Vec3::from(0.0, -sqrt_2_over_2, sqrt_2_over_2),
            );

            let hit = Hit::from(sqrt_2, floor);
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let i = prepare_computations(hit, &r, hit_test.hits());

            let c = shade_hit(&w, &i, REFLECT_LIMIT);
            assert_relative_eq!(0.93391, c.r, epsilon = EPSILON);
            assert_relative_eq!(0.69643, c.g, epsilon = EPSILON);
            assert_relative_eq!(0.69243, c.b, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_color_at() {
        let mut w = World::new();
        w.set_light(PointLight::from(
            Vec3::from(-10.0, 10.0, -10.0),
            color::WHITE,
        ));

        let mut s1 = Entity::sphere();
        s1.material.color = Color::from(0.8, 1.0, 0.6);
        s1.material.diffuse = 0.7;
        s1.material.specular = 0.2;

        let mut s2 = Entity::sphere();
        s2.transform = Matrix4::op_scale(0.5, 0.5, 0.5);

        let _ = w.add_entity(s1);
        let _ = w.add_entity(s2);

        {
            let r = Ray::from(Vec3::from(0.0, 0.0, -5.0), Vec3::y_axis());

            let c = color_at(&w, &r, REFLECT_LIMIT);

            assert_eq!(0.0, c.r);
            assert_eq!(0.0, c.g);
            assert_eq!(0.0, c.b);
        }

        {
            let r = Ray::from(Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis());

            let c = color_at(&w, &r, REFLECT_LIMIT);

            assert_relative_eq!(0.38066, c.r, epsilon = EPSILON);
            assert_relative_eq!(0.47583, c.g, epsilon = EPSILON);
            assert_relative_eq!(0.2855, c.b, epsilon = EPSILON);
        }

        {
            let mut w = World::new();
            w.set_light(PointLight::from(
                Vec3::from(-10.0, 10.0, -10.0),
                color::WHITE,
            ));

            let mut s1 = Entity::sphere();
            s1.material.color = Color::from(0.8, 1.0, 0.6);
            s1.material.ambient = 1.0;
            s1.material.diffuse = 0.7;
            s1.material.specular = 0.2;

            let mut s2 = Entity::sphere();
            s2.transform = Matrix4::op_scale(0.5, 0.5, 0.5);
            s2.material.ambient = 1.0;

            let _ = w.add_entity(s1);
            let s2 = w.add_entity(s2);

            let r = Ray::from(Vec3::from(0.0, 0.0, 0.75), Vec3::neg_z_axis());

            let c = color_at(&w, &r, REFLECT_LIMIT);

            assert_eq!(s2.borrow().material.color, c);
        }

        {
            let mut w = World::new();
            w.set_light(PointLight::from(Vec3::zero(), color::WHITE));

            let mut p1 = Entity::plane();
            p1.material.reflective = 1.0;
            p1.transform = Matrix4::op_translate(0.0, -1.0, 0.0);

            let mut p2 = Entity::sphere();
            p2.material.reflective = 1.0;
            p2.transform = Matrix4::op_translate(0.0, 1.0, 0.0);

            let _ = w.add_entity(p1);
            let _ = w.add_entity(p2);

            let r = Ray::from(Vec3::origin(), Vec3::y_axis());

            let _ = color_at(&w, &r, REFLECT_LIMIT);
        }
    }

    #[test]
    fn test_reflected_color() {
        let sqrt_2 = 2.0_f32.sqrt();
        let sqrt_2_over_2 = sqrt_2 / 2.0;

        {
            let mut w = World::new();
            w.set_light(PointLight::from(
                Vec3::from(-10.0, 10.0, -10.0),
                color::WHITE,
            ));

            let mut s1 = Entity::sphere();
            s1.material.color = Color::from(0.8, 1.0, 0.6);
            s1.material.diffuse = 0.7;
            s1.material.specular = 0.2;

            let mut s2 = Entity::sphere();
            s2.transform = Matrix4::op_scale(0.5, 0.5, 0.5);
            s2.material.ambient = 1.0;

            let _ = w.add_entity(s1);
            let s2 = w.add_entity(s2);

            let r = Ray::from(Vec3::zero(), Vec3::z_axis());

            let hit = Hit::from(1.0, s2);
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let i = prepare_computations(hit, &r, hit_test.hits());

            let c = reflected_color(&w, &i, REFLECT_LIMIT);
            assert_eq!(0.0, c.r);
            assert_eq!(0.0, c.g);
            assert_eq!(0.0, c.b);
        }

        {
            let mut w = World::new();
            w.set_light(PointLight::from(
                Vec3::from(-10.0, 10.0, -10.0),
                color::WHITE,
            ));

            let mut s1 = Entity::sphere();
            s1.material.color = Color::from(0.8, 1.0, 0.6);
            s1.material.diffuse = 0.7;
            s1.material.specular = 0.2;

            let mut s2 = Entity::sphere();
            s2.transform = Matrix4::op_scale(0.5, 0.5, 0.5);

            let _ = w.add_entity(s1);
            let _ = w.add_entity(s2);

            let mut p = Entity::plane();
            p.material.reflective = 0.5;
            p.transform = Matrix4::op_translate(0.0, -1.0, 0.0);

            let p = w.add_entity(p);

            let r = Ray::from(
                Vec3::from(0.0, 0.0, -3.0),
                Vec3::from(0.0, -sqrt_2_over_2, sqrt_2_over_2),
            );

            let hit = Hit::from(sqrt_2, p);
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let i = prepare_computations(hit, &r, hit_test.hits());

            let c = reflected_color(&w, &i, REFLECT_LIMIT);
            assert_relative_eq!(0.19032, c.r, epsilon = EPSILON);
            assert_relative_eq!(0.2379, c.g, epsilon = EPSILON);
            assert_relative_eq!(0.14274, c.b, epsilon = EPSILON);
        }

        {
            let mut w = World::new();
            w.set_light(PointLight::from(
                Vec3::from(-10.0, 10.0, -10.0),
                color::WHITE,
            ));

            let mut s1 = Entity::sphere();
            s1.material.color = Color::from(0.8, 1.0, 0.6);
            s1.material.diffuse = 0.7;
            s1.material.specular = 0.2;

            let mut s2 = Entity::sphere();
            s2.transform = Matrix4::op_scale(0.5, 0.5, 0.5);

            let _ = w.add_entity(s1);
            let _ = w.add_entity(s2);

            let mut p = Entity::plane();
            p.material.reflective = 0.5;
            p.transform = Matrix4::op_translate(0.0, -1.0, 0.0);

            let p = w.add_entity(p);

            let r = Ray::from(
                Vec3::from(0.0, 0.0, -3.0),
                Vec3::from(0.0, -sqrt_2_over_2, sqrt_2_over_2),
            );

            let hit = Hit::from(sqrt_2, p);
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let i = prepare_computations(hit, &r, hit_test.hits());

            let c = reflected_color(&w, &i, 0);
            assert_eq!(color::BLACK, c);
        }
    }

    #[test]
    fn test_refracted_color() {
        let sqrt_2_over_2 = 2.0_f32.sqrt() / 2.0;

        let mut w = World::new();
        w.set_light(PointLight::from(
            Vec3::from(-10.0, 10.0, -10.0),
            color::WHITE,
        ));

        let mut s1 = Entity::sphere();
        s1.material.color = Color::from(0.8, 1.0, 0.6);
        s1.material.diffuse = 0.7;
        s1.material.specular = 0.2;

        let mut s2 = Entity::sphere();
        s2.transform = Matrix4::op_scale(0.5, 0.5, 0.5);

        let s1 = w.add_entity(s1);
        let _ = w.add_entity(s2);

        {
            let r = Ray::from(Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis());

            let hit_test = HitTest::builder()
                .add_hit(Hit::from(4.0, Rc::clone(&s1)))
                .add_hit(Hit::from(6.0, Rc::clone(&s1)))
                .build();

            let hit = &hit_test.hits()[0];
            let i = prepare_computations(hit, &r, hit_test.hits());

            let c = refracted_color(&w, &i, REFLECT_LIMIT);
            assert_eq!(color::BLACK, c);
        }

        {
            let r = Ray::from(Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis());

            let hit_test = HitTest::builder()
                .add_hit(Hit::from(4.0, Rc::clone(&s1)))
                .add_hit(Hit::from(6.0, Rc::clone(&s1)))
                .build();

            let hit = &hit_test.hits()[0];
            let i = prepare_computations(hit, &r, hit_test.hits());

            let c = refracted_color(&w, &i, 0);
            assert_eq!(color::BLACK, c);
        }

        {
            let mut w = World::new();
            w.set_light(PointLight::from(
                Vec3::from(-10.0, 10.0, -10.0),
                color::WHITE,
            ));

            let mut s1 = Entity::sphere();
            s1.material.color = Color::from(0.8, 1.0, 0.6);
            s1.material.diffuse = 0.7;
            s1.material.specular = 0.2;
            s1.material.transparency = 1.0;
            s1.material.refractive_index = 1.5;

            let mut s2 = Entity::sphere();
            s2.transform = Matrix4::op_scale(0.5, 0.5, 0.5);

            let s1 = w.add_entity(s1);
            let _ = w.add_entity(s2);

            let r = Ray::from(Vec3::from(0.0, 0.0, sqrt_2_over_2), Vec3::y_axis());

            let hit_test = HitTest::builder()
                .add_hit(Hit::from(-sqrt_2_over_2, Rc::clone(&s1)))
                .add_hit(Hit::from(sqrt_2_over_2, Rc::clone(&s1)))
                .build();

            let hit = &hit_test.hits()[1];
            let i = prepare_computations(hit, &r, hit_test.hits());

            let c = refracted_color(&w, &i, 0);
            assert_eq!(color::BLACK, c);
        }

        {
            let mut w = World::new();
            w.set_light(PointLight::from(
                Vec3::from(-10.0, 10.0, -10.0),
                color::WHITE,
            ));

            let pattern = Rc::new(TestPattern::new());

            let mut s1 = Entity::sphere();
            s1.material.color = Color::from(0.8, 1.0, 0.6);
            s1.material.diffuse = 0.7;
            s1.material.specular = 0.2;
            s1.material.ambient = 1.0;
            s1.material.pattern = Some(pattern);

            let mut s2 = Entity::sphere();
            s2.transform = Matrix4::op_scale(0.5, 0.5, 0.5);
            s2.material.transparency = 1.0;
            s2.material.refractive_index = 1.5;

            let s1 = w.add_entity(s1);
            let s2 = w.add_entity(s2);

            let r = Ray::from(Vec3::from(0.0, 0.0, 0.1), Vec3::y_axis());

            let hit_test = HitTest::builder()
                .add_hit(Hit::from(-0.9899, Rc::clone(&s1)))
                .add_hit(Hit::from(-0.4899, Rc::clone(&s2)))
                .add_hit(Hit::from(0.4899, Rc::clone(&s2)))
                .add_hit(Hit::from(0.9899, Rc::clone(&s1)))
                .build();

            let hit = &hit_test.hits()[2];
            let i = prepare_computations(hit, &r, hit_test.hits());

            let c = refracted_color(&w, &i, REFLECT_LIMIT);
            assert_relative_eq!(0.0, c.r, epsilon = EPSILON);
            assert_relative_eq!(0.99888, c.g, epsilon = 0.1); // epsilon is different by one tenth
            assert_relative_eq!(0.04725, c.b, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_schlick() {
        let sqrt_2_over_2 = 2.0_f32.sqrt() / 2.0;

        {
            let s = Rc::new(RefCell::new(Entity::glass_sphere()));
            let r = Ray::from(Vec3::from(0.0, 0.0, -sqrt_2_over_2), Vec3::y_axis());

            let hit_test = HitTest::builder()
                .add_hit(Hit::from(-sqrt_2_over_2, Rc::clone(&s)))
                .add_hit(Hit::from(sqrt_2_over_2, Rc::clone(&s)))
                .build();

            let i = prepare_computations(&hit_test.hits()[1], &r, hit_test.hits());

            let reflectance = schlick(&i);
            assert_eq!(1.0, reflectance);
        }

        {
            let s = Rc::new(RefCell::new(Entity::glass_sphere()));
            let r = Ray::from(Vec3::origin(), Vec3::y_axis());

            let hit_test = HitTest::builder()
                .add_hit(Hit::from(-1.0, Rc::clone(&s)))
                .add_hit(Hit::from(1.0, Rc::clone(&s)))
                .build();

            let i = prepare_computations(&hit_test.hits()[1], &r, hit_test.hits());

            let reflectance = schlick(&i);
            assert_relative_eq!(0.04, reflectance, epsilon = EPSILON);
        }

        {
            let s = Rc::new(RefCell::new(Entity::glass_sphere()));
            let r = Ray::from(Vec3::from(0.0, 0.99, -2.0), Vec3::z_axis());

            let hit_test = HitTest::builder().add_hit(Hit::from(1.8589, s)).build();

            let i = prepare_computations(&hit_test.hits()[0], &r, hit_test.hits());

            let reflectance = schlick(&i);
            assert_relative_eq!(0.48873, reflectance, epsilon = EPSILON);
        }
    }
}
