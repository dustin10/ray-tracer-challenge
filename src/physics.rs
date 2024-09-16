use crate::{
    entity::{Entity, World},
    math::{Matrix4, Ray, Vec3},
    EPSILON,
};

use std::{cell::RefCell, rc::Rc};

/// Enumeration of the supported shapes which can be used to render objects in a scene.
pub enum Shape {
    Sphere {
        origin: Vec3,
        radius: f32,
    },
    Plane,
    Cube,
    Cylinder {
        minimum: f32,
        maximum: f32,
        closed: bool,
    },
    Cone {
        minimum: f32,
        maximum: f32,
        closed: bool,
    },
    Group,
    Triangle {
        p1: Vec3,
        p2: Vec3,
        p3: Vec3,
        e1: Vec3,
        e2: Vec3,
        normal: Vec3,
    },
    SmoothTriangle {
        p1: Vec3,
        p2: Vec3,
        p3: Vec3,
        e1: Vec3,
        e2: Vec3,
        n1: Vec3,
        n2: Vec3,
        n3: Vec3,
    },
}

/// Intermediate data that is the result of the local intersection of a shape with a ray. The
/// `uv` value will be present in the case of a triangle only.
#[derive(Default)]
struct LocalIntersect {
    ts: Vec<f32>,
    uv: Option<(f32, f32)>,
}

impl LocalIntersect {
    /// Creates a new [LocalIntersect] fron the given values.
    fn from(ts: Vec<f32>, uv: Option<(f32, f32)>) -> Self {
        Self { ts, uv }
    }
}

/// Calculates the positions of intersections along the [Ray] for the [Shape] if any.
fn local_intersect(shape: &Shape, ray: &Ray) -> LocalIntersect {
    match shape {
        Shape::Sphere { origin, .. } => local_intersect_sphere(ray, origin),
        Shape::Plane => local_intersect_plane(ray),
        Shape::Cube => local_intersect_cube(ray),
        Shape::Cylinder {
            minimum,
            maximum,
            closed,
        } => local_intersect_cylinder(ray, *minimum, *maximum, *closed),
        Shape::Cone {
            minimum,
            maximum,
            closed,
        } => local_intersect_cone(ray, *minimum, *maximum, *closed),
        Shape::Group => panic!("local_intersect is not valid for Group"),
        Shape::Triangle { p1, e1, e2, .. } | Shape::SmoothTriangle { p1, e1, e2, .. } => {
            local_intersect_triangle(ray, p1, e1, e2)
        }
    }
}

/// Calculates the normal at the given point on the [Shape].
fn local_normal_at(shape: &Shape, point: &Vec3, hit: Option<&Hit>) -> Vec3 {
    const ORIGIN: Vec3 = Vec3::zero();

    match shape {
        Shape::Sphere { .. } => *point - ORIGIN,
        Shape::Plane => Vec3::y_axis(),
        Shape::Cube => {
            let maxc = point.x.abs().max(point.y.abs().max(point.z.abs()));

            if maxc == point.x.abs() {
                Vec3::from(point.x, 0.0, 0.0)
            } else if maxc == point.y.abs() {
                Vec3::from(0.0, point.y, 0.0)
            } else {
                Vec3::from(0.0, 0.0, point.z)
            }
        }
        Shape::Cylinder {
            minimum, maximum, ..
        } => {
            let dist = point.x.powi(2) + point.z.powi(2);
            if dist < 1.0 && point.y >= maximum - EPSILON {
                Vec3::y_axis()
            } else if dist < 1.0 && point.y <= minimum + EPSILON {
                Vec3::neg_y_axis()
            } else {
                Vec3::from(point.x, 0.0, point.z)
            }
        }
        Shape::Cone {
            minimum, maximum, ..
        } => {
            let dist = point.x.powi(2) + point.z.powi(2);
            if dist < 1.0 && point.y >= maximum - EPSILON {
                Vec3::y_axis()
            } else if dist < 1.0 && point.y <= minimum + EPSILON {
                Vec3::neg_y_axis()
            } else {
                let mut y = (point.x.powi(2) + point.z.powi(2)).sqrt();
                if point.y > 0.0 {
                    y = -y;
                }

                Vec3::from(point.x, y, point.z)
            }
        }
        Shape::Group => panic!("local_normal_at is not valid for Group"),
        Shape::Triangle { normal, .. } => *normal,
        Shape::SmoothTriangle { n1, n2, n3, .. } => {
            if let Some((u, v)) = hit.and_then(|h| h.uv) {
                Vec3::from_scaled(n2, u)
                    + Vec3::from_scaled(n3, v)
                    + Vec3::from_scaled(n1, 1.0 - u - v)
            } else {
                panic!("expected uv");
            }
        }
    }
}

/// Calculates the positions of intersections along the [Ray] for a sphere.
fn local_intersect_sphere(ray: &Ray, origin: &Vec3) -> LocalIntersect {
    let shape_to_ray = ray.origin - *origin;

    let a = ray.direction.dot(&ray.direction);
    let b = 2.0 * ray.direction.dot(&shape_to_ray);
    let c = shape_to_ray.dot(&shape_to_ray) - 1.0;

    let disc = (b * b) - (4.0 * a * c);

    let mut hits = Vec::with_capacity(2);
    if disc < 0.0 {
        return LocalIntersect::default();
    }

    let t1 = (-b - disc.sqrt()) / (2.0 * a);
    let t2 = (-b + disc.sqrt()) / (2.0 * a);

    hits.push(t1);
    hits.push(t2);

    LocalIntersect::from(hits, None)
}

/// Calculates the positions of intersections along the [Ray] for a plane.
fn local_intersect_plane(ray: &Ray) -> LocalIntersect {
    let mut hits = Vec::with_capacity(1);
    if ray.direction.y.abs() < EPSILON {
        return LocalIntersect::default();
    }

    let t = -ray.origin.y / ray.direction.y;
    hits.push(t);

    LocalIntersect::from(hits, None)
}

/// Calculates the positions of intersections along the [Ray] for a cube.
fn local_intersect_cube(ray: &Ray) -> LocalIntersect {
    let (xtmin, xtmax) = check_axis_cube(ray.origin.x, ray.direction.x);
    let (ytmin, ytmax) = check_axis_cube(ray.origin.y, ray.direction.y);
    let (ztmin, ztmax) = check_axis_cube(ray.origin.z, ray.direction.z);

    let tmin = xtmin.max(ytmin).max(ztmin);
    let tmax = xtmax.min(ytmax).min(ztmax);

    let mut hits = Vec::with_capacity(1);
    if tmin > tmax {
        return LocalIntersect::default();
    }

    hits.push(tmin);
    hits.push(tmax);

    LocalIntersect::from(hits, None)
}

fn check_axis_cube(o: f32, d: f32) -> (f32, f32) {
    let tmin_numerator = -1.0 - o;
    let tmax_numerator = 1.0 - o;

    let (mut tmin, mut tmax) = if d.abs() > EPSILON {
        (tmin_numerator / d, tmax_numerator / d)
    } else {
        (
            tmin_numerator * f32::INFINITY,
            tmax_numerator * f32::INFINITY,
        )
    };

    if tmin > tmax {
        std::mem::swap(&mut tmin, &mut tmax);
    }

    (tmin, tmax)
}

/// Calculates the positions of intersections along the [Ray] for a cylinder.
fn local_intersect_cylinder(ray: &Ray, minimum: f32, maximum: f32, closed: bool) -> LocalIntersect {
    let mut hits = Vec::new();

    let a = ray.direction.x.powi(2) + ray.direction.z.powi(2);
    if a.abs() > EPSILON {
        let b = (2.0 * ray.origin.x * ray.direction.x) + (2.0 * ray.origin.z * ray.direction.z);
        let c = ray.origin.x.powi(2) + ray.origin.z.powi(2) - 1.0;

        let disc = b.powi(2) - (4.0 * a * c);
        if disc < 0.0 {
            return LocalIntersect::default();
        }

        let mut t0 = (-b - disc.sqrt()) / (2.0 * a);
        let mut t1 = (-b + disc.sqrt()) / (2.0 * a);
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }

        let y0 = ray.origin.y + (t0 * ray.direction.y);
        if minimum < y0 && y0 < maximum {
            hits.push(t0);
        }

        let y1 = ray.origin.y + (t1 * ray.direction.y);
        if minimum < y1 && y1 < maximum {
            hits.push(t1);
        }
    }

    let local_intersect = local_intersect_caps_cylinder(ray, minimum, maximum, closed);
    for hit in local_intersect.ts.into_iter() {
        hits.push(hit);
    }

    LocalIntersect::from(hits, None)
}

fn local_intersect_caps_cylinder(
    ray: &Ray,
    minimum: f32,
    maximum: f32,
    closed: bool,
) -> LocalIntersect {
    let mut hits = Vec::with_capacity(2);

    if !closed || ray.direction.y.abs() < EPSILON {
        return LocalIntersect::default();
    }

    let t = (minimum - ray.origin.y) / ray.direction.y;
    if check_cap_cylinder(ray, t) {
        hits.push(t);
    }

    let t = (maximum - ray.origin.y) / ray.direction.y;
    if check_cap_cylinder(ray, t) {
        hits.push(t);
    }

    LocalIntersect::from(hits, None)
}

fn check_cap_cylinder(ray: &Ray, t: f32) -> bool {
    let x = ray.origin.x + (t * ray.direction.x);
    let z = ray.origin.z + (t * ray.direction.z);

    x.powi(2) + z.powi(2) <= 1.0 + EPSILON
}

/// Calculates the positions of intersections along the [Ray] for a cone.
fn local_intersect_cone(ray: &Ray, minimum: f32, maximum: f32, closed: bool) -> LocalIntersect {
    let mut hits = Vec::new();

    let a = ray.direction.x.powi(2) - ray.direction.y.powi(2) + ray.direction.z.powi(2);

    let b = (2.0 * ray.origin.x * ray.direction.x) - (2.0 * ray.origin.y * ray.direction.y)
        + (2.0 * ray.origin.z * ray.direction.z);

    let c = ray.origin.x.powi(2) - ray.origin.y.powi(2) + ray.origin.z.powi(2);

    if a.abs() < EPSILON {
        if b.abs() >= EPSILON {
            let t = -c / (2.0 * b);

            hits.push(t);
        } else {
            return LocalIntersect::default();
        }
    } else {
        let disc = b.powi(2) - (4.0 * a * c);
        if disc < 0.0 {
            return LocalIntersect::default();
        }

        let mut t0 = (-b - disc.sqrt()) / (2.0 * a);
        let mut t1 = (-b + disc.sqrt()) / (2.0 * a);
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }

        let y0 = ray.origin.y + (t0 * ray.direction.y);
        if minimum < y0 && y0 < maximum {
            hits.push(t0);
        }

        let y1 = ray.origin.y + (t1 * ray.direction.y);
        if minimum < y1 && y1 < maximum {
            hits.push(t1);
        }
    }

    let cap_hits = local_intersect_caps_cone(ray, minimum, maximum, closed);
    for hit in cap_hits.into_iter() {
        hits.push(hit);
    }

    LocalIntersect::from(hits, None)
}

fn local_intersect_caps_cone(ray: &Ray, minimum: f32, maximum: f32, closed: bool) -> Vec<f32> {
    let mut hits = Vec::with_capacity(2);

    if !closed || ray.direction.y.abs() < EPSILON {
        return hits;
    }

    let t = (minimum - ray.origin.y) / ray.direction.y;
    if check_cap_cone(ray, t, minimum.abs()) {
        hits.push(t);
    }

    let t = (maximum - ray.origin.y) / ray.direction.y;
    if check_cap_cone(ray, t, maximum.abs()) {
        hits.push(t);
    }

    hits
}

fn check_cap_cone(ray: &Ray, t: f32, y: f32) -> bool {
    let x = ray.origin.x + (t * ray.direction.x);
    let z = ray.origin.z + (t * ray.direction.z);

    x.powi(2) + z.powi(2) <= y + EPSILON
}

/// Calculates the positions of intersections along the [Ray] for a triangle.
fn local_intersect_triangle(ray: &Ray, p1: &Vec3, e1: &Vec3, e2: &Vec3) -> LocalIntersect {
    let dir_cross_e2 = Vec3::from_cross(&ray.direction, e2);
    let det = e1.dot(&dir_cross_e2);

    if det.abs() < EPSILON {
        return LocalIntersect::default();
    }

    let f = 1.0 / det;
    let p1_to_origin = ray.origin - *p1;

    let u = f * p1_to_origin.dot(&dir_cross_e2);
    if !(0.0..=1.0).contains(&u) {
        return LocalIntersect::default();
    }

    let origin_cross_e1 = Vec3::from_cross(&p1_to_origin, e1);

    let v = f * ray.direction.dot(&origin_cross_e1);
    if v < 0.0 || (u + v) > 1.0 {
        return LocalIntersect::default();
    }

    let t = f * e2.dot(&origin_cross_e1);

    LocalIntersect::from(vec![t], Some((u, v)))
}

/// Builder struct that allows for ease of creating a [HitTest] that has any hits sorted with the
/// correct ordering.
#[derive(Clone, Default)]
pub struct HitTestBuilder {
    hits: Vec<Hit>,
}

impl HitTestBuilder {
    /// Creates a new default [HitTestBuilder].
    pub fn new() -> Self {
        HitTestBuilder::default()
    }
    /// Adds a [Hit] to the builder.
    pub fn add_hit(mut self, hit: Hit) -> Self {
        self.hits.push(hit);

        self
    }
    /// Sorts any [Hit]s that were added and then creates the [HitTest].
    pub fn build(mut self) -> HitTest {
        self.hits
            .sort_by(|a, b| a.t.partial_cmp(&b.t).expect("f32 has ordering"));

        HitTest::with_hits(self.hits)
    }
}

/// Represents a hit on an entity of a ray that was cast onto a scene.
#[derive(Clone)]
pub struct Hit {
    pub t: f32,
    pub entity: Rc<RefCell<Entity>>,
    pub uv: Option<(f32, f32)>,
}

impl Hit {
    /// Creates a new [Hit] from the specified location and entity.
    pub fn from(t: f32, entity: Rc<RefCell<Entity>>) -> Self {
        Hit {
            t,
            entity,
            uv: None,
        }
    }
    pub fn on_triangle(t: f32, entity: Rc<RefCell<Entity>>, u: f32, v: f32) -> Self {
        Hit {
            t,
            entity,
            uv: Some((u, v)),
        }
    }
}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.t == other.t && self.entity.borrow().id == other.entity.borrow().id
    }
}

/// Represents the results of a hit test from casting a ray at an entity in a scene.
#[derive(Clone)]
pub struct HitTest {
    hits: Vec<Hit>,
}

impl HitTest {
    /// Creates a new default [HitTestBuilder].
    pub fn builder() -> HitTestBuilder {
        HitTestBuilder::new()
    }
    /// Creates a new empty [HitTest].
    pub fn empty() -> Self {
        Self::with_hits(vec![])
    }
    /// Creates a new [HitTest] initialized with the given hits.
    fn with_hits(hits: Vec<Hit>) -> Self {
        Self { hits }
    }
    /// Returns all of the [Hit]s from the test.
    pub fn hits(&self) -> &Vec<Hit> {
        &self.hits
    }
    /// Returns the valid [Hit] from the hit test if it exists.
    pub fn hit(&self) -> Option<&Hit> {
        for hit in self.hits.iter() {
            if hit.t < 0.0 {
                continue;
            }

            return Some(hit);
        }

        None
    }
}

/// Contains relevant data for an intersection point on an entity by a ray.
#[derive(Clone)]
pub struct Intersection {
    pub t: f32,
    pub entity: Rc<RefCell<Entity>>,
    pub inside: bool,
    pub point: Vec3,
    pub over_point: Vec3,
    pub under_point: Vec3,
    pub eyev: Vec3,
    pub normalv: Vec3,
    pub reflectv: Vec3,
    pub n1: f32,
    pub n2: f32,
}

/// Performs a hit test for the [Ray] against the specified [Entity].
pub fn intersect(ray: &Ray, entity: Rc<RefCell<Entity>>) -> HitTest {
    let e = entity.borrow();

    let inv = Matrix4::from_inverse(&e.transform).expect("matrix is invertible");
    let local_ray = Ray::from_transformed(ray, &inv);

    match e.shape {
        Shape::Group => {
            let mut builder = HitTest::builder();
            for child in e.children.iter() {
                let hit_test = intersect(&local_ray, Rc::clone(child));
                for hit in hit_test.hits() {
                    builder = builder.add_hit(hit.clone());
                }
            }

            builder.build()
        }
        Shape::Triangle { .. } | Shape::SmoothTriangle { .. } => {
            let local_intersect = local_intersect(&entity.borrow().shape, &local_ray);

            let mut builder = HitTest::builder();
            for t in local_intersect.ts {
                let (u, v) = local_intersect.uv.expect("uv is set for triangle");
                builder = builder.add_hit(Hit::on_triangle(t, Rc::clone(&entity), u, v));
            }

            builder.build()
        }
        _ => {
            let local_intersect = local_intersect(&entity.borrow().shape, &local_ray);

            let mut builder = HitTest::builder();
            for t in local_intersect.ts {
                builder = builder.add_hit(Hit::from(t, Rc::clone(&entity)));
            }

            builder.build()
        }
    }
}

/// Computes the normal vector for the given [Entity] at the point in world space.
pub fn normal_at(entity: Rc<RefCell<Entity>>, world_point: &Vec3, hit: &Hit) -> Vec3 {
    let local_point = world_to_object(Rc::clone(&entity), world_point);
    let local_normal = local_normal_at(&entity.borrow().shape, &local_point, Some(hit));

    normal_to_world(entity, &local_normal)
}

/// Reflects the given [Vec3] across the normal vector.
pub fn reflect(v: &Vec3, normal: &Vec3) -> Vec3 {
    let scaled_normal = Vec3::from_scaled(normal, 2.0 * Vec3::dot(v, normal));
    *v - scaled_normal
}

/// Performs a hit test for each entity in the [World] with the given [Ray].
pub fn intersect_at(w: &World, r: &Ray) -> HitTest {
    let mut world_test = HitTest::builder();
    for obj in w.entities() {
        let obj_test = intersect(r, Rc::clone(obj));
        for hit in obj_test.hits {
            world_test = world_test.add_hit(hit);
        }
    }

    world_test.build()
}

/// Given a [Hit] and the [Ray], computes all of the data about the intersetion required for
/// rendering.
pub fn prepare_computations(hit: &Hit, r: &Ray, hits: &Vec<Hit>) -> Intersection {
    let point = Ray::position(r, hit.t);
    let eyev = Vec3::from_scaled(&r.direction, -1.0);
    let mut normalv = normal_at(Rc::clone(&hit.entity), &point, hit);
    let mut inside = false;

    if normalv.dot(&eyev) < 0.0 {
        normalv.scale(-1.0);
        inside = true;
    }

    let over_point = point + Vec3::from_scaled(&normalv, EPSILON);
    let under_point = point - Vec3::from_scaled(&normalv, EPSILON);

    let reflectv = reflect(&r.direction, &normalv);

    let mut n1 = 0.0;
    let mut n2 = 0.0;

    let mut containers: Vec<Rc<RefCell<Entity>>> = Vec::with_capacity(hits.len());
    for h in hits {
        if h == hit {
            if containers.is_empty() {
                n1 = 1.0;
            } else {
                n1 = containers
                    .last()
                    .expect("not empty")
                    .borrow()
                    .material
                    .refractive_index;
            }
        }

        let obj_id = h.entity.borrow().id;
        let position = containers.iter().position(|v| v.borrow().id == obj_id);
        if let Some(idx) = position {
            containers.remove(idx);
        } else {
            containers.push(Rc::clone(&h.entity));
        }

        if h == hit {
            if containers.is_empty() {
                n2 = 1.0;
            } else {
                n2 = containers
                    .last()
                    .expect("not empty")
                    .borrow()
                    .material
                    .refractive_index;
            }

            break;
        }
    }

    Intersection {
        t: hit.t,
        entity: Rc::clone(&hit.entity),
        inside,
        point,
        over_point,
        under_point,
        eyev,
        normalv,
        reflectv,
        n1,
        n2,
    }
}

/// Transforms the point given in world space to object space.
pub fn world_to_object(entity: Rc<RefCell<Entity>>, point: &Vec3) -> Vec3 {
    let e = entity.borrow();

    let mut p = *point;
    if let Some(parent_ref) = &e.parent {
        let parent = parent_ref.upgrade().expect("parent is valid");
        p = world_to_object(Rc::clone(&parent), &p);
    }

    let inv_transform = Matrix4::from_inverse(&e.transform).expect("transform is invertible");

    Matrix4::mul_vec3_point(&inv_transform, &p)
}

/// Transforms the normal vector at the object to world space.
pub fn normal_to_world(entity: Rc<RefCell<Entity>>, normal: &Vec3) -> Vec3 {
    let e = entity.borrow();
    let inv_transform = Matrix4::from_inverse(&e.transform).expect("transform is invertible");
    let inv_transpose = Matrix4::from_transpose(&inv_transform);
    let mut result = Matrix4::mul_vec3_dir(&inv_transpose, normal);
    result.normalize();

    if let Some(parent_ref) = &e.parent {
        result = normal_to_world(parent_ref.upgrade().expect("parent is valid"), &result);
    }

    result
}

#[cfg(test)]
mod tests {
    use crate::{
        color::{self, Color},
        entity::{Entity, World},
        math::{Matrix4, Ray, Vec3},
        physics::{
            local_intersect_cone, local_intersect_cube, local_intersect_cylinder,
            local_intersect_plane, local_normal_at,
        },
        render::PointLight,
        EPSILON,
    };

    use super::{
        intersect, intersect_at, local_intersect, normal_at, normal_to_world, prepare_computations,
        reflect, world_to_object, Hit, HitTest, Shape,
    };

    use approx::assert_relative_eq;
    use std::{
        cell::RefCell,
        f32::consts::{FRAC_1_SQRT_2, PI},
        rc::Rc,
    };

    #[test]
    fn test_normal_at() {
        {
            let s = Rc::new(RefCell::new(Entity::sphere()));
            let hit = Hit::from(1.0, Rc::clone(&s));
            let n = normal_at(s, &Vec3::x_axis(), &hit);

            assert_eq!(1.0, n.x);
            assert_eq!(0.0, n.y);
            assert_eq!(0.0, n.z);
        }

        {
            let s = Rc::new(RefCell::new(Entity::sphere()));
            let hit = Hit::from(1.0, Rc::clone(&s));
            let n = normal_at(s, &Vec3::y_axis(), &hit);

            assert_eq!(0.0, n.x);
            assert_eq!(1.0, n.y);
            assert_eq!(0.0, n.z);
        }

        {
            let s = Rc::new(RefCell::new(Entity::sphere()));
            let hit = Hit::from(1.0, Rc::clone(&s));
            let n = normal_at(s, &Vec3::z_axis(), &hit);

            assert_eq!(0.0, n.x);
            assert_eq!(0.0, n.y);
            assert_eq!(1.0, n.z);
        }

        {
            let sqrt_3_over_3 = 3.0_f32.sqrt() / 3.0;

            let s = Rc::new(RefCell::new(Entity::sphere()));
            let hit = Hit::from(1.0, Rc::clone(&s));
            let n = normal_at(
                s,
                &Vec3::from(sqrt_3_over_3, sqrt_3_over_3, sqrt_3_over_3),
                &hit,
            );

            assert_relative_eq!(sqrt_3_over_3, n.x);
            assert_relative_eq!(sqrt_3_over_3, n.y);
            assert_relative_eq!(sqrt_3_over_3, n.z);
        }

        {
            let mut s = Entity::sphere();
            s.transform = Matrix4::op_translate(0.0, 1.0, 0.0);
            let s = Rc::new(RefCell::new(s));
            let hit = Hit::from(1.0, Rc::clone(&s));

            let n = normal_at(s, &Vec3::from(0.0, 1.70711, -FRAC_1_SQRT_2), &hit);

            assert_eq!(0.0, n.x);
            assert_relative_eq!(FRAC_1_SQRT_2, n.y, epsilon = EPSILON);
            assert_relative_eq!(-FRAC_1_SQRT_2, n.z, epsilon = EPSILON);
        }

        {
            let sqrt_2_over_2 = 2.0_f32.sqrt() / 2.0;

            let scale = Matrix4::op_scale(1.0, 0.5, 1.0);
            let rotate = Matrix4::op_rotate_z(PI / 5.0);

            let mut s = Entity::sphere();
            s.transform = scale * rotate;
            let s = Rc::new(RefCell::new(s));
            let hit = Hit::from(1.0, Rc::clone(&s));

            let n = normal_at(s, &Vec3::from(0.0, sqrt_2_over_2, -sqrt_2_over_2), &hit);

            assert_relative_eq!(0.0, n.x, epsilon = EPSILON);
            assert_relative_eq!(0.97014, n.y, epsilon = EPSILON);
            assert_relative_eq!(-0.24254, n.z, epsilon = EPSILON);
        }

        {
            let g1 = Rc::new(RefCell::new(Entity::group()));
            g1.borrow_mut().transform = Matrix4::op_rotate_y(PI / 2.0);

            let g2 = Rc::new(RefCell::new(Entity::group()));
            g2.borrow_mut().transform = Matrix4::op_scale(1.0, 2.0, 3.0);

            Entity::add_child(Rc::clone(&g1), Rc::clone(&g2));

            let s1 = Rc::new(RefCell::new(Entity::sphere()));
            s1.borrow_mut().transform = Matrix4::op_translate(5.0, 0.0, 0.0);
            let hit = Hit::from(1.0, Rc::clone(&s1));

            Entity::add_child(Rc::clone(&g2), Rc::clone(&s1));

            let p = Vec3::from(1.7321, 1.1547, -5.5774);

            let n = normal_at(Rc::clone(&s1), &p, &hit);
            assert_relative_eq!(0.2857, n.x, epsilon = EPSILON);
            assert_relative_eq!(0.4286, n.y, epsilon = EPSILON);
            assert_relative_eq!(-0.8571, n.z, epsilon = EPSILON);
        }

        {
            let t = Rc::new(RefCell::new(Entity::smooth_triangle()));
            let hit = Hit::on_triangle(1.0, Rc::clone(&t), 0.45, 0.25);

            let n = normal_at(t, &Vec3::zero(), &hit);
            assert_relative_eq!(-0.5547, n.x, epsilon = EPSILON);
            assert_relative_eq!(0.83205, n.y, epsilon = EPSILON);
            assert_relative_eq!(0.0, n.z, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_hit_test() {
        let s = Rc::new(RefCell::new(Entity::sphere()));

        let test = HitTest::builder()
            .add_hit(Hit::from(5.0, Rc::clone(&s)))
            .add_hit(Hit::from(7.0, Rc::clone(&s)))
            .add_hit(Hit::from(-3.0, Rc::clone(&s)))
            .add_hit(Hit::from(2.0, Rc::clone(&s)))
            .build();

        let hit = test.hit().expect("hit exists");
        assert_eq!(2.0, hit.t);
    }

    #[test]
    fn test_ray_transform() {
        {
            let r = Ray::from(Vec3::from(1.0, 2.0, 3.0), Vec3::y_axis());

            let m = Matrix4::op_translate(3.0, 4.0, 5.0);

            let t = Ray::from_transformed(&r, &m);
            assert_eq!(4.0, t.origin.x);
            assert_eq!(6.0, t.origin.y);
            assert_eq!(8.0, t.origin.z);

            assert_eq!(0.0, t.direction.x);
            assert_eq!(1.0, t.direction.y);
            assert_eq!(0.0, t.direction.z);
        }

        {
            let r = Ray::from(Vec3::from(1.0, 2.0, 3.0), Vec3::y_axis());

            let m = Matrix4::op_scale(2.0, 3.0, 4.0);

            let t = Ray::from_transformed(&r, &m);
            assert_eq!(2.0, t.origin.x);
            assert_eq!(6.0, t.origin.y);
            assert_eq!(12.0, t.origin.z);

            assert_eq!(0.0, t.direction.x);
            assert_eq!(3.0, t.direction.y);
            assert_eq!(0.0, t.direction.z);
        }
    }

    #[test]
    fn test_intersect() {
        {
            let r = Ray::from(Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis());

            let s = Rc::new(RefCell::new(Entity::sphere()));

            let test = intersect(&r, s);
            assert_eq!(2, test.hits.len());
            assert_eq!(4.0, test.hits[0].t);
            assert_eq!(6.0, test.hits[1].t);
        }

        {
            let r = Ray::from(Vec3::from(0.0, 1.0, -5.0), Vec3::z_axis());

            let s = Rc::new(RefCell::new(Entity::sphere()));

            let test = intersect(&r, s);
            assert_eq!(2, test.hits.len());
            assert_eq!(5.0, test.hits[0].t);
            assert_eq!(5.0, test.hits[1].t);
        }

        {
            let r = Ray::from(Vec3::from(0.0, 2.0, -5.0), Vec3::z_axis());

            let s = Rc::new(RefCell::new(Entity::sphere()));

            let test = intersect(&r, s);
            assert!(test.hits.is_empty());
        }

        {
            let r = Ray::from(Vec3::zero(), Vec3::z_axis());

            let s = Rc::new(RefCell::new(Entity::sphere()));

            let test = intersect(&r, s);
            assert_eq!(2, test.hits.len());
            assert_eq!(-1.0, test.hits[0].t);
            assert_eq!(1.0, test.hits[1].t);
        }

        {
            let r = Ray::from(Vec3::from(0.0, 0.0, 5.0), Vec3::z_axis());

            let s = Rc::new(RefCell::new(Entity::sphere()));

            let test = intersect(&r, s);
            assert_eq!(2, test.hits.len());
            assert_eq!(-6.0, test.hits[0].t);
            assert_eq!(-4.0, test.hits[1].t);
        }

        {
            let r = Ray::from(Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis());

            let mut s = Entity::sphere();
            s.transform = Matrix4::op_scale(2.0, 2.0, 2.0);
            let s = Rc::new(RefCell::new(s));

            let test = intersect(&r, s);
            assert_eq!(2, test.hits.len());
            assert_eq!(3.0, test.hits[0].t);
            assert_eq!(7.0, test.hits[1].t);
        }

        {
            let r = Ray::from(Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis());

            let mut s = Entity::sphere();
            s.transform = Matrix4::op_translate(5.0, 0.0, 0.0);
            let s = Rc::new(RefCell::new(s));

            let test = intersect(&r, s);
            assert_eq!(0, test.hits.len());
        }

        {
            let g = Rc::new(RefCell::new(Entity::group()));

            let r = Ray::from(Vec3::origin(), Vec3::z_axis());

            let hit_test = intersect(&r, g);
            assert!(hit_test.hits().is_empty());
        }

        {
            let g = Rc::new(RefCell::new(Entity::group()));

            let s1 = Rc::new(RefCell::new(Entity::sphere()));

            let s2 = Rc::new(RefCell::new(Entity::sphere()));
            s2.borrow_mut().transform = Matrix4::op_translate(0.0, 0.0, -3.0);

            let s3 = Rc::new(RefCell::new(Entity::sphere()));
            s3.borrow_mut().transform = Matrix4::op_translate(5.0, 0.0, 0.0);

            Entity::add_child(Rc::clone(&g), Rc::clone(&s1));
            Entity::add_child(Rc::clone(&g), Rc::clone(&s2));
            Entity::add_child(Rc::clone(&g), Rc::clone(&s3));

            let r = Ray::from(Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis());

            let hit_test = intersect(&r, Rc::clone(&g));
            let hits = hit_test.hits();

            assert_eq!(4, hits.len());
            assert_eq!(s2.borrow().id, hits[0].entity.borrow().id);
            assert_eq!(s2.borrow().id, hits[1].entity.borrow().id);
            assert_eq!(s1.borrow().id, hits[2].entity.borrow().id);
            assert_eq!(s1.borrow().id, hits[3].entity.borrow().id);
        }

        {
            let g = Rc::new(RefCell::new(Entity::group()));
            g.borrow_mut().transform = Matrix4::op_scale(2.0, 2.0, 2.0);

            let s1 = Rc::new(RefCell::new(Entity::sphere()));
            s1.borrow_mut().transform = Matrix4::op_translate(5.0, 0.0, 0.0);

            Entity::add_child(Rc::clone(&g), Rc::clone(&s1));

            let r = Ray::from(Vec3::from(10.0, 0.0, -10.0), Vec3::z_axis());

            let hit_test = intersect(&r, Rc::clone(&g));
            let hits = hit_test.hits();

            assert_eq!(2, hits.len());
        }
    }

    #[test]
    fn test_reflect() {
        {
            let v = Vec3::from(1.0, -1.0, 0.0);
            let n = Vec3::y_axis();

            let r = reflect(&v, &n);
            assert_eq!(1.0, r.x);
            assert_eq!(1.0, r.y);
            assert_eq!(0.0, r.z);
        }

        {
            let sqrt_2_over_2 = 2.0_f32.sqrt() / 2.0;

            let v = Vec3::neg_y_axis();
            let n = Vec3::from(sqrt_2_over_2, sqrt_2_over_2, 0.0);

            let r = reflect(&v, &n);
            assert_relative_eq!(1.0, r.x);
            assert_relative_eq!(0.0, r.y);
            assert_relative_eq!(0.0, r.z);
        }
    }

    #[test]
    fn test_intersect_at() {
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
            let r = Ray::from(Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis());

            let hit_test = intersect_at(&w, &r);

            assert_eq!(4, hit_test.hits.len());
            assert_eq!(4.0, hit_test.hits[0].t);
            assert_eq!(4.5, hit_test.hits[1].t);
            assert_eq!(5.5, hit_test.hits[2].t);
            assert_eq!(6.0, hit_test.hits[3].t);
        }
    }

    #[test]
    fn test_prepare_computations() {
        let sqrt_2_over_2 = 2.0_f32.sqrt() / 2.0;

        {
            let r = Ray::from(Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis());

            let s = Rc::new(RefCell::new(Entity::sphere()));

            let hit = Hit::from(4.0, s);
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let i = prepare_computations(hit, &r, &hit_test.hits);

            assert_eq!(hit.t, i.t);
            assert_eq!(Vec3::neg_z_axis(), i.point);
            assert_eq!(Vec3::neg_z_axis(), i.eyev);
            assert_eq!(Vec3::neg_z_axis(), i.normalv);
            assert!(!i.inside);
        }

        {
            let r = Ray::from(Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis());

            let s = Rc::new(RefCell::new(Entity::sphere()));

            let hit = Hit::from(4.0, s);
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let i = prepare_computations(hit, &r, &hit_test.hits);

            assert_eq!(hit.t, i.t);
            assert!(!i.inside);
        }

        {
            let r = Ray::from(Vec3::zero(), Vec3::z_axis());

            let s = Rc::new(RefCell::new(Entity::sphere()));

            let hit = Hit::from(1.0, s);
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let i = prepare_computations(hit, &r, &hit_test.hits);

            assert_eq!(hit.t, i.t);
            assert_eq!(Vec3::z_axis(), i.point);
            assert_eq!(Vec3::neg_z_axis(), i.eyev);
            assert_eq!(Vec3::neg_z_axis(), i.normalv);
            assert!(i.inside);
        }

        {
            let r = Ray::from(Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis());

            let mut s = Entity::sphere();
            s.transform = Matrix4::op_translate(0.0, 0.0, 1.0);
            let s = Rc::new(RefCell::new(s));

            let hit = Hit::from(5.0, s);
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let i = prepare_computations(hit, &r, &hit_test.hits);

            assert!(i.over_point.z < -0.0001 / 2.0);
            assert!(i.point.z > i.over_point.z);
        }

        {
            let r = Ray::from(
                Vec3::from(0.0, 1.0, -1.0),
                Vec3::from(0.0, -sqrt_2_over_2, sqrt_2_over_2),
            );

            let s = Rc::new(RefCell::new(Entity::plane()));

            let hit_test = intersect(&r, s);
            let hit = hit_test.hit().expect("ray hits plane");

            let i = prepare_computations(hit, &r, &hit_test.hits);

            assert_eq!(Vec3::from(0.0, sqrt_2_over_2, sqrt_2_over_2), i.reflectv);
        }

        {
            let mut a = Entity::glass_sphere();
            a.transform = Matrix4::op_scale(2.0, 2.0, 2.0);
            a.material.refractive_index = 1.5;
            let a = Rc::new(RefCell::new(a));

            let mut b = Entity::glass_sphere();
            b.transform = Matrix4::op_translate(0.0, 0.0, -0.25);
            b.material.refractive_index = 2.0;
            let b = Rc::new(RefCell::new(b));

            let mut c = Entity::glass_sphere();
            c.transform = Matrix4::op_translate(0.0, 0.0, 0.25);
            c.material.refractive_index = 2.5;
            let c = Rc::new(RefCell::new(c));

            let hits = vec![
                Hit::from(2.0, Rc::clone(&a)),
                Hit::from(2.75, Rc::clone(&b)),
                Hit::from(3.25, Rc::clone(&c)),
                Hit::from(4.75, Rc::clone(&b)),
                Hit::from(5.25, Rc::clone(&c)),
                Hit::from(6.0, Rc::clone(&a)),
            ];

            let expected = [
                (1.0, 1.5),
                (1.5, 2.0),
                (2.0, 2.5),
                (2.5, 2.5),
                (2.5, 1.5),
                (1.5, 1.0),
            ];

            assert_eq!(hits.len(), expected.len());

            let r = Ray::from(Vec3::from(0.0, 0.0, -4.0), Vec3::z_axis());

            for i in 0..hits.len() {
                let intersection = prepare_computations(&hits[i], &r, &hits);
                let exp = expected[i];

                assert_eq!(exp.0, intersection.n1);
                assert_eq!(exp.1, intersection.n2);
            }
        }

        {
            let r = Ray::from(Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis());

            let mut s = Entity::glass_sphere();
            s.transform = Matrix4::op_translate(0.0, 0.0, 1.0);
            let s = Rc::new(RefCell::new(s));

            let hit = Hit::from(5.0, s);
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let intersection = prepare_computations(hit, &r, hit_test.hits());
            assert!(intersection.under_point.z > EPSILON / 2.0);
            assert!(intersection.point.z < intersection.under_point.z);
        }

        {
            let t = Rc::new(RefCell::new(Entity::smooth_triangle()));

            let r = Ray::from(Vec3::from(-0.2, 0.3, -2.0), Vec3::z_axis());

            let hit = Hit::on_triangle(1.0, t, 0.45, 0.25);
            let hit_test = HitTest::builder().add_hit(hit).build();
            let hit = hit_test.hit().expect("hit exists");

            let intersection = prepare_computations(hit, &r, hit_test.hits());
            assert_relative_eq!(-0.5547, intersection.normalv.x, epsilon = EPSILON);
            assert_relative_eq!(0.83205, intersection.normalv.y, epsilon = EPSILON);
            assert_relative_eq!(0.0, intersection.normalv.z, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_local_normal_at_plane() {
        let p = Entity::plane();

        {
            let n = local_normal_at(&p.shape, &Vec3::zero(), None);
            assert_eq!(Vec3::y_axis(), n);
        }

        {
            let n = local_normal_at(&p.shape, &Vec3::from(10.0, 0.0, -10.0), None);
            assert_eq!(Vec3::y_axis(), n);
        }

        {
            let n = local_normal_at(&p.shape, &Vec3::from(-5.0, 0.0, 150.0), None);
            assert_eq!(Vec3::y_axis(), n);
        }
    }

    #[test]
    fn test_local_intersect_plane() {
        let p = Rc::new(RefCell::new(Entity::plane()));

        {
            let r = Ray::from(Vec3::from(0.0, 10.0, 0.0), Vec3::z_axis());

            let local_intersect = local_intersect_plane(&r);
            let hit_test = create_hit_test(Rc::clone(&p), &local_intersect.ts);

            assert!(hit_test.hit().is_none());
        }

        {
            let r = Ray::from(Vec3::zero(), Vec3::z_axis());

            let local_intersect = local_intersect_plane(&r);
            let hit_test = create_hit_test(Rc::clone(&p), &local_intersect.ts);

            assert!(hit_test.hit().is_none());
        }

        {
            let r = Ray::from(Vec3::y_axis(), Vec3::neg_y_axis());

            let local_intersect = local_intersect_plane(&r);
            let hit_test = create_hit_test(Rc::clone(&p), &local_intersect.ts);
            let hit = hit_test.hit();

            assert!(hit.is_some());

            let h = hit.unwrap();
            assert_eq!(1.0, h.t);
        }

        {
            let r = Ray::from(Vec3::neg_y_axis(), Vec3::y_axis());

            let local_intersect = local_intersect_plane(&r);
            let hit_test = create_hit_test(Rc::clone(&p), &local_intersect.ts);
            let hit = hit_test.hit();

            assert!(hit.is_some());

            let h = hit.unwrap();
            assert_eq!(1.0, h.t);
        }
    }

    #[test]
    fn test_local_intersect_cube() {
        {
            let c = Rc::new(RefCell::new(Entity::cube()));

            let rays = [
                Ray::from(Vec3::from(5.0, 0.5, 0.0), Vec3::neg_x_axis()),
                Ray::from(Vec3::from(-5.0, 0.5, 0.0), Vec3::x_axis()),
                Ray::from(Vec3::from(0.5, 5.0, 0.0), Vec3::neg_y_axis()),
                Ray::from(Vec3::from(0.5, -5.0, 0.0), Vec3::y_axis()),
                Ray::from(Vec3::from(0.5, 0.0, 5.0), Vec3::neg_z_axis()),
                Ray::from(Vec3::from(0.5, 0.0, -5.0), Vec3::z_axis()),
                Ray::from(Vec3::from(0.0, 0.5, 0.0), Vec3::z_axis()),
            ];

            let expected = [
                (4.0, 6.0),
                (4.0, 6.0),
                (4.0, 6.0),
                (4.0, 6.0),
                (4.0, 6.0),
                (4.0, 6.0),
                (-1.0, 1.0),
            ];

            assert_eq!(rays.len(), expected.len());

            for i in 0..rays.len() {
                let r = &rays[i];
                let exp = expected[i];

                let local_intersect = local_intersect_cube(r);
                let hit_test = create_hit_test(Rc::clone(&c), &local_intersect.ts);

                let hits = hit_test.hits();
                assert_eq!(2, hits.len());
                assert_eq!(exp.0, hits[0].t);
                assert_eq!(exp.1, hits[1].t);
            }
        }

        {
            let rays = [
                Ray::from(
                    Vec3::from(-2.0, 0.0, 0.0),
                    Vec3::from(0.2673, 0.5345, 0.8018),
                ),
                Ray::from(
                    Vec3::from(0.0, -2.0, 0.0),
                    Vec3::from(0.8018, 0.2673, 0.5345),
                ),
                Ray::from(
                    Vec3::from(0.0, 0.0, -2.0),
                    Vec3::from(0.5345, 0.8018, 0.2673),
                ),
                Ray::from(Vec3::from(2.0, 0.0, 2.0), Vec3::neg_z_axis()),
                Ray::from(Vec3::from(0.0, 2.0, 2.0), Vec3::neg_y_axis()),
                Ray::from(Vec3::from(2.0, 2.0, 0.0), Vec3::neg_x_axis()),
            ];

            for r in rays.iter() {
                let local_intersect = local_intersect_cube(r);
                assert_eq!(0, local_intersect.ts.len());
            }
        }
    }

    #[test]
    fn test_local_normal_at_cube() {
        {
            let c = Entity::cube();

            let tests = [
                (Vec3::from(1.0, 0.5, -0.8), Vec3::x_axis()),
                (Vec3::from(-1.0, -0.2, 0.9), Vec3::neg_x_axis()),
                (Vec3::from(-0.4, 1.0, -0.1), Vec3::y_axis()),
                (Vec3::from(0.3, -1.0, -0.7), Vec3::neg_y_axis()),
                (Vec3::from(-0.6, 0.3, 1.0), Vec3::z_axis()),
                (Vec3::from(0.4, 0.4, -1.0), Vec3::neg_z_axis()),
                (Vec3::from(1.0, 1.0, 1.0), Vec3::x_axis()),
                (Vec3::from(-1.0, -1.0, -1.0), Vec3::neg_x_axis()),
            ];

            for test in tests.iter() {
                let normal = local_normal_at(&c.shape, &test.0, None);

                assert_eq!(test.1, normal);
            }
        }
    }

    #[test]
    fn test_local_intersect_cylinder() {
        {
            let tests = [
                (Vec3::x_axis(), Vec3::y_axis()),
                (Vec3::zero(), Vec3::y_axis()),
                (Vec3::from(0.0, 0.0, -5.0), Vec3::from(1.0, 1.0, 1.0)),
            ];

            for test in tests.iter() {
                let direction = Vec3::from_normalized(&test.1);
                let r = Ray::from(test.0, direction);

                let local_intersect =
                    local_intersect_cylinder(&r, -f32::INFINITY, f32::INFINITY, false);
                assert!(local_intersect.ts.is_empty());
            }
        }

        {
            let c = Rc::new(RefCell::new(Entity::cylinder()));

            let tests = [
                (Vec3::from(1.0, 0.0, -5.0), Vec3::z_axis()),
                (Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis()),
                (Vec3::from(0.5, 0.0, -5.0), Vec3::from(0.1, 1.0, 1.0)),
            ];

            let expected = [(5.0, 5.0), (4.0, 6.0), (6.80798, 7.08872)];

            assert_eq!(tests.len(), expected.len());

            for i in 0..tests.len() {
                let test = tests[i];
                let exp = expected[i];

                let direction = Vec3::from_normalized(&test.1);
                let r = Ray::from(test.0, direction);

                let local_intersect =
                    local_intersect_cylinder(&r, -f32::INFINITY, f32::INFINITY, false);
                let hit_test = create_hit_test(Rc::clone(&c), &local_intersect.ts);

                assert_eq!(2, hit_test.hits().len());
                assert_relative_eq!(exp.0, hit_test.hits()[0].t, epsilon = EPSILON);
                assert_relative_eq!(exp.1, hit_test.hits()[1].t, epsilon = EPSILON);
            }
        }

        {
            let tests = [
                (Vec3::from(0.0, 1.5, 0.0), Vec3::from(0.1, 1.0, 0.0)),
                (Vec3::from(0.0, 3.0, -5.0), Vec3::z_axis()),
                (Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis()),
                (Vec3::from(0.0, 2.0, -5.0), Vec3::z_axis()),
                (Vec3::from(0.0, 1.0, -5.0), Vec3::z_axis()),
                (Vec3::from(0.0, 1.5, -2.0), Vec3::z_axis()),
            ];

            let expected = [0, 0, 0, 0, 0, 2];

            assert_eq!(tests.len(), expected.len());

            for i in 0..tests.len() {
                let test = tests[i];
                let exp = expected[i];

                let direction = Vec3::from_normalized(&test.1);
                let r = Ray::from(test.0, direction);

                let local_intersect = local_intersect_cylinder(&r, 1.0, 2.0, false);
                assert_eq!(exp, local_intersect.ts.len());
            }
        }

        {
            let tests = [
                (Vec3::from(0.0, 3.0, 0.0), Vec3::neg_y_axis()),
                (Vec3::from(0.0, 3.0, -2.0), Vec3::from(0.0, -1.0, 2.0)),
                (Vec3::from(0.0, 4.0, -2.0), Vec3::from(0.0, -1.0, 1.0)),
                (Vec3::from(0.0, 0.0, -2.0), Vec3::from(0.0, 1.0, 2.0)),
                (Vec3::from(0.0, -1.0, -2.0), Vec3::from(0.0, 1.0, 1.0)),
            ];

            for test in tests.iter() {
                let direction = Vec3::from_normalized(&test.1);
                let r = Ray::from(test.0, direction);

                let local_intersect = local_intersect_cylinder(&r, 1.0, 2.0, true);
                assert_eq!(2, local_intersect.ts.len());
            }
        }
    }

    #[test]
    fn test_local_normal_at_cylinder() {
        {
            let c = Entity::cylinder();

            let tests = [
                (Vec3::x_axis(), Vec3::x_axis()),
                (Vec3::from(0.0, 5.0, -1.0), Vec3::neg_z_axis()),
                (Vec3::from(0.0, -2.0, 1.0), Vec3::z_axis()),
                (Vec3::from(-1.0, 1.0, 0.0), Vec3::neg_x_axis()),
            ];

            for test in tests.iter() {
                let n = local_normal_at(&c.shape, &test.0, None);
                assert_eq!(test.1, n);
            }
        }

        {
            let c = Entity::cylinder_from(1.0, 2.0, true);

            let tests = [
                (Vec3::y_axis(), Vec3::neg_y_axis()),
                (Vec3::from(0.5, 1.0, 0.0), Vec3::neg_y_axis()),
                (Vec3::from(0.0, 1.0, 0.5), Vec3::neg_y_axis()),
                (Vec3::from(0.0, 2.0, 0.0), Vec3::y_axis()),
                (Vec3::from(0.5, 2.0, 0.0), Vec3::y_axis()),
                (Vec3::from(0.0, 2.0, 0.5), Vec3::y_axis()),
            ];

            for test in tests.iter() {
                let n = local_normal_at(&c.shape, &test.0, None);
                assert_eq!(test.1, n);
            }
        }
    }

    #[test]
    fn test_local_intersect_cone() {
        {
            let c = Rc::new(RefCell::new(Entity::cone()));

            let tests = [
                (Vec3::from(0.0, 0.0, -5.0), Vec3::z_axis()),
                (Vec3::from(1.0, 1.0, -5.0), Vec3::from(-0.5, -1.0, 1.0)),
            ];

            let expected = [(5.0, 5.0), (4.55006, 49.44994)];

            assert_eq!(tests.len(), expected.len());

            for i in 0..tests.len() {
                let test = tests[i];
                let exp = expected[i];

                let direction = Vec3::from_normalized(&test.1);
                let r = Ray::from(test.0, direction);

                let local_intersect =
                    local_intersect_cone(&r, -f32::INFINITY, f32::INFINITY, false);
                let hit_test = create_hit_test(Rc::clone(&c), &local_intersect.ts);
                let hits = hit_test.hits();

                assert_eq!(2, hits.len());
                assert_relative_eq!(exp.0, hits[0].t, epsilon = EPSILON);
                assert_relative_eq!(exp.1, hits[1].t, epsilon = EPSILON);
            }
        }

        {
            let c = Rc::new(RefCell::new(Entity::cone()));

            let direction = Vec3::from_normalized(&Vec3::from(0.0, 1.0, 1.0));
            let r = Ray::from(Vec3::neg_z_axis(), direction);

            let local_intersect = local_intersect_cone(&r, -f32::INFINITY, f32::INFINITY, false);
            let hit_test = create_hit_test(Rc::clone(&c), &local_intersect.ts);

            let hits = hit_test.hits();
            assert_eq!(1, hits.len());
            assert_relative_eq!(0.35355, hits[0].t, epsilon = EPSILON);
        }

        {
            let c = Rc::new(RefCell::new(Entity::cone_from(-0.5, 0.5, true)));

            let tests = [
                (Vec3::from(0.0, 0.0, -5.0), Vec3::y_axis()),
                (Vec3::from(0.0, 0.0, -0.25), Vec3::from(0.0, 1.0, 1.0)),
                (Vec3::from(0.0, 1.0, -0.25), Vec3::y_axis()),
            ];

            let expected = [0, 2, 4];

            assert_eq!(tests.len(), expected.len());

            for i in 0..tests.len() {
                let test = tests[i];
                let exp = expected[i];

                let direction = Vec3::from_normalized(&test.1);
                let r = Ray::from(test.0, direction);

                let local_intersect = local_intersect_cone(&r, -0.5, 0.5, true);
                let hit_test = create_hit_test(Rc::clone(&c), &local_intersect.ts);

                let hits = hit_test.hits();
                assert_eq!(exp, hits.len());
            }
        }
    }

    #[test]
    fn test_local_normal_at_cone() {
        let sqrt_2 = 2.0_f32.sqrt();

        {
            let c = Entity::cone();

            let tests = [
                (Vec3::origin(), Vec3::origin()),
                (Vec3::from(1.0, 1.0, 1.0), Vec3::from(1.0, -sqrt_2, 1.0)),
                (Vec3::from(-1.0, -1.0, 0.0), Vec3::from(-1.0, 1.0, 0.0)),
            ];

            for test in tests.iter() {
                let normal = local_normal_at(&c.shape, &test.0, None);
                assert_relative_eq!(test.1.x, normal.x, epsilon = EPSILON);
                assert_relative_eq!(test.1.y, normal.y, epsilon = EPSILON);
                assert_relative_eq!(test.1.z, normal.z, epsilon = EPSILON);
            }
        }
    }

    #[test]
    fn test_local_intersect_triangle() {
        {
            let t = Rc::new(RefCell::new(Entity::triangle_from(
                Vec3::y_axis(),
                Vec3::neg_x_axis(),
                Vec3::x_axis(),
            )));
            let r = Ray::from(Vec3::from(1.0, 1.0, -2.0), Vec3::z_axis());

            let local_intersect = local_intersect(&t.borrow().shape, &r);
            assert!(local_intersect.ts.is_empty());
        }

        {
            let t = Rc::new(RefCell::new(Entity::triangle_from(
                Vec3::y_axis(),
                Vec3::neg_x_axis(),
                Vec3::x_axis(),
            )));
            let r = Ray::from(Vec3::from(-1.0, 1.0, -2.0), Vec3::z_axis());

            let local_intersect = local_intersect(&t.borrow().shape, &r);
            assert!(local_intersect.ts.is_empty());
        }

        {
            let t = Rc::new(RefCell::new(Entity::triangle_from(
                Vec3::y_axis(),
                Vec3::neg_x_axis(),
                Vec3::x_axis(),
            )));
            let r = Ray::from(Vec3::from(0.0, -1.0, -2.0), Vec3::z_axis());

            let local_intersect = local_intersect(&t.borrow().shape, &r);
            assert!(local_intersect.ts.is_empty());
        }

        {
            let t = Rc::new(RefCell::new(Entity::triangle_from(
                Vec3::y_axis(),
                Vec3::neg_x_axis(),
                Vec3::x_axis(),
            )));
            let r = Ray::from(Vec3::from(0.0, 0.5, -2.0), Vec3::z_axis());

            let local_intersect = local_intersect(&t.borrow().shape, &r);
            let hits = local_intersect.ts;
            assert_eq!(1, hits.len());
            assert_eq!(2.0, hits[0]);
        }

        {
            let t = Rc::new(RefCell::new(Entity::smooth_triangle()));
            let r = Ray::from(Vec3::from(-0.2, 0.3, -2.0), Vec3::z_axis());

            let local_intersect = local_intersect(&t.borrow().shape, &r);
            let hits = local_intersect.ts;
            assert_eq!(1, hits.len());
            if let Some((u, v)) = local_intersect.uv {
                assert_eq!(0.45, u);
                assert_eq!(0.25, v);
            } else {
                panic!("expected uv");
            }
        }
    }

    #[test]
    fn test_local_normal_at_triangle() {
        let t = Rc::new(RefCell::new(Entity::triangle_from(
            Vec3::y_axis(),
            Vec3::neg_x_axis(),
            Vec3::x_axis(),
        )));
        let hit = Hit::on_triangle(1.0, Rc::clone(&t), 0.45, 0.25);
        let expected = match &t.borrow().shape {
            Shape::Triangle { normal, .. } => *normal,
            _ => panic!("expected Triangle"),
        };

        {
            let p = Vec3::from(0.0, 0.5, 0.0);
            let n = local_normal_at(&t.borrow().shape, &p, Some(&hit));
            assert_eq!(expected, n);
        }

        {
            let p = Vec3::from(-0.5, 0.75, 0.0);
            let n = local_normal_at(&t.borrow().shape, &p, Some(&hit));
            assert_eq!(expected, n);
        }

        {
            let p = Vec3::from(0.5, 0.25, 0.0);
            let n = local_normal_at(&t.borrow().shape, &p, Some(&hit));
            assert_eq!(expected, n);
        }
    }

    #[test]
    fn test_world_to_object() {
        {
            let g1 = Rc::new(RefCell::new(Entity::group()));
            g1.borrow_mut().transform = Matrix4::op_rotate_y(PI / 2.0);

            let g2 = Rc::new(RefCell::new(Entity::group()));
            g2.borrow_mut().transform = Matrix4::op_scale(2.0, 2.0, 2.0);

            Entity::add_child(Rc::clone(&g1), Rc::clone(&g2));

            let s1 = Rc::new(RefCell::new(Entity::sphere()));
            s1.borrow_mut().transform = Matrix4::op_translate(5.0, 0.0, 0.0);

            Entity::add_child(Rc::clone(&g2), Rc::clone(&s1));

            let p = Vec3::from(-2.0, 0.0, -10.0);

            let result = world_to_object(Rc::clone(&s1), &p);
            assert_relative_eq!(0.0, result.x, epsilon = EPSILON);
            assert_relative_eq!(0.0, result.y, epsilon = EPSILON);
            assert_relative_eq!(-1.0, result.z, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_normal_to_world() {
        let sqrt_3_over_3 = 3.0_f32.sqrt() / 3.0;

        {
            let g1 = Rc::new(RefCell::new(Entity::group()));
            g1.borrow_mut().transform = Matrix4::op_rotate_y(PI / 2.0);

            let g2 = Rc::new(RefCell::new(Entity::group()));
            g2.borrow_mut().transform = Matrix4::op_scale(1.0, 2.0, 3.0);

            Entity::add_child(Rc::clone(&g1), Rc::clone(&g2));

            let s1 = Rc::new(RefCell::new(Entity::sphere()));
            s1.borrow_mut().transform = Matrix4::op_translate(5.0, 0.0, 0.0);

            Entity::add_child(Rc::clone(&g2), Rc::clone(&s1));

            let normal = Vec3::from(sqrt_3_over_3, sqrt_3_over_3, sqrt_3_over_3);

            let r = normal_to_world(Rc::clone(&s1), &normal);
            assert_relative_eq!(0.2857, r.x, epsilon = EPSILON);
            assert_relative_eq!(0.4286, r.y, epsilon = EPSILON);
            assert_relative_eq!(-0.8571, r.z, epsilon = EPSILON);
        }
    }

    fn create_hit_test(entity: Rc<RefCell<Entity>>, ts: &[f32]) -> HitTest {
        let mut builder = HitTest::builder();
        for &t in ts.iter() {
            builder = builder.add_hit(Hit::from(t, Rc::clone(&entity)));
        }

        builder.build()
    }
}
