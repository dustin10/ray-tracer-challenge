use crate::{
    math::{Matrix4, Vec3},
    physics::Shape,
    render::{Material, PointLight},
};

use std::{
    cell::RefCell,
    rc::{Rc, Weak},
    sync::atomic::{AtomicU32, Ordering},
};

/// Used to assign a unique integer identifier to each new entity created.
static NEXT_ENTITY_ID: AtomicU32 = AtomicU32::new(1);

/// Represents an object in the scene which is to be rendered by casting rays.
pub struct Entity {
    pub id: u32,
    pub transform: Matrix4,
    pub material: Material,
    pub shape: Shape,
    pub parent: Option<Weak<RefCell<Entity>>>,
    pub children: Vec<Rc<RefCell<Entity>>>,
}

impl Entity {
    /// Creates a new [Entity] whose [Shape] is set to a [Shape::Sphere] of radius one.
    pub fn sphere() -> Self {
        Self::sphere_from(Vec3::origin(), 1.0)
    }
    /// Creates a new [Entity] whose [Shape] is set to a [Shape::Sphere] with the given radius.
    pub fn sphere_from(origin: Vec3, radius: f32) -> Self {
        Self::new(Shape::Sphere { origin, radius })
    }
    /// Creates a new [Entity] whose [Shape] is set to a [Shape::Sphere] of radius one and whose
    /// [Material] is configured to have a glass effect.
    pub fn glass_sphere() -> Self {
        let mut s = Self::sphere();
        s.material.transparency = 1.0;
        s.material.refractive_index = 1.5;

        s
    }
    /// Creates a new [Entity] whose [Shape] is set to a [Shape::Plane].
    pub fn plane() -> Self {
        Self::new(Shape::Plane)
    }
    /// Creates a new [Entity] whose [Shape] is set to a [Shape::Cube].
    pub fn cube() -> Self {
        Self::new(Shape::Cube)
    }
    /// Creates a new [Entity] whose [Shape] is set to a closed [Shape::Cylinder] with min and max
    /// set to negative and positive infinity repsectively.
    pub fn cylinder() -> Self {
        Self::cylinder_from(-f32::INFINITY, f32::INFINITY, false)
    }
    /// Creates a new [Entity] whose [Shape] is set to a [Shape::Cylinder] with created from the
    /// given values.
    pub fn cylinder_from(minimum: f32, maximum: f32, closed: bool) -> Self {
        Self::new(Shape::Cylinder {
            minimum,
            maximum,
            closed,
        })
    }
    /// Creates a new [Entity] whose [Shape] is set to a closed [Shape::Cone] with min and max
    /// set to negative and positive infinity repsectively.
    pub fn cone() -> Self {
        Self::cone_from(-f32::INFINITY, f32::INFINITY, false)
    }
    /// Creates a new [Entity] whose [Shape] is set to a [Shape::Cone] with created from the
    /// given values.
    pub fn cone_from(minimum: f32, maximum: f32, closed: bool) -> Self {
        Self::new(Shape::Cone {
            minimum,
            maximum,
            closed,
        })
    }
    /// Creates a new [Entity] whose [Shape] is set to a [Shape::Group].
    pub fn group() -> Self {
        Self::new(Shape::Group)
    }
    /// Creates a new [Entity] whose [Shape] is set to a [Shape::Triangle] whose points are at -1
    /// and 1 on the x-axis as well as 1 on the y-axis.
    pub fn triangle() -> Self {
        Self::triangle_from(Vec3::neg_x_axis(), Vec3::y_axis(), Vec3::x_axis())
    }
    /// Creates a new [Entity] whose [Shape] is set to a [Shape::Triangle] with the given points.
    pub fn triangle_from(p1: Vec3, p2: Vec3, p3: Vec3) -> Self {
        let e1 = p2 - p1;
        let e2 = p3 - p1;

        let mut normal = Vec3::from_cross(&e2, &e1);
        normal.normalize();

        Self::new(Shape::Triangle {
            p1,
            p2,
            p3,
            e1,
            e2,
            normal,
        })
    }
    /// Creates a new [Entity] whose [Shape] is set to a [Shape::SmoothTriangle] whose points are
    /// at -1 and 1 on the x-axis as well as 1 on the y-axis.
    pub fn smooth_triangle() -> Self {
        Self::smooth_triangle_from(
            Vec3::y_axis(),
            Vec3::neg_x_axis(),
            Vec3::x_axis(),
            Vec3::y_axis(),
            Vec3::neg_x_axis(),
            Vec3::x_axis(),
        )
    }
    /// Creates a new [Entity] whose [Shape] is set to a [Shape::SmoothTriangle] with the given
    /// values.
    pub fn smooth_triangle_from(
        p1: Vec3,
        p2: Vec3,
        p3: Vec3,
        n1: Vec3,
        n2: Vec3,
        n3: Vec3,
    ) -> Self {
        let e1 = p2 - p1;
        let e2 = p3 - p1;

        let mut normal = Vec3::from_cross(&e2, &e1);
        normal.normalize();

        Self::new(Shape::SmoothTriangle {
            p1,
            p2,
            p3,
            e1,
            e2,
            n1,
            n2,
            n3,
        })
    }
    /// Creates a new [Entity] with the given [Shape].
    fn new(shape: Shape) -> Self {
        Self {
            id: NEXT_ENTITY_ID.fetch_add(1, Ordering::SeqCst),
            transform: Matrix4::identity(),
            material: Material::default(),
            parent: None,
            children: Vec::new(),
            shape,
        }
    }
    /// Adds an entity to the children of the parent while also setting a back pointer to parent on
    /// the child.
    pub fn add_child(parent: Rc<RefCell<Entity>>, child: Rc<RefCell<Entity>>) {
        let mut p = parent.borrow_mut();
        match p.shape {
            Shape::Group => {
                p.children.push(Rc::clone(&child));
                child.borrow_mut().parent = Some(Rc::downgrade(&parent));
            }
            _ => panic!("add_child is valid only on Group"),
        }
    }
}

impl Default for Entity {
    /// Creates a default [Entity]. The default [Entity] is a sphere of radius one at the origin.
    fn default() -> Self {
        Entity::sphere()
    }
}

/// Contains all of the entities, lights, etc. that make up the world to be rendered.
#[derive(Default)]
pub struct World {
    entities: Vec<Rc<RefCell<Entity>>>,
    light: PointLight,
}

impl World {
    /// Creates a new empty [World].
    pub fn new() -> Self {
        Self::default()
    }
    /// Returns a reference to the [Vec] containing the [Entity] values that exist in the world.
    pub fn entities(&self) -> &Vec<Rc<RefCell<Entity>>> {
        &self.entities
    }
    /// Adds a raw [Entity] to the world and returns a mutable reference to it.
    pub fn add_entity(&mut self, entity: Entity) -> Rc<RefCell<Entity>> {
        let entity = Rc::new(RefCell::new(entity));
        self.entities.push(Rc::clone(&entity));
        entity
    }
    /// Adds an existing [Entity] reference to the world.
    pub fn add_entity_ref(&mut self, entity: Rc<RefCell<Entity>>) {
        self.entities.push(Rc::clone(&entity));
    }
    /// Retruns a reference to the [PointLight] which is lighting the world.
    pub fn light(&self) -> &PointLight {
        &self.light
    }
    /// Sets the [PointLight] which is lighting the world.
    pub fn set_light(&mut self, light: PointLight) {
        self.light = light;
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        math::{Matrix4, Vec3},
        physics::Shape,
    };

    use super::{Entity, World};

    use core::panic;
    use std::{cell::RefCell, rc::Rc};

    #[test]
    fn test_sphere_default() {
        let s = Entity::sphere();

        assert!(s.id > 0);
        assert_eq!(s.transform, Matrix4::identity());
        assert!(s.children.is_empty());
        assert!(s.parent.is_none());

        match s.shape {
            Shape::Sphere { origin, radius } => {
                assert_eq!(origin, Vec3::zero());
                assert_eq!(radius, 1.0);
            }
            _ => panic!("expected sphere"),
        }
    }

    #[test]
    fn test_sphere_of_radius() {
        let s = Entity::sphere_from(Vec3::origin(), 10.0);

        assert!(s.id > 0);
        assert_eq!(s.transform, Matrix4::identity());
        assert!(s.children.is_empty());
        assert!(s.parent.is_none());

        match s.shape {
            Shape::Sphere { origin, radius } => {
                assert_eq!(origin, Vec3::zero());
                assert_eq!(radius, 10.0);
            }
            _ => panic!("expected sphere"),
        }
    }

    #[test]
    fn test_world_default() {
        let w = World::default();
        assert!(w.entities.is_empty());
    }

    #[test]
    fn test_entity_add_child() {
        let g = Rc::new(RefCell::new(Entity::group()));
        let s = Rc::new(RefCell::new(Entity::sphere()));

        Entity::add_child(Rc::clone(&g), Rc::clone(&s));

        assert_eq!(1, g.borrow().children.len());
        assert_eq!(
            g.borrow().id,
            s.borrow()
                .parent
                .as_ref()
                .expect("parent is set")
                .upgrade()
                .expect("parent is valid")
                .borrow()
                .id
        );
    }

    #[test]
    fn test_create_triangle() {
        let t = Entity::triangle_from(Vec3::y_axis(), Vec3::neg_x_axis(), Vec3::x_axis());

        match t.shape {
            Shape::Triangle {
                p1,
                p2,
                p3,
                e1,
                e2,
                normal,
            } => {
                assert_eq!(Vec3::y_axis(), p1);
                assert_eq!(Vec3::neg_x_axis(), p2);
                assert_eq!(Vec3::x_axis(), p3);
                assert_eq!(Vec3::from(-1.0, -1.0, 0.0), e1);
                assert_eq!(Vec3::from(1.0, -1.0, 0.0), e2);
                assert_eq!(Vec3::neg_z_axis(), normal);
            }
            _ => panic!("expected Triangle"),
        }
    }
}
