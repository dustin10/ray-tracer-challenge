use crate::{entity::Entity, math::Vec3};

use std::{cell::RefCell, path::Path, rc::Rc};

/// Results of parsing an OBJ file that can be transformed into an [Entity].
struct Obj {
    unrecognized: u32,
    vertices: Vec<Vec3>,
    normals: Vec<Vec3>,
    default_group: Rc<RefCell<Entity>>,
    groups: Vec<Rc<RefCell<Entity>>>,
}

impl Obj {
    /// Creates a new default [Obj].
    fn new() -> Self {
        Self::default()
    }
    /// Transforms the [Obj] to an [Entity] which can be rendered in a scene.
    fn into_entity(self) -> Rc<RefCell<Entity>> {
        // TODO: just add them all?
        let entity = Rc::new(RefCell::new(Entity::group()));
        if !self.groups.is_empty() {
            for group in self.groups {
                Entity::add_child(Rc::clone(&entity), group);
            }
        } else {
            Entity::add_child(Rc::clone(&entity), self.default_group);
        }

        entity
    }
}

impl Default for Obj {
    /// Creates a new default [Obj].
    fn default() -> Self {
        Self {
            unrecognized: 0,
            vertices: Vec::new(),
            normals: Vec::new(),
            default_group: Rc::new(RefCell::new(Entity::group())),
            groups: Vec::new(),
        }
    }
}

/// Contains the vertex and optional normal data for a face defined in the obj file.
struct FaceVertex {
    vertex: Vec3,
    normal: Option<Vec3>,
}

/// Parses an OBJ file from disk at the given path and transforms it into an [Entity] which can be
/// rendered in a scene by the ray tracer.
pub fn load_entity_from_obj_file(path: impl AsRef<Path>) -> Rc<RefCell<Entity>> {
    let content = std::fs::read_to_string(path).expect("valid file path");
    load_entity_from_obj(content)
}

/// Parses the given [String] containing OBJ file formatted text and transforms it into an [Entity]
/// which can be rendered in a scene by the ray tracer.
pub fn load_entity_from_obj(content: String) -> Rc<RefCell<Entity>> {
    let obj = parse_obj(content);
    obj.into_entity()
}

/// Parses the given [String] containing OBJ file formatted text into an [Obj].
fn parse_obj(content: String) -> Obj {
    let mut obj = Obj::new();

    let mut active_group = Rc::clone(&obj.default_group);

    for line in content.lines().map(|l| l.trim()).filter(|l| !l.is_empty()) {
        if line.starts_with("v ") {
            let tokens: Vec<&str> = line.split(" ").collect();

            let mut ns: [f32; 3] = [0.0, 0.0, 0.0];
            for (i, token) in tokens.iter().enumerate().skip(1) {
                let n: f32 = token.parse().expect("value is valid f32");
                ns[i - 1] = n;
            }

            let vertex = Vec3::from(ns[0], ns[1], ns[2]);

            obj.vertices.push(vertex);
        } else if line.starts_with("vn ") {
            let tokens: Vec<&str> = line.split(" ").collect();

            let mut ns: [f32; 3] = [0.0, 0.0, 0.0];
            for (i, token) in tokens.iter().enumerate().skip(1) {
                let n: f32 = token.parse().expect("value is valid f32");
                ns[i - 1] = n;
            }

            let normal = Vec3::from(ns[0], ns[1], ns[2]);

            obj.normals.push(normal);
        } else if line.starts_with("f ") {
            let tokens: Vec<&str> = line.split(" ").collect();

            let mut vertex_indices: Vec<usize> = Vec::new();
            let mut normal_indices: Vec<usize> = Vec::new();
            for token in tokens.iter().skip(1) {
                let sub_tokens: Vec<&str> = token.split('/').collect();

                match sub_tokens.len() {
                    1 => {
                        let vertex_index: usize =
                            sub_tokens[0].parse().expect("value is valid usize");
                        vertex_indices.push(vertex_index);
                    }
                    3 => {
                        let vertex_index: usize =
                            sub_tokens[0].parse().expect("value is valid usize");
                        vertex_indices.push(vertex_index);

                        let normal_index: usize =
                            sub_tokens[2].parse().expect("value is valid usize");
                        normal_indices.push(normal_index);
                    }
                    _ => panic!("unexpected face vertex token"),
                }
            }

            let mut face_vertices = Vec::new();

            for (i, index) in vertex_indices.iter().enumerate() {
                let vertex = obj.vertices[index - 1];

                // assumption is that if a vertex index is specified then so is a normal index
                let mut normal = None;
                if !normal_indices.is_empty() {
                    normal = Some(obj.normals[normal_indices[i] - 1])
                }

                face_vertices.push(FaceVertex { vertex, normal });
            }

            let triangles = fan_triangulation(&face_vertices);
            for triangle in triangles {
                Entity::add_child(Rc::clone(&active_group), Rc::new(RefCell::new(triangle)));
            }
        } else if line.starts_with("g ") {
            active_group = Rc::new(RefCell::new(Entity::group()));
            obj.groups.push(Rc::clone(&active_group));
        } else {
            obj.unrecognized += 1;
        }
    }

    obj
}

/// Computes the triangles for a face given the resolved vertices.
fn fan_triangulation(vertices: &[FaceVertex]) -> Vec<Entity> {
    let mut triangles = Vec::new();
    for i in 1..vertices.len() - 1 {
        let v1 = &vertices[0];
        let v2 = &vertices[i];
        let v3 = &vertices[i + 1];

        // assumption is that if one vertex has a normal then all others have one as well
        if v1.normal.is_none() {
            triangles.push(Entity::triangle_from(v1.vertex, v2.vertex, v3.vertex));
        } else {
            triangles.push(Entity::smooth_triangle_from(
                v1.vertex,
                v2.vertex,
                v3.vertex,
                v1.normal.expect("vertex normal exists"),
                v2.normal.expect("vertex normal exists"),
                v3.normal.expect("vertex normal exists"),
            ));
        }
    }

    triangles
}

#[cfg(test)]
mod tests {
    use super::parse_obj;

    use crate::{math::Vec3, physics::Shape};

    use core::panic;

    #[test]
    fn test_parse_obj() {
        {
            let content = r#"There was a young lady named Bright
            who traveled much faster than light.
            She set out one day
            in a relative way,
            and came back the previous night."#;

            let content = String::from(content);

            let obj = parse_obj(content);
            assert_eq!(5, obj.unrecognized);
        }

        {
            let content = r#"v -1 1 0
            v -1.0000 0.5000 0.0000
            v 1 0 0
            v 1 1 0"#;

            let content = String::from(content);

            let obj = parse_obj(content);
            assert_eq!(Vec3::from(-1.0, 1.0, 0.0), obj.vertices[0]);
            assert_eq!(Vec3::from(-1.0, 0.5, 0.0), obj.vertices[1]);
            assert_eq!(Vec3::x_axis(), obj.vertices[2]);
            assert_eq!(Vec3::from(1.0, 1.0, 0.0), obj.vertices[3]);
        }

        {
            let content = r#"v -1 1 0
            v -1 0 0
            v 1 0 0
            v 1 1 0

            f 1 2 3
            f 1 3 4"#;

            let content = String::from(content);

            let obj = parse_obj(content);

            match obj.default_group.borrow().children[0].borrow().shape {
                Shape::Triangle { p1, p2, p3, .. } => {
                    assert_eq!(obj.vertices[0], p1);
                    assert_eq!(obj.vertices[1], p2);
                    assert_eq!(obj.vertices[2], p3);
                }
                _ => panic!("expected Triangle"),
            };

            match obj.default_group.borrow().children[1].borrow().shape {
                Shape::Triangle { p1, p2, p3, .. } => {
                    assert_eq!(obj.vertices[0], p1);
                    assert_eq!(obj.vertices[2], p2);
                    assert_eq!(obj.vertices[3], p3);
                }
                _ => panic!("expected Triangle"),
            };
        }

        {
            let content = r#"v -1 1 0
            v -1 0 0
            v 1 0 0
            v 1 1 0
            v 0 2 0

            f 1 2 3 4 5"#;

            let content = String::from(content);

            let obj = parse_obj(content);

            match obj.default_group.borrow().children[0].borrow().shape {
                Shape::Triangle { p1, p2, p3, .. } => {
                    assert_eq!(obj.vertices[0], p1);
                    assert_eq!(obj.vertices[1], p2);
                    assert_eq!(obj.vertices[2], p3);
                }
                _ => panic!("expected Triangle"),
            };

            match obj.default_group.borrow().children[1].borrow().shape {
                Shape::Triangle { p1, p2, p3, .. } => {
                    assert_eq!(obj.vertices[0], p1);
                    assert_eq!(obj.vertices[2], p2);
                    assert_eq!(obj.vertices[3], p3);
                }
                _ => panic!("expected Triangle"),
            };

            match obj.default_group.borrow().children[2].borrow().shape {
                Shape::Triangle { p1, p2, p3, .. } => {
                    assert_eq!(obj.vertices[0], p1);
                    assert_eq!(obj.vertices[3], p2);
                    assert_eq!(obj.vertices[4], p3);
                }
                _ => panic!("expected Triangle"),
            };
        }

        {
            let content = r#"v -1 1 0
            v -1 0 0
            v 1 0 0
            v 1 1 0

            g FirstGroup
            f 1 2 3

            g SecondGroup
            f 1 3 4"#;

            let content = String::from(content);

            let obj = parse_obj(content);

            assert_eq!(2, obj.groups.len());

            match obj.groups[0].borrow().children[0].borrow().shape {
                Shape::Triangle { p1, p2, p3, .. } => {
                    assert_eq!(obj.vertices[0], p1);
                    assert_eq!(obj.vertices[1], p2);
                    assert_eq!(obj.vertices[2], p3);
                }
                _ => panic!("expected Triangle"),
            };

            match obj.groups[1].borrow().children[0].borrow().shape {
                Shape::Triangle { p1, p2, p3, .. } => {
                    assert_eq!(obj.vertices[0], p1);
                    assert_eq!(obj.vertices[2], p2);
                    assert_eq!(obj.vertices[3], p3);
                }
                _ => panic!("expected Triangle"),
            };
        }

        {
            let content = r#"vn 0 0 1
            vn 0.707 0 -0.707
            vn 1 2 3"#;

            let content = String::from(content);

            let obj = parse_obj(content);

            assert_eq!(3, obj.normals.len());
            assert_eq!(Vec3::z_axis(), obj.normals[0]);
            assert_eq!(Vec3::from(0.707, 0.0, -0.707), obj.normals[1]);
            assert_eq!(Vec3::from(1.0, 2.0, 3.0), obj.normals[2]);
        }

        {
            let content = r#"v 0 1 0
            v -1 0 0
            v 1 0 0

            vn -1 0 0
            vn 1 0 0
            vn 0 1 0

            f 1//3 2//1 3//2
            f 1/0/3 2/102/1 3/14/2"#;

            let content = String::from(content);

            let obj = parse_obj(content);

            match obj.default_group.borrow().children[0].borrow().shape {
                Shape::SmoothTriangle {
                    p1,
                    p2,
                    p3,
                    n1,
                    n2,
                    n3,
                    ..
                } => {
                    assert_eq!(obj.vertices[0], p1);
                    assert_eq!(obj.vertices[1], p2);
                    assert_eq!(obj.vertices[2], p3);

                    assert_eq!(obj.normals[2], n1);
                    assert_eq!(obj.normals[0], n2);
                    assert_eq!(obj.normals[1], n3);
                }
                _ => panic!("expected SmoothTriangle"),
            };

            match obj.default_group.borrow().children[1].borrow().shape {
                Shape::SmoothTriangle {
                    p1,
                    p2,
                    p3,
                    n1,
                    n2,
                    n3,
                    ..
                } => {
                    assert_eq!(obj.vertices[0], p1);
                    assert_eq!(obj.vertices[1], p2);
                    assert_eq!(obj.vertices[2], p3);

                    assert_eq!(obj.normals[2], n1);
                    assert_eq!(obj.normals[0], n2);
                    assert_eq!(obj.normals[1], n3);
                }
                _ => panic!("expected SmoothTriangle"),
            };
        }
    }
}
