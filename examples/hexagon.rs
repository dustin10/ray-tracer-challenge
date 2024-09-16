use clap::Parser;
use rtc::{
    color,
    entity::{Entity, World},
    math::{Matrix4, Vec3},
    render::{render, Camera, PointLight},
};
use std::{cell::RefCell, rc::Rc};
use tracing::metadata::LevelFilter;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Cli {}

const PI_OVER_2: f32 = 3.141592 / 2.0;
const PI_OVER_3: f32 = 3.141592 / 3.0;
const PI_OVER_6: f32 = 3.141592 / 6.0;

fn main() {
    let _cli = Cli::parse();

    init_logging();

    let hexagon = hexagon();

    let mut world = World::new();
    world.set_light(PointLight::from(
        Vec3::from(-10.0, 10.0, -10.0),
        color::WHITE,
    ));

    let _ = world.add_entity_ref(hexagon);

    let mut camera = Camera::from(400, 200, PI_OVER_3);
    camera.transform = Matrix4::view_transform(
        &Vec3::from(0.0, 2.0, -6.0),
        &Vec3::y_axis(),
        &Vec3::y_axis(),
    );

    let canvas = render(&camera, &world);

    if let Err(e) = std::fs::write("hexagon.ppm", canvas.export_to_ppm()) {
        tracing::error!("failed to export canvas: {}", e);
    }
}

fn hexagon_corner() -> Rc<RefCell<Entity>> {
    let mut e = Entity::sphere();
    e.transform = Matrix4::op_translate(0.0, 0.0, -1.0) * Matrix4::op_scale(0.25, 0.25, 0.25);

    Rc::new(RefCell::new(e))
}

fn hexagon_edge() -> Rc<RefCell<Entity>> {
    let mut e = Entity::cylinder_from(0.0, 1.0, false);
    e.transform = Matrix4::op_translate(0.0, 0.0, -1.0)
        * Matrix4::op_rotate_y(-PI_OVER_6)
        * Matrix4::op_rotate_z(-PI_OVER_2)
        * Matrix4::op_scale(0.25, 1.0, 0.25);

    Rc::new(RefCell::new(e))
}

fn hexagon_side() -> Rc<RefCell<Entity>> {
    let g = Rc::new(RefCell::new(Entity::group()));

    let corner = hexagon_corner();
    let edge = hexagon_edge();

    Entity::add_child(Rc::clone(&g), Rc::clone(&corner));
    Entity::add_child(Rc::clone(&g), Rc::clone(&edge));

    g
}

fn hexagon() -> Rc<RefCell<Entity>> {
    let hex = Rc::new(RefCell::new(Entity::group()));

    for i in 0..6 {
        let side = hexagon_side();
        side.borrow_mut().transform = Matrix4::op_rotate_y(i as f32 * PI_OVER_3);

        Entity::add_child(Rc::clone(&hex), side);
    }

    hex
}

fn init_logging() {
    tracing_subscriber::fmt()
        .pretty()
        .with_level(true)
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .init();
}
