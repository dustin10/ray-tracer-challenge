use clap::Parser;
use rtc::{
    color::{self, Color},
    entity::{Entity, World},
    math::{Matrix4, Vec3},
    render::{render, Camera, Checkers, Gradient, PointLight, Rings, Stripes},
};
use std::rc::Rc;
use tracing::metadata::LevelFilter;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Cli {}

const PI_OVER_3: f32 = 3.141592 / 3.0;

fn main() {
    let _cli = Cli::parse();

    init_logging();

    let checker = Rc::new(Checkers::new());

    let mut floor = Entity::plane();
    floor.material.pattern = Some(checker);

    let mut stripe_pattern = Stripes::from(color::RED, color::BLUE);
    stripe_pattern.transform = Matrix4::op_scale(0.3, 0.3, 0.3);
    let stripe_pattern = Rc::new(stripe_pattern);

    let mut middle = Entity::sphere();
    middle.transform = Matrix4::op_translate(-0.5, 1.0, 0.5);
    middle.material.color = Color::from(0.1, 1.0, 0.5);
    middle.material.diffuse = 0.7;
    middle.material.specular = 0.3;
    middle.material.pattern = Some(stripe_pattern);

    let gradient = Rc::new(Gradient::from(color::YELLOW, color::GREEN));

    let mut right = Entity::sphere();
    right.transform = Matrix4::op_translate(1.5, 0.5, -0.5) * Matrix4::op_scale(0.5, 0.5, 0.5);
    right.material.color = Color::from(0.5, 1.0, 0.1);
    right.material.diffuse = 0.7;
    right.material.specular = 0.3;
    right.material.pattern = Some(gradient);

    let mut rings = Rings::from(color::ORANGE, color::WHITE);
    rings.transform = Matrix4::op_scale(0.1, 0.1, 0.1);
    let rings = Rc::new(rings);

    let mut left = Entity::sphere();
    left.transform = Matrix4::op_translate(-1.5, 0.33, -0.75) * Matrix4::op_scale(0.33, 0.33, 0.33);
    left.material.color = Color::from(1.0, 0.8, 0.1);
    left.material.diffuse = 0.7;
    left.material.specular = 0.3;
    left.material.pattern = Some(rings);

    let mut world = World::new();
    world.set_light(PointLight::from(
        Vec3::from(-10.0, 10.0, -10.0),
        color::WHITE,
    ));

    let _ = world.add_entity(floor);
    let _ = world.add_entity(middle);
    let _ = world.add_entity(left);
    let _ = world.add_entity(right);

    let mut camera = Camera::from(400, 200, PI_OVER_3);
    camera.transform = Matrix4::view_transform(
        &Vec3::from(0.0, 1.5, -5.0),
        &Vec3::y_axis(),
        &Vec3::y_axis(),
    );

    let canvas = render(&camera, &world);

    if let Err(e) = std::fs::write("pattern.ppm", canvas.export_to_ppm()) {
        tracing::error!("failed to export canvas: {}", e);
    }
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
