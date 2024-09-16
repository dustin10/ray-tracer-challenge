use clap::Parser;
use rtc::{
    color::{self, Color},
    entity::{Entity, World},
    math::{Matrix4, Vec3},
    render::{render, Camera, Material, PointLight},
};
use std::f32::consts::PI;
use tracing::metadata::LevelFilter;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Cli {}

fn main() {
    let _cli = Cli::parse();

    init_logging();

    let mut white_material = Material::new();
    white_material.color = color::WHITE;
    white_material.diffuse = 0.7;
    white_material.ambient = 0.1;
    white_material.specular = 0.0;
    white_material.reflective = 0.1;

    let mut blue_material = Material::new();
    blue_material.color = Color::from(0.537, 0.831, 0.914);
    blue_material.diffuse = white_material.diffuse;
    blue_material.ambient = white_material.ambient;
    blue_material.specular = white_material.specular;
    blue_material.reflective = white_material.specular;

    let mut red_material = Material::new();
    red_material.color = Color::from(0.941, 0.322, 0.388);
    red_material.diffuse = white_material.diffuse;
    red_material.ambient = white_material.ambient;
    red_material.specular = white_material.specular;
    red_material.reflective = white_material.specular;

    let mut purple_material = Material::new();
    purple_material.color = Color::from(0.373, 0.404, 0.550);
    purple_material.diffuse = white_material.diffuse;
    purple_material.ambient = white_material.ambient;
    purple_material.specular = white_material.specular;
    purple_material.reflective = white_material.specular;

    let standard_transform =
        Matrix4::op_scale(0.5, 0.5, 0.5) * Matrix4::op_translate(1.0, -1.0, 1.0);

    let large_obj_transform = Matrix4::op_scale(3.5, 3.5, 3.5) * standard_transform;

    let medium_obj_transform = Matrix4::op_scale(3.0, 3.0, 3.0) * standard_transform;

    let small_obj_transform = Matrix4::op_scale(2.0, 2.0, 2.0) * standard_transform;

    let mut plane = Entity::plane();
    plane.material = Material::new();
    plane.material.color = color::WHITE;
    plane.material.ambient = 1.0;
    plane.material.diffuse = 0.0;
    plane.material.specular = 0.0;
    plane.transform = Matrix4::op_translate(0.0, 0.0, 500.0) * Matrix4::op_rotate_x(PI / 2.0);

    let mut sphere = Entity::sphere();
    sphere.material = Material::new();
    sphere.material.color = Color::from(0.373, 0.404, 0.550);
    sphere.material.diffuse = 0.2;
    sphere.material.ambient = 0.0;
    sphere.material.specular = 1.0;
    sphere.material.shininess = 200.0;
    sphere.material.reflective = 0.7;
    sphere.material.transparency = 0.7;
    sphere.material.refractive_index = 1.5;
    sphere.transform = large_obj_transform;

    let mut cube1 = Entity::cube();
    cube1.material = white_material.clone();
    cube1.transform = Matrix4::op_translate(4.0, 0.0, 0.0) * medium_obj_transform;

    let mut cube2 = Entity::cube();
    cube2.material = blue_material.clone();
    cube2.transform = Matrix4::op_translate(8.5, 1.5, -0.5) * large_obj_transform;

    let mut cube3 = Entity::cube();
    cube3.material = red_material.clone();
    cube3.transform = Matrix4::op_translate(0.0, 0.0, 4.0) * large_obj_transform;

    let mut cube4 = Entity::cube();
    cube4.material = white_material.clone();
    cube4.transform = Matrix4::op_translate(4.0, 0.0, 4.0) * small_obj_transform;

    let mut cube5 = Entity::cube();
    cube5.material = purple_material.clone();
    cube5.transform = Matrix4::op_translate(7.5, 0.5, 4.0) * medium_obj_transform;

    let mut cube6 = Entity::cube();
    cube6.material = white_material.clone();
    cube6.transform = Matrix4::op_translate(-0.25, 0.25, 8.0) * medium_obj_transform;

    let mut cube7 = Entity::cube();
    cube7.material = blue_material.clone();
    cube7.transform = Matrix4::op_translate(4.0, 1.0, 7.5) * large_obj_transform;

    let mut cube8 = Entity::cube();
    cube8.material = red_material.clone();
    cube8.transform = Matrix4::op_translate(10.0, 2.0, 7.5) * medium_obj_transform;

    let mut cube9 = Entity::cube();
    cube9.material = white_material.clone();
    cube9.transform = Matrix4::op_translate(8.0, 2.0, 12.0) * small_obj_transform;

    let mut cube10 = Entity::cube();
    cube10.material = white_material.clone();
    cube10.transform = Matrix4::op_translate(20.0, 1.0, 12.0) * small_obj_transform;

    let mut cube11 = Entity::cube();
    cube11.material = blue_material.clone();
    cube11.transform = Matrix4::op_translate(-0.5, -5.0, 0.25) * large_obj_transform;

    let mut cube12 = Entity::cube();
    cube12.material = red_material.clone();
    cube12.transform = Matrix4::op_translate(4.0, -4.0, 0.0) * large_obj_transform;

    let mut cube13 = Entity::cube();
    cube13.material = white_material.clone();
    cube13.transform = Matrix4::op_translate(8.5, -4.0, 0.0) * large_obj_transform;

    let mut cube14 = Entity::cube();
    cube14.material = white_material.clone();
    cube14.transform = Matrix4::op_translate(0.0, -4.0, 4.0) * large_obj_transform;

    let mut cube15 = Entity::cube();
    cube15.material = purple_material.clone();
    cube15.transform = Matrix4::op_translate(-0.5, -4.5, 8.0) * large_obj_transform;

    let mut cube16 = Entity::cube();
    cube16.material = white_material.clone();
    cube16.transform = Matrix4::op_translate(0.0, -8.0, 4.0) * large_obj_transform;

    let mut cube17 = Entity::cube();
    cube17.material = white_material.clone();
    cube17.transform = Matrix4::op_translate(-0.5, -8.5, 8.0) * large_obj_transform;

    let mut world = World::new();
    world.set_light(PointLight::from(
        Vec3::from(-50.0, 100.0, -50.0),
        color::WHITE,
    ));

    let _ = world.add_entity(plane);
    let _ = world.add_entity(sphere);
    let _ = world.add_entity(cube1);
    let _ = world.add_entity(cube2);
    let _ = world.add_entity(cube3);
    let _ = world.add_entity(cube4);
    let _ = world.add_entity(cube5);
    let _ = world.add_entity(cube6);
    let _ = world.add_entity(cube7);
    let _ = world.add_entity(cube8);
    let _ = world.add_entity(cube9);
    let _ = world.add_entity(cube10);
    let _ = world.add_entity(cube11);
    let _ = world.add_entity(cube12);
    let _ = world.add_entity(cube13);
    let _ = world.add_entity(cube14);
    let _ = world.add_entity(cube15);
    let _ = world.add_entity(cube16);
    let _ = world.add_entity(cube17);

    let mut camera = Camera::from(640, 480, 0.785);
    camera.transform = Matrix4::view_transform(
        &Vec3::from(-6.0, 6.0, -10.0),
        &Vec3::from(6.0, 0.0, 6.0),
        &Vec3::from(-0.45, 1.0, 0.0),
    );

    let canvas = render(&camera, &world);

    if let Err(e) = std::fs::write("cover.ppm", canvas.export_to_ppm()) {
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
