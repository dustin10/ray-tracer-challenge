use clap::Parser;
use rtc::{
    color::{self, Color},
    entity::Entity,
    math::{Ray, Vec3},
    physics,
    render::{self, Canvas, PointLight},
};
use std::{cell::RefCell, rc::Rc};
use tracing::metadata::LevelFilter;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Cli {}

fn main() {
    let _cli = Cli::parse();

    init_logging();

    let origin = Vec3::from(0.0, 0.0, -5.0);

    let mut sphere = Entity::sphere();
    sphere.material.color = Color::from(1.0, 0.2, 1.0);
    let sphere = Rc::new(RefCell::new(sphere));

    let light_color = color::WHITE;
    let light_position = Vec3::from(-10.0, 10.0, -10.0);
    let light = PointLight::from(light_position, light_color);

    let wall_z = 10.0;
    let wall_size = 7.0;

    let pixel_size = wall_size / 300.0;
    let half = wall_size / 2.0;

    let mut canvas = Canvas::new(300, 300);

    for y in 0..canvas.height {
        let world_y = half - (pixel_size * y as f32);

        for x in 0..canvas.width {
            let world_x = -half + (pixel_size * x as f32);

            let target = Vec3::from(world_x, world_y, wall_z);
            let mut dir = target - origin;
            dir.normalize();

            let ray = Ray::from(origin, dir);

            let interstections = physics::intersect(&ray, Rc::clone(&sphere));

            if let Some(hit) = interstections.hit() {
                let point = Ray::position(&ray, hit.t);
                let normal = physics::normal_at(Rc::clone(&hit.entity), &point, hit);
                let eye = Vec3::from_scaled(&ray.direction, -1.0);

                let color = render::lighting(
                    &hit.entity.borrow().material,
                    Rc::clone(&hit.entity),
                    &light,
                    &point,
                    &eye,
                    &normal,
                    false,
                );

                canvas.write_pixel(x, y, &color);
            }
        }
    }

    if let Err(e) = std::fs::write("reflection.ppm", canvas.export_to_ppm()) {
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
