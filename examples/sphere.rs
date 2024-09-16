use clap::Parser;
use rtc::{entity::Entity, math::{Ray, Vec3}, render::Canvas, color, physics};
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
    let sphere = Rc::new(RefCell::new(Entity::sphere()));

    let wall_z = 10.0;
    let wall_size = 7.0;

    let pixel_size = wall_size / 100.0;
    let half = wall_size / 2.0;

    let mut canvas = Canvas::new(100, 100);

    for y in 0..canvas.height {
        let world_y = half - (pixel_size * y as f32);

        for x in 0..canvas.width {
            let world_x = -half + (pixel_size * x as f32);

            let target = Vec3::from(world_x, world_y, wall_z);
            let mut dir = target - origin;
            dir.normalize();

            let ray = Ray::from(origin, dir);

            let interstections = physics::intersect(&ray, Rc::clone(&sphere));

            if let Some(_) = interstections.hit() {
                canvas.write_pixel(x, y, &color::RED);
            }
        }
    }

    if let Err(e) = std::fs::write("sphere.ppm", canvas.export_to_ppm()) {
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
