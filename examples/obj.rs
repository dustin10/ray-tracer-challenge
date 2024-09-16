use clap::Parser;
use rtc::{
    color,
    entity::World,
    math::{Matrix4, Vec3},
    model::load_entity_from_obj_file,
    render::{render, Camera, PointLight},
};
use tracing::metadata::LevelFilter;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    file: String,
}

const PI_OVER_3: f32 = 3.141592 / 3.0;

fn main() {
    let args = Args::parse();

    init_logging();

    let obj = load_entity_from_obj_file(args.file);
    obj.borrow_mut().material.color = color::CYAN;
    obj.borrow_mut().transform = Matrix4::op_scale(0.25, 0.25, 0.25);

    let mut world = World::new();
    world.set_light(PointLight::from(
        Vec3::from(-20.0, 20.0, -20.0),
        color::WHITE,
    ));

    let _ = world.add_entity_ref(obj);

    let mut camera = Camera::from(640, 480, PI_OVER_3);
    camera.transform = Matrix4::view_transform(
        &Vec3::from(0.0, 2.0, -15.0),
        &Vec3::y_axis(),
        &Vec3::y_axis(),
    );

    let canvas = render(&camera, &world);

    if let Err(e) = std::fs::write("obj.ppm", canvas.export_to_ppm()) {
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
