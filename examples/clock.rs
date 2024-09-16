use clap::Parser;
use rtc::{color, math::{Matrix4, Vec3}, render::Canvas};
use tracing::metadata::LevelFilter;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Cli {}

fn main() {
    let _cli = Cli::parse();

    init_logging();

    let mut canvas = Canvas::new(600, 600);

    let noon = Vec3::z_axis();

    for i in 0..12 {
        let rot = Matrix4::op_rotate_y(i as f32 * (std::f32::consts::PI / 6.0));
        let h = Matrix4::mul_vec3(&rot, &noon, 0.0);

        let cx = ((h.x * canvas.width as f32 * 0.4) + (canvas.width / 2) as f32) as usize;
        let cy = ((h.z * canvas.width as f32 * 0.4) + (canvas.width / 2) as f32) as usize;

        canvas.write_pixel(cx, cy, &color::GREEN);
    }

    if let Err(e) = std::fs::write("clock.ppm", canvas.export_to_ppm()) {
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
