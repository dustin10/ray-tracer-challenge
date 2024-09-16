use clap::Parser;
use rtc::{color, math::Vec3, render::Canvas};
use tracing::metadata::LevelFilter;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Cli {}

struct Projectile {
    position: Vec3,
    velocity: Vec3,
}

impl Projectile {
    fn new(position: Vec3, velocity: Vec3) -> Self {
        Projectile { position, velocity }
    }

    fn is_airborne(&self) -> bool {
        self.position.y > 0.0
    }

    fn apply_env(&mut self, env: &Environment) {
        self.position = self.position + self.velocity;
        self.velocity = self.velocity + env.gravity + env.wind;
    }
}

struct Environment {
    gravity: Vec3,
    wind: Vec3,
}

impl Environment {
    fn new(gravity: Vec3, wind: Vec3) -> Self {
        Environment { gravity, wind }
    }
}

fn main() {
    let _cli = Cli::parse();

    init_logging();

    let position = Vec3::y_axis();

    let mut velocity = Vec3::from(1.0, 1.8, 0.0);
    velocity.normalize();

    // adjust to alter flight path
    velocity.scale(11.25);

    let mut proj = Projectile::new(position, velocity);

    let gravity = Vec3::from(0.0, -0.1, 0.0);
    let wind = Vec3::from(-0.01, 0.0, 0.0);

    let env = Environment::new(gravity, wind);

    let mut canvas = Canvas::new(900, 550);

    while proj.is_airborne() {
        proj.apply_env(&env);

        let px = proj.position.x as usize;
        let py = (canvas.height as f32 - proj.position.y) as usize;

        canvas.write_pixel(px, py, &color::GREEN);
    }

    if let Err(e) = std::fs::write("projectile.ppm", canvas.export_to_ppm()) {
        tracing::error!("failed to export canvas: {}", e)
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
