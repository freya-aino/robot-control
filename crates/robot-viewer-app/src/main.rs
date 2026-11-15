//! The offline replay viewer (ADR 0019 first deliverable): a Bevy window
//! over `robot-viewer`'s recorded runs.
//!
//! Loads a committed frame-dump directory or a `viewer-frames/1` run from a
//! blob store, renders the robot's collision geometry — the same 33 shapes
//! the simulator contacts, hulls included — with a skeleton overlay, and
//! plays it back.
//!
//! Controls: `Space` play/pause, `Left`/`Right` step one frame,
//! `Up`/`Down` speed, `Home` restart, left-mouse drag orbits, scroll zooms.
//! The window title carries the frame, time and speed.
//!
//! The project's canonical frame is Z-up (ADR 0022) while Bevy renders
//! Y-up; every robot-frame entity lives under one root rotated −90° about
//! X, so robot coordinates are used as-is below it.
//!
//! `--dump` prints the loaded run's shape and frame summary and exits
//! without opening a window — the headless path CI can exercise.

use bevy::ecs::message::MessageReader;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use robot_model::pose::RigidPose;
use robot_viewer_app::{bones, ReplayData, ShapeKind};
use std::path::PathBuf;

/// Parsed command line.
struct Args {
    path: PathBuf,
    robot: String,
    store_id: Option<String>,
    dump: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut path = None;
    let mut robot = "g1-usd".to_owned();
    let mut store_id = None;
    let mut dump = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--robot" => robot = args.next().ok_or("--robot needs a value")?,
            "--id" => store_id = Some(args.next().ok_or("--id needs a value")?),
            "--dump" => dump = true,
            other if !other.starts_with("--") => path = Some(PathBuf::from(other)),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(Args {
        path: path.ok_or("usage: robot-viewer-app <run-dir|store-dir> [--id sha256:<hex>] [--robot g1-usd|g1] [--dump]")?,
        robot,
        store_id,
        dump,
    })
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(64);
        }
    };
    let replay = match &args.store_id {
        Some(id) => ReplayData::load_store(&args.path, id, &args.robot),
        None => ReplayData::load_dir(&args.path, &args.robot),
    };
    let replay = match replay {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "loaded {} frames over {:.3} s, {} shapes, robot {}",
        replay.frame_count(),
        replay.time_of(replay.frame_count().saturating_sub(1)),
        replay.shapes().len(),
        args.robot
    );
    if args.dump {
        if let Ok(poses) = replay.link_poses(0) {
            for (link, pose) in replay.profile.links.iter().zip(&poses) {
                println!(
                    "  {} at ({:.3}, {:.3}, {:.3})",
                    link.name.as_str(),
                    pose.translation.x,
                    pose.translation.y,
                    pose.translation.z
                );
            }
        }
        return;
    }

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "robot replay".to_owned(),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .insert_resource(Replay(replay))
        .insert_resource(Playback {
            frame: 0,
            playing: true,
            speed: 1.0,
            accumulator: 0.0,
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                controls,
                advance,
                pose_shapes,
                draw_overlay,
                orbit_camera,
                title,
            ),
        )
        .run();
}

/// The loaded run.
#[derive(Resource)]
struct Replay(ReplayData);

/// Where playback stands.
#[derive(Resource)]
struct Playback {
    frame: usize,
    playing: bool,
    speed: f64,
    accumulator: f64,
}

/// A rendered collision shape: which link carries it and where on the link.
#[derive(Component)]
struct LinkShape {
    link: usize,
    local: RigidPose,
}

/// The Z-up root every robot-frame entity hangs under.
#[derive(Component)]
struct ZUpRoot;

/// The orbiting camera's spherical state, in Bevy (Y-up) coordinates.
#[derive(Component)]
struct Orbit {
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: Vec3,
}

fn to_vec3(p: robot_types::frames::Vec3) -> Vec3 {
    Vec3::new(p.x as f32, p.y as f32, p.z as f32)
}

fn to_quat(q: robot_types::frames::Quat) -> Quat {
    Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32)
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    replay: Res<Replay>,
) {
    // The canonical Z-up world under one rotated root (module doc).
    let root = commands
        .spawn((
            ZUpRoot,
            Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
            Visibility::default(),
        ))
        .id();

    let body = materials.add(StandardMaterial {
        base_color: Color::srgb(0.75, 0.78, 0.82),
        perceptual_roughness: 0.6,
        ..Default::default()
    });
    let ground = materials.add(StandardMaterial {
        base_color: Color::srgb(0.25, 0.3, 0.33),
        perceptual_roughness: 0.95,
        ..Default::default()
    });

    for shape in replay.0.shapes() {
        let mesh = match &shape.kind {
            ShapeKind::Sphere { radius } => meshes.add(Sphere::new(*radius as f32)),
            ShapeKind::Box { half } => meshes.add(Cuboid::new(
                2.0 * half[0] as f32,
                2.0 * half[1] as f32,
                2.0 * half[2] as f32,
            )),
            // Bevy's cylinder stands along Y; the shape frame's is along Z.
            // The extra local rotation is folded into the entity transform
            // each frame, so here the mesh is built pre-rotated instead:
            // a cylinder mesh rotated +90° about X puts its axis on Z.
            ShapeKind::Cylinder {
                radius,
                half_height,
            } => {
                let mut m = Mesh::from(Cylinder::new(*radius as f32, 2.0 * *half_height as f32));
                m.rotate_by(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
                meshes.add(m)
            }
            ShapeKind::Hull { triangles } => meshes.add(hull_mesh(triangles)),
        };
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(body.clone()),
            Transform::IDENTITY,
            LinkShape {
                link: shape.link,
                local: shape.local,
            },
            ChildOf(root),
        ));
    }

    // The ground plane at z = 0 in robot coordinates.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(8.0, 8.0, 0.02))),
        MeshMaterial3d(ground),
        Transform::from_xyz(0.0, 0.0, -0.01),
        ChildOf(root),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadow_maps_enabled: true,
            ..Default::default()
        },
        Transform::from_xyz(3.0, 6.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Camera3d::default(),
        // Ambient light is per-camera in this Bevy release.
        AmbientLight {
            color: Color::WHITE,
            brightness: 250.0,
            ..Default::default()
        },
        Transform::from_xyz(2.5, 1.6, 2.5).looking_at(Vec3::new(0.0, 0.8, 0.0), Vec3::Y),
        Orbit {
            yaw: 0.8,
            pitch: 0.35,
            distance: 3.5,
            target: Vec3::new(0.0, 0.8, 0.0),
        },
    ));
}

/// A flat-shaded mesh from the hull's outward-wound triangles.
fn hull_mesh(triangles: &[[f64; 3]]) -> Mesh {
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::PrimitiveTopology;
    let positions: Vec<[f32; 3]> = triangles
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    let mut normals = Vec::with_capacity(positions.len());
    for tri in positions.chunks_exact(3) {
        let a = Vec3::from(tri[0]);
        let b = Vec3::from(tri[1]);
        let c = Vec3::from(tri[2]);
        let n = (b - a).cross(c - a).normalize_or(Vec3::Z).to_array();
        normals.extend([n, n, n]);
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
}

fn controls(keys: Res<ButtonInput<KeyCode>>, mut playback: ResMut<Playback>, replay: Res<Replay>) {
    let last = replay.0.frame_count().saturating_sub(1);
    if keys.just_pressed(KeyCode::Space) {
        playback.playing = !playback.playing;
    }
    if keys.just_pressed(KeyCode::ArrowRight) {
        playback.playing = false;
        playback.frame = (playback.frame + 1).min(last);
    }
    if keys.just_pressed(KeyCode::ArrowLeft) {
        playback.playing = false;
        playback.frame = playback.frame.saturating_sub(1);
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        playback.speed = (playback.speed * 2.0).min(64.0);
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        playback.speed = (playback.speed * 0.5).max(1.0 / 64.0);
    }
    if keys.just_pressed(KeyCode::Home) {
        playback.frame = 0;
        playback.accumulator = 0.0;
    }
}

fn advance(time: Res<Time>, mut playback: ResMut<Playback>, replay: Res<Replay>) {
    if !playback.playing || replay.0.frame_count() < 2 {
        return;
    }
    // The recording's own step, inferred from its first two timestamps.
    let dt = (replay.0.time_of(1) - replay.0.time_of(0)).max(1e-6);
    playback.accumulator += time.delta_secs_f64() * playback.speed;
    while playback.accumulator >= dt {
        playback.accumulator -= dt;
        playback.frame += 1;
        if playback.frame >= replay.0.frame_count() {
            playback.frame = 0; // loop
        }
    }
}

fn pose_shapes(
    playback: Res<Playback>,
    replay: Res<Replay>,
    mut shapes: Query<(&LinkShape, &mut Transform)>,
) {
    let Ok(poses) = replay.0.link_poses(playback.frame) else {
        return;
    };
    for (shape, mut transform) in &mut shapes {
        let Some(link) = poses.get(shape.link) else {
            continue;
        };
        let world = link.compose(&shape.local);
        transform.translation = to_vec3(world.translation);
        transform.rotation = to_quat(world.rotation);
    }
}

fn draw_overlay(
    playback: Res<Playback>,
    replay: Res<Replay>,
    roots: Query<&Transform, With<ZUpRoot>>,
    mut gizmos: Gizmos,
) {
    let Ok(root) = roots.single() else {
        return;
    };
    let Ok(poses) = replay.0.link_poses(playback.frame) else {
        return;
    };
    // Bones, drawn in world (Bevy) space through the Z-up root's transform.
    let map = |p: [f64; 3]| root.transform_point(Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32));
    for (a, b) in bones(&replay.0.profile, &poses) {
        gizmos.line(map(a), map(b), Color::srgb(0.95, 0.55, 0.15));
    }
    // A metre grid on the ground.
    for i in -4..=4 {
        let i = i as f32;
        gizmos.line(
            map([i as f64, -4.0, 0.0]),
            map([i as f64, 4.0, 0.0]),
            Color::srgba(1.0, 1.0, 1.0, 0.12),
        );
        gizmos.line(
            map([-4.0, i as f64, 0.0]),
            map([4.0, i as f64, 0.0]),
            Color::srgba(1.0, 1.0, 1.0, 0.12),
        );
    }
}

fn orbit_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut cameras: Query<(&mut Orbit, &mut Transform)>,
) {
    let Ok((mut orbit, mut transform)) = cameras.single_mut() else {
        return;
    };
    if buttons.pressed(MouseButton::Left) {
        for m in motion.read() {
            orbit.yaw -= m.delta.x * 0.01;
            orbit.pitch = (orbit.pitch + m.delta.y * 0.01).clamp(-1.5, 1.5);
        }
    } else {
        motion.clear();
    }
    for w in wheel.read() {
        orbit.distance = (orbit.distance * (1.0 - w.y * 0.1)).clamp(0.5, 30.0);
    }
    let (sy, cy) = orbit.yaw.sin_cos();
    let (sp, cp) = orbit.pitch.sin_cos();
    let offset = Vec3::new(cy * cp, sp, sy * cp) * orbit.distance;
    *transform =
        Transform::from_translation(orbit.target + offset).looking_at(orbit.target, Vec3::Y);
}

fn title(
    playback: Res<Playback>,
    replay: Res<Replay>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    let Ok(mut window) = windows.single_mut() else {
        return;
    };
    window.title = format!(
        "robot replay — frame {}/{} · t={:.3}s · {}x {}",
        playback.frame + 1,
        replay.0.frame_count(),
        replay.0.time_of(playback.frame),
        playback.speed,
        if playback.playing { "▶" } else { "⏸" },
    );
}
