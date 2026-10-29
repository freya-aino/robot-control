//! Reproducible, transfer-inclusive accounting for the implemented CubeCL
//! pendulum RK4 and equality batch paths. This is a debug/dev-build workload,
//! not a throughput benchmark.
//!
//! Run with `cargo run -p robot-sim-gpu --features cubecl --example gpu_resource_repeatability`.

use cubecl::Device;
use robot_sim::constraint::{solve_equality, EqualityRow};
use robot_sim_gpu::{gpu_rk4_pendulum_step, gpu_solve_equality};
use std::hint::black_box;
use std::time::{Duration, Instant};

const ENVS: usize = 4097;
const RUNS: usize = 3;
const RK4_STEPS: usize = 1000;
const RK4_DT: f32 = 0.001;
const G_OVER_L: f32 = 9.81;
const NV: usize = 2;
const NROW: usize = 2;
const EQUALITY_DT: f32 = 0.001;

fn report_times(label: &str, samples: &[Duration]) {
    let mut ordered = samples.to_vec();
    ordered.sort();
    let median = ordered[ordered.len() / 2];
    println!(
        "{label}: timed trials={} (after one untimed warm-up) median={:?} range={:?}..{:?}",
        ordered.len(),
        median,
        ordered[0],
        ordered[ordered.len() - 1]
    );
}

fn rk4_input() -> (Vec<f32>, Vec<f32>) {
    (
        (0..ENVS).map(|i| i as f32 * 0.001 - 2.0).collect(),
        (0..ENVS).map(|i| i as f32 * 0.0005).collect(),
    )
}

fn run_gpu_rk4(theta0: &[f32], omega0: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut theta = theta0.to_vec();
    let mut omega = omega0.to_vec();
    for _ in 0..RK4_STEPS {
        gpu_rk4_pendulum_step(&mut theta, &mut omega, G_OVER_L, RK4_DT).unwrap();
    }
    (theta, omega)
}

fn run_cpu_rk4(theta0: &[f32], omega0: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let mut theta: Vec<f64> = theta0.iter().map(|&x| f64::from(x)).collect();
    let mut omega: Vec<f64> = omega0.iter().map(|&x| f64::from(x)).collect();
    for _ in 0..RK4_STEPS {
        robot_sim::integrators::rk4_step(&mut theta, &mut omega, f64::from(RK4_DT), |q, _| {
            q.iter().map(|x| -f64::from(G_OVER_L) * x.sin()).collect()
        });
    }
    (theta, omega)
}

struct EqualityInput {
    mass: Vec<f32>,
    qvel: Vec<f32>,
    smooth: Vec<f32>,
    jac: Vec<f32>,
    pos: Vec<f32>,
    margin: Vec<f32>,
    solref: Vec<f32>,
    solimp: Vec<f32>,
}

impl EqualityInput {
    fn new() -> Self {
        let mass = [1.5f32, 0.1, 0.1, 1.2].repeat(ENVS);
        let qvel = (0..ENVS * NV)
            .map(|i| (i % 13) as f32 * 0.001 - 0.005)
            .collect();
        let smooth = (0..ENVS * NV)
            .map(|i| (i % 17) as f32 * 0.01 - 0.08)
            .collect();
        let jac = (0..ENVS)
            .flat_map(|e| [1.0, (e % 11) as f32 * 0.01, (e % 7) as f32 * 0.01, 1.0])
            .collect();
        let pos = (0..ENVS * NROW)
            .map(|i| (i % 9) as f32 * 0.0001 - 0.0004)
            .collect();
        Self {
            mass,
            qvel,
            smooth,
            jac,
            pos,
            margin: vec![0.0; ENVS * NROW],
            solref: [0.02, 1.0].repeat(ENVS * NROW),
            solimp: [0.5, 0.5, 0.001, 0.5, 2.0].repeat(ENVS * NROW),
        }
    }

    fn run_gpu(&self) -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>) {
        let mut force = vec![123.0; ENVS * NROW];
        let mut qacc = vec![456.0; ENVS * NV];
        let mut clamped = vec![9u32; ENVS * NROW];
        let mut status = vec![9u32; ENVS];
        gpu_solve_equality(
            NV,
            NROW,
            &self.mass,
            &self.qvel,
            &self.smooth,
            &self.jac,
            &self.pos,
            &self.margin,
            &self.solref,
            &self.solimp,
            EQUALITY_DT,
            &mut force,
            &mut qacc,
            &mut clamped,
            &mut status,
        )
        .unwrap();
        (force, qacc, clamped, status)
    }

    fn run_cpu(&self) -> f64 {
        let mut checksum = 0.0;
        for env in 0..ENVS {
            let rows: Vec<EqualityRow> = (0..NROW)
                .map(|row| {
                    let ri = env * NROW + row;
                    EqualityRow {
                        jacobian: self.jac[ri * NV..(ri + 1) * NV]
                            .iter()
                            .map(|&x| f64::from(x))
                            .collect(),
                        pos: f64::from(self.pos[ri]),
                        solref: [
                            f64::from(self.solref[ri * 2]),
                            f64::from(self.solref[ri * 2 + 1]),
                        ],
                        solimp: std::array::from_fn(|i| f64::from(self.solimp[ri * 5 + i])),
                    }
                })
                .collect();
            let mass: Vec<f64> = self.mass[env * NV * NV..(env + 1) * NV * NV]
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            let qvel: Vec<f64> = self.qvel[env * NV..(env + 1) * NV]
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            let smooth: Vec<f64> = self.smooth[env * NV..(env + 1) * NV]
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            let solution =
                solve_equality(&mass, &qvel, &smooth, &rows, f64::from(EQUALITY_DT)).unwrap();
            checksum += solution.force.iter().sum::<f64>() + solution.qacc.iter().sum::<f64>();
        }
        black_box(checksum)
    }
}

fn main() {
    let client = Device::default().client();
    let properties = client.properties();
    println!("CubeCL backend/compiler: {}", client.name());
    println!("CubeCL adapter: {}", properties.identity.name);
    println!(
        "CubeCL physical adapter identity: {:?}",
        properties.identity.physical
    );
    println!(
        "CubeCL-reported device properties: max_page_size={} B, max_bindings={}, max_memory={:?}",
        properties.memory.max_page_size,
        properties.hardware.max_bindings,
        properties.memory.max_memory()
    );
    println!(
        "These are CubeCL-exposed properties, not live allocation counters; max_bindings is not necessarily a storage-binding-specific limit."
    );
    let profile = if cfg!(debug_assertions) {
        "dev/debug"
    } else {
        "release"
    };
    println!("build profile: cargo run default ({profile})");

    let (theta0, omega0) = rk4_input();
    black_box(run_gpu_rk4(&theta0, &omega0)); // device/client and kernel warm-up
    black_box(run_cpu_rk4(&theta0, &omega0)); // CPU warm-up
    let mut gpu_rk4_times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let start = Instant::now();
        black_box(run_gpu_rk4(&theta0, &omega0));
        gpu_rk4_times.push(start.elapsed());
    }
    let mut cpu_rk4_times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let start = Instant::now();
        black_box(run_cpu_rk4(&theta0, &omega0));
        cpu_rk4_times.push(start.elapsed());
    }
    report_times(
        "RK4 GPU 4097 env x 1000 steps (uploads+readbacks included)",
        &gpu_rk4_times,
    );
    report_times("RK4 CPU f64 4097 env x 1000 steps", &cpu_rk4_times);

    let rk4 = robot_sim_gpu::rk4_resources(ENVS).unwrap();
    println!(
        "RK4 per dispatch (re-allocated every step, not persistent): {} storage bindings, \
         buffers theta_in={} B omega_in={} B theta_out={} B omega_out={} B, total={} B",
        rk4.bindings,
        rk4.theta_in_bytes,
        rk4.omega_in_bytes,
        rk4.theta_out_bytes,
        rk4.omega_out_bytes,
        rk4.allocated_bytes()
    );
    println!(
        "RK4 transfer per dispatch: upload={} B readback={} B; over {RK4_STEPS} steps: upload+readback={} B",
        rk4.upload_bytes(),
        rk4.readback_bytes(),
        (rk4.upload_bytes() + rk4.readback_bytes()) * RK4_STEPS
    );

    let equality = EqualityInput::new();
    black_box(equality.run_gpu()); // warm-up
    black_box(equality.run_cpu()); // CPU warm-up
    let mut gpu_equality_times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let start = Instant::now();
        black_box(equality.run_gpu());
        gpu_equality_times.push(start.elapsed());
    }
    let mut cpu_equality_times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let start = Instant::now();
        black_box(equality.run_cpu());
        cpu_equality_times.push(start.elapsed());
    }
    report_times(
        "Equality GPU 4097 env, nv=2, nrow=2 (packing+transfers included)",
        &gpu_equality_times,
    );
    report_times(
        "Equality CPU f64 solve per environment",
        &cpu_equality_times,
    );

    let eq = robot_sim_gpu::equality_resources(NV, NROW, ENVS).unwrap();
    println!(
        "Equality per dispatch: {} storage bindings, input={} B output={} B flags={} B, total={} B",
        eq.bindings,
        eq.input_bytes,
        eq.output_bytes,
        eq.flags_bytes,
        eq.allocated_bytes()
    );
    println!(
        "Equality transfer per dispatch: upload={} B readback={} B, total={} B",
        eq.upload_bytes(),
        eq.readback_bytes(),
        eq.upload_bytes() + eq.readback_bytes()
    );
    println!("These are count-only formulas from the launch packing, not adapter memory counters.");
}
