//! Measured facts for the GPU pendulum RK4 and constraint impedance kernels:
//! max error against the CPU f64 reference (with index), bitwise determinism
//! over repeated identical launches, and wall-clock latency of a fixed
//! workload on GPU and CPU. Not a throughput benchmark.
//!
//! Run: `cargo run -p robot-sim-gpu --features cubecl --example gpu_latency`

use robot_sim_gpu::{gpu_constraint_impedance_with_derivative, gpu_rk4_pendulum_step};
use std::time::Instant;

const ENVS: usize = 4097;
const STEPS: usize = 1000;
const G_OVER_L: f32 = 9.81;
const DT: f32 = 0.001;
const DETERMINISM_RUNS: usize = 4;
const IMPEDANCE_LAUNCHES: usize = 100;

fn max_abs_with_index(gpu: &[f64], cpu: &[f64]) -> (f64, usize) {
    gpu.iter()
        .zip(cpu)
        .enumerate()
        .map(|(i, (g, c))| ((g - c).abs(), i))
        .fold((0.0, 0), |acc, (d, i)| if d > acc.0 { (d, i) } else { acc })
}

fn gpu_rk4_trajectory(theta0: &[f32], omega0: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut th = theta0.to_vec();
    let mut om = omega0.to_vec();
    for _ in 0..STEPS {
        gpu_rk4_pendulum_step(&mut th, &mut om, G_OVER_L, DT).unwrap();
    }
    (th, om)
}

fn main() {
    let theta0: Vec<f32> = (0..ENVS).map(|i| (i as f32) * 0.001 - 2.0).collect();
    let omega0: Vec<f32> = (0..ENVS).map(|i| (i as f32) * 0.0005).collect();

    // Warm-up launch: includes device/client initialisation and kernel compile.
    let mut warm_th = theta0.clone();
    let mut warm_om = omega0.clone();
    let t = Instant::now();
    gpu_rk4_pendulum_step(&mut warm_th, &mut warm_om, G_OVER_L, DT).unwrap();
    let first_launch = t.elapsed();

    let t = Instant::now();
    let (gpu_th, gpu_om) = gpu_rk4_trajectory(&theta0, &omega0);
    let gpu_steady = t.elapsed();

    let mut cpu_th: Vec<f64> = theta0.iter().map(|&x| x as f64).collect();
    let mut cpu_om: Vec<f64> = omega0.iter().map(|&x| x as f64).collect();
    let t = Instant::now();
    for _ in 0..STEPS {
        robot_sim::integrators::rk4_step(&mut cpu_th, &mut cpu_om, DT as f64, |q, _| {
            q.iter().map(|x| -(G_OVER_L as f64) * x.sin()).collect()
        });
    }
    let cpu_time = t.elapsed();

    let gpu_th64: Vec<f64> = gpu_th.iter().map(|&x| x as f64).collect();
    let gpu_om64: Vec<f64> = gpu_om.iter().map(|&x| x as f64).collect();
    let (th_err, th_idx) = max_abs_with_index(&gpu_th64, &cpu_th);
    let (om_err, om_idx) = max_abs_with_index(&gpu_om64, &cpu_om);
    println!("rk4 pendulum envs={ENVS} steps={STEPS} dt={DT} g/L={G_OVER_L}");
    println!("rk4 max |theta_gpu - theta_cpu| = {th_err:.6e} at env {th_idx}");
    println!("rk4 max |omega_gpu - omega_cpu| = {om_err:.6e} at env {om_idx}");
    println!("rk4 first launch (incl. init/compile) = {first_launch:?}");
    println!("rk4 GPU 1000-step trajectory (host round trip per step) = {gpu_steady:?}");
    println!("rk4 CPU f64 1000-step trajectory = {cpu_time:?}");

    let mut identical = 0usize;
    for _ in 0..DETERMINISM_RUNS {
        let (th, om) = gpu_rk4_trajectory(&theta0, &omega0);
        if th == gpu_th && om == gpu_om {
            identical += 1;
        }
    }
    println!("determinism: {identical}/{DETERMINISM_RUNS} repeated trajectories bitwise identical");

    let solimp: Vec<[f32; 5]> = (0..ENVS)
        .map(|i| {
            let w = 0.001 + (i % 7) as f32 * 0.002;
            [0.9, 0.95, w, 0.5, 1.0 + (i % 3) as f32]
        })
        .collect();
    let pos: Vec<f32> = (0..ENVS).map(|i| (i as f32) * 1e-4 - 0.2).collect();
    let margin = vec![0.0f32; ENVS];
    let mut imp = vec![0.0f32; ENVS];
    let mut imp_prime = vec![0.0f32; ENVS];

    gpu_constraint_impedance_with_derivative(&solimp, &pos, &margin, &mut imp, &mut imp_prime)
        .unwrap();

    let mut cpu_imp = vec![0.0f64; ENVS];
    let mut cpu_imp_prime = vec![0.0f64; ENVS];
    for i in 0..ENVS {
        let s = robot_sim::constraint::clamp_solimp([
            solimp[i][0] as f64,
            solimp[i][1] as f64,
            solimp[i][2] as f64,
            solimp[i][3] as f64,
            solimp[i][4] as f64,
        ]);
        let (d, dp) = robot_sim::constraint::impedance(s, pos[i] as f64, margin[i] as f64);
        cpu_imp[i] = d;
        cpu_imp_prime[i] = dp;
    }
    let imp64: Vec<f64> = imp.iter().map(|&x| x as f64).collect();
    let imp_prime64: Vec<f64> = imp_prime.iter().map(|&x| x as f64).collect();
    let (d_err, d_idx) = max_abs_with_index(&imp64, &cpu_imp);
    let (dp_err, dp_idx) = max_abs_with_index(&imp_prime64, &cpu_imp_prime);
    println!("impedance rows={ENVS}");
    println!("impedance max |d_gpu - d_cpu| = {d_err:.6e} at row {d_idx}");
    println!("impedance max |impP_gpu - impP_cpu| = {dp_err:.6e} at row {dp_idx}");

    let t = Instant::now();
    for _ in 0..IMPEDANCE_LAUNCHES {
        gpu_constraint_impedance_with_derivative(&solimp, &pos, &margin, &mut imp, &mut imp_prime)
            .unwrap();
    }
    let gpu_imp = t.elapsed() / IMPEDANCE_LAUNCHES as u32;

    let t = Instant::now();
    for _ in 0..IMPEDANCE_LAUNCHES {
        for i in 0..ENVS {
            let s = robot_sim::constraint::clamp_solimp([
                solimp[i][0] as f64,
                solimp[i][1] as f64,
                solimp[i][2] as f64,
                solimp[i][3] as f64,
                solimp[i][4] as f64,
            ]);
            let _ = robot_sim::constraint::impedance(s, pos[i] as f64, margin[i] as f64);
        }
    }
    let cpu_imp_time = t.elapsed() / IMPEDANCE_LAUNCHES as u32;
    println!("impedance GPU mean per launch over {IMPEDANCE_LAUNCHES} = {gpu_imp:?}");
    println!("impedance CPU mean per pass over {IMPEDANCE_LAUNCHES} = {cpu_imp_time:?}");
}
