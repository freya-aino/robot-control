#[cfg(feature = "cubecl")]
use robot_sim_gpu::gpu_rk4_pendulum_step;

#[cfg(feature = "cubecl")]
#[test]
fn rk4_4097_environment_1000_step_trajectories_repeat_bitwise() {
    const ENVS: usize = 4097;
    const STEPS: usize = 1000;
    const RUNS: usize = 5;
    const G_OVER_L: f32 = 9.81;
    const DT: f32 = 0.001;

    let theta0: Vec<f32> = (0..ENVS).map(|i| i as f32 * 0.001 - 2.0).collect();
    let omega0: Vec<f32> = (0..ENVS).map(|i| i as f32 * 0.0005).collect();
    let trajectory = || {
        let mut theta = theta0.clone();
        let mut omega = omega0.clone();
        for _ in 0..STEPS {
            gpu_rk4_pendulum_step(&mut theta, &mut omega, G_OVER_L, DT).unwrap();
        }
        (theta, omega)
    };

    let (expected_theta, expected_omega) = trajectory();
    for run in 1..RUNS {
        let (theta, omega) = trajectory();
        assert_eq!(
            theta.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected_theta
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            "theta differs bitwise on identical trajectory run {run}"
        );
        assert_eq!(
            omega.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected_omega
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            "omega differs bitwise on identical trajectory run {run}"
        );
    }
}

#[cfg(feature = "cubecl")]
#[test]
fn equality_4097_environment_batch_repeats_bitwise() {
    use robot_sim_gpu::gpu_solve_equality;

    const ENVS: usize = 4097;
    const NV: usize = 2;
    const NROW: usize = 2;
    const RUNS: usize = 5;
    let mass = [1.5f32, 0.1, 0.1, 1.2].repeat(ENVS);
    let qvel: Vec<f32> = (0..ENVS * NV)
        .map(|i| (i % 13) as f32 * 0.001 - 0.005)
        .collect();
    let smooth: Vec<f32> = (0..ENVS * NV)
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect();
    let jac: Vec<f32> = (0..ENVS)
        .flat_map(|e| [1.0, (e % 11) as f32 * 0.01, (e % 7) as f32 * 0.01, 1.0])
        .collect();
    let pos: Vec<f32> = (0..ENVS * NROW)
        .map(|i| (i % 9) as f32 * 0.0001 - 0.0004)
        .collect();
    let margin = vec![0.0; ENVS * NROW];
    let mut solref = [0.02, 1.0].repeat(ENVS * NROW);
    // Env 0 has solref[0] = 0.001 < 2*dt, so its rows must be clamped.
    for row in 0..NROW {
        solref[row * 2] = 0.001;
    }
    let solimp = [0.5, 0.5, 0.001, 0.5, 2.0].repeat(ENVS * NROW);
    let initial_force = vec![123.0; ENVS * NROW];
    let initial_qacc = vec![456.0; ENVS * NV];
    let initial_clamped = vec![9u32; ENVS * NROW];
    let initial_status = vec![9u32; ENVS];

    let solve = || {
        let mut force = initial_force.clone();
        let mut qacc = initial_qacc.clone();
        let mut clamped = initial_clamped.clone();
        let mut status = initial_status.clone();
        gpu_solve_equality(
            NV,
            NROW,
            &mass,
            &qvel,
            &smooth,
            &jac,
            &pos,
            &margin,
            &solref,
            &solimp,
            0.001,
            &mut force,
            &mut qacc,
            &mut clamped,
            &mut status,
        )
        .unwrap();
        (force, qacc, clamped, status)
    };

    let expected = solve();
    // Guard: first run must be physically valid (status=1) and not sentinel (123/456).
    assert!(
        expected.3.iter().all(|&s| s == 1),
        "First equality run status invalid"
    );
    assert!(
        !expected.0.contains(&123.0),
        "First equality run contains sentinel force"
    );
    assert!(
        !expected.1.contains(&456.0),
        "First equality run contains sentinel qacc"
    );
    assert!(
        expected.2[..NROW].iter().all(|&c| c == 1),
        "Env 0 with solref[0] < 2*dt must set clamped flags"
    );
    assert!(
        !expected.2.contains(&9),
        "First equality run left a sentinel clamped flag"
    );

    for run in 1..RUNS {
        let actual = solve();
        assert_eq!(
            actual.0.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.0.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "force differs bitwise on identical equality run {run}"
        );
        assert_eq!(
            actual.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "qacc differs bitwise on identical equality run {run}"
        );
        assert_eq!(actual.2, expected.2, "clamped flags differ on run {run}");
        assert_eq!(actual.3, expected.3, "status flags differ on run {run}");
    }
}
