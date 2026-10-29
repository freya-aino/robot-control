//! CPU-vs-GPU agreement for the implicit-damping Euler step
//! (`robot_sim::integrators::implicit_damping_euler_step`, MuJoCo 3.17.0
//! `engine_forward.c`, `mj_EulerSkip`). The CPU reference is `f64`; the GPU
//! runs `f32`. Source-faithful port with oracle checks; not measured parity
//! with MuJoCo.

use robot_sim_gpu::*;

mod common;

fn expect_invalid(r: Result<(), GpuSimError>) {
    assert!(
        matches!(r, Err(GpuSimError::InvalidParameter { .. })),
        "{r:?}"
    );
}

// Validation runs before the device is touched, so these need no GPU and
// hold with the `cubecl` feature on or off.

#[test]
fn nv_outside_one_to_six_is_rejected() {
    for nv in [0usize, 7] {
        let mut status = [0u32; 1];
        expect_invalid(gpu_implicit_damping_euler_step(
            nv,
            &mut [],
            &mut [],
            &[],
            &[],
            &[],
            0.1,
            &mut status,
        ));
    }
}

#[test]
fn malformed_inputs_are_rejected_without_touching_state() {
    let mut status = [9u32; 2];
    let mass = [2.0f32, 0.0, 0.0, 2.0];
    let damping = [0.0f32, 0.0];
    let bias = [1.0f32, 1.0];

    // Length mismatch: `q` should hold 2 envs x 1 dof.
    let mut q = [0.5f32; 3];
    let mut qd = [0.25f32; 2];
    assert_eq!(
        gpu_implicit_damping_euler_step(
            1,
            &mut q,
            &mut qd,
            &mass[..2],
            &damping,
            &bias,
            0.1,
            &mut status
        ),
        Err(GpuSimError::DimensionMismatch {
            what: "q",
            expected: 2,
            found: 3
        })
    );

    let mut q = [0.5f32; 2];
    assert_eq!(
        gpu_implicit_damping_euler_step(
            1,
            &mut q,
            &mut qd,
            &mass,
            &damping,
            &bias,
            0.1,
            &mut status
        ),
        Err(GpuSimError::DimensionMismatch {
            what: "mass",
            expected: 2,
            found: 4
        })
    );

    let mass1 = [2.0f32, 3.0];
    assert!(matches!(
        gpu_implicit_damping_euler_step(
            1,
            &mut q,
            &mut qd,
            &mass1,
            &[f32::NAN, 0.0],
            &bias,
            0.1,
            &mut status
        ),
        Err(GpuSimError::NonFinite { what: "damping" })
    ));
    assert!(matches!(
        gpu_implicit_damping_euler_step(
            1,
            &mut q,
            &mut qd,
            &[f32::INFINITY, 1.0],
            &damping,
            &bias,
            0.1,
            &mut status
        ),
        Err(GpuSimError::NonFinite { what: "mass" })
    ));
    expect_invalid(gpu_implicit_damping_euler_step(
        1,
        &mut q,
        &mut qd,
        &mass1,
        &[-0.5, 0.0],
        &bias,
        0.1,
        &mut status,
    ));
    expect_invalid(gpu_implicit_damping_euler_step(
        1,
        &mut q,
        &mut qd,
        &mass1,
        &damping,
        &bias,
        -0.1,
        &mut status,
    ));
    expect_invalid(gpu_implicit_damping_euler_step(
        1,
        &mut q,
        &mut qd,
        &mass1,
        &damping,
        &bias,
        f32::NAN,
        &mut status,
    ));

    // Asymmetric 2-dof mass for one env.
    let mut q2 = [0.0f32; 2];
    let mut qd2 = [0.0f32; 2];
    let mut status1 = [9u32; 1];
    expect_invalid(gpu_implicit_damping_euler_step(
        2,
        &mut q2,
        &mut qd2,
        &[2.0, 1.0, 0.5, 2.0],
        &[0.0, 0.0],
        &[1.0, 1.0],
        0.1,
        &mut status1,
    ));

    assert_eq!(q, [0.5, 0.5]);
    assert_eq!(qd, [0.25, 0.25]);
    assert_eq!(q2, [0.0, 0.0]);
    assert_eq!(status, [9, 9]);
    assert_eq!(status1, [9]);
}

#[test]
fn oversized_dense_batch_is_rejected_by_count_only_preflight() {
    // 4M envs x 36 f32 = 576 MB for `mass`, far above the 128 MiB limit.
    // Nothing is allocated: the check uses counts only.
    assert!(matches!(
        check_implicit_batch(6, 4_000_000),
        Err(GpuSimError::InvalidParameter { .. })
    ));
    assert_eq!(
        check_implicit_batch(6, 4097).unwrap(),
        (4097 * 6, 4097 * 36)
    );
    assert!(check_implicit_batch(7, 1).is_err());
    assert!(check_implicit_batch(0, 1).is_err());
}

#[cfg(not(feature = "cubecl"))]
#[test]
fn valid_batch_reports_gpu_disabled_without_the_feature() {
    let mut q = [0.0f32];
    let mut qd = [1.0f32];
    let mut status = [0u32];
    assert_eq!(
        gpu_implicit_damping_euler_step(
            1,
            &mut q,
            &mut qd,
            &[2.0],
            &[1.0],
            &[-1.0],
            0.1,
            &mut status
        ),
        Err(GpuSimError::GpuDisabled)
    );
}

#[cfg(feature = "cubecl")]
mod gpu_tests {
    use super::*;
    use robot_sim::integrators::implicit_damping_euler_step;

    /// Hand-derived cases: `f32` against exact arithmetic.
    const HAND_TOLERANCE: f32 = 1e-6;
    /// Random SPD batches: the absolute plus relative bound on `f32` vs `f64`
    /// is read from the `implicit_random_spd_q_qd` record in `tolerances.toml`.

    #[test]
    fn one_dof_matches_the_discrete_identity() {
        // M = 2, B = 1, qd0 = 1, bias = -B*qd0 = -1, h = 0.1:
        // qacc = -1 / 2.1, qd1 = 1 - 0.1/2.1 = 2/2.1, q1 = 0.1 * qd1.
        let mut q = [0.0f32];
        let mut qd = [1.0f32];
        let mut status = [0u32];
        gpu_implicit_damping_euler_step(
            1,
            &mut q,
            &mut qd,
            &[2.0],
            &[1.0],
            &[-1.0],
            0.1,
            &mut status,
        )
        .unwrap();
        assert_eq!(status, [1]);
        assert!((qd[0] - 2.0 / 2.1).abs() < HAND_TOLERANCE, "qd = {}", qd[0]);
        assert!((q[0] - 0.2 / 2.1).abs() < HAND_TOLERANCE, "q = {}", q[0]);
        // The explicit answer 0.95 must not come out.
        assert!((qd[0] - 0.95).abs() > 1e-3);
    }

    #[test]
    fn two_dof_off_diagonal_inertia() {
        // M = [[2,1],[1,2]], B = 0, bias = [3,3] gives qacc = [1,1]. With
        // qd0 = 0 and h = 0.5: qd1 = [0.5,0.5], q1 = [0.25,0.25].
        let mut q = [0.0f32; 2];
        let mut qd = [0.0f32; 2];
        let mut status = [0u32];
        gpu_implicit_damping_euler_step(
            2,
            &mut q,
            &mut qd,
            &[2.0, 1.0, 1.0, 2.0],
            &[0.0, 0.0],
            &[3.0, 3.0],
            0.5,
            &mut status,
        )
        .unwrap();
        assert_eq!(status, [1]);
        for i in 0..2 {
            assert!((qd[i] - 0.5).abs() < HAND_TOLERANCE);
            assert!((q[i] - 0.25).abs() < HAND_TOLERANCE);
        }
    }

    #[test]
    fn three_dof_with_damping_and_coupling() {
        // M = [[4,2,0],[2,3,1],[0,1,2]], B = [1,2,0], h = 0.5, so
        // A = M + h*diag(B) = [[4.5,2,0],[2,4,1],[0,1,2]]. Choose
        // qacc = [1,-1,2]; then bias = A*qacc = [4.5-2, 2-4+2, -1+4] = [2.5,0,3].
        // qd0 = [0.2,-0.4,0.6]: qd1 = qd0 + 0.5*qacc = [0.7,-0.9,1.6].
        // q0 = [0,1,-1]: q1 = q0 + 0.5*qd1 = [0.35,0.55,-0.2].
        let mut q = [0.0f32, 1.0, -1.0];
        let mut qd = [0.2f32, -0.4, 0.6];
        let mut status = [0u32];
        gpu_implicit_damping_euler_step(
            3,
            &mut q,
            &mut qd,
            &[4.0, 2.0, 0.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0],
            &[1.0, 2.0, 0.0],
            &[2.5, 0.0, 3.0],
            0.5,
            &mut status,
        )
        .unwrap();
        assert_eq!(status, [1]);
        let qd_expected = [0.7f32, -0.9, 1.6];
        let q_expected = [0.35f32, 0.55, -0.2];
        for i in 0..3 {
            assert!(
                (qd[i] - qd_expected[i]).abs() < HAND_TOLERANCE,
                "qd[{i}] = {}",
                qd[i]
            );
            assert!(
                (q[i] - q_expected[i]).abs() < HAND_TOLERANCE,
                "q[{i}] = {}",
                q[i]
            );
        }
    }

    /// Deterministic generator (LCG, fixed seed), uniform in `[-1, 1)`.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// Runs `n_env` random SPD systems of size `nv` on the GPU and on the CPU
    /// (fed the same `f32`-rounded inputs, widened to `f64`), returning the
    /// worst error ratio against an absolute + relative tolerance.
    fn random_batch_error_ratio(nv: usize, n_env: usize, seed: u64) -> (f64, f64, usize) {
        let record = common::record("implicit_random_spd_q_qd");
        let abs_tol = common::number(&record, "tolerance_abs");
        let rel_tol = common::number(&record, "tolerance_rel");
        let mut rng = Lcg(seed);
        let mut mass = vec![0.0f32; n_env * nv * nv];
        let mut damping = vec![0.0f32; n_env * nv];
        let mut bias = vec![0.0f32; n_env * nv];
        let mut q = vec![0.0f32; n_env * nv];
        let mut qd = vec![0.0f32; n_env * nv];
        for e in 0..n_env {
            // M = L L^T + 0.5 I is symmetric positive definite.
            let l: Vec<f64> = (0..nv * nv).map(|_| rng.next()).collect();
            for i in 0..nv {
                for j in 0..nv {
                    let mut s = 0.0;
                    for k in 0..nv {
                        s += l[i * nv + k] * l[j * nv + k];
                    }
                    if i == j {
                        s += 0.5;
                    }
                    mass[e * nv * nv + i * nv + j] = s as f32;
                }
            }
            for i in 0..nv {
                damping[e * nv + i] = (rng.next() + 1.0) as f32;
                bias[e * nv + i] = (5.0 * rng.next()) as f32;
                q[e * nv + i] = rng.next() as f32;
                qd[e * nv + i] = rng.next() as f32;
            }
        }
        let dt = 0.01f32;

        let mut q_gpu = q.clone();
        let mut qd_gpu = qd.clone();
        let mut status = vec![0u32; n_env];
        gpu_implicit_damping_euler_step(
            nv,
            &mut q_gpu,
            &mut qd_gpu,
            &mass,
            &damping,
            &bias,
            dt,
            &mut status,
        )
        .unwrap();
        assert!(status.iter().all(|&s| s == 1), "every SPD env must step");

        let widen = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();
        let mut worst = 0.0f64;
        let mut max_abs = (0.0f64, 0usize);
        for e in 0..n_env {
            let mut qc = widen(&q[e * nv..(e + 1) * nv]);
            let mut qdc = widen(&qd[e * nv..(e + 1) * nv]);
            let r = implicit_damping_euler_step(
                &mut qc,
                &mut qdc,
                &widen(&mass[e * nv * nv..(e + 1) * nv * nv]),
                &widen(&damping[e * nv..(e + 1) * nv]),
                &widen(&bias[e * nv..(e + 1) * nv]),
                f64::from(dt),
            );
            assert_eq!(r, Some(()));
            for i in 0..nv {
                let k = e * nv + i;
                for (gpu, cpu) in [(q_gpu[k], qc[i]), (qd_gpu[k], qdc[i])] {
                    let err = (f64::from(gpu) - cpu).abs();
                    worst = worst.max(err / (abs_tol + rel_tol * cpu.abs()));
                    if err > max_abs.0 {
                        max_abs = (err, k);
                    }
                }
            }
        }
        (worst, max_abs.0, max_abs.1)
    }

    #[test]
    fn random_spd_batches_of_4097_envs_match_the_cpu_reference() {
        // 4097 envs = 16 full workgroups of 256 plus one thread in the 17th,
        // so the partial-tail guard is exercised for every nv.
        for (nv, seed) in [(1usize, 11u64), (2, 12), (3, 13), (6, 14)] {
            let (ratio, max_abs, at) = random_batch_error_ratio(nv, 4097, seed);
            eprintln!("implicit euler nv = {nv}: worst error / bound = {ratio:.4}, max abs error {max_abs:.6e} at index {at}");
            assert!(
                ratio < 1.0,
                "nv = {nv}: error ratio {ratio} exceeds the bound"
            );
        }
    }

    #[test]
    fn non_spd_envs_are_reported_and_keep_their_state() {
        // 300 envs of nv = 2 span two workgroups. Env 7 is symmetric but
        // indefinite: [[1,2],[2,1]] gives pivots 1 then 1 - 4 = -3. Env 299 (in
        // the partial tail) has a negative first pivot. All others are 2*I.
        let n = 300usize;
        let nv = 2usize;
        let mut mass = vec![0.0f32; n * 4];
        for e in 0..n {
            mass[e * 4] = 2.0;
            mass[e * 4 + 3] = 2.0;
        }
        mass[7 * 4..8 * 4].copy_from_slice(&[1.0, 2.0, 2.0, 1.0]);
        mass[299 * 4..300 * 4].copy_from_slice(&[-1.0, 0.0, 0.0, 1.0]);
        let damping = vec![0.0f32; n * nv];
        let bias = vec![2.0f32; n * nv];
        let mut q = vec![0.3f32; n * nv];
        let mut qd = vec![0.7f32; n * nv];
        let mut status = vec![9u32; n];

        gpu_implicit_damping_euler_step(
            nv,
            &mut q,
            &mut qd,
            &mass,
            &damping,
            &bias,
            0.5,
            &mut status,
        )
        .unwrap();

        for (e, &st) in status.iter().enumerate() {
            let failed = e == 7 || e == 299;
            assert_eq!(st, u32::from(!failed), "env {e}");
            for i in 0..nv {
                let k = e * nv + i;
                if failed {
                    assert_eq!((q[k], qd[k]), (0.3, 0.7), "env {e} must be unchanged");
                } else {
                    // qacc = 2/2 = 1: qd1 = 0.7 + 0.5, q1 = 0.3 + 0.5*qd1.
                    assert!((qd[k] - 1.2).abs() < HAND_TOLERANCE);
                    assert!((q[k] - 0.9).abs() < HAND_TOLERANCE);
                }
            }
        }

        // The CPU reference refuses the same systems.
        for bad in [[1.0, 2.0, 2.0, 1.0], [-1.0, 0.0, 0.0, 1.0]] {
            let mut qc = [0.3f64, 0.3];
            let mut qdc = [0.7f64, 0.7];
            let r =
                implicit_damping_euler_step(&mut qc, &mut qdc, &bad, &[0.0, 0.0], &[2.0, 2.0], 0.5);
            assert_eq!(r, None);
        }
    }

    #[test]
    fn empty_batch_is_a_no_op() {
        let mut status: [u32; 0] = [];
        gpu_implicit_damping_euler_step(3, &mut [], &mut [], &[], &[], &[], 0.1, &mut status)
            .unwrap();
    }
}
