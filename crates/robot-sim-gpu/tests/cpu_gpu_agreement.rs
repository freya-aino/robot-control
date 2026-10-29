//! CPU-vs-GPU agreement tests comparing CubeCL GPU outputs against pure CPU references.

use robot_sim_gpu::*;

mod common;

// Count-only preflight: these allocate nothing, do not need the `cubecl`
// feature, and never touch a device.

#[test]
fn workgroup_cap_is_rejected_before_any_upload() {
    let n = 65_535 * 256 + 1;
    assert!(matches!(
        check_batch_size(n, 1),
        Err(GpuSimError::InvalidParameter { .. })
    ));
}

#[test]
fn strided_buffer_above_binding_limit_is_rejected_before_any_upload() {
    // 8M rows pass the workgroup cap, but solimp is 5 * 8M f32 = 160 MB,
    // above the 128 MiB storage binding limit. The check is count-only.
    let n = 8_000_000usize;
    assert!(check_batch_size(n, 1).is_ok());
    assert!(matches!(
        check_batch_size(n, 5),
        Err(GpuSimError::InvalidParameter { .. })
    ));
}

#[cfg(feature = "cubecl")]
mod gpu_tests {
    use super::*;

    use crate::common::{number, record, tolerance_abs};

    const TOLERANCE_F32: f32 = 1e-5;

    fn impp_tolerance() -> f32 {
        tolerance_abs("constraint_impedance_impp_4097_rows")
    }

    fn max_abs_with_index(gpu: &[f32], cpu: &[f64]) -> (f64, usize) {
        gpu.iter()
            .zip(cpu)
            .enumerate()
            .map(|(i, (g, c))| ((*g as f64 - c).abs(), i))
            .fold((0.0, 0), |acc, (d, i)| if d > acc.0 { (d, i) } else { acc })
    }

    #[test]
    fn semi_implicit_euler_matches_cpu_reference() {
        let q_init = [1.0f32, 2.0, -0.5, 0.0, core::f32::consts::PI];
        let qd_init = [0.1f32, -0.2, 0.5, 1.0, -0.5];
        let qacc = [0.5f32, -0.5, 1.0, -2.0, 0.0];
        let dt = 0.002f32;

        let mut q_gpu = q_init;
        let mut qd_gpu = qd_init;
        gpu_semi_implicit_euler_step(&mut q_gpu, &mut qd_gpu, &qacc, dt).unwrap();

        let mut q_cpu: Vec<f64> = q_init.iter().map(|&x| x as f64).collect();
        let mut qd_cpu: Vec<f64> = qd_init.iter().map(|&x| x as f64).collect();
        let qacc_cpu: Vec<f64> = qacc.iter().map(|&x| x as f64).collect();
        robot_sim::integrators::semi_implicit_euler_step(
            &mut q_cpu,
            &mut qd_cpu,
            &qacc_cpu,
            dt as f64,
        );

        for i in 0..q_init.len() {
            let diff_q = (q_gpu[i] - q_cpu[i] as f32).abs();
            let diff_qd = (qd_gpu[i] - qd_cpu[i] as f32).abs();
            assert!(
                diff_q < TOLERANCE_F32,
                "q[{i}] mismatch: GPU={}, CPU={}, diff={diff_q}",
                q_gpu[i],
                q_cpu[i]
            );
            assert!(
                diff_qd < TOLERANCE_F32,
                "qd[{i}] mismatch: GPU={}, CPU={}, diff={diff_qd}",
                qd_gpu[i],
                qd_cpu[i]
            );
        }
    }

    #[test]
    fn joint_spring_forces_match_cpu_reference() {
        let q = [0.3f32, 0.1, 7.0, -1.5];
        let q_ref = [0.1f32, 0.1, -2.0, 0.0];
        let stiffness = [5.0f32, 5.0, 0.0, 10.0];
        let mut out_gpu = [0.0f32; 4];

        gpu_joint_spring_forces(&q, &q_ref, &stiffness, &mut out_gpu).unwrap();

        for i in 0..q.len() {
            let cpu_force = robot_sim::passive::joint_spring_force(
                q[i] as f64,
                q_ref[i] as f64,
                stiffness[i] as f64,
            ) as f32;
            let diff = (out_gpu[i] - cpu_force).abs();
            assert!(
                diff < TOLERANCE_F32,
                "spring[{i}] mismatch: GPU={}, CPU={}, diff={diff}",
                out_gpu[i],
                cpu_force
            );
        }
    }

    #[test]
    fn joint_damper_forces_match_cpu_reference() {
        let qd = [2.0f32, -2.0, 3.0, 0.0];
        let damping = [0.5f32, 0.5, 0.0, 1.2];
        let mut out_gpu = [0.0f32; 4];

        gpu_joint_damper_forces(&qd, &damping, &mut out_gpu).unwrap();

        for i in 0..qd.len() {
            let cpu_force =
                robot_sim::passive::joint_damper_force(qd[i] as f64, damping[i] as f64) as f32;
            let diff = (out_gpu[i] - cpu_force).abs();
            assert!(
                diff < TOLERANCE_F32,
                "damper[{i}] mismatch: GPU={}, CPU={}, diff={diff}",
                out_gpu[i],
                cpu_force
            );
        }
    }

    #[test]
    fn affine_actuator_forces_match_cpu_reference() {
        let gain_prm = [[10.0f32, 0.0, 0.0], [4.0, 0.0, 0.0], [1.0, 2.0, 3.0]];
        let bias_prm = [[0.0f32, -10.0, -2.0], [0.0, 0.0, -4.0], [4.0, 5.0, 6.0]];
        let ctrl = [0.5f32, 0.5, 2.0];
        let length = [0.2f32, 0.7, 1.0];
        let velocity = [1.0f32, 0.25, -1.0];
        let mut out_gpu = [0.0f32; 3];

        gpu_affine_actuator_forces(
            &gain_prm,
            &bias_prm,
            &ctrl,
            &length,
            &velocity,
            &mut out_gpu,
        )
        .unwrap();

        for i in 0..ctrl.len() {
            let gp = [
                gain_prm[i][0] as f64,
                gain_prm[i][1] as f64,
                gain_prm[i][2] as f64,
            ];
            let bp = [
                bias_prm[i][0] as f64,
                bias_prm[i][1] as f64,
                bias_prm[i][2] as f64,
            ];
            let cpu_force = robot_sim::passive::affine_actuator_force(
                gp,
                bp,
                ctrl[i] as f64,
                length[i] as f64,
                velocity[i] as f64,
            ) as f32;
            let diff = (out_gpu[i] - cpu_force).abs();
            assert!(
                diff < TOLERANCE_F32,
                "actuator[{i}] mismatch: GPU={}, CPU={}, diff={diff}",
                out_gpu[i],
                cpu_force
            );
        }
    }

    #[test]
    fn joint_equality_eval_matches_cpu_reference() {
        let q1 = [0.5f32, 1.2];
        let q2 = [0.25f32, -0.4];
        let ref1 = [0.0f32, 0.1];
        let ref2 = [0.0f32, -0.2];
        let coef = [[0.1f32, 0.5, -0.2, 0.05, -0.01], [0.0, 1.0, 0.0, 0.0, 0.0]];

        let mut pos_out = [0.0f32; 2];
        let mut jac2_out = [0.0f32; 2];

        gpu_joint_equality_eval(&q1, &q2, &ref1, &ref2, &coef, &mut pos_out, &mut jac2_out)
            .unwrap();

        for i in 0..q1.len() {
            let eq = robot_sim::constraint::JointEquality {
                joint1: 0,
                joint2: Some(1),
                reference1: ref1[i] as f64,
                reference2: ref2[i] as f64,
                coef: [
                    coef[i][0] as f64,
                    coef[i][1] as f64,
                    coef[i][2] as f64,
                    coef[i][3] as f64,
                    coef[i][4] as f64,
                ],
                solref: [0.02, 1.0],
                solimp: [0.9, 0.95, 0.001, 0.5, 2.0],
            };
            let row = robot_sim::constraint::joint_equality_row(&eq, &[q1[i] as f64, q2[i] as f64])
                .unwrap();

            let diff_pos = (pos_out[i] - row.pos as f32).abs();
            let diff_jac = (jac2_out[i] - row.jacobian[1] as f32).abs();

            assert!(
                diff_pos < TOLERANCE_F32,
                "eq pos[{i}] mismatch: GPU={}, CPU={}, diff={diff_pos}",
                pos_out[i],
                row.pos
            );
            assert!(
                diff_jac < TOLERANCE_F32,
                "eq jac2[{i}] mismatch: GPU={}, CPU={}, diff={diff_jac}",
                jac2_out[i],
                row.jacobian[1]
            );
        }
    }

    #[test]
    fn constraint_impedance_matches_cpu_reference() {
        let solimp = [
            [0.9f32, 0.95, 0.001, 0.5, 2.0],
            [0.1, 0.8, 0.01, 0.5, 1.0],
            [0.5, 0.5, 0.0, 0.5, 2.0],
        ];
        let pos = [0.0005f32, 0.005, 0.02];
        let margin = [0.0f32, 0.0, 0.0];
        let mut out_imp = [0.0f32; 3];
        let mut out_imp_prime = [0.0f32; 3];

        gpu_constraint_impedance_with_derivative(
            &solimp,
            &pos,
            &margin,
            &mut out_imp,
            &mut out_imp_prime,
        )
        .unwrap();

        for i in 0..pos.len() {
            let s_clamped = robot_sim::constraint::clamp_solimp([
                solimp[i][0] as f64,
                solimp[i][1] as f64,
                solimp[i][2] as f64,
                solimp[i][3] as f64,
                solimp[i][4] as f64,
            ]);
            let (cpu_imp, cpu_imp_prime) =
                robot_sim::constraint::impedance(s_clamped, pos[i] as f64, margin[i] as f64);
            let diff = (out_imp[i] - cpu_imp as f32).abs();
            assert!(
                diff < TOLERANCE_F32,
                "impedance[{i}] mismatch: GPU={}, CPU={}, diff={diff}",
                out_imp[i],
                cpu_imp
            );
            let diff_prime = (out_imp_prime[i] - cpu_imp_prime as f32).abs();
            assert!(
                diff_prime < TOLERANCE_F32 * 10.0,
                "impP[{i}] mismatch: GPU={}, CPU={}, diff={diff_prime}",
                out_imp_prime[i],
                cpu_imp_prime
            );
        }
    }

    #[test]
    fn constraint_impedance_branches_match_cpu_including_impp_sign() {
        // Rows cover: pos below margin (negative x, impP sign flip), interior
        // power-1 and power-2 branches, saturated low and high, and the
        // degenerate width and dmin == dmax branches. X > mid coverage is in
        // `constraint_impedance_above_mid_impp_matches_cpu`.
        let solimp = [
            [0.9f32, 0.95, 0.05, 0.5, 2.0],
            [0.9, 0.95, 0.05, 0.5, 2.0],
            [0.9, 0.95, 0.05, 0.3, 1.0],
            [0.9, 0.95, 0.05, 0.7, 3.0],
            [0.9, 0.95, 0.05, 0.5, 2.0],
            [0.9, 0.95, 0.05, 0.5, 2.0],
            [0.5, 0.5, 0.05, 0.5, 2.0],
            [0.9, 0.95, 0.0, 0.5, 2.0],
        ];
        let pos = [-0.01f32, 0.01, 0.01, 0.02, 0.0, 0.2, 0.0, 0.0];
        let margin = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let n = pos.len();
        let mut out_imp = vec![0.0f32; n];
        let mut out_imp_prime = vec![0.0f32; n];

        gpu_constraint_impedance_with_derivative(
            &solimp,
            &pos,
            &margin,
            &mut out_imp,
            &mut out_imp_prime,
        )
        .unwrap();

        for i in 0..n {
            let s = robot_sim::constraint::clamp_solimp([
                solimp[i][0] as f64,
                solimp[i][1] as f64,
                solimp[i][2] as f64,
                solimp[i][3] as f64,
                solimp[i][4] as f64,
            ]);
            let (cpu_imp, cpu_imp_prime) =
                robot_sim::constraint::impedance(s, pos[i] as f64, margin[i] as f64);
            let diff = (out_imp[i] - cpu_imp as f32).abs();
            assert!(
                diff < TOLERANCE_F32,
                "branch {i}: impedance GPU={}, CPU={cpu_imp}, diff={diff}",
                out_imp[i]
            );
            let diff_prime = (out_imp_prime[i] - cpu_imp_prime as f32).abs();
            assert!(
                diff_prime < TOLERANCE_F32 * 10.0,
                "branch {i}: impP GPU={}, CPU={cpu_imp_prime}, diff={diff_prime}",
                out_imp_prime[i]
            );
        }
    }

    #[test]
    fn rk4_pendulum_matches_cpu_closure_reference() {
        let g_over_l = 9.81f32;
        let dt = 0.01f32;
        let mut theta = [0.3f32, -1.2, 2.5, 0.0];
        let mut omega = [0.0f32, 0.5, -0.7, 3.0];
        let mut theta_cpu: Vec<f64> = theta.iter().map(|x| *x as f64).collect();
        let mut omega_cpu: Vec<f64> = omega.iter().map(|x| *x as f64).collect();

        for _ in 0..50 {
            gpu_rk4_pendulum_step(&mut theta, &mut omega, g_over_l, dt).unwrap();
            robot_sim::integrators::rk4_step(
                &mut theta_cpu,
                &mut omega_cpu,
                dt as f64,
                |q, _qd| q.iter().map(|x| -(g_over_l as f64) * x.sin()).collect(),
            );
        }

        for i in 0..theta.len() {
            let dth = (theta[i] as f64 - theta_cpu[i]).abs();
            let dom = (omega[i] as f64 - omega_cpu[i]).abs();
            assert!(
                dth < tolerance_abs("rk4_pendulum_theta_1000_steps") as f64,
                "rk4 theta[{i}] GPU={}, CPU={}",
                theta[i],
                theta_cpu[i]
            );
            assert!(
                dom < tolerance_abs("rk4_pendulum_omega_1000_steps") as f64,
                "rk4 omega[{i}] GPU={}, CPU={}",
                omega[i],
                omega_cpu[i]
            );
        }
    }

    #[test]
    fn repeated_identical_launches_are_bitwise_identical() {
        let g_over_l = 9.81f32;
        let dt = 0.005f32;
        let base_theta: Vec<f32> = (0..4097).map(|i| (i as f32) * 0.001 - 2.0).collect();
        let base_omega: Vec<f32> = (0..4097).map(|i| (i as f32) * 0.0005).collect();

        let mut first_theta = base_theta.clone();
        let mut first_omega = base_omega.clone();
        gpu_rk4_pendulum_step(&mut first_theta, &mut first_omega, g_over_l, dt).unwrap();

        for _ in 0..5 {
            let mut theta = base_theta.clone();
            let mut omega = base_omega.clone();
            gpu_rk4_pendulum_step(&mut theta, &mut omega, g_over_l, dt).unwrap();
            assert_eq!(
                theta, first_theta,
                "rk4 theta differs between identical launches"
            );
            assert_eq!(
                omega, first_omega,
                "rk4 omega differs between identical launches"
            );
        }
    }

    #[test]
    fn plane_sphere_contact_matches_cpu_reference() {
        let centers = [
            [0.0f32, 0.0, 0.05], // penetrates plane (height=0, radius=0.1)
            [1.0, 2.0, 0.11],    // within margin (0.02)
            [0.5, -0.5, 1.0],    // far above
        ];
        let radii = [0.1f32, 0.1, 0.1];
        let plane_height = 0.0f32;
        let margin = 0.02f32;
        let mut contacts = vec![None; 3];

        gpu_plane_sphere_contact(&centers, &radii, plane_height, margin, &mut contacts).unwrap();

        for i in 0..centers.len() {
            let cpu_contact = robot_sim::contact::plane_sphere_contact(
                [
                    centers[i][0] as f64,
                    centers[i][1] as f64,
                    centers[i][2] as f64,
                ],
                radii[i] as f64,
                plane_height as f64,
                margin as f64,
            )
            .unwrap();

            match (contacts[i], cpu_contact) {
                (Some(gpu_c), Some(cpu_c)) => {
                    let diff_dist = (gpu_c.dist - cpu_c.dist as f32).abs();
                    assert!(
                        diff_dist < TOLERANCE_F32,
                        "contact[{i}] dist mismatch: GPU={}, CPU={}",
                        gpu_c.dist,
                        cpu_c.dist
                    );
                    for k in 0..3 {
                        let diff_pos = (gpu_c.pos[k] - cpu_c.pos[k] as f32).abs();
                        assert!(
                            diff_pos < TOLERANCE_F32,
                            "contact[{i}] pos[{k}] mismatch: GPU={}, CPU={}",
                            gpu_c.pos[k],
                            cpu_c.pos[k]
                        );
                        let diff_normal = (gpu_c.normal[k] - cpu_c.normal[k] as f32).abs();
                        assert!(
                            diff_normal < TOLERANCE_F32,
                            "contact[{i}] normal[{k}] mismatch: GPU={}, CPU={}",
                            gpu_c.normal[k],
                            cpu_c.normal[k]
                        );
                    }
                }
                (None, None) => {}
                _ => panic!(
                    "contact[{i}] existence mismatch: GPU={:?}, CPU={:?}",
                    contacts[i], cpu_contact
                ),
            }
        }
    }

    #[test]
    fn invalid_and_mismatched_inputs_are_rejected() {
        let mut q = [1.0f32, 2.0];
        let mut qd = [0.1f32]; // Mismatch
        let qacc = [0.5f32, 0.5];

        let err = gpu_semi_implicit_euler_step(&mut q, &mut qd, &qacc, 0.01).unwrap_err();
        assert!(matches!(err, GpuSimError::DimensionMismatch { .. }));

        let mut q_nan = [f32::NAN, 2.0];
        let mut qd_ok = [0.1f32, 0.2];
        let err_nan =
            gpu_semi_implicit_euler_step(&mut q_nan, &mut qd_ok, &qacc, 0.01).unwrap_err();
        assert!(matches!(err_nan, GpuSimError::NonFinite { .. }));

        let err_dt =
            gpu_semi_implicit_euler_step(&mut q, &mut [0.1f32, 0.2], &qacc, -0.01).unwrap_err();
        assert!(matches!(err_dt, GpuSimError::InvalidParameter { .. }));

        let mut contacts = vec![None; 1];
        let err_rad =
            gpu_plane_sphere_contact(&[[0.0, 0.0, 1.0]], &[-0.1], 0.0, 0.01, &mut contacts)
                .unwrap_err();
        assert!(matches!(err_rad, GpuSimError::InvalidParameter { .. }));
    }

    #[test]
    fn malformed_wrapper_inputs_are_rejected() {
        let mut pos = [0.0f32; 1];
        let mut jac = [0.0f32; 1];
        let err = gpu_joint_equality_eval(
            &[0.1],
            &[0.1, 0.2],
            &[0.0],
            &[0.0],
            &[[1.0, 0.0, 0.0, 0.0, 0.0]],
            &mut pos,
            &mut jac,
        )
        .unwrap_err();
        assert!(matches!(err, GpuSimError::DimensionMismatch { .. }));

        let err_coef = gpu_joint_equality_eval(
            &[f32::NAN],
            &[0.1],
            &[0.0],
            &[0.0],
            &[[1.0, 0.0, 0.0, 0.0, 0.0]],
            &mut pos,
            &mut jac,
        )
        .unwrap_err();
        assert!(matches!(err_coef, GpuSimError::NonFinite { .. }));

        let mut out = [0.0f32; 1];
        let err_act = gpu_affine_actuator_forces(
            &[[1.0, 0.0, 0.0]],
            &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            &[0.5],
            &[0.0],
            &[0.0],
            &mut out,
        )
        .unwrap_err();
        assert!(matches!(err_act, GpuSimError::DimensionMismatch { .. }));

        let err_imp = gpu_constraint_impedance(
            &[[0.9, 0.95, 0.001, 0.5, 2.0]],
            &[f32::INFINITY],
            &[0.0],
            &mut [0.0f32; 1],
        )
        .unwrap_err();
        assert!(matches!(err_imp, GpuSimError::NonFinite { .. }));

        let err_spring =
            gpu_joint_spring_forces(&[0.1, 0.2], &[0.0], &[1.0, 1.0], &mut [0.0f32; 2])
                .unwrap_err();
        assert!(matches!(err_spring, GpuSimError::DimensionMismatch { .. }));
    }

    #[test]
    fn batch_of_4096_envs_matches_cpu_reference() {
        const N_ENV: usize = 4096;
        const N_DOF: usize = 7;
        const LEN: usize = N_ENV * N_DOF;

        let q: Vec<f32> = (0..LEN).map(|i| ((i as f32) * 0.001).sin()).collect();
        let q_ref = vec![0.0f32; LEN];
        let stiffness: Vec<f32> = (0..LEN).map(|i| 1.0 + (i % 5) as f32).collect();
        let mut out_gpu = vec![0.0f32; LEN];

        gpu_joint_spring_forces(&q, &q_ref, &stiffness, &mut out_gpu).unwrap();

        for i in 0..LEN {
            let cpu = robot_sim::passive::joint_spring_force(
                q[i] as f64,
                q_ref[i] as f64,
                stiffness[i] as f64,
            ) as f32;
            let diff = (out_gpu[i] - cpu).abs();
            assert!(
                diff < TOLERANCE_F32,
                "batch spring[{i}] mismatch: GPU={}, CPU={}, diff={diff}",
                out_gpu[i],
                cpu
            );
        }
    }

    #[test]
    fn batch_of_4097_partial_tail_matches_cpu_for_strided_kernels() {
        // 4097 = 16 full workgroups of 256 plus one partial, so the last
        // cube has 255 threads past `n`. Strided buffers (3 and 5 wide) are
        // the ones an out-of-range index would corrupt.
        const N: usize = 4097;
        let centers: Vec<[f32; 3]> = (0..N)
            .map(|i| {
                let f = i as f32 * 0.001;
                [f.sin(), f.cos(), 0.02 + (i % 7) as f32 * 0.01]
            })
            .collect();
        let radii: Vec<f32> = (0..N).map(|i| 0.01 + (i % 3) as f32 * 0.005).collect();
        let mut contacts = vec![None; N];
        gpu_plane_sphere_contact(&centers, &radii, 0.0, 0.02, &mut contacts).unwrap();
        for i in 0..N {
            let cpu = robot_sim::contact::plane_sphere_contact(
                [
                    centers[i][0] as f64,
                    centers[i][1] as f64,
                    centers[i][2] as f64,
                ],
                radii[i] as f64,
                0.0,
                0.02,
            )
            .unwrap();
            match (contacts[i], cpu) {
                (Some(g), Some(c)) => {
                    assert!(
                        (g.dist - c.dist as f32).abs() < TOLERANCE_F32,
                        "tail contact[{i}] dist: GPU={}, CPU={}",
                        g.dist,
                        c.dist
                    );
                    for k in 0..3 {
                        assert!(
                            (g.pos[k] - c.pos[k] as f32).abs() < TOLERANCE_F32,
                            "tail contact[{i}] pos[{k}]"
                        );
                    }
                }
                (None, None) => {}
                _ => panic!("tail contact[{i}] existence mismatch"),
            }
        }

        let gain: Vec<[f32; 3]> = (0..N).map(|i| [10.0 + (i % 4) as f32, 0.5, 0.0]).collect();
        let bias: Vec<[f32; 3]> = (0..N).map(|i| [0.0, -1.0, (i % 2) as f32]).collect();
        let ctrl: Vec<f32> = (0..N).map(|i| (i as f32 * 0.01).sin()).collect();
        let length: Vec<f32> = (0..N).map(|i| 0.2 + (i % 5) as f32 * 0.1).collect();
        let velocity: Vec<f32> = (0..N).map(|i| (i % 3) as f32 - 1.0).collect();
        let mut out = vec![0.0f32; N];
        gpu_affine_actuator_forces(&gain, &bias, &ctrl, &length, &velocity, &mut out).unwrap();
        for i in 0..N {
            let cpu = robot_sim::passive::affine_actuator_force(
                [gain[i][0] as f64, gain[i][1] as f64, gain[i][2] as f64],
                [bias[i][0] as f64, bias[i][1] as f64, bias[i][2] as f64],
                ctrl[i] as f64,
                length[i] as f64,
                velocity[i] as f64,
            ) as f32;
            assert!(
                (out[i] - cpu).abs() < TOLERANCE_F32,
                "tail actuator[{i}]: GPU={}, CPU={}",
                out[i],
                cpu
            );
        }
    }

    #[test]
    fn nonlinear_impedance_branches_match_cpu_reference() {
        // Width 0.1, mid 0.5, so x = pos / 0.1:
        //   case 0: x = 0.2, x <= mid, power 2 (the first power branch)
        //   case 1: x = 0.8, x > mid, power 3 (the second power branch)
        //   case 2: x = 2.0, x >= 1 (saturation returns dmax)
        //   case 3: pos = -0.04, x folds to 0.4 with power 2 (negative residual)
        //   case 4: out-of-range solimp (dmin 1.5, dmax -0.2, mid 2.0, power 0.5,
        //           with pos = 0.05 so x = 0.5); the clamp runs before the formula
        let solimp: [[f32; 5]; 5] = [
            [0.9, 0.95, 0.1, 0.5, 2.0],
            [0.9, 0.95, 0.1, 0.5, 3.0],
            [0.9, 0.95, 0.1, 0.5, 2.0],
            [0.9, 0.95, 0.1, 0.5, 2.0],
            [1.5, -0.2, 0.1, 2.0, 0.5],
        ];
        let pos = [0.02f32, 0.08, 0.2, -0.04, 0.05];
        let margin = [0.0f32, 0.0, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 5];
        gpu_constraint_impedance(&solimp, &pos, &margin, &mut out).unwrap();

        for i in 0..5 {
            // `impedance` takes an already clamped `solimp` (see its doc). The
            // GPU kernel clamps in-kernel, so the CPU side clamps first.
            let clamped = robot_sim::constraint::clamp_solimp(solimp[i].map(|v| v as f64));
            let (cpu, _imp_p) =
                robot_sim::constraint::impedance(clamped, pos[i] as f64, margin[i] as f64);
            assert!(
                (out[i] - cpu as f32).abs() < TOLERANCE_F32,
                "impedance case {i}: GPU={}, CPU={}",
                out[i],
                cpu
            );
        }
    }

    fn assert_impedance_within_record(solimp: &[[f32; 5]], pos: &[f32], margin: &[f32]) {
        let n = pos.len();
        let mut out_imp = vec![0.0f32; n];
        let mut out_imp_prime = vec![0.0f32; n];
        gpu_constraint_impedance_with_derivative(
            solimp,
            pos,
            margin,
            &mut out_imp,
            &mut out_imp_prime,
        )
        .unwrap();

        let mut cpu_imp = vec![0.0f64; n];
        let mut cpu_imp_prime = vec![0.0f64; n];
        for i in 0..n {
            let s = robot_sim::constraint::clamp_solimp(solimp[i].map(f64::from));
            let (d, dp) = robot_sim::constraint::impedance(s, pos[i] as f64, margin[i] as f64);
            cpu_imp[i] = d;
            cpu_imp_prime[i] = dp;
        }

        let (d_err, d_idx) = max_abs_with_index(&out_imp, &cpu_imp);
        let (dp_err, dp_idx) = max_abs_with_index(&out_imp_prime, &cpu_imp_prime);
        let d_tol = tolerance_abs("constraint_impedance_d_4097_rows") as f64;
        let dp_tol = impp_tolerance() as f64;
        eprintln!(
            "impedance rows={n}: d max={d_err:e} at {d_idx}, impP max={dp_err:e} at {dp_idx}"
        );
        assert!(
            d_err <= d_tol,
            "impedance d: max |GPU - CPU| = {d_err:e} at row {d_idx}, tolerance {d_tol:e}"
        );
        assert!(
            dp_err <= dp_tol,
            "impP: max |GPU - CPU| = {dp_err:e} at row {dp_idx}, tolerance {dp_tol:e}"
        );
    }

    #[test]
    fn constraint_impedance_above_mid_impp_matches_cpu() {
        // Rows 0-4 have x = (pos - margin) / width > mid, with p = 2 and p = 3,
        // so they exercise the second power branch and its impP. Row 5 has
        // x = 0.4 <= mid = 0.7 as the lower-branch contrast.
        let solimp = [
            [0.9f32, 0.95, 0.05, 0.5, 2.0],
            [0.9, 0.95, 0.05, 0.5, 2.0],
            [0.9, 0.95, 0.05, 0.5, 3.0],
            [0.9, 0.95, 0.05, 0.5, 3.0],
            [0.9, 0.95, 0.05, 0.3, 3.0],
            [0.9, 0.95, 0.05, 0.7, 2.0],
        ];
        let pos = [0.03f32, 0.04, 0.03, 0.045, 0.02, 0.02];
        let margin = [0.0f32; 6];
        assert_impedance_within_record(&solimp, &pos, &margin);
    }

    #[test]
    fn constraint_impedance_4097_rows_match_cpu_within_recorded_tolerance() {
        let n = number(&record("constraint_impedance_d_4097_rows"), "envs") as usize;
        let solimp: Vec<[f32; 5]> = (0..n)
            .map(|i| {
                let w = 0.001 + (i % 7) as f32 * 0.002;
                [0.9, 0.95, w, 0.5, 1.0 + (i % 3) as f32]
            })
            .collect();
        let pos: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-4 - 0.2).collect();
        let margin = vec![0.0f32; n];
        assert_impedance_within_record(&solimp, &pos, &margin);
    }

    #[test]
    fn rk4_pendulum_4097_envs_1000_steps_match_cpu_within_recorded_tolerance() {
        let theta_rec = record("rk4_pendulum_theta_1000_steps");
        let envs = number(&theta_rec, "envs") as usize;
        let steps = number(&theta_rec, "steps") as usize;
        let dt = number(&theta_rec, "dt") as f32;
        let g_over_l = 9.81f32;

        let theta0: Vec<f32> = (0..envs).map(|i| (i as f32) * 0.001 - 2.0).collect();
        let omega0: Vec<f32> = (0..envs).map(|i| (i as f32) * 0.0005).collect();
        let mut theta = theta0.clone();
        let mut omega = omega0.clone();
        let mut theta_cpu: Vec<f64> = theta0.iter().map(|&x| x as f64).collect();
        let mut omega_cpu: Vec<f64> = omega0.iter().map(|&x| x as f64).collect();

        for _ in 0..steps {
            gpu_rk4_pendulum_step(&mut theta, &mut omega, g_over_l, dt).unwrap();
            robot_sim::integrators::rk4_step(&mut theta_cpu, &mut omega_cpu, dt as f64, |q, _| {
                q.iter().map(|x| -(g_over_l as f64) * x.sin()).collect()
            });
        }

        let (th_err, th_idx) = max_abs_with_index(&theta, &theta_cpu);
        let (om_err, om_idx) = max_abs_with_index(&omega, &omega_cpu);
        let th_tol = tolerance_abs("rk4_pendulum_theta_1000_steps") as f64;
        let om_tol = tolerance_abs("rk4_pendulum_omega_1000_steps") as f64;
        eprintln!(
            "rk4 envs={envs} steps={steps}: theta max={th_err:e} at {th_idx}, omega max={om_err:e} at {om_idx}"
        );
        assert!(
            th_err <= th_tol,
            "rk4 theta: max |GPU - CPU| = {th_err:e} at env {th_idx}, tolerance {th_tol:e}"
        );
        assert!(
            om_err <= om_tol,
            "rk4 omega: max |GPU - CPU| = {om_err:e} at env {om_idx}, tolerance {om_tol:e}"
        );
    }

    #[test]
    fn multistep_trajectory_tracks_cpu_f64_reference() {
        const STEPS: usize = 1000;
        const DT: f32 = 0.001;
        let q_init = [1.0f32, -0.5, 0.25];
        let qd_init = [0.0f32, 0.3, -0.1];
        let qacc = [-1.0f32, 0.2, 0.05];

        let mut q_gpu = q_init;
        let mut qd_gpu = qd_init;
        let mut q_cpu: Vec<f64> = q_init.iter().map(|&x| x as f64).collect();
        let mut qd_cpu: Vec<f64> = qd_init.iter().map(|&x| x as f64).collect();
        let qacc_cpu: Vec<f64> = qacc.iter().map(|&x| x as f64).collect();

        for _ in 0..STEPS {
            gpu_semi_implicit_euler_step(&mut q_gpu, &mut qd_gpu, &qacc, DT).unwrap();
            robot_sim::integrators::semi_implicit_euler_step(
                &mut q_cpu,
                &mut qd_cpu,
                &qacc_cpu,
                DT as f64,
            );
        }

        // f32 rounding is ~6e-8 relative per operation. Over 1000 steps of a
        // linear update the accumulated error stays well inside 1e-4 absolute.
        const TRAJ_TOLERANCE_F32: f32 = 1e-4;
        for i in 0..q_init.len() {
            let diff_q = (q_gpu[i] - q_cpu[i] as f32).abs();
            let diff_qd = (qd_gpu[i] - qd_cpu[i] as f32).abs();
            assert!(
                diff_q < TRAJ_TOLERANCE_F32,
                "trajectory q[{i}] drift: GPU={}, CPU={}, diff={diff_q}",
                q_gpu[i],
                q_cpu[i]
            );
            assert!(
                diff_qd < TRAJ_TOLERANCE_F32,
                "trajectory qd[{i}] drift: GPU={}, CPU={}, diff={diff_qd}",
                qd_gpu[i],
                qd_cpu[i]
            );
        }
    }
}

#[cfg(not(feature = "cubecl"))]
mod no_gpu_tests {
    use super::*;

    #[test]
    fn disabled_feature_returns_typed_error() {
        let mut q = [1.0f32];
        let mut qd = [0.1f32];
        let qacc = [0.5f32];
        let err = gpu_semi_implicit_euler_step(&mut q, &mut qd, &qacc, 0.01).unwrap_err();
        assert_eq!(err, GpuSimError::GpuDisabled);
    }

    #[test]
    fn disabled_feature_still_rejects_malformed_inputs() {
        let q = [1.0f32];
        let q_ref = [0.0f32];
        let k = [10.0f32];
        let mut out = [0.0f32; 2];
        let err = gpu_joint_spring_forces(&q, &q_ref, &k, &mut out).unwrap_err();
        assert!(matches!(
            err,
            GpuSimError::DimensionMismatch {
                what: "out",
                expected: 1,
                found: 2
            }
        ));
    }
}
