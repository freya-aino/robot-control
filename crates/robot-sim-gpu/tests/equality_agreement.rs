use robot_sim_gpu::{check_equality_batch, gpu_solve_equality, GpuSimError};

mod common;

#[derive(Clone)]
struct Batch {
    nv: usize,
    nr: usize,
    mass: Vec<f32>,
    vel: Vec<f32>,
    smooth: Vec<f32>,
    jac: Vec<f32>,
    pos: Vec<f32>,
    margin: Vec<f32>,
    solref: Vec<f32>,
    solimp: Vec<f32>,
    force: Vec<f32>,
    qacc: Vec<f32>,
    clamp: Vec<u32>,
    status: Vec<u32>,
}

impl Batch {
    fn new(n: usize, nv: usize, nr: usize) -> Self {
        let mut mass = vec![0.; n * nv * nv];
        for e in 0..n {
            for i in 0..nv {
                mass[e * nv * nv + i * nv + i] = 1.;
            }
        }
        Self {
            nv,
            nr,
            mass,
            vel: vec![0.; n * nv],
            smooth: vec![0.; n * nv],
            jac: vec![0.; n * nr * nv],
            pos: vec![0.; n * nr],
            margin: vec![0.; n * nr],
            solref: [0.02, 1.].repeat(n * nr),
            solimp: [0.5, 0.5, 0.001, 0.5, 2.].repeat(n * nr),
            force: vec![123.; n * nr],
            qacc: vec![456.; n * nv],
            clamp: vec![9; n * nr],
            status: vec![9; n],
        }
    }

    fn run(&mut self) -> Result<(), GpuSimError> {
        gpu_solve_equality(
            self.nv,
            self.nr,
            &self.mass,
            &self.vel,
            &self.smooth,
            &self.jac,
            &self.pos,
            &self.margin,
            &self.solref,
            &self.solimp,
            0.001,
            &mut self.force,
            &mut self.qacc,
            &mut self.clamp,
            &mut self.status,
        )
    }
}

#[test]
fn validates_before_device_or_output_change() {
    assert!(check_equality_batch(6, 6, 4097).is_ok());
    assert!(matches!(
        check_equality_batch(7, 1, 1),
        Err(GpuSimError::InvalidParameter { .. })
    ));
    // 300k * (36+12+36+48) * 4 = 158.4 MB: the packed buffer
    // exceeds the binding limit although every separate input fits it.
    assert!(matches!(
        check_equality_batch(6, 6, 300_000),
        Err(GpuSimError::InvalidParameter { .. })
    ));
    let mut batch = Batch::new(1, 1, 1);
    batch.jac[0] = 1.;
    batch.margin[0] = 0.1;
    assert!(matches!(
        batch.run(),
        Err(GpuSimError::InvalidParameter { .. })
    ));
    batch.margin[0] = 0.;
    batch.solref[1] = -1.;
    assert!(matches!(
        batch.run(),
        Err(GpuSimError::InvalidParameter { .. })
    ));
    batch.solref[1] = 1.;
    batch.mass[0] = f32::NAN;
    assert!(matches!(batch.run(), Err(GpuSimError::NonFinite { .. })));
    batch.mass[0] = 1.;
    batch.jac.clear();
    assert!(matches!(
        batch.run(),
        Err(GpuSimError::DimensionMismatch { what: "jac", .. })
    ));
    assert_eq!(batch.force, [123.]);
    assert_eq!(batch.qacc, [456.]);
    assert_eq!(batch.clamp, [9]);
    assert_eq!(batch.status, [9]);
}

#[cfg(not(feature = "cubecl"))]
#[test]
fn valid_batch_is_not_cpu_emulated() {
    let mut batch = Batch::new(1, 1, 1);
    batch.jac[0] = 1.;
    assert_eq!(batch.run(), Err(GpuSimError::GpuDisabled));
}

#[cfg(feature = "cubecl")]
mod device {
    use super::*;
    use robot_sim::constraint::{solve_equality, EqualityRow};

    fn close(got: f32, expected: f64, abs: f64, rel: f64) {
        assert!(
            (f64::from(got) - expected).abs() <= abs + rel * expected.abs(),
            "GPU {got} CPU {expected}"
        );
    }

    #[test]
    fn scalar_derived_and_clamped_timeconst() {
        // From `mj_makeImpedance`: imp=.5, K=1/(.5²*.02²)=10000,
        // B=2/(.5*.02)=200; pos=.001, J=M=1, vel=0:
        // aref=-5, A=1, R=(1-imp)*A/imp=1, force=-5/2=-2.5.
        let mut b = Batch::new(2, 1, 1);
        b.jac = vec![1., 1.];
        b.pos = vec![0.001, 0.001];
        b.solref[2] = 0.0005; // clamp to 2*dt=.002: K=1e6, aref=-500.
        b.run().unwrap();
        close(b.force[0], -2.5, 1e-5, 1e-6);
        close(b.qacc[0], -2.5, 1e-5, 1e-6);
        close(b.force[1], -250., 2e-4, 1e-6);
        assert_eq!(b.clamp, [0, 1]);
        assert_eq!(b.status, [1, 1]);
    }

    #[test]
    fn coupled_mass_and_row_system_derived() {
        // M=[[2,1],[1,2]], M^-1 J^T = [1,-1] for J=[1,-1].
        // pos=.002, vel=J*[.1,-.2]=.3, imp=.5 => aref=-200*.3-10000*.5*.002=-70.
        // A=2, R=2, f=-70/4=-17.5, qacc=[-17.5,17.5].
        let mut b = Batch::new(1, 2, 1);
        b.mass = vec![2., 1., 1., 2.];
        b.jac = vec![1., -1.];
        b.vel = vec![0.1, -0.2];
        b.pos[0] = 0.002;
        b.run().unwrap();
        close(b.force[0], -17.5, 5e-5, 1e-6);
        close(b.qacc[0], -17.5, 5e-5, 1e-6);
        close(b.qacc[1], 17.5, 5e-5, 1e-6);
        assert_eq!(b.status, [1]);
    }

    #[test]
    fn random_cpu_agreement_across_partial_tail_and_impedance_branches() {
        // Deterministic f32 inputs; CPU receives exactly their f64 conversion.
        // M=L L^T + I (well conditioned); nrow>nv cases have dependent
        // rows, whose diagonal regularization still makes A+R SPD.
        // Nonlinear solimp positions span x<=mid, x>mid, saturation, and
        // sign cases. Fixed edge rows also exercise clamping, p==1, and
        // non-positive/tiny widths; 4097 envs exercises the partial tail.
        let record = common::record("equality_random_spd_force_qacc");
        let abs_tol = common::number(&record, "tolerance_abs");
        let rel_tol = common::number(&record, "tolerance_rel");
        for (nv, nr) in [(1, 1), (1, 2), (2, 2), (2, 4), (2, 5), (3, 2), (6, 6)] {
            let n = 4097;
            let mut b = Batch::new(n, nv, nr);
            let mut seed = 0x5120_5eed_u64;
            let mut random = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 32) as u32 as f32) / (u32::MAX as f32) - 0.5
            };
            for e in 0..n {
                let mut l = vec![0f32; nv * nv];
                for i in 0..nv {
                    for j in 0..=i {
                        l[i * nv + j] = if i == j {
                            0.7 + random().abs()
                        } else {
                            random() * 0.2
                        };
                    }
                }
                for i in 0..nv {
                    b.vel[e * nv + i] = random() * 0.02;
                    b.smooth[e * nv + i] = random() * 0.1;
                    for j in 0..nv {
                        let mut sum = if i == j { 1. } else { 0. };
                        for k in 0..nv {
                            sum += l[i * nv + k] * l[j * nv + k];
                        }
                        b.mass[e * nv * nv + i * nv + j] = sum;
                    }
                }
                for r in 0..nr {
                    let ri = e * nr + r;
                    for i in 0..nv {
                        b.jac[ri * nv + i] = if i == r % nv { 1. } else { random() * 0.1 };
                    }
                    b.pos[ri] = match (e + r) % 6 {
                        0 => 0.0002,
                        1 => -0.0008,
                        2 => 0.0012,
                        3 => 0.,
                        4 => -0.001,
                        _ => random() * 0.0009,
                    };
                    b.solimp[ri * 5..ri * 5 + 5].copy_from_slice(&[0.2, 0.8, 0.001, 0.4, 2.]);
                    if (e + r) % 7 == 0 {
                        b.solref[ri * 2..ri * 2 + 2].copy_from_slice(&[-50., -3.]);
                    }
                    if (e + r) % 11 == 0 {
                        b.solref[ri * 2] = 0.0005;
                        b.solref[ri * 2 + 1] = 1.;
                    }
                    if nv == 2 && nr == 5 && e == 0 {
                        match r {
                            0 => {
                                // Clamp d0, d1, mid and p; p becomes one.
                                b.pos[ri] = 0.005;
                                b.solimp[ri * 5..ri * 5 + 5]
                                    .copy_from_slice(&[-0.2, 1.2, 0.01, 1.5, 0.5]);
                            }
                            1 => {
                                // The kernel must take its width <= MINVAL branch.
                                b.pos[ri] = 0.005;
                                b.solimp[ri * 5..ri * 5 + 5]
                                    .copy_from_slice(&[0.2, 0.8, 1e-16, 0.4, 2.]);
                            }
                            2 => {
                                // x=abs(pos/width)=0.7 > mid=0.4.
                                b.pos[ri] = 0.007;
                                b.solimp[ri * 5..ri * 5 + 5]
                                    .copy_from_slice(&[0.2, 0.8, 0.01, 0.4, 2.]);
                            }
                            3 => {
                                // Negative width is clamped to zero.
                                b.pos[ri] = 0.005;
                                b.solimp[ri * 5..ri * 5 + 5]
                                    .copy_from_slice(&[0.2, 0.8, -0.01, 0.4, 2.]);
                            }
                            4 => {
                                // Mid clamp check: pos=0.005, width=0.01 => x=0.5.
                                // Mid=1.5, p=2. With clamp(0.9999), mid=0.9999.
                                b.pos[ri] = 0.005;
                                b.solimp[ri * 5..ri * 5 + 5]
                                    .copy_from_slice(&[0.2, 0.8, 0.01, 1.5, 2.0]);
                            }
                            _ => unreachable!(),
                        }
                        b.solref[ri * 2..ri * 2 + 2].copy_from_slice(&[0.02, 1.]);
                    }
                }
            }
            b.run().unwrap();
            let mut worst = 0f64;
            let mut max_abs = (0f64, 0usize);
            for (e, &status) in b.status.iter().enumerate() {
                assert_eq!(status, 1, "env {e} nv {nv} nr {nr}");
                let rows: Vec<_> = (0..nr)
                    .map(|r| {
                        let ri = e * nr + r;
                        EqualityRow {
                            jacobian: b.jac[ri * nv..(ri + 1) * nv]
                                .iter()
                                .map(|&x| f64::from(x))
                                .collect(),
                            pos: f64::from(b.pos[ri]),
                            solref: <[f32; 2]>::try_from(&b.solref[ri * 2..ri * 2 + 2])
                                .unwrap()
                                .map(f64::from),
                            solimp: <[f32; 5]>::try_from(&b.solimp[ri * 5..ri * 5 + 5])
                                .unwrap()
                                .map(f64::from),
                        }
                    })
                    .collect();
                let cpu = solve_equality(
                    &b.mass[e * nv * nv..(e + 1) * nv * nv]
                        .iter()
                        .map(|&x| f64::from(x))
                        .collect::<Vec<_>>(),
                    &b.vel[e * nv..(e + 1) * nv]
                        .iter()
                        .map(|&x| f64::from(x))
                        .collect::<Vec<_>>(),
                    &b.smooth[e * nv..(e + 1) * nv]
                        .iter()
                        .map(|&x| f64::from(x))
                        .collect::<Vec<_>>(),
                    &rows,
                    f64::from(0.001f32),
                )
                .unwrap();
                for r in 0..nr {
                    let expected = cpu.force[r];
                    let got = b.force[e * nr + r];
                    let err = (f64::from(got) - expected).abs();
                    let ratio = err / (abs_tol + rel_tol * expected.abs());
                    worst = worst.max(ratio);
                    if err > max_abs.0 {
                        max_abs = (err, e);
                    }
                    assert!(
                        ratio <= 1.,
                        "force env {e} row {r} nv {nv} nr {nr}: {got} vs {expected} ratio {ratio}"
                    );
                    assert_eq!(
                        b.clamp[e * nr + r],
                        u32::from(cpu.clamped_rows.contains(&r))
                    );
                }
                for i in 0..nv {
                    let expected = cpu.qacc[i];
                    let got = b.qacc[e * nv + i];
                    let err = (f64::from(got) - expected).abs();
                    let ratio = err / (abs_tol + rel_tol * expected.abs());
                    worst = worst.max(ratio);
                    if err > max_abs.0 {
                        max_abs = (err, e);
                    }
                    assert!(
                        ratio <= 1.,
                        "qacc env {e} dof {i} nv {nv} nr {nr}: {got} vs {expected} ratio {ratio}"
                    );
                }
            }
            eprintln!("equality nv={nv} nr={nr}: worst error/bound {worst:.4}, max abs error {:.6e} at env {}", max_abs.0, max_abs.1);
        }
    }

    #[test]
    fn non_spd_environments_keep_outputs() {
        let mut b = Batch::new(300, 2, 1);
        for e in 0..300 {
            b.jac[e * 2] = 1.;
            b.pos[e] = 0.001;
        }
        b.mass[7 * 4..7 * 4 + 4].copy_from_slice(&[1., 2., 2., 1.]);
        b.mass[299 * 4] = -1.;
        b.run().unwrap();
        for (e, &st) in b.status.iter().enumerate() {
            if e == 7 || e == 299 {
                assert_eq!(st, 0);
                assert_eq!(b.force[e], 123.);
                assert_eq!(b.qacc[e * 2..e * 2 + 2], [456., 456.]);
                assert_eq!(b.clamp[e], 9);
            } else {
                assert_eq!(st, 1);
                close(b.force[e], -2.5, 1e-5, 1e-6);
            }
        }
    }
}
