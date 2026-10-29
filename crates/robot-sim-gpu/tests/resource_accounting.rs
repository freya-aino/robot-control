#[test]
fn resource_accounting_matches_known_4097_env_totals() {
    let rk4 = robot_sim_gpu::rk4_resources(4097).unwrap();
    assert_eq!(rk4.allocated_bytes(), 65552);
    assert_eq!(rk4.upload_bytes(), 32776);
    assert_eq!(rk4.readback_bytes(), 32776);
    assert_eq!(rk4.bindings, 4);

    let eq = robot_sim_gpu::equality_resources(2, 2, 4097).unwrap();
    assert_eq!(eq.input_bytes, 458_864);
    assert_eq!(eq.output_bytes, 65_552);
    assert_eq!(eq.flags_bytes, 49_164);
    assert_eq!(eq.allocated_bytes(), 573_580);
    assert_eq!(eq.upload_bytes(), 573_580);
    assert_eq!(eq.readback_bytes(), 114_716);
    assert_eq!(eq.bindings, 3);
}
