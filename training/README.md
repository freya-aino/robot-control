# Description

# Goal

# Process

### Sensor Processing

- [ ] UnitreeSdk2Python connector for collected async data extraction
    - [ ] Lidar Point Cloud
    - [ ] IMU
    - [ ] Global position (#todo: verify calibration / initialization)
    - [ ] Actuator State
    - [ ] Foot Preasuer (go2)
    - [ ] (#todo: other)
- [ ] Move to `state_real(t_real)`
- [ ] External connector
    - [ ] (Depth Camera)
    - [ ] Camera
- [ ] Add Message queue

### Data Control Association

- [ ] `state_real(t_real) <-> [usd/usrd/mjcf] state_sim(t_sim)` semantic mapping (#todo: try using LLM for general purpose static transcription generation).
- [ ] #todo: what happens to the initial pose in 3D space, can we expect to reset sim or real state position.

### Pose Optimization (Constraint Enforcement & sim2sim)

- [ ] `f(state_real[t]) -> state_sim[t]` (observe)
- [ ] `c(state_sim[t], priors, constraints) -> state_sim[t+1]` (simulate, constraint)
- [ ] `p(state_sim[t+1]) -> state_real[t+1]*` (action)
- [ ] apply `state_real[t+1]*`
- [ ] `g(state_real[t+1], state_real[t+1]*)` (feedback)


- [ ] `f(state_real, t_real, state_sim*) -> d_state_sim*/d_t_sim`
- [ ] `constraint_fn(d_state_sim*/d_t_sim) -> d_state_sim/d_t_sim`
- [ ] `simulation_fn(state_sim, d_state_sim/d_t_sim) -> state_sim(t_sim+1)`
- [ ] `c(d_state_real*/d_t_real, state_real, constraint_model, ) -> d_state_real/d_t_real`
