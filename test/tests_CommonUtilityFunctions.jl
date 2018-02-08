@test DualOptimalFiltering.normalise(1:4) == (1:4)/10
tmp = DualOptimalFiltering.get_quantiles_from_mass(0.95)
@test tmp[1] ≈ 0.025 atol=10.0^(-10)
@test tmp[2] ≈ 0.975 atol=10.0^(-10)
