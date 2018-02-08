@testset "update WF Tests" begin
    srand(2)
    p = rand(Dirichlet(ones(5)*1./5))
    data = rand(Multinomial(10, p),7)'
    tmp = DualOptimalFiltering.update_WF_params_debug([1., 0.1], ones(5)*1./5, [0*collect(1:5), collect(1:5)], data)
    @test tmp[1] == [[4, 0, 17, 23, 26], [5, 2, 20, 27, 31]]
    # @test tmp[2] == [0.578617, 0.421383]
    @test tmp[2][1] ≈ 0.578617 atol=10.0^(-5)
    @test tmp[2][2] ≈ 0.421383 atol=10.0^(-5)
end;

@testset "predict WF Tests" begin
    tmp = DualOptimalFiltering.WF_prediction_for_one_m_debug_mem2([0,4,2,1], ones(4) |> sum, 1.5)
    # @test tmp[1] ≈ 1.072578883495754 atol=10.0^(-5)
    @test length(keys(tmp)) == 30
    @test sum(values(tmp)) ≈ 1.0 atol=10.0^(-5)

    tmp = DualOptimalFiltering.WF_prediction_for_one_m_debug_mem2([0,4,2,1], ones(4) |> sum, 1.5; wm = 0.5)
    @test sum(values(tmp)) ≈ 0.5 atol=10.0^(-5)

    Λ_test = [[6, 9], [5, 10], [4, 27], [3, 2], [9, 8], [1, 7], [5, 17], [8, 3], [2, 18], [3, 28]]
    num_m = length(Λ_test)


    tmp = DualOptimalFiltering.predict_WF_params_debug_mem2(rand(Dirichlet(ones(num_m)*1./num_m)), 2., Λ_test, 1.)
    @test sum(tmp[2]) ≈ 1.0 atol=10.0^(-5)
    # [@test tmp[3][k] ≈ [0.686096, 0.268465, 0.0420196, 0.0032884, 0.000128673, 2.01396e-6][k] atol=10.0^(-5) for k in 1:6]
end;


@test size(rand(Dirichlet(4,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, 4)[:,2:end]*10 |> x -> round.(x) |> x -> Int64.(x) ) == (4,4)


@testset "WF filtering tests" begin
    srand(4)
    wfchain = rand(Dirichlet(4,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, 3)[:,2:end]*10 |> x -> round.(x) |> x -> Int64.(x)
    data = Dict(zip(linspace(0, 5, size(wfchain,2)), [wfchain[:,t:t]' for t in 1:size(wfchain,2)]))
    Λ_of_t, wms_of_t = DualOptimalFiltering.filter_WF_mem2(ones(4), data)
    @test length(keys(Λ_of_t)) == 3
    @test length(keys(wms_of_t)) == 3
    @test_throws AssertionError filter_WF(ones(2), data)
end;

using Nemo

@testset "WF filtering Nemo arbitrary precision standard function tests" begin
    @test typeof(DualOptimalFiltering.gamma_arb(1.1)) == Nemo.arb
    @test isequal(DualOptimalFiltering.λm_arb(6, 2.1), 21.2999999999999971578290569595992565155029296875)
    @test (DualOptimalFiltering.normalise_arb(DualOptimalFiltering.RR.(1:3)) |> x -> Float64.(x)) == (1/DualOptimalFiltering.RR(6) .* DualOptimalFiltering.RR.(1:3) |> x -> Float64.(x))
end;

@testset "WF filtering Nemo arbitrary precision prediction tests" begin
    @test DualOptimalFiltering.first_term_pmmi_arb(4, 6, 2.1) |> Float64 ≈ DualOptimalFiltering.RR(20376.2722499999940960186961547156920718805894724003203720448156050099359060884) |> Float64
    @test DualOptimalFiltering.denominator_Cmmi_arb(4, 3, 6, 2.1) |> Float64 ≈ DualOptimalFiltering.RR(-1702.97741249999966627208802805171764191348064322509729761737644302245150973277) |> Float64
    @test DualOptimalFiltering.Cmmi_arb(6, 4, 1., 2.1) |> Float64 ≈ DualOptimalFiltering.RR(8.177847065599812060536888397722834719127428496901059731705003930090871998367e-6) |> Float64

    K = 3
    α = ones(K)
    Pop_size = 15
    Ntimes = 3
    srand(4)
    wfchain = rand(Dirichlet(K,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, Ntimes)[:,2:end]
    wfobs = [rand(Multinomial(Pop_size, wfchain[:,k])) for k in 1:size(wfchain,2)] |> l -> hcat(l...)
    data = Dict(zip(linspace(0, 5, size(wfobs,2)) , [wfobs[:,t:t]' for t in 1:size(wfobs,2)]))

    # @test typeof(DualOptimalFiltering.precompute_first_term_arb(data, 2.1)) == Dict{Tuple{Int64,Int64},Nemo.arb}
    # @test typeof(DualOptimalFiltering.precompute_Cmmi_arb(data, 2.1; digits_after_comma_for_time_precision = 4)) == Dict{Tuple{Int64,Int64},Nemo.arb}
    # @test typeof(DualOptimalFiltering.precompute_binomial_coefficients_arb(data))  == Dict{Tuple{Int64,Int64},Nemo.fmpz}

    ν_dict_arb, Cmmi_dict_arb, precomputed_binomial_coefficients_arb = DualOptimalFiltering.precompute_terms_arb(data, sum(α); digits_after_comma_for_time_precision = 4)
    @test typeof(ν_dict_arb) == Dict{Tuple{Int64,Int64},Nemo.arb}
    @test typeof(Cmmi_dict_arb) == Dict{Tuple{Int64,Int64},Nemo.arb}
    @test typeof(precomputed_binomial_coefficients_arb) == Dict{Tuple{Int64,Int64},Nemo.fmpz}

    @test DualOptimalFiltering.hypergeom_pdf_using_precomputed_arb([5,4], [6,15], 9, 21, precomputed_binomial_coefficients_arb) |> Float64 ≈ DualOptimalFiltering.RR(0.027863777089783281733746130030959752321981424148606811145510835913312693498452) |> Float64
    @test DualOptimalFiltering.pmmi_raw_precomputed_arb([5,4], [6,15], 21, 9, 1., ν_dict_arb, Cmmi_dict_arb, precomputed_binomial_coefficients_arb) |> Float64 ≈ DualOptimalFiltering.RR(5.955457872806034870233019152630824866413482875372599635147999459272720706012e-90) |> Float64
    @test DualOptimalFiltering.pmmi_precomputed_arb([6,15], [6,15], 21, 21, 1., 2.1, ν_dict_arb, Cmmi_dict_arb, precomputed_binomial_coefficients_arb) |> Float64 ≈ DualOptimalFiltering.RR(0.9179738501810576933654436631471958181369642368022600097063707951783738991813) |> Float64

    @test typeof(DualOptimalFiltering.WF_prediction_for_one_m_precomputed_arb([6,15], 2.1, 1., ν_dict_arb, Cmmi_dict_arb, precomputed_binomial_coefficients_arb; wm = 1)) == Dict{Array{Int64,1},Nemo.arb}

    tst = DualOptimalFiltering.WF_prediction_for_one_m_precomputed_arb([6,15], 2.1, 1., ν_dict_arb, Cmmi_dict_arb, precomputed_binomial_coefficients_arb; wm = 1)
    DualOptimalFiltering.inc_arb!(tst, [1000,1000], DualOptimalFiltering.RR(-1))
    @test tst[[1000,1000]]  |> Float64 == -1.

    tst = DualOptimalFiltering.predict_WF_params_precomputed_arb([DualOptimalFiltering.RR(1.)], 2.1, [data[0] |> vec], 1., ν_dict_arb, Cmmi_dict_arb, precomputed_binomial_coefficients_arb)
    @test typeof(tst[1]) == Array{Array{Int64,1},1}
    @test length(tst[1]) == data[0] |> vec |> x -> x + 1 |> prod
    @test typeof(tst[2]) == Array{Nemo.arb,1}
    @test tst[2][1] |> Float64 ≈ DualOptimalFiltering.RR(2.625750848545005910749955597435596715743893777059224787555683102089318453052e-25) |> Float64


end;

@testset "WF filtering Nemo arbitrary precision update tests" begin
    K = 3
    α = ones(K)
    Pop_size = 15
    Ntimes = 3
    srand(4)
    wfchain = rand(Dirichlet(K,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, Ntimes)[:,2:end]
    wfobs = [rand(Multinomial(Pop_size, wfchain[:,k])) for k in 1:size(wfchain,2)] |> l -> hcat(l...)
    data = Dict(zip(linspace(0, 5, size(wfobs,2)) , [wfobs[:,t:t]' for t in 1:size(wfobs,2)]))

    tst = DualOptimalFiltering.update_WF_params_arb([DualOptimalFiltering.RR(1.)], α, [data[0] |> vec], data[0])
    @test (tst[1][1]) == [4, 18, 8]
    @test tst[2][1] |> Float64 == 1.
    tst = DualOptimalFiltering.update_WF_params_arb([DualOptimalFiltering.RR(.7), DualOptimalFiltering.RR(.3)], α, [data[0] |> vec, data[0] |> vec |> x -> x+2], data[0])
    @test sum(tst[2]) |> Float64 == 1.

end;
@testset "WF filtering Nemo arbitrary precision whole filtering tests" begin
    K = 3
    α = ones(K)
    Pop_size = 15
    Ntimes = 3
    srand(4)
    wfchain = rand(Dirichlet(K,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, Ntimes)[:,2:end]
    wfobs = [rand(Multinomial(Pop_size, wfchain[:,k])) for k in 1:size(wfchain,2)] |> l -> hcat(l...)
    data = Dict(zip(linspace(0, 5, size(wfobs,2)) , [wfobs[:,t:t]' for t in 1:size(wfobs,2)]))

    ν_dict_arb, Cmmi_dict_arb, precomputed_binomial_coefficients_arb = DualOptimalFiltering.precompute_terms_arb(data, sum(α); digits_after_comma_for_time_precision = 4)
    tst = DualOptimalFiltering.get_next_filtering_distribution_precomputed_arb([data[0] |> vec, data[0] |> vec |> x -> x+2], [DualOptimalFiltering.RR(.7), DualOptimalFiltering.RR(.3)], 0.3, 0.7, α, sum(α), data[0], ν_dict_arb, Cmmi_dict_arb, precomputed_binomial_coefficients_arb)
    # println(tst[1])
    @test length(tst[1]) == 420
    @test sum(tst[2]) |> Float64 == 1.

    Λ_of_t_arb, wms_of_t_arb = DualOptimalFiltering.filter_WF_precomputed_arb(α, data, ν_dict_arb, Cmmi_dict_arb, precomputed_binomial_coefficients_arb)
    times = keys(Λ_of_t_arb) |> collect |> sort
    @test length.([Λ_of_t_arb[t] for t in times]) == [1, 150, 800]
    @test Float64.(sum.([wms_of_t_arb[t] for t in times])) == [1., 1., 1.]

    # Λ_of_t, wms_of_t = DualOptimalFiltering.filter_WF_mem2(ones(4), data)
    # @test length(keys(Λ_of_t)) == 3
    # @test length(keys(wms_of_t)) == 3
    # @test_throws AssertionError filter_WF(ones(2), data)
end;
