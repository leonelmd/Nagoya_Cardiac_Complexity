using CSV
using DataFrames
using Statistics
using Dates
using LinearAlgebra

# --- Core Algorithm Implementations ---

# 1. Sample Entropy Matches
function get_matches_fast(signal, m, r)
    N = length(signal)
    if N <= m return 0, 0 end
    B, A = 0, 0
    @inbounds for i in 1:(N - m)
        for j in (i + 1):(N - m)
            match_m = true
            for k in 0:(m-1)
                if abs(signal[i+k] - signal[j+k]) > r
                    match_m = false; break
                end
            end
            if match_m
                B += 1
                if abs(signal[i+m] - signal[j+m]) <= r
                    A += 1
                end
            end
        end
    end
    return A, B
end

# 2. Refined Composite Multiscale Entropy (RC-MSE)
function rc_mse(signal, m, r_factor, scales)
    N = length(signal)
    en_list = Float64[]
    for tau in scales
        sum_A, sum_B = 0.0, 0.0
        for k in 1:tau
            len_cg = trunc(Int, (N - k + 1) / tau)
            if len_cg <= m continue end
            cg_sig = Vector{Float64}(undef, len_cg)
            @inbounds for i in 1:len_cg
                s = 0.0
                for j in ((i-1)*tau + k):(i*tau + k - 1)
                    s += signal[j]
                end
                cg_sig[i] = s / tau
            end
            r_scale = r_factor * std(cg_sig)
            a_k, b_k = get_matches_fast(cg_sig, m, r_scale)
            sum_A += a_k; sum_B += b_k
        end
        push!(en_list, (sum_A > 0 && sum_B > 0) ? -log(sum_A/sum_B) : NaN)
    end
    return en_list
end

# 3. DFA alpha1
function dfa_alpha1(x; n_range=4:16)
    N = length(x)
    if N < 20 return NaN end
    y = cumsum(x .- mean(x))
    fluctuations = Float64[]
    for n in n_range
        n_segments = floor(Int, N/n)
        if n_segments < 1 continue end
        f_n_sum = 0.0
        for i in 1:n_segments
            start = (i-1)*n + 1
            stop = i*n
            segment = y[start:stop]
            t = 1:n
            coef = [ones(n) t] \ segment
            trend = coef[1] .+ coef[2] .* t
            f_n_sum += mean((segment .- trend).^2)
        end
        push!(fluctuations, sqrt(f_n_sum / n_segments))
    end
    if length(fluctuations) < 2 return NaN end
    log_n = log.(n_range[1:length(fluctuations)])
    log_f = log.(fluctuations)
    coefs = [ones(length(log_n)) log_n] \ log_f
    return coefs[2]
end

# --- Time Handling ---
function get_seconds_from_any(st)
    if st isa Time
        return hour(st)*3600 + minute(st)*60 + second(st)
    elseif st isa String
        parts = parse.(Int, split(st, ':'))
        return parts[1]*3600 + parts[2]*60 + parts[3]
    else
        return 0
    end
end

# --- Main Pipeline ---

function run_pipeline()
    base_dir = joinpath(@__DIR__, "..")
    db_path = joinpath(base_dir, "data/metadata/metadata.csv")
    data_dir = joinpath(base_dir, "data/processed_rri")
    out_dir = joinpath(base_dir, "calculations")
    mkpath(out_dir)
    
    df_meta = CSV.read(db_path, DataFrame)
    
    # Standard analysis parameters
    win_size_sec = 30 * 60
    step_size_sec = 60  # 1-minute step for high resolution
    total_sec = 24 * 3600
    
    all_subject_data = Vector{DataFrame}(undef, nrow(df_meta))
    
    println(">>> Starting 24h Evolution Analysis...")
    Threads.@threads for i in 1:nrow(df_meta)
        row = df_meta[i, :]
        file_path = joinpath(data_dir, row.Filename)
        if !isfile(file_path) 
            all_subject_data[i] = DataFrame()
            continue 
        end
        
        # Load raw RRI
        raw_data = CSV.read(file_path, DataFrame, header=false, skipto=1, delim=' ', ignorerepeated=true, types=[Float64, Float64])
        rename!(raw_data, [:rel_time, :rri])
        
        # Artifact filtering (0.3s to 2.0s)
        mask_clean = (raw_data.rri .> 0.3) .& (raw_data.rri .< 2.0)
        rel_time = raw_data.rel_time[mask_clean]
        rri = raw_data.rri[mask_clean]
        
        # Sync with clock time
        start_sec_offset = get_seconds_from_any(row.Start_Time)
        time_of_day = (rel_time .+ start_sec_offset) .% total_sec
        
        subject_results = []
        
        # Sliding Window loop
        for t_start in 0:step_size_sec:(total_sec - step_size_sec)
            t_end = t_start + win_size_sec
            if t_end <= total_sec
                mask = (time_of_day .>= t_start) .& (time_of_day .< t_end)
            else # spans midnight
                mask = (time_of_day .>= t_start) .| (time_of_day .< (t_end % total_sec))
            end
            
            win_rri = rri[mask]
            if length(win_rri) < 400 continue end
            
            # --- Metrics ---
            hr = 60.0 / mean(win_rri)
            sdnn = std(win_rri)
            rmssd = sqrt(mean(diff(win_rri).^2))
            sd2 = sqrt(max(0, 2 * sdnn^2 - 0.5 * std(diff(win_rri).^2)))
            
            # MSE Scales 1-5
            mse = rc_mse(win_rri, 2, 0.2, 1:5)
            comp = mean(mse)
            norm_comp = comp / hr
            
            alpha1 = dfa_alpha1(win_rri)
            
            push!(subject_results, (
                Subject=row.Subject_ID, 
                Group=row.Group, 
                Time_h=Float64(t_start/3600),
                HR=hr, SDNN=sdnn, RMSSD=rmssd, SD2=sd2,
                Complexity=comp, Norm_Comp=norm_comp, Alpha1=alpha1
            ))
        end
        println("  Finished Subject $(row.Subject_ID)")
        all_subject_data[i] = isempty(subject_results) ? DataFrame() : DataFrame(subject_results)
    end
    
    # Save Evolution Data
    final_df = vcat(filter(x -> !isempty(x), all_subject_data)...)
    CSV.write(joinpath(out_dir, "Full_HRV_Evolution_1min.csv"), final_df)
    println(">>> 24h Evolution saved.")
    
    # --- Discrete Clinical Windows ---
    println(">>> Extracting Day (10a-4p) and Night (12a-6a) Windows...")
    
    day_mask = (final_df.Time_h .>= 10) .& (final_df.Time_h .< 16)
    night_mask = (final_df.Time_h .>= 0) .& (final_df.Time_h .< 6)
    
    day_subj = combine(groupby(final_df[day_mask, :], [:Group, :Subject]), 
        [:HR, :SDNN, :RMSSD, :Complexity, :Norm_Comp] .=> mean .=> [:HR, :SDNN, :RMSSD, :Complexity, :Norm_Comp])
    night_subj = combine(groupby(final_df[night_mask, :], [:Group, :Subject]), 
        [:HR, :SDNN, :RMSSD, :Complexity, :Norm_Comp] .=> mean .=> [:HR, :SDNN, :RMSSD, :Complexity, :Norm_Comp])
        
    CSV.write(joinpath(out_dir, "Day_Window_Stats.csv"), day_subj)
    CSV.write(joinpath(out_dir, "Night_Window_Stats.csv"), night_subj)
    println(">>> Pipeline Complete.")
end

run_pipeline()
