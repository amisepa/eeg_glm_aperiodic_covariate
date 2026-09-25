% Demo: Comparing aperiodic estimation methods for GLM ERSP analysis
%
% This simulation tests whether different methods for estimating the aperiodic
% component can reduce the background-dependent bias observed in v2, and whether
% any method is agnostic to the additive vs. multiplicative generative model.
%
% Methods compared:
% 1. Original: Exclude only alpha band (8-12 Hz)
%    - Standard approach: log-log linear regression (log₁₀(Power) ~ log₁₀(Freq))
%      fitted to all frequencies except 8-12 Hz to estimate offset and exponent
%    - Assumes 1/f^β relationship: Power(f) = offset × f^(-exponent)
%    - Simplest approach but assumes we know exactly where the peak is
%    
% 2. Wide exclusion: Exclude 6-14 Hz
%    - Rationale: Alpha peaks vary across individuals (8-13 Hz) and have 
%      bandwidth extending beyond nominal range. Excluding 6-14 Hz ensures
%      we don't fit through the "skirts" of the alpha peak
%    - Conservative approach when peak location/width is uncertain
%    
% 3. Multi-band: Exclude theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz)
%    - Rationale: Most neural oscillations fall in these ranges across individuals
%    - Fits only the "clean" aperiodic regions below theta and above beta
%    - Assumes these bands capture most oscillatory contamination
%    
% 4. Robust regression: Iterative reweighting to downweight peaks
%    - Rationale: Treat spectral peaks as "outliers" to be downweighted
%    - Uses bisquare weights: iteratively reduces influence of points far
%      from the fitted line (which should be peaks in log-log space)
%    - Assumption: peaks are deviations from an underlying smooth 1/f trend
%    
% 5. FOOOF-style: Iterative fit-subtract-refit approach
%    - Rationale: Don't pre-specify which frequencies to exclude; let the
%      data tell you where peaks are
%    - Process: (1) fit initial aperiodic, (2) subtract to flatten spectrum,
%      (3) detect remaining elevated regions as peaks, (4) refit excluding peaks
%    - Most adaptive approach; doesn't require prior knowledge of peak locations
%    
% 6. Low-frequency only: Fit 2-6 Hz, extrapolate to higher frequencies
%    - Rationale: Avoid ALL oscillatory peaks by fitting only the low-frequency
%      range where structured oscillations are rare/low-amplitude
%    - Tests whether local fitting (low freq) vs. global fitting (exclude peaks)
%      produces different results
%    - Risk: Fails if aperiodic properties change with frequency (e.g., spectral
%      "knee"). Assumes 1/f^β relationship holds constant across all frequencies
%    
% 7. IRASA: Irregular-Resampling Auto-Spectral Analysis
%    - Rationale: Separate aperiodic from periodic through geometric resampling
%      rather than curve fitting. Exploits the fact that resampling preserves
%      fractal (aperiodic) but not oscillatory (periodic) components
%    - Method: Resample at multiple non-integer factors (h = 1.1 to 1.9), compute
%      geometric mean of upsampled/downsampled spectra. Periodic components cancel,
%      aperiodic components are preserved
%    - IMPORTANT: This is a SIMPLIFIED approximation for pre-computed spectra.
%      We simulate the resampling effect through frequency-axis transformations
%      (interpolation). True IRASA operates on time-domain signals with actual
%      resampling. For real EEG data, use proper IRASA toolboxes (e.g., Fieldtrip,
%      MNE-Python). Our approximation tests the conceptual approach but may not
%      fully capture IRASA's performance on real data
%
% For each method, we assess on BOTH additive and multiplicative data:
% - Recovery accuracy in linear space (additive correct) and dB space (mult correct)
% - Background dependency (correlation between estimated offset and recovered change)
% - Model-agnosticism: Does any method work correctly regardless of generative model?
%
% Key question: Can we find a method that achieves r ≈ 0 (background-invariant)
% AND works correctly in both linear and dB space (model-agnostic)?
%
% Cedric Cannard, January 2026
clear; close all; clc
repo_path = 'C:\Users\ccann\Documents\MATLAB\eeg_glm_aperiodic_covariate';
cd(repo_path)
rng(42);

%% Setup
f = 2:0.5:40;
alpha_mu = 10; alpha_bw = 1.5;
N = 200;

% Ground truth: SAME alpha change across all trials
alpha_change_true = 0.25;

% Variable aperiodic backgrounds
off0 = 0.8 + 0.6*rand(N,1);
exp0 = 1.3 + 0.3*randn(N,1);
A0 = 0.5 + 0.15*randn(N,1);

% Post-stimulus: backgrounds change variably, alpha change is constant
off1 = off0 .* (1.2 + 0.4*rand(N,1));
exp1 = exp0 - (0.1 + 0.2*rand(N,1));
A1 = A0 + alpha_change_true;

%% Generate BOTH additive and multiplicative spectra

% Additive model: oscillation adds to aperiodic
toSpec_add = @(off,expo,A) off.*(f.^(-expo)) + ...
                           A.*exp(-0.5*((f-alpha_mu)/alpha_bw).^2);

% Multiplicative model: oscillation modulates aperiodic
toSpec_mult = @(off,expo,A) off.*(f.^(-expo)) .* ...
                            (1 + A.*exp(-0.5*((f-alpha_mu)/alpha_bw).^2));

P0_add = zeros(N,numel(f)); P1_add = P0_add;
P0_mult = P0_add; P1_mult = P0_add;

for k = 1:N
    % Additive
    P0_add(k,:) = toSpec_add(off0(k),exp0(k),A0(k)) .* exp(0.05*randn(1,numel(f)));
    P1_add(k,:) = toSpec_add(off1(k),exp1(k),A1(k)) .* exp(0.05*randn(1,numel(f)));
    
    % Multiplicative
    P0_mult(k,:) = toSpec_mult(off0(k),exp0(k),A0(k)) .* exp(0.05*randn(1,numel(f)));
    P1_mult(k,:) = toSpec_mult(off1(k),exp1(k),A1(k)) .* exp(0.05*randn(1,numel(f)));
end

%% Define all methods

methods_info = {
    'Original: Exclude 8-12 Hz', @(P) method_standard(P, f, f>=8 & f<=12);
    'Wide exclusion: 6-14 Hz', @(P) method_standard(P, f, f>=6 & f<=14);
    'Multi-band: theta+alpha+beta', @(P) method_standard(P, f, (f>=4 & f<=8) | (f>=8 & f<=13) | (f>=13 & f<=30));
    'Robust regression', @(P) method_robust(P, f);
    'FOOOF-style iterative', @(P) method_fooof(P, f);
    'Low-freq only: 2-6 Hz', @(P) method_standard(P, f, f>=2 & f<=6, true);
    'IRASA resampling', @(P) method_irasa(P, f);
};

n_methods = size(methods_info, 1);

%% Test all methods on both generative models

fprintf('\n========================================\n');
fprintf('ADDITIVE DATA (linear space is correct)\n');
fprintf('========================================\n');

results_add_lin = test_all_methods(methods_info, P0_add, P1_add, f, false, alpha_change_true);

fprintf('\n========================================\n');
fprintf('ADDITIVE DATA (dB space - WRONG model)\n');
fprintf('========================================\n');

results_add_db = test_all_methods(methods_info, P0_add, P1_add, f, true, alpha_change_true);

fprintf('\n========================================\n');
fprintf('MULTIPLICATIVE DATA (linear - WRONG model)\n');
fprintf('========================================\n');

results_mult_lin = test_all_methods(methods_info, P0_mult, P1_mult, f, false, alpha_change_true);

fprintf('\n========================================\n');
fprintf('MULTIPLICATIVE DATA (dB space is correct)\n');
fprintf('========================================\n');

% For multiplicative, need expected dB change
alpha0_mult = mean(P0_mult(:,f>=8 & f<=12),2);
alpha1_mult = mean(P1_mult(:,f>=8 & f<=12),2);
true_db_change = mean(10*log10(alpha1_mult ./ alpha0_mult));

results_mult_db = test_all_methods(methods_info, P0_mult, P1_mult, f, true, true_db_change);

%% Summary comparison figure

figure('Color','w','Position',[50 50 1600 1100]);
tiledlayout(3,4,'Padding','compact','TileSpacing','compact');

% Top row: Summary bars
nexttile; 
plot_summary(results_add_lin, 'Additive data, Linear GLM (CORRECT)', alpha_change_true);

nexttile;
plot_summary(results_add_db, 'Additive data, dB GLM (WRONG)', NaN);

nexttile;
plot_summary(results_mult_lin, 'Multiplicative data, Linear GLM (WRONG)', alpha_change_true);

nexttile;
plot_summary(results_mult_db, 'Multiplicative data, dB GLM (CORRECT)', true_db_change);

% Bottom rows: Correlation plots
for i = 1:n_methods
    nexttile;
    plot_correlation(results_add_lin(i), methods_info{i,1}, alpha_change_true);
end

sgtitle('Aperiodic Estimation Methods: Additive vs Multiplicative', 'FontSize', 14, 'FontWeight', 'bold');
exportgraphics(gcf, fullfile(repo_path, 'results_v3.png'), 'Resolution', 300);

%% Helper Functions

function results = test_all_methods(methods_info, P0, P1, f, use_log, ground_truth)
    n_methods = size(methods_info, 1);
    results = struct('name', {}, 'recovery', {}, 'error', {}, 'corr_r', {}, 'corr_p', {}, ...
                     'offsets', {}, 'delta', {});
    
    for i = 1:n_methods
        method_name = methods_info{i,1};
        method_func = methods_info{i,2};
        
        % Estimate aperiodic parameters
        [off_hat, exp_hat] = method_func(P0);
        
        % Remove any complex values (shouldn't happen but safety check)
        off_hat = real(off_hat);
        exp_hat = real(exp_hat);
        
        % Run GLM
        [recovery, delta, offsets] = run_glm_analysis(P0, P1, f, off_hat, exp_hat, use_log);
        
        % Ensure real values for correlation
        offsets = real(offsets);
        delta = real(delta);
        
        % Statistics
        error_val = recovery - ground_truth;
        [r, p] = corr(offsets, delta);
        
        % Store results
        results(i).name = method_name;
        results(i).recovery = recovery;
        results(i).error = error_val;
        results(i).corr_r = r;
        results(i).corr_p = p;
        results(i).offsets = offsets;
        results(i).delta = delta;
        
        % Print
        if use_log
            fprintf('%s\n  Recovery: %.3f dB (error: %.3f)\n', ...
                method_name, recovery, error_val);
        else
            fprintf('%s\n  Recovery: %.3f uV^2/Hz (error: %.3f, %.1f%%)\n', ...
                method_name, recovery, error_val, abs(error_val/ground_truth)*100);
        end
        fprintf('  Background correlation: r = %.3f, p = %.4f\n\n', r, p);
    end
end

function [recovery, delta, offsets] = run_glm_analysis(P0, P1, f, off_hat, exp_hat, use_log)
    alpha_mask = f>=8 & f<=12;
    alpha0 = mean(P0(:,alpha_mask),2);
    alpha1 = mean(P1(:,alpha_mask),2);
    
    if use_log
        y = 10*log10(alpha1 ./ alpha0);
        baseline_power = 10*log10(alpha0);
    else
        y = alpha1 - alpha0;
        baseline_power = alpha0;
    end
    
    N = length(y);
    Xglm = [ones(N,1) zscore(off_hat) zscore(exp_hat) zscore(baseline_power)];
    betas = Xglm \ y;
    recovery = betas(1);
    delta = y;
    offsets = off_hat;
end

%% Method implementations

function [off_hat, exp_hat] = method_standard(P, f, exclude_mask, extrapolate_only)
    if nargin < 4
        extrapolate_only = false;
    end
    
    N = size(P, 1);
    off_hat = zeros(N,1);
    exp_hat = zeros(N,1);
    
    if extrapolate_only
        % Use only the specified range for fitting
        fit_mask = exclude_mask;
    else
        % Exclude the specified range
        fit_mask = ~exclude_mask;
    end
    
    lf_fit = log10(f(fit_mask)).';
    
    for k = 1:N
        lp_fit = log10(P(k,fit_mask)).';
        X = [ones(numel(lf_fit),1) lf_fit];
        beta = X \ lp_fit;
        off_hat(k) = 10^beta(1);
        exp_hat(k) = -beta(2);
    end
end

function [off_hat, exp_hat] = method_robust(P, f)
    N = size(P, 1);
    off_hat = zeros(N,1);
    exp_hat = zeros(N,1);
    
    lf = log10(f).';
    X = [ones(numel(lf),1) lf];
    
    for k = 1:N
        lp = log10(P(k,:)).';
        
        % Iterative robust fitting with bisquare weights
        beta = X \ lp;
        for iter = 1:3
            resid = lp - X*beta;
            mad_resid = median(abs(resid - median(resid)));
            weights = 1 ./ (1 + (resid/(1.4826*mad_resid)).^2);
            beta = (X'*diag(weights)*X) \ (X'*diag(weights)*lp);
        end
        
        off_hat(k) = 10^beta(1);
        exp_hat(k) = -beta(2);
    end
end

function [off_hat, exp_hat] = method_fooof(P, f)
    N = size(P, 1);
    off_hat = zeros(N,1);
    exp_hat = zeros(N,1);
    
    alpha_mask = f>=8 & f<=12;
    lf = log10(f).';
    X = [ones(numel(lf),1) lf];
    
    for k = 1:N
        lp = log10(P(k,:)).';
        
        % Initial fit excluding alpha
        beta = X(~alpha_mask,:) \ lp(~alpha_mask);
        
        % Subtract aperiodic, detect peaks
        aperiodic_full = X * beta;
        flattened = lp - aperiodic_full;
        
        % Exclude frequencies with elevated power (peaks)
        threshold = mean(flattened) + 0.5*std(flattened);
        peak_mask = flattened > threshold;
        
        % Refit excluding detected peaks
        beta = X(~peak_mask,:) \ lp(~peak_mask);
        
        off_hat(k) = 10^beta(1);
        exp_hat(k) = -beta(2);
    end
end

function [off_hat, exp_hat] = method_irasa(P, f)
    N = size(P, 1);
    off_hat = zeros(N,1);
    exp_hat = zeros(N,1);
    
    % IRASA parameters
    resample_factors = 1.1:0.05:1.9;
    n_factors = length(resample_factors);
    
    for k = 1:N
        psd_orig = P(k,:);
        psd_resampled = zeros(n_factors, numel(f));
        
        for r_idx = 1:n_factors
            h = resample_factors(r_idx);
            
            % Simulate resampling effect on spectrum
            f_up = f * h;
            f_down = f / h;
            
            % Use 'pchip' for smooth interpolation, bounded extrapolation
            psd_up = interp1(f, psd_orig, f_up, 'pchip', psd_orig(end));
            psd_down = interp1(f, psd_orig, f_down, 'pchip', psd_orig(1));
            
            % Ensure positive values before geometric mean
            psd_up = max(psd_up, eps);
            psd_down = max(psd_down, eps);
            
            % Geometric mean preserves aperiodic, cancels periodic
            psd_resampled(r_idx,:) = sqrt(psd_up .* psd_down);
        end
        
        % Median across resampling factors
        aperiodic_irasa = median(psd_resampled, 1);
        
        % Ensure positive and real
        aperiodic_irasa = max(real(aperiodic_irasa), eps);
        
        % Fit slope to extracted aperiodic
        lf = log10(f).';
        lp_irasa = log10(aperiodic_irasa).';
        X = [ones(numel(lf),1) lf];
        beta = X \ lp_irasa;
        
        off_hat(k) = 10^beta(1);
        exp_hat(k) = -beta(2);
    end
end


%% Plotting functions

function plot_summary(results, title_str, ground_truth)
    n_methods = length(results);
    recoveries = [results.recovery];
    corrs = [results.corr_r];
    
    yyaxis left
    bar(1:n_methods, recoveries);
    ylabel('Recovery');
    if ~isnan(ground_truth)
        hold on;
        yline(ground_truth, 'r--', 'LineWidth', 2);
        ylabel('Recovery (uV^2/Hz or dB)');
    end
    
    yyaxis right
    plot(1:n_methods, abs(corrs), 'ko-', 'LineWidth', 2, 'MarkerFaceColor', 'k');
    ylabel('|r| with background');
    ylim([0 0.6]);
    
    xticks(1:n_methods);
    xticklabels(cellfun(@(x) strrep(x, ': ', '\n'), {results.name}, 'UniformOutput', false));
    xtickangle(45);
    title(title_str, 'FontSize', 10);
    grid on;
end

function plot_correlation(result, method_name, ground_truth)
    scatter(result.offsets, result.delta, 20, 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    yline(ground_truth, 'r--', 'LineWidth', 2);
    
    % Fit line
    p = polyfit(result.offsets, result.delta, 1);
    x_fit = linspace(min(result.offsets), max(result.offsets), 100);
    plot(x_fit, polyval(p, x_fit), 'k-', 'LineWidth', 1);
    
    xlabel('Offset');
    ylabel('\Delta Alpha');
    title(sprintf('%s\nr=%.2f, p=%.3f', method_name, result.corr_r, result.corr_p), ...
        'FontSize', 8);
end