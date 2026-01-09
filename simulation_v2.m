% Demo: ERSP GLM with aperiodic covariates - Testing additive vs multiplicative models
%
% This simulation addresses the fundamental question: can GLM recover the same
% oscillatory change when it occurs on top of different aperiodic background levels?
%
% Key tests:
% - Same alpha change (0.250 uV²/Hz) across all trials
% - Variable aperiodic backgrounds (offset 0.8-1.4, variable exponents)
% - Both additive and multiplicative generative models
% - Analysis in both linear and dB space
% - Assesses whether recovery is invariant to background level
%
% The simulation reveals that GLM method is model-dependent: it only works
% correctly when the analysis space (linear/dB) matches the generative model
% (additive/multiplicative). Even with correct matching, ~10-20% residual
% background-dependent bias remains.
%
% Cedric Cannard, January 2025

clear; close all; clc
repo_path = 'C:\Users\ccann\Documents\MATLAB\eeg_glm_aperiodic_covariate';
cd(repo_path)


%% Setup

rng(42);

f = 2:0.5:40;  % Hz
alpha_mu = 10; alpha_bw = 1.5;
N = 200;  % trials

%% Ground truth: SAME alpha change, DIFFERENT aperiodic backgrounds
% This is the key test: can GLM recover constant oscillatory change 
% regardless of background level?

alpha_change_true = 0.25;  % CONSTANT alpha increase (uV^2/Hz)

% Variable aperiodic backgrounds at baseline
off0 = 0.8 + 0.6*rand(N,1);     % offset varies 0.8-1.4
exp0 = 1.3 + 0.3*randn(N,1);    % exponent ~1.3±0.3

% Baseline alpha amplitude (also variable)
A0 = 0.5 + 0.15*randn(N,1);

% POST-stimulus: same backgrounds change, but alpha change is CONSTANT
off1 = off0 .* (1.2 + 0.4*rand(N,1));  % backgrounds increase variably
exp1 = exp0 - (0.1 + 0.2*rand(N,1));   % slopes flatten variably
A1 = A0 + alpha_change_true;            % CONSTANT alpha increase

%% Generate spectra under TWO generative models

% Model 1: ADDITIVE (neural oscillation adds to aperiodic)
toSpec_add = @(off,expo,A) off.*(f.^(-expo)) + ...
                           A.*exp(-0.5*((f-alpha_mu)/alpha_bw).^2);

% Model 2: MULTIPLICATIVE (oscillation modulates aperiodic)
toSpec_mult = @(off,expo,A) off.*(f.^(-expo)) .* ...
                            (1 + A.*exp(-0.5*((f-alpha_mu)/alpha_bw).^2));

P0_add = zeros(N,numel(f)); P1_add = P0_add;
P0_mult = P0_add; P1_mult = P0_add;

for k = 1:N
    % Additive spectra
    P0_add(k,:) = toSpec_add(off0(k),exp0(k),A0(k)) .* exp(0.05*randn(1,numel(f)));
    P1_add(k,:) = toSpec_add(off1(k),exp1(k),A1(k)) .* exp(0.05*randn(1,numel(f)));
    
    % Multiplicative spectra  
    P0_mult(k,:) = toSpec_mult(off0(k),exp0(k),A0(k)) .* exp(0.05*randn(1,numel(f)));
    P1_mult(k,:) = toSpec_mult(off1(k),exp1(k),A1(k)) .* exp(0.05*randn(1,numel(f)));
end

%% Helper function: estimate aperiodic and run GLM
function [recovered_change, betas, off_hat, exp_hat] = ...
    run_glm_analysis(P0, P1, f, use_log)
    
    N = size(P0,1);
    alpha_mask = f>=8 & f<=12;
    fit_mask = ~alpha_mask;
    lf_fit = log10(f(fit_mask)).';
    
    % Estimate aperiodic from baseline
    off_hat = zeros(N,1); exp_hat = zeros(N,1);
    for k = 1:N
        lp_fit = log10(P0(k,fit_mask)).';
        X = [ones(numel(lf_fit),1) lf_fit];
        beta = X \ lp_fit;
        off_hat(k) = 10^beta(1);
        exp_hat(k) = -beta(2);
    end
    
    % Extract alpha power
    alpha0 = mean(P0(:,alpha_mask),2);
    alpha1 = mean(P1(:,alpha_mask),2);
    
    % GLM in requested space
    if use_log
        % dB space (assumes multiplicative)
        y = 10*log10(alpha1 ./ alpha0);
        baseline_power = 10*log10(alpha0);
    else
        % Linear space (assumes additive)
        y = alpha1 - alpha0;
        baseline_power = alpha0;
    end
    
    Xglm = [ones(N,1) zscore(off_hat) zscore(exp_hat) zscore(baseline_power)];
    betas = Xglm \ y;
    recovered_change = betas(1);  % intercept = mean change after controlling
end

%% Test all 4 combinations: {additive, multiplicative} × {linear, dB}

fprintf('\n=== GROUND TRUTH: alpha change = %.3f uV^2/Hz ===\n\n', alpha_change_true);

% Additive data, linear GLM (MATCHED)
[rec_add_lin, ~, off_add, exp_add] = run_glm_analysis(P0_add, P1_add, f, false);
fprintf('Additive data → Linear GLM:  %.3f uV^2/Hz  (error: %.3f)\n', ...
    rec_add_lin, rec_add_lin - alpha_change_true);

% Additive data, dB GLM (MISMATCHED)
[rec_add_db, ~] = run_glm_analysis(P0_add, P1_add, f, true);
fprintf('Additive data → dB GLM:      %.3f dB        (MISMATCHED model)\n', rec_add_db);

% Multiplicative data, linear GLM (MISMATCHED)
[rec_mult_lin, ~, off_mult, exp_mult] = run_glm_analysis(P0_mult, P1_mult, f, false);
fprintf('\nMultiplicative data → Linear GLM: %.3f uV^2/Hz (MISMATCHED model)\n', rec_mult_lin);

% Multiplicative data, dB GLM (MATCHED)
[rec_mult_db, ~] = run_glm_analysis(P0_mult, P1_mult, f, true);
fprintf('Multiplicative data → dB GLM:     %.3f dB\n', rec_mult_db);

%% Convert multiplicative ground truth to dB for comparison
% In multiplicative model: P_post/P_base = (1+A1)/(1+A0)
% Since backgrounds vary, need to compute expected dB change
alpha1_mult = mean(P1_mult(:,f>=8 & f<=12),2);
alpha0_mult = mean(P0_mult(:,f>=8 & f<=12),2);
true_db_change = mean(10*log10(alpha1_mult ./ alpha0_mult));
fprintf('                              (expected dB: %.3f, error: %.3f)\n', ...
    true_db_change, rec_mult_db - true_db_change);

%% Key test: Does recovery depend on aperiodic background level?
% Split trials by aperiodic offset into low/high groups

% For additive data
[~, sort_idx] = sort(off_add);
low_idx = sort_idx(1:floor(N/2));
high_idx = sort_idx(ceil(N/2)+1:end);

alpha0_add = mean(P0_add(:,f>=8 & f<=12),2);
alpha1_add = mean(P1_add(:,f>=8 & f<=12),2);

fprintf('\n=== Critical test: Same alpha change on different backgrounds ===\n');
fprintf('Additive data, linear GLM:\n');
fprintf('  Low offset trials:  %.3f uV^2/Hz (mean offset: %.2f)\n', ...
    mean(alpha1_add(low_idx) - alpha0_add(low_idx)), mean(off_add(low_idx)));
fprintf('  High offset trials: %.3f uV^2/Hz (mean offset: %.2f)\n', ...
    mean(alpha1_add(high_idx) - alpha0_add(high_idx)), mean(off_add(high_idx)));
fprintf('  Difference: %.4f (should be ~0 if method is invariant)\n', ...
    mean(alpha1_add(high_idx)-alpha0_add(high_idx)) - mean(alpha1_add(low_idx)-alpha0_add(low_idx)));

fprintf('\nAdditive data, dB GLM:\n');
fprintf('  Low offset trials:  %.3f dB\n', ...
    mean(10*log10(alpha1_add(low_idx)./alpha0_add(low_idx))));
fprintf('  High offset trials: %.3f dB\n', ...
    mean(10*log10(alpha1_add(high_idx)./alpha0_add(high_idx))));
fprintf('  Difference: %.4f (MISMATCHED model shows background dependency)\n', ...
    mean(10*log10(alpha1_add(high_idx)./alpha0_add(high_idx))) - ...
    mean(10*log10(alpha1_add(low_idx)./alpha0_add(low_idx))));

%% Visualization
figure('Color','w','Position',[100 100 1200 800]);
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

% Additive spectra
nexttile; hold on;
plot(f, mean(P0_add,1), 'LineWidth', 1.5);
plot(f, mean(P1_add,1), 'LineWidth', 1.5);
patch([8 12 12 8], [0 0 max(mean(P1_add,1))*1.2 max(mean(P1_add,1))*1.2], ...
    [0.85 0.85 0.95], 'EdgeColor','none','FaceAlpha',0.5);
set(gca,'YScale','log');
xlabel('Frequency (Hz)'); ylabel('Power');
title('Additive model: mean spectra');
legend({'Baseline','Post','Alpha band'},'Location','best');

% Multiplicative spectra
nexttile; hold on;
plot(f, mean(P0_mult,1), 'LineWidth', 1.5);
plot(f, mean(P1_mult,1), 'LineWidth', 1.5);
patch([8 12 12 8], [0 0 max(mean(P1_mult,1))*1.2 max(mean(P1_mult,1))*1.2], ...
    [0.85 0.85 0.95], 'EdgeColor','none','FaceAlpha',0.5);
set(gca,'YScale','log');
xlabel('Frequency (Hz)'); ylabel('Power');
title('Multiplicative model: mean spectra');
legend({'Baseline','Post','Alpha band'},'Location','best');

% Recovery accuracy scatter
nexttile; hold on;
scatter(off_add, alpha1_add - alpha0_add, 30, 'filled', 'MarkerFaceAlpha', 0.6);
yline(alpha_change_true, 'r--', 'LineWidth', 2);
xlabel('Aperiodic offset'); ylabel('\Delta Alpha (linear)');
title('Additive: change vs. background');
legend({'Trials','True change'},'Location','best');

% Low vs high offset comparison
nexttile; hold on;
bar([1 2], [mean(alpha1_add(low_idx)-alpha0_add(low_idx)), ...
            mean(alpha1_add(high_idx)-alpha0_add(high_idx))]);
yline(alpha_change_true, 'r--', 'LineWidth', 2);
xticks([1 2]); xticklabels({'Low offset','High offset'});
ylabel('\Delta Alpha (linear, uV^2/Hz)');
title('Additive data: linear GLM recovery');
ylim([0 alpha_change_true*1.5]);

% dB space comparison (showing the problem)
nexttile; hold on;
bar([1 2], [mean(10*log10(alpha1_add(low_idx)./alpha0_add(low_idx))), ...
            mean(10*log10(alpha1_add(high_idx)./alpha0_add(high_idx)))]);
xticks([1 2]); xticklabels({'Low offset','High offset'});
ylabel('\Delta Alpha (dB)');
title('Additive data: dB space (WRONG)');

% Summary text
nexttile; axis off;
text(0.1, 0.9, 'Key Results:', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.75, sprintf('Ground truth: +%.3f uV^2/Hz', alpha_change_true), 'FontSize', 10);
text(0.1, 0.60, sprintf('Additive→Linear: %.3f', rec_add_lin), 'FontSize', 10, 'Color', 'g');
text(0.1, 0.50, sprintf('Additive→dB: %.3f dB', rec_add_db), 'FontSize', 10, 'Color', 'r');
text(0.1, 0.35, sprintf('Mult→Linear: %.3f', rec_mult_lin), 'FontSize', 10, 'Color', 'r');
text(0.1, 0.25, sprintf('Mult→dB: %.3f dB', rec_mult_db), 'FontSize', 10, 'Color', 'g');

exportgraphics(gcf, fullfile(repo_path, 'results_v2.png'), 'Resolution', 300);


%% Statistical test: does background level interact with estimated change?
fprintf('\n=== Regression test: does recovery depend on background? ===\n');
% If method is correct, background shouldn't predict residual error

delta_add_lin = alpha1_add - alpha0_add;
X_test = [ones(N,1) zscore(off_add)];
b_test = X_test \ delta_add_lin;
fprintf('Linear GLM on additive data: background β = %.4f (p from t-test)\n', b_test(2));
[~, p] = corrcoef(off_add, delta_add_lin);
fprintf('Correlation: r = %.3f, p = %.4f\n', corr(off_add, delta_add_lin), p(1,2));

delta_add_db = 10*log10(alpha1_add ./ alpha0_add);
b_test_db = X_test \ delta_add_db;
fprintf('\ndB GLM on additive data: background β = %.4f\n', b_test_db(2));
[~, p_db] = corrcoef(off_add, delta_add_db);
fprintf('Correlation: r = %.3f, p = %.4f (should be significant if WRONG model)\n', ...
    corr(off_add, delta_add_db), p_db(1,2));