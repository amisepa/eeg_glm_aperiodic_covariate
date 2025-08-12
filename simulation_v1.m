% Demo: ERSP dB vs GLM with aperiodic covariates
% When does dB disagree with the true oscillatory change, and how does 
% aperiodic-aware GLM behave?
% 
% Cedric Cannard, Aug 2025

clear; close all; clc

repo_path = '/Users/cedriccannard/Documents/MATLAB/eeg_glm_aperiodic_covariate';
cd(repo_path)

rng(7);

% Frequencies
f = 2:0.5:40;                      % Hz
alpha_mu = 10; alpha_bw = 1.5;     % alpha center and width

% Trials and baseline variability
N = 200;
off0 = 1.0 * exp(0.2*randn(N,1));  % aperiodic offset (baseline)
exp0 = 1.5 + 0.15*randn(N,1);      % aperiodic exponent (baseline)
A0   = 0.60 + 0.10*randn(N,1);     % alpha peak amplitude (baseline)

% Post changes: true alpha ERD plus broadband shift
A1   = 0.80*A0;                    % 20% ERD
off1 = 1.5*off0;                   % offset increase
exp1 = exp0 - 0.20;                % flatter slope
true_change = mean(A1 - A0);       % ground truth in uV^2/Hz

% Spectrum generator
toSpec = @(off,expo,A) off.*(f.^(-expo)) + A.*exp(-0.5*((f-alpha_mu)/alpha_bw).^2);

% Build spectra with mild multiplicative noise
P0 = zeros(N,numel(f)); P1 = P0;
for k = 1:N
    P0(k,:) = toSpec(off0(k),exp0(k),A0(k)) .* exp(0.05*randn(1,numel(f)));
    P1(k,:) = toSpec(off1(k),exp1(k),A1(k)) .* exp(0.05*randn(1,numel(f)));
end

% Alpha-band power (8–12 Hz)
alpha_mask = f>=8 & f<=12;
alpha0 = mean(P0(:,alpha_mask),2);
alpha1 = mean(P1(:,alpha_mask),2);

% Classic ERSP in dB
ersp_db = 10*log10(alpha1 ./ alpha0);
fprintf('Classic ERSP (dB): mean = %.2f dB\n', mean(ersp_db));

% Estimate baseline aperiodic offset and exponent via log-log regression
fit_mask = true(size(f));
fit_mask(alpha_mask) = false;
lf_fit = log10(f(fit_mask)).';

b0 = zeros(N,1); b1 = zeros(N,1);
for k = 1:N
    lp_fit = log10(P0(k,fit_mask)).';
    X = [ones(numel(lf_fit),1) lf_fit];
    beta = X \ lp_fit;
    b0(k) = beta(1);
    b1(k) = beta(2);
end
off_hat = 10.^b0;             % aperiodic offset proxy
exp_hat = -b1;                 % aperiodic exponent proxy

% GLM on alpha change in linear units
delta_alpha = alpha1 - alpha0;
Xglm = [ones(N,1) off_hat exp_hat alpha0];
betas = Xglm \ delta_alpha;
delta_adj_mean = betas(1);
fprintf('GLM adjusted change:  mean = %.4f uV^2/Hz (intercept)\n', delta_adj_mean);


%% Figure 1: spectra, differences, and GLM-adjusted views (4x2 tiles)

% Precompute means and raw differences
m0     = mean(P0,1);
m1     = mean(P1,1);
m0_db  = 10*log10(m0);
m1_db  = 10*log10(m1);
diff_lin = m1 - m0;
diff_db  = 10*log10(m1 ./ m0);

% Per-frequency GLM with aperiodic covariates + baseline power
nF = numel(f);
glm_lin_int = zeros(1,nF);   % adjusted mean change (linear)
glm_db_int  = zeros(1,nF);   % adjusted mean change (dB)
for j = 1:nF
    pj0 = P0(:,j); pj1 = P1(:,j);

    % Linear delta model
    y_lin = pj1 - pj0;
    X_lin = [ones(N,1) off_hat exp_hat pj0];
    b_lin = X_lin \ y_lin;
    glm_lin_int(j) = b_lin(1);

    % dB delta model
    y_db = 10*log10(pj1 ./ pj0);
    X_db = [ones(N,1) off_hat exp_hat 10*log10(pj0)];
    b_db = X_db \ y_db;
    glm_db_int(j) = b_db(1);
end

% Reconstruct GLM-adjusted mean PSDs
m1_glm_lin = m0 + glm_lin_int;       % adjusted post PSD in linear units
m1_glm_db  = m0_db + glm_db_int;     % adjusted post PSD in dB

figure('Color','w');
tiledlayout(4,2,'Padding','compact','TileSpacing','compact');

% Row 1: mean spectra (linear)
nexttile; hold on;
yl = [min([m0 m1])*0.7, max([m0 m1])*1.3];
patch([8 12 12 8],[yl(1) yl(1) yl(2) yl(2)],[0.85 0.85 0.95], 'EdgeColor','none','FaceAlpha',0.5);
semilogy(f, m0, 'LineWidth',1.5);
semilogy(f, m1, 'LineWidth',1.5);
ylim(yl); xlim([min(f) max(f)]);
xlabel('Frequency (Hz)'); ylabel('Power (\muV^2/Hz)');
title('Mean spectra (linear). Shaded = 8–12 Hz');
legend({'Alpha band','Baseline','Post'},'Location','best');

% Row 1: mean spectra (dB)
nexttile; hold on;
yl_db = [min([m0_db m1_db])-1, max([m0_db m1_db])+1];
patch([8 12 12 8],[yl_db(1) yl_db(1) yl_db(2) yl_db(2)],[0.85 0.85 0.95], 'EdgeColor','none','FaceAlpha',0.5);
plot(f, m0_db, 'LineWidth',1.5);
plot(f, m1_db, 'LineWidth',1.5);
ylim(yl_db); xlim([min(f) max(f)]);
xlabel('Frequency (Hz)'); ylabel('Power (dB)');
title('Mean spectra (dB). Shaded = 8–12 Hz');
legend({'Alpha band','Baseline','Post'},'Location','best');

% Row 2: raw difference (linear)
nexttile; hold on;
yld = [min(diff_lin)*1.1, max(diff_lin)*1.1]; if yld(1)==yld(2), yld = yld + [-1 1]*max(1e-6,abs(yld(1))*0.1); end
patch([8 12 12 8],[yld(1) yld(1) yld(2) yld(2)],[0.85 0.85 0.95], 'EdgeColor','none','FaceAlpha',0.5);
plot(f, diff_lin, 'k', 'LineWidth',1.5);
yline(0,'--','HandleVisibility','off');
xlim([min(f) max(f)]); ylim(yld);
xlabel('Frequency (Hz)'); ylabel('\Delta Power (\muV^2/Hz)');
title('Post − Baseline (linear)');

% Row 2: raw difference (dB)
nexttile; hold on;
yld_db = [min(diff_db)*1.1, max(diff_db)*1.1]; if yld_db(1)==yld_db(2), yld_db = yld_db + [-1 1]*0.1; end
patch([8 12 12 8],[yld_db(1) yld_db(1) yld_db(2) yld_db(2)],[0.85 0.85 0.95], 'EdgeColor','none','FaceAlpha',0.5);
plot(f, diff_db, 'k', 'LineWidth',1.5);
yline(0,'--','HandleVisibility','off');
xlim([min(f) max(f)]); ylim(yld_db);
xlabel('Frequency (Hz)'); ylabel('\Delta Power (dB)');
title('Post vs Baseline (dB = 10·log10(Post/Baseline))');

% Row 3: GLM-adjusted PSDs (linear)
nexttile; hold on;
yl_lin2 = [min([m0 m1_glm_lin])*0.7, max([m0 m1_glm_lin])*1.3];
patch([8 12 12 8],[yl_lin2(1) yl_lin2(1) yl_lin2(2) yl_lin2(2)],[0.85 0.85 0.95], 'EdgeColor','none','FaceAlpha',0.5);
semilogy(f, m0, 'LineWidth',1.5);
semilogy(f, m1_glm_lin, 'LineWidth',1.5);
xlim([min(f) max(f)]); ylim(yl_lin2);
xlabel('Frequency (Hz)'); ylabel('Power (\muV^2/Hz)');
title('GLM-adjusted PSD (Baseline vs Post adj, linear)');
legend({'Alpha band','Baseline','Post (GLM adj.)'},'Location','best');

% Row 3: GLM-adjusted PSDs (dB)
nexttile; hold on;
yl_db2 = [min([m0_db m1_glm_db])-1, max([m0_db m1_glm_db])+1];
patch([8 12 12 8],[yl_db2(1) yl_db2(1) yl_db2(2) yl_db2(2)],[0.85 0.85 0.95], 'EdgeColor','none','FaceAlpha',0.5);
plot(f, m0_db, 'LineWidth',1.5);
plot(f, m1_glm_db, 'LineWidth',1.5);
xlim([min(f) max(f)]); ylim(yl_db2);
xlabel('Frequency (Hz)'); ylabel('Power (dB)');
title('GLM-adjusted PSD (Baseline vs Post adj, dB)');
legend({'Alpha band','Baseline','Post (GLM adj.)'},'Location','best');

% Row 4: GLM-adjusted difference (linear)
nexttile; hold on;
yld_glm = [min(glm_lin_int)*1.1, max(glm_lin_int)*1.1];
if yld_glm(1)==yld_glm(2), yld_glm = yld_glm + [-1 1]*max(1e-6,abs(yld_glm(1))*0.1); end
patch([8 12 12 8],[yld_glm(1) yld_glm(1) yld_glm(2) yld_glm(2)],[0.85 0.85 0.95], 'EdgeColor','none','FaceAlpha',0.5);
plot(f, glm_lin_int, 'k', 'LineWidth',1.5);
yline(0,'--','HandleVisibility','off');
xlim([min(f) max(f)]); ylim(yld_glm);
xlabel('Frequency (Hz)'); ylabel('\Delta Power adj (\muV^2/Hz)');
title('GLM-adjusted post − baseline (linear)');

% Row 4: GLM-adjusted difference (dB)
nexttile; hold on;
yld_db_glm = [min(glm_db_int)*1.1, max(glm_db_int)*1.1];
if yld_db_glm(1)==yld_db_glm(2), yld_db_glm = yld_db_glm + [-1 1]*0.1; end
patch([8 12 12 8],[yld_db_glm(1) yld_db_glm(1) yld_db_glm(2) yld_db_glm(2)],[0.85 0.85 0.95], 'EdgeColor','none','FaceAlpha',0.5);
plot(f, glm_db_int, 'k', 'LineWidth',1.5);
yline(0,'--','HandleVisibility','off');
xlim([min(f) max(f)]); ylim(yld_db_glm);
xlabel('Frequency (Hz)'); ylabel('\Delta Power adj (dB)');
title('GLM-adjusted 10·log10(Post/Baseline)');

exportgraphics(gcf, 'fig1.png', 'Resolution', 300);

%% Diagnostics: per-frequency GLM coefficients (linear and dB)

% Refit frequency-wise GLMs with z-scored predictors to get comparable betas
glm_lin_b = zeros(nF,4);   % columns: [intercept, offset, exponent, baseline]
glm_db_b  = zeros(nF,4);

for j = 1:nF
    pj0 = P0(:,j); pj1 = P1(:,j);

    % z-score predictors once per frequency
    z_off   = zscore(off_hat);
    z_exp   = zscore(exp_hat);
    z_pj0   = zscore(pj0);
    z_pj0db = zscore(10*log10(pj0));

    % Linear delta model
    y_lin = pj1 - pj0;
    X_lin = [ones(N,1) z_off z_exp z_pj0];
    b_lin = X_lin \ y_lin;
    glm_lin_b(j,:) = b_lin.';   % keep intercept too

    % dB delta model
    y_db = 10*log10(pj1 ./ pj0);
    X_db = [ones(N,1) z_off z_exp z_pj0db];
    b_db = X_db \ y_db;
    glm_db_b(j,:) = b_db.';
end

% Plot coefficient traces across frequency
figure('Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% Linear model betas
nexttile; hold on;
plot(f, glm_lin_b(:,2), 'LineWidth', 1.2);   % offset
plot(f, glm_lin_b(:,3), 'LineWidth', 1.2);   % exponent
plot(f, glm_lin_b(:,4), 'LineWidth', 1.2);   % baseline
yline(0,'--','HandleVisibility','off');
xlabel('Frequency (Hz)'); ylabel('\beta (linear model)');
title('GLM coefficients vs frequency (linear \Delta)');
legend({'offset','exponent','baseline'},'Location','best');

% dB model betas
nexttile; hold on;
plot(f, glm_db_b(:,2), 'LineWidth', 1.2);    % offset
plot(f, glm_db_b(:,3), 'LineWidth', 1.2);    % exponent
plot(f, glm_db_b(:,4), 'LineWidth', 1.2);    % baseline dB
yline(0,'--','HandleVisibility','off');
xlabel('Frequency (Hz)'); ylabel('\beta (dB model)');
title('GLM coefficients vs frequency (dB \Delta)');
legend({'offset','exponent','baseline dB'},'Location','best');

exportgraphics(gcf,'fig2.png','Resolution',300);
