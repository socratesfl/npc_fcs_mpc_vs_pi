% =========================================================================
%  Comparison: FCS-MPC  vs  PI (dq) + PD-SPWM
%  3-level NPC inverter feeding a three-phase RL load (isolated neutral)
% =========================================================================
%  Controllers compared
%    1) FCS-MPC with one-step delay compensation and a neutral-point (NP)
%       balancing term in the cost function
%    2) PI in the dq frame (with decoupling and anti-windup) + level-shifted
%       SPWM (phase disposition), WITHOUT neutral-point balancing
%    3) Same as 2), plus a common-mode offset injection that balances the
%       neutral point (a fairer comparison against the MPC)
%
%  Model
%    - Ideal DC source (Vdc) across two equal capacitors (C each).
%      Vc1 + Vc2 = Vdc, and dV = Vc1 - Vc2 evolves with the neutral-point
%      current:  C * d(dV)/dt = i_np
%    - Pole voltage: +Vc1 (state +1), 0 (state 0), -Vc2 (state -1)
%    - Wye-connected RL load with isolated neutral:
%      v_phase = v_pole - mean(v_pole)
%    - The plant is discretized exactly (ZOH); the MPC uses forward Euler
%
%  Requirements: MATLAB R2016b+ (implicit expansion) or GNU Octave.
%                No toolboxes required.
% =========================================================================
clear; clc; close all;

%% 1. Parameters
par.R   = 5;      par.L = 10e-3;    % RL load per phase (ohm, H)
par.Vdc = 600;    par.C = 2e-3;     % DC link (V); capacitance of each capacitor (F)
par.f   = 60;     par.w = 2*pi*par.f;

par.dt  = 2e-6;   % plant simulation step (s)
par.Ts  = 20e-6;  % FCS-MPC sampling period (s)
par.fsw = 5e3;    % SPWM carrier frequency (Hz)

T_sim  = 0.1;     % simulation length (6 cycles at 60 Hz)
t_step = 0.05;    % instant of the reference step (exactly 3 cycles)
cfg.plot = true;

% Ratios between the sampling rates (must be integers)
Nmpc  = round(par.Ts/par.dt);          % plant steps per MPC sample
Ncar  = round(1/(par.fsw*par.dt));     % plant steps per carrier period
Npi   = Ncar/2;                        % PI samples at carrier peaks and valleys
Ts_pi = Npi*par.dt;
N     = round(T_sim/par.dt);
t     = (0:N-1)*par.dt;

% Exact (ZOH) discretization of the RL plant
par.a = exp(-par.R*par.dt/par.L);
par.b = (1 - par.a)/par.R;

% Three-phase reference with an amplitude step from 10 A to 15 A
I_amp = @(tt) 10 + 5*(tt >= t_step);
phi   = [0; -2*pi/3; 2*pi/3];
i_ref = @(tt) I_amp(tt) .* sin(par.w*tt + phi);

%% 2. FCS-MPC configuration
[Sa, Sb, Sc] = ndgrid([-1 0 1], [-1 0 1], [-1 0 1]);
cand = [Sa(:) Sb(:) Sc(:)];            % 27 switching states (27x3)

lambda     = 0.05;     % NP balancing weight (with lambda = 0, dV diverges)
par.I_base = 15;       % base used to normalize the current error (A)
par.V_base = 10;       % base used to normalize the imbalance dV (V)

%% 3. PI (dq) + PD-SPWM configuration
f_bw = 500;                            % desired current-loop bandwidth (Hz)
Kp   = par.L*(2*pi*f_bw);              % pole cancellation: Kp = L*wc
Ki   = par.R*(2*pi*f_bw);              %                    Ki = R*wc
Vmax = par.Vdc/2;                      % voltage vector limit (m = 1)
k_np = 0.3;                            % NP offset gain (pu/V) - controller 3

%% 4. States and histories  (controller index: 1 = MPC, 2 = PI, 3 = PI + NP)
names = {'FCS-MPC', 'PI-dq + SPWM', 'PI-dq + SPWM + NP bal.'};
Ic    = zeros(3,3);                    % currents (phase x controller)
DVc   = zeros(1,3);                    % imbalance dV = Vc1 - Vc2
Sc_   = zeros(3,3);                    % applied switching state
Ih    = zeros(3,N,3);
DVh   = zeros(3,N);
Sh    = zeros(3,N,3);

S_next = zeros(3,1);                   % optimal state computed in the previous step (MPC)
xi     = zeros(2,3);                   % PI dq integrators
m_ref  = zeros(3,3);                   % modulating signals held between samples

%% 5. Main loop
disp('Simulating... Please wait.');
for k = 1:N
    tk = (k-1)*par.dt;

    % ---- FCS-MPC (runs every Ts) ---------------------------------------
    if mod(k-1, Nmpc) == 0
        S_apply = S_next;                                          % state decided in the previous step
        [i1, dv1] = prediction(Ic(:,1).', DVc(1), S_apply.', par); % delay compensation (k+1)
        [i2, dv2] = prediction(i1, dv1, cand, par);                % evaluate candidates (k+2)
        iref2 = i_ref(tk + 2*par.Ts).';
        g = sum(((iref2 - i2)/par.I_base).^2, 2) + lambda*abs(dv2)/par.V_base;
        [~, idx] = min(g);
        S_next   = cand(idx,:).';
        Sc_(:,1) = S_apply;
    end

    % ---- PI-dq (runs at carrier peaks and valleys) ---------------------
    if mod(k-1, Npi) == 0
        th  = par.w*tk - pi/2;                              % aligns the d axis with the reference
        rdq = park(i_ref(tk), th);
        for c = 2:3
            idq  = park(Ic(:,c), th);
            e    = rdq - idq;
            vff  = par.w*par.L*[-idq(2); idq(1)];           % decoupling
            xi_n = xi(:,c) + Ki*Ts_pi*e;
            vdq  = Kp*e + xi_n + vff;
            if norm(vdq) > Vmax                             % saturation + anti-windup
                vdq = vdq*Vmax/norm(vdq);
            else
                xi(:,c) = xi_n;
            end
            m = ipark(vdq, th)/(par.Vdc/2);
            if c == 3
                m = m + k_np*DVc(c);                        % common-mode offset (balances the NP)
            end
            m_ref(:,c) = max(min(m, 1), -1);
        end
    end

    % ---- PD modulator (in-phase carriers, level-shifted) ---------------
    ph  = mod(k-1, Ncar)/Ncar;
    tri = 1 - 4*abs(ph - 0.5);                              % triangle from -1 to 1
    c1  = 0.5*tri + 0.5;                                    % upper carrier (0 to 1)
    c2  = 0.5*tri - 0.5;                                    % lower carrier (-1 to 0)
    for c = 2:3
        S = -ones(3,1);
        S(m_ref(:,c) >= c2) = 0;
        S(m_ref(:,c) >= c1) = 1;
        Sc_(:,c) = S;
    end

    % ---- Plant + logging -----------------------------------------------
    for c = 1:3
        [Ic(:,c), DVc(c)] = plant(Ic(:,c), DVc(c), Sc_(:,c), par);
        Ih(:,k,c) = Ic(:,c);
        DVh(c,k)  = DVc(c);
        Sh(:,k,c) = Sc_(:,c);
    end
end
disp('Simulation complete!');

%% 6. Metrics
% Windows with whole cycles in steady state: [1/60, 3/60] (10 A) and [4/60, 6/60] (15 A)
win = round([1/60 3/60; 4/60 6/60]/par.dt);
th_all = par.w*t - pi/2;

fprintf('\n%-24s %8s %8s %10s %10s %9s %9s %10s\n', 'Controller', ...
        'THD10[%]', 'THD15[%]', 'fsw[kHz]', 'RMSE15[A]', 'max|dV|', 'rms dV', 't_set[ms]');
met = struct();
for c = 1:3
    ia = squeeze(Ih(1,:,c));
    thd1 = calc_thd(ia(win(1,1)+1:win(1,2)), t(win(1,1)+1:win(1,2)), par.f);
    thd2 = calc_thd(ia(win(2,1)+1:win(2,2)), t(win(2,1)+1:win(2,2)), par.f);
    w2   = win(2,1)+1:win(2,2);
    fsw  = calc_fsw(Sh(:,w2,c), par.dt);
    iref_w = I_amp(t(w2)) .* sin(par.w*t(w2) + phi);
    rmse = sqrt(mean(mean((squeeze(Ih(:,w2,c)) - iref_w).^2)));
    dv   = DVh(c, win(1,1)+1:end);
    idq  = [(2/3)*sum(squeeze(Ih(:,:,c)).*[cos(th_all); cos(th_all-2*pi/3); cos(th_all+2*pi/3)], 1);
           -(2/3)*sum(squeeze(Ih(:,:,c)).*[sin(th_all); sin(th_all-2*pi/3); sin(th_all+2*pi/3)], 1)];
    tset = settling_time(idq(1,:), t, t_step, 15, 0.05);
    fprintf('%-24s %8.2f %8.2f %10.2f %10.3f %9.2f %9.2f %10.2f\n', names{c}, ...
            thd1, thd2, fsw/1e3, rmse, max(abs(dv)), sqrt(mean(dv.^2)), tset*1e3);
    met(c).name = names{c};  met(c).thd10 = thd1;  met(c).thd15 = thd2;
    met(c).fsw = fsw;  met(c).rmse15 = rmse;  met(c).dv_max = max(abs(dv));
    met(c).dv_rms = sqrt(mean(dv.^2));  met(c).t_set = tset;
    Idq{c} = idq; %#ok<SAGROW>
end
fprintf(['\nTHD: current residual (all components except the fundamental) / fundamental.\n' ...
         'fsw: level transitions per phase / (2*T), equivalent to the per-switch switching frequency.\n' ...
         't_set: time until id stays within +-5%% of 15 A after the step.\n']);

%% 7. Plots
if cfg.plot
    ref_all = I_amp(t) .* sin(par.w*t + phi);
    colors  = {[0 0.45 0.74], [0.85 0.33 0.10], [0.47 0.67 0.19]};

    figure('Name', 'Currents - Phase A', 'Position', [100 100 850 520]);
    for c = 1:2
        subplot(2,1,c);
        plot(t*1e3, ref_all(1,:), 'k--', 'LineWidth', 1.5); hold on;
        plot(t*1e3, squeeze(Ih(1,:,c)), 'Color', colors{c}, 'LineWidth', 1);
        grid on; ylabel('Current (A)'); title([names{c} ': phase A']);
        legend('Reference', 'Measured', 'Location', 'southeast');
    end
    xlabel('Time (ms)');

    figure('Name', 'Step response (dq)', 'Position', [130 130 850 520]);
    jn = t >= t_step - 1e-3 & t <= t_step + 3e-3;
    subplot(2,1,1); hold on;
    for c = 1:2, plot(t(jn)*1e3, Idq{c}(1,jn), 'Color', colors{c}, 'LineWidth', 1.2); end
    plot(t(jn)*1e3, I_amp(t(jn)), 'k--'); grid on;
    ylabel('i_d (A)'); title('Step response: 10 A to 15 A'); legend(names{1:2}, 'Reference', 'Location', 'southeast');
    subplot(2,1,2); hold on;
    for c = 1:2, plot(t(jn)*1e3, Idq{c}(2,jn), 'Color', colors{c}, 'LineWidth', 1.2); end
    grid on; ylabel('i_q (A)'); xlabel('Time (ms)');

    figure('Name', 'Neutral-point imbalance', 'Position', [160 160 850 400]);
    hold on;
    for c = 1:3, plot(t*1e3, DVh(c,:), 'Color', colors{c}, 'LineWidth', 1.3); end
    grid on; xlabel('Time (ms)'); ylabel('V_{c1} - V_{c2} (V)');
    title('Neutral-point imbalance'); legend(names, 'Location', 'northwest');
end

%% ========================= Local functions ==============================
function [i_new, dv_new] = plant(i, dv, S, par)
% One plant step (dt): wye RL load, isolated neutral, DC link with dynamic dV
    vc1   = par.Vdc/2 + dv/2;
    vc2   = par.Vdc/2 - dv/2;
    vpole = (S == 1)*vc1 - (S == -1)*vc2;       % pole voltage w.r.t. the DC-link midpoint
    vph   = vpole - mean(vpole);                % remove the common-mode voltage
    i_new = par.a*i + par.b*vph;
    inp   = sum((1 - abs(S)).*(i + i_new)/2);   % current injected into the neutral point
    dv_new = dv + par.dt/par.C*inp;
end

function [i_nx, dv_nx] = prediction(i, dv, S, par)
% One-step (Ts) MPC prediction, vectorized over the rows of S (n x 3)
% i: 1x3, dv: scalar, S: n x 3
    vc1   = par.Vdc/2 + dv/2;
    vc2   = par.Vdc/2 - dv/2;
    vpole = (S == 1)*vc1 - (S == -1)*vc2;
    vph   = vpole - mean(vpole, 2);
    i_nx  = (1 - par.R*par.Ts/par.L)*i + (par.Ts/par.L)*vph;
    inp   = sum((1 - abs(S)).*(i + i_nx)/2, 2);
    dv_nx = dv + par.Ts/par.C*inp;
end

function dq = park(abc, th)
    c  = [cos(th); cos(th - 2*pi/3); cos(th + 2*pi/3)];
    s  = [sin(th); sin(th - 2*pi/3); sin(th + 2*pi/3)];
    dq = (2/3)*[c.'*abc; -s.'*abc];
end

function abc = ipark(dq, th)
    c   = [cos(th); cos(th - 2*pi/3); cos(th + 2*pi/3)];
    s   = [sin(th); sin(th - 2*pi/3); sin(th + 2*pi/3)];
    abc = c*dq(1) - s*dq(2);
end

function thd = calc_thd(x, tt, f)
% THD (%) by projection onto the fundamental (window with whole cycles)
    x   = x(:).' - mean(x);
    w   = 2*pi*f;
    a1  = 2*mean(x.*cos(w*tt));
    b1  = 2*mean(x.*sin(w*tt));
    I1  = sqrt((a1^2 + b1^2)/2);
    Irm = sqrt(mean(x.^2));
    thd = 100*sqrt(max(Irm^2 - I1^2, 0))/I1;
end

function fsw = calc_fsw(S, dt)
% Average switching frequency per phase (level transitions / 2T)
    trans = sum(sum(abs(diff(S, 1, 2))));
    T     = size(S, 2)*dt;
    fsw   = trans/3/(2*T);
end

function tset = settling_time(x, tt, t0, xf, band)
% Time after t0 until x stays within +-band*xf
    idx = find(tt >= t0);
    out = find(abs(x(idx) - xf) > band*xf, 1, 'last');
    if isempty(out)
        tset = 0;
    elseif out == numel(idx)
        tset = NaN;
    else
        tset = tt(idx(out + 1)) - t0;
    end
end
