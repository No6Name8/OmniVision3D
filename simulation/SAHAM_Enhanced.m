% =========================================================================
%  SAHAM — Counter-Drone Defense System
%  ENHANCED 3D SIMULATION — Full Mission Sequence
%  Version 4.0 — Competition Grade — MATLAB Online Compatible
% =========================================================================
%  Upload to MATLAB Online, then type:  SAHAM_Enhanced
% =========================================================================

function SAHAM_Enhanced()

clc; clear; close all;

fprintf('\n');
fprintf('=================================================\n');
fprintf('  SAHAM  COUNTER-DRONE DEFENSE SYSTEM           \n');
fprintf('  Enhanced 3D Mission Simulation v4.0           \n');
fprintf('  SWARM INTERCEPT + RADAR SWEEP + BLAST FX      \n');
fprintf('=================================================\n\n');

% ── SCENARIO: 3-drone swarm attack ───────────────────────────────
N_THREATS = 3;
threat_starts = [4900 4900 820; 4600 4200 650; 5000 3800 750];
threat_ends   = [100  100  180; 300  500  120; 200  800  160];
threat_spd    = 51.4;   % m/s = 185 km/h  (Shahed-136)
sahm_spd      = 138.9;  % m/s = 500 km/h

% Compute flight time from longest path
t_flights = zeros(N_THREATS,1);
for k=1:N_THREATS
    t_flights(k) = norm(threat_ends(k,:)-threat_starts(k,:))/threat_spd;
end

% Phase timeline
T_SCAN=0; T_LWIR=5; T_MIC=10; T_LASER=15;
T_LOCK=22; T_DATATX=27; T_LAUNCH=32; T_END=90;

% Precompute intercept points
pod_pos    = [2500 2500 0];
launch_pos = [800  800  0];
hit_pts    = zeros(N_THREATS,3);
for k=1:N_THREATS
    tvec = threat_ends(k,:)-threat_starts(k,:);
    fh   = min((T_LAUNCH+norm(tvec)*0.55/sahm_spd)/t_flights(k),0.86);
    fh   = 0.5*(1-cos(pi*fh));
    hit_pts(k,:) = threat_starts(k,:) + fh*tvec;
end

% ── COLORS ───────────────────────────────────────────────────────
C_BG   = [0.02 0.05 0.02];
C_G1   = [0.3  1.0  0.5 ];
C_G2   = [0.15 0.55 0.2 ];
C_G3   = [0.08 0.28 0.10];
C_RED  = [0.80 0.18 0.18];
C_YEL  = [1.0  0.88 0.22];
C_CYN  = [0.20 0.85 0.90];
C_ORG  = [1.0  0.55 0.10];

% ── FIGURE ───────────────────────────────────────────────────────
fig = figure('Name','SAHAM v4.0 — Swarm Intercept Simulation', ...
    'Color',[0.02 0.04 0.02],'NumberTitle','off', ...
    'Units','normalized','OuterPosition',[0.01 0.01 0.98 0.96]);

% ── TITLE BAR ────────────────────────────────────────────────────
ax_t = axes('Parent',fig,'Position',[0 0.94 1 0.06], ...
    'Color',[0.01 0.03 0.01],'XTick',[],'YTick',[], ...
    'XColor',C_G3,'YColor',C_G3,'Box','on');
text(0.5,0.65,'SAHAM   |   COUNTER-DRONE DEFENSE SYSTEM   |   SWARM INTERCEPT v4.0', ...
    'Parent',ax_t,'HorizontalAlignment','center', ...
    'FontName','Courier New','FontSize',13,'FontWeight','bold','Color',C_G1);
text(0.5,0.18,'3-DRONE SALVO  |  HYBRID SENSOR FUSION  |  AUTONOMOUS MULTI-INTERCEPT  |  CONCEPT DEMO', ...
    'Parent',ax_t,'HorizontalAlignment','center', ...
    'FontName','Courier New','FontSize',7,'Color',C_G2);

% ── 3D MAIN AXES ─────────────────────────────────────────────────
ax = axes('Parent',fig,'Position',[0.01 0.21 0.67 0.72]);
hold(ax,'on'); grid(ax,'on'); view(ax,-42,28);
ax.Color=C_BG; ax.XColor=[0.15 0.5 0.2]; ax.YColor=[0.15 0.5 0.2]; ax.ZColor=[0.15 0.5 0.2];
ax.GridColor=C_G3; ax.GridAlpha=0.30;
ax.FontName='Courier New'; ax.FontSize=8;
ax.XLim=[-200 5500]; ax.YLim=[-200 5500]; ax.ZLim=[0 1100];
xlabel(ax,'East  X (m)','Color',C_G2,'FontSize',9,'FontName','Courier New');
ylabel(ax,'North Y (m)','Color',C_G2,'FontSize',9,'FontName','Courier New');
zlabel(ax,'Altitude Z (m)','Color',C_G2,'FontSize',9,'FontName','Courier New');
title(ax,'SAHAM — LIVE 3D SWARM ENGAGEMENT', ...
    'Color',C_G1,'FontSize',11,'FontWeight','bold','FontName','Courier New');

% ── TELEMETRY PANEL ──────────────────────────────────────────────
ax_tl = axes('Parent',fig,'Position',[0.69 0.21 0.30 0.72]);
ax_tl.Color=[0.01 0.03 0.01]; ax_tl.XTick=[]; ax_tl.YTick=[];
ax_tl.XLim=[0 1]; ax_tl.YLim=[0 1]; ax_tl.Box='on';
ax_tl.XColor=C_G3; ax_tl.YColor=C_G3;
hold(ax_tl,'on');
text(0.5,0.978,'MISSION TELEMETRY','Parent',ax_tl, ...
    'HorizontalAlignment','center','FontName','Courier New', ...
    'FontSize',10,'FontWeight','bold','Color',C_G1);
line([0.03 0.97],[0.958 0.958],'Parent',ax_tl,'Color',C_G3,'LineWidth',0.8);

% ── PHASE BAR ────────────────────────────────────────────────────
ax_ph = axes('Parent',fig,'Position',[0.01 0.13 0.98 0.07]);
ax_ph.Color=[0.01 0.03 0.01]; ax_ph.XTick=[]; ax_ph.YTick=[];
ax_ph.XLim=[0 1]; ax_ph.YLim=[0 1]; ax_ph.Box='on';
ax_ph.XColor=C_G3; ax_ph.YColor=C_G3;
hold(ax_ph,'on');

% ── MINI PLOTS ───────────────────────────────────────────────────
ax_spd  = axes('Parent',fig,'Position',[0.01  0.01 0.23 0.11]);
ax_rng  = axes('Parent',fig,'Position',[0.26  0.01 0.23 0.11]);
ax_conf = axes('Parent',fig,'Position',[0.51  0.01 0.23 0.11]);
ax_alt  = axes('Parent',fig,'Position',[0.76  0.01 0.23 0.11]);
mini_ax  = [ax_spd ax_rng ax_conf ax_alt];
mini_ttl = {'THREAT SPEED (km/h)','RANGE TO POD (km)', ...
            'SENSOR CONFIDENCE (%)','THREAT ALTITUDE (m)'};
mini_col = {[1.0 0.35 0.35],[0.25 0.90 0.42],[0.25 0.90 0.42],[1.0 0.60 0.10]};
for i=1:4
    mini_ax(i).Color=C_BG; mini_ax(i).XColor=[0.12 0.40 0.15];
    mini_ax(i).YColor=[0.12 0.40 0.15]; mini_ax(i).GridColor=[0.07 0.22 0.09];
    mini_ax(i).GridAlpha=0.4; mini_ax(i).FontName='Courier New'; mini_ax(i).FontSize=7;
    grid(mini_ax(i),'on'); hold(mini_ax(i),'on');
    title(mini_ax(i),mini_ttl{i},'Color',[0.25 0.9 0.4], ...
        'FontSize',7,'FontWeight','bold','FontName','Courier New');
    xlabel(mini_ax(i),'Time (s)','Color',C_G2,'FontSize',6,'FontName','Courier New');
end

% ── GROUND PLANE ─────────────────────────────────────────────────
[gx,gy] = meshgrid(linspace(0,5200,14),linspace(0,5200,14));
surf(ax,gx,gy,zeros(size(gx)), ...
    'FaceColor',[0.02 0.07 0.02],'EdgeColor',[0.05 0.16 0.06], ...
    'FaceAlpha',0.75,'EdgeAlpha',0.20);

% Grid range rings
for r=[1000 2000 3000 4000]
    th=linspace(0,2*pi,100);
    plot3(ax,2500+r*cos(th),2500+r*sin(th),zeros(1,100),'--', ...
        'Color',[0.05 0.18 0.07],'LineWidth',0.5);
end

% Altitude rings
for alt_r=[200 400 600 800 1000]
    th=linspace(0,2*pi,80);
    plot3(ax,2500+2700*cos(th),2500+2700*sin(th),repmat(alt_r,1,80),'--', ...
        'Color',[0.06 0.20 0.08],'LineWidth',0.35);
    text(ax,5300,2500,alt_r,sprintf('%dm',alt_r), ...
        'Color',[0.10 0.38 0.14],'FontSize',7,'FontName','Courier New');
end

% ── INFRASTRUCTURE: buildings on ground ──────────────────────────
bld = [500 4200 0 80 80 120; 700 4000 0 60 60 90; ...
       4200 600 0 70 70 100; 4400 400 0 50 50 80; ...
       4000 4500 0 90 90 140; 3800 4700 0 55 55 75];
for b=1:size(bld,1)
    draw_box3(ax,bld(b,1),bld(b,2),bld(b,3),bld(b,4),bld(b,5),bld(b,6), ...
        [0.04 0.12 0.05],[0.08 0.25 0.10]);
end

% ── DETECTION POD ────────────────────────────────────────────────
plot3(ax,[pod_pos(1) pod_pos(1)],[pod_pos(2) pod_pos(2)],[0 95], ...
    '-','Color',C_G1,'LineWidth',3);
draw_box3(ax,pod_pos(1),pod_pos(2),95,70,45,35, ...
    [0.04 0.16 0.06],[0.28 0.95 0.45]);
th2=linspace(0,2*pi,32);
fill3(ax,pod_pos(1)+18*cos(th2),pod_pos(2)+10+10*sin(th2), ...
    repmat(112,1,32),[0.03 0.10 0.04], ...
    'EdgeColor',C_G1,'LineWidth',0.9);
fill3(ax,pod_pos(1)+9*cos(th2),pod_pos(2)+10+10*sin(th2), ...
    repmat(112,1,32),[0.07 0.22 0.09], ...
    'EdgeColor',[0.22 0.75 0.35],'LineWidth',0.5);
for mi=-1:1
    plot3(ax,pod_pos(1)-12,pod_pos(2)+mi*12,106,'o', ...
        'MarkerSize',5,'MarkerFaceColor',C_G1,'MarkerEdgeColor',C_G1);
end
text(ax,pod_pos(1)-320,pod_pos(2),160,'SAHAM DETECTION POD', ...
    'Color',C_G1,'FontWeight','bold','FontSize',8,'FontName','Courier New');
text(ax,pod_pos(1)-320,pod_pos(2),128,'LWIR + MIC ARRAY + Nd:YAG LASER', ...
    'Color',C_G2,'FontSize',7,'FontName','Courier New');
text(ax,pod_pos(1)+25,pod_pos(2),95,'GPS: 24.6880N / 46.7210E', ...
    'Color',C_G2,'FontSize',6.5,'FontName','Courier New');

% ── LAUNCHER ─────────────────────────────────────────────────────
draw_box3(ax,launch_pos(1),launch_pos(2),0,90,60,25, ...
    [0.03 0.12 0.05],[0.28 0.95 0.45]);
plot3(ax,[launch_pos(1)-25 launch_pos(1)+45], ...
        [launch_pos(2)     launch_pos(2)],[22 90], ...
    '-','Color',C_G1,'LineWidth',6);
text(ax,launch_pos(1)-450,launch_pos(2),120,'TUBE LAUNCHER', ...
    'Color',C_G1,'FontWeight','bold','FontSize',8,'FontName','Courier New');
text(ax,launch_pos(1)-450,launch_pos(2),88,'3x SAHAM INTERCEPTORS', ...
    'Color',C_G2,'FontSize',7,'FontName','Courier New');

% ── RADAR SWEEP (rotating line) ──────────────────────────────────
radar_r = 4500;
h_radar = plot3(ax,[pod_pos(1) pod_pos(1)+radar_r], ...
                   [pod_pos(2) pod_pos(2)],[100 100], ...
    '-','Color',[0.15 0.65 0.22],'LineWidth',1.2,'LineStyle','--');

% ── STATIC THREAT PATHS ──────────────────────────────────────────
threat_colors = {C_RED,[0.85 0.25 0.10],[0.70 0.15 0.20]};
for k=1:N_THREATS
    plot3(ax,[threat_starts(k,1) threat_ends(k,1)], ...
            [threat_starts(k,2) threat_ends(k,2)], ...
            [threat_starts(k,3) threat_ends(k,3)], ...
        '--','Color',threat_colors{k},'LineWidth',0.7);
    % Intercept markers
    plot3(ax,hit_pts(k,1),hit_pts(k,2),hit_pts(k,3),'x', ...
        'MarkerSize',20,'LineWidth',2.5,'Color',C_G1);
    th3=linspace(0,2*pi,50);
    plot3(ax,hit_pts(k,1)+80*cos(th3),hit_pts(k,2)+80*sin(th3), ...
        repmat(hit_pts(k,3),1,50),'-','Color',C_G1,'LineWidth',0.8);
end
plot3(ax,[launch_pos(1) hit_pts(1,1)],[launch_pos(2) hit_pts(1,2)], ...
        [launch_pos(3) hit_pts(1,3)],':','Color',[0.14 0.55 0.22],'LineWidth',0.8);

text(ax,hit_pts(1,1)+140,hit_pts(1,2),hit_pts(1,3)+70,'PIP-1', ...
    'Color',C_G1,'FontWeight','bold','FontSize',8,'FontName','Courier New');
text(ax,hit_pts(2,1)+140,hit_pts(2,2),hit_pts(2,3)+70,'PIP-2', ...
    'Color',C_G1,'FontWeight','bold','FontSize',8,'FontName','Courier New');
text(ax,hit_pts(3,1)+140,hit_pts(3,2),hit_pts(3,3)+70,'PIP-3', ...
    'Color',C_G1,'FontWeight','bold','FontSize',8,'FontName','Courier New');

% ── BUILD MESHES ─────────────────────────────────────────────────
[sh_f,sh_v]   = build_shahed_mesh();
[sahm_f,sahm_v] = build_saham_mesh();

% Threat patches
h_threats = gobjects(N_THREATS,1);
for k=1:N_THREATS
    h_threats(k) = patch(ax,'Faces',sh_f,'Vertices',sh_v+threat_starts(k,:), ...
        'FaceColor',threat_colors{k},'EdgeColor','none', ...
        'FaceLighting','gouraud','FaceAlpha',0.92);
end

% Sahm patches (one per threat)
h_sahms = gobjects(N_THREATS,1);
sahm_colors = {[0.15 0.80 0.40],[0.20 0.90 0.55],[0.10 0.70 0.35]};
for k=1:N_THREATS
    h_sahms(k) = patch(ax,'Faces',sahm_f,'Vertices',sahm_v+launch_pos, ...
        'FaceColor',sahm_colors{k},'EdgeColor','none', ...
        'FaceLighting','gouraud','FaceAlpha',0.92,'Visible','off');
end

light(ax,'Position',[2 2 5],'Style','infinite','Color',[0.85 1.0 0.88]);
light(ax,'Position',[-1 0 2],'Style','infinite','Color',[0.2 0.3 0.22]);
material(ax,'dull');

% ── SENSOR BEAMS (per threat) ─────────────────────────────────────
h_lwirs  = gobjects(N_THREATS,1);
h_lasers = gobjects(N_THREATS,1);
beam_cols = {[1.0 0.55 0.10],[0.90 0.45 0.08],[0.80 0.35 0.06]};
for k=1:N_THREATS
    h_lwirs(k)  = plot3(ax,[0 0],[0 0],[0 0],'--', ...
        'Color',beam_cols{k},'LineWidth',1.6,'Visible','off');
    h_lasers(k) = plot3(ax,[0 0],[0 0],[0 0],'-', ...
        'Color',C_YEL,'LineWidth',1.8,'Visible','off');
end
h_dlink = plot3(ax,[pod_pos(1) launch_pos(1)],[pod_pos(2) launch_pos(2)],[95 22], ...
    ':','Color',C_G1,'LineWidth',1.2,'Visible','off');

% ── ACOUSTIC CONE (visible at Phase 3) ───────────────────────────
h_cone = fill3(ax,[0 0 0],[0 0 0],[0 0 0],[0.15 0.60 0.20], ...
    'FaceAlpha',0.08,'EdgeColor',[0.15 0.60 0.20],'EdgeAlpha',0.25,'Visible','off');

% ── TRAILS ───────────────────────────────────────────────────────
h_tt = gobjects(N_THREATS,1);
h_st = gobjects(N_THREATS,1);
for k=1:N_THREATS
    h_tt(k) = plot3(ax,nan,nan,nan,'-','Color',threat_colors{k},'LineWidth',1.4);
    h_st(k) = plot3(ax,nan,nan,nan,'-','Color',sahm_colors{k},'LineWidth',1.4);
end

h_stat3d = text(ax,2700,100,1080,'INITIALIZING SYSTEMS...', ...
    'HorizontalAlignment','center','FontName','Courier New', ...
    'FontSize',10,'FontWeight','bold','Color',C_G1);

% ── TELEMETRY FIELDS ─────────────────────────────────────────────
tl_keys = { ...
  'SCENARIO','THREAT TYPE','THREAT COUNT','SEP0', ...
  'PRIMARY SPEED','PRIMARY ALT','GPS LATITUDE','GPS LONGITUDE', ...
  'BEARING TO POD','RANGE TO POD','CLOSING SPEED','SEP1', ...
  'SENSOR','LASER TYPE','LASER WL','LASER RANGE','LASER CEP','SEP2', ...
  'SAHM SPEED','INTERCEPTS DONE','THREATS ACTIVE','T-NEXT HIT','CONFIDENCE'};

tl_init = { ...
  'SWARM ALPHA-3','SHAHED-136 UAV','3 DRONES','', ...
  '- km/h','- m AGL','-','-', ...
  '- deg','- m','- m/s','', ...
  'HYBRID FUSION','Nd:YAG pulsed','1064 nm (IR)','- m','+/-8 cm','', ...
  '500 km/h','0 / 3','3','- s','- %'};

n_tl  = length(tl_keys);
tl_y  = linspace(0.945, 0.022, n_tl);
tl_dh = gobjects(n_tl,1);

sep_idx = find(strcmp(tl_keys,'SEP0') | strcmp(tl_keys,'SEP1') | strcmp(tl_keys,'SEP2'));

for i=1:n_tl
    k=tl_keys{i}; v=tl_init{i};
    if any(i==sep_idx)
        line([0.03 0.97],[tl_y(i) tl_y(i)],'Parent',ax_tl,'Color',C_G3,'LineWidth',0.6);
        tl_dh(i)=text(0.5,tl_y(i),'','Parent',ax_tl,'FontSize',1,'Color',[0 0 0],'FontName','Courier New');
        continue;
    end
    rectangle('Parent',ax_tl,'Position',[0.03 tl_y(i)-0.018 0.94 0.036], ...
        'EdgeColor',[0.06 0.22 0.08],'FaceColor',[0.012 0.035 0.012], ...
        'LineWidth',0.4,'Curvature',0.05);
    text(0.05,tl_y(i)+0.007,k,'Parent',ax_tl,'FontName','Courier New', ...
        'FontSize',5.8,'Color',[0.13 0.48 0.17],'FontWeight','bold');
    col = C_G1;
    if contains(k,'LASER') || contains(k,'WL') || contains(k,'CEP'), col=C_YEL; end
    if contains(k,'INTERCEPT') || contains(k,'CONFIDENCE'), col=C_CYN; end
    if contains(k,'THREAT') || contains(k,'SPEED') && contains(k,'PRIMARY'), col=[1.0 0.55 0.55]; end
    tl_dh(i) = text(0.97,tl_y(i)-0.005,v,'Parent',ax_tl, ...
        'HorizontalAlignment','right','FontName','Courier New', ...
        'FontSize',8.0,'FontWeight','bold','Color',col);
end

h_timer = text(0.5,0.005,'T +  0:00.0','Parent',ax_tl, ...
    'HorizontalAlignment','center','FontName','Courier New', ...
    'FontSize',8,'Color',C_G2);

% ── PHASE BAR ────────────────────────────────────────────────────
phases  = {'POWER ON','LWIR DETECT','ACOUSTIC','LASER RANGE', ...
           'SWARM LOCK','DATA TX','LAUNCH x3','INTERCEPT'};
ph_x    = linspace(0.04,0.96,length(phases));
ph_dots = gobjects(length(phases),1);
ph_lbls = gobjects(length(phases),1);
for i=1:length(phases)
    if i<length(phases)
        line([ph_x(i)+0.04 ph_x(i+1)-0.04],[0.42 0.42],'Parent',ax_ph, ...
            'Color',C_G3,'LineWidth',1);
    end
    ph_dots(i)=plot(ph_x(i),0.42,'o','Parent',ax_ph,'MarkerSize',12, ...
        'MarkerFaceColor',[0.02 0.08 0.03],'MarkerEdgeColor',[0.09 0.32 0.13],'LineWidth',1.5);
    ph_lbls(i)=text(ph_x(i),0.10,phases{i},'Parent',ax_ph, ...
        'HorizontalAlignment','center','FontName','Courier New', ...
        'FontSize',6.8,'Color',[0.11 0.40 0.15],'FontWeight','bold');
end
h_ph_stat = text(0.5,0.82,'SYSTEM STANDBY','Parent',ax_ph, ...
    'HorizontalAlignment','center','FontName','Courier New', ...
    'FontSize',9,'FontWeight','bold','Color',C_G1);

% ── MINI PLOT LINES ──────────────────────────────────────────────
h_spd_plt = plot(ax_spd, nan,nan,'-','Color',mini_col{1},'LineWidth',1.4);
h_rng_plt = plot(ax_rng, nan,nan,'-','Color',mini_col{2},'LineWidth',1.4);
h_cnf_plt = plot(ax_conf,nan,nan,'-','Color',mini_col{3},'LineWidth',1.4);
h_alt_plt = plot(ax_alt, nan,nan,'-','Color',mini_col{4},'LineWidth',1.4);
yline(ax_spd, 185,'--','Color',[0.25 0.9 0.42],'LineWidth',0.8);
yline(ax_conf,90, '--','Color',C_YEL,'LineWidth',0.8);
ax_spd.YLim=[0 600]; ax_rng.YLim=[0 9];
ax_conf.YLim=[0 105]; ax_alt.YLim=[0 1000];
set([ax_spd ax_rng ax_conf ax_alt],'XLim',[0 T_END]);

% ── SIMULATION STATE ─────────────────────────────────────────────
dt          = 0.06;
t_arr=[]; spd_arr=[]; rng_arr=[]; cnf_arr=[]; alt_arr=[];
tx_h = cell(N_THREATS,1); ty_h=tx_h; tz_h=tx_h;
sx_h = cell(N_THREATS,1); sy_h=sx_h; sz_h=sx_h;
sahm_pos    = repmat(launch_pos,N_THREATS,1);
sahm_vis    = false(N_THREATS,1);
intercept_ok= false(N_THREATS,1);
blast_drawn = false(N_THREATS,1);
conf_val    = 0;
phase_done  = false(1,8);
seed_n      = 0;
radar_ang   = 0;
intercepts_done = 0;

fprintf('  Animation running — swarm of %d drones...\n\n',N_THREATS);

for t = 0:dt:T_END

    if ~ishandle(fig), break; end

    seed_n  = seed_n+1;
    noise_s = sin(seed_n*7.3)*0.4+sin(seed_n*13.1)*0.3;

    % Radar sweep
    radar_ang = mod(radar_ang+4,360);
    rx = pod_pos(1)+radar_r*cosd(radar_ang);
    ry = pod_pos(2)+radar_r*sind(radar_ang);
    set(h_radar,'XData',[pod_pos(1) rx],'YData',[pod_pos(2) ry]);

    % Threat positions
    tpos = zeros(N_THREATS,3);
    for k=1:N_THREATS
        if intercept_ok(k), tpos(k,:)=hit_pts(k,:); continue; end
        frac_t=min(t/t_flights(k),1.0);
        frac_t=0.5*(1-cos(pi*frac_t));
        % Add evasive jink to threats 2 and 3
        jink=0;
        if k>1 && t>T_LOCK
            jink=80*sin(t*1.8+(k-1)*2.1);
        end
        base=threat_starts(k,:)+frac_t*(threat_ends(k,:)-threat_starts(k,:));
        tpos(k,:)=[base(1)+jink base(2) base(3)];
    end

    % Primary threat telemetry
    range_m  = norm(tpos(1,:)-pod_pos);
    spd_kmh  = 185+noise_s*3.5;
    bearing  = mod(atan2d(tpos(1,2)-pod_pos(2),tpos(1,1)-pod_pos(1)),360);
    clos_ms  = (spd_kmh/3.6)*cosd(abs(bearing-225));
    lat      = 24.600+tpos(1,2)*0.000009;
    lon      = 46.600+tpos(1,1)*0.0000114;

    % Confidence
    if     t<T_LWIR,  conf_val=0;
    elseif t<T_MIC,   conf_val=(t-T_LWIR)/(T_MIC-T_LWIR)*42;
    elseif t<T_LASER, conf_val=42+(t-T_MIC)/(T_LASER-T_MIC)*26;
    elseif t<T_LOCK,  conf_val=68+(t-T_LASER)/(T_LOCK-T_LASER)*22;
    else,             conf_val=min(97,90+(t-T_LOCK)*0.35);
    end

    % ── PHASES ───────────────────────────────────────────────────
    if t>=T_SCAN && ~phase_done(1)
        phase_done(1)=true;
        activate_phase(ph_dots,ph_lbls,1,h_ph_stat, ...
            'PHASE 1 - SYSTEM ACTIVE: SCANNING AIRSPACE FOR THREATS',h_stat3d,'SCANNING AIRSPACE');
        fprintf('  [T+%4.1fs] System active\n',t);
    end

    if t>=T_LWIR && ~phase_done(2)
        phase_done(2)=true;
        for k=1:N_THREATS, set(h_lwirs(k),'Visible','on'); end
        activate_phase(ph_dots,ph_lbls,2,h_ph_stat, ...
            'PHASE 2 - LWIR: THERMAL SIGNATURES DETECTED  3 TARGETS',h_stat3d,'LWIR: 3 TARGETS DETECTED');
        set(tl_dh(2),'String','SHAHED-136 UAV');
        fprintf('  [T+%4.1fs] LWIR: 3 thermal contacts  Bearing:%.1fdeg\n',t,bearing);
    end

    if t>=T_MIC && ~phase_done(3)
        phase_done(3)=true;
        set(h_cone,'Visible','on');
        activate_phase(ph_dots,ph_lbls,3,h_ph_stat, ...
            'PHASE 3 - ACOUSTIC: PROPELLER SIGNATURE 3x CONFIRMED  58-63 dB',h_stat3d,'ACOUSTIC: SWARM CONFIRMED');
        fprintf('  [T+%4.1fs] Acoustic confirmed  58-63 dB  3 units\n',t);
    end

    if t>=T_LASER && ~phase_done(4)
        phase_done(4)=true;
        for k=1:N_THREATS, set(h_lasers(k),'Visible','on'); end
        activate_phase(ph_dots,ph_lbls,4,h_ph_stat, ...
            'PHASE 4 - Nd:YAG LASER 1064nm RANGING ALL 3 TARGETS  +/-8cm CEP',h_stat3d,'LASER RANGING x3');
        set(tl_dh(14),'String','Nd:YAG pulsed');
        fprintf('  [T+%4.1fs] Laser ranging  1064nm  +/-8cm  3 targets\n',t);
    end

    if t>=T_LOCK && ~phase_done(5)
        phase_done(5)=true;
        activate_phase(ph_dots,ph_lbls,5,h_ph_stat, ...
            'PHASE 5 - SWARM LOCKED  3 PIP SOLUTIONS  CONFIDENCE 97%',h_stat3d,'SWARM LOCKED - 97%');
        fprintf('  [T+%4.1fs] SWARM LOCKED  3 intercept solutions computed\n',t);
    end

    if t>=T_DATATX && ~phase_done(6)
        phase_done(6)=true;
        set(h_dlink,'Visible','on');
        activate_phase(ph_dots,ph_lbls,6,h_ph_stat, ...
            sprintf('PHASE 6 - COORD TX: 3x PIP  PRIMARY %.4fN/%.4fE  ALT:%.0fm', ...
            lat,lon,hit_pts(1,3)),h_stat3d,'TX: 3x PIP TO LAUNCHER');
        fprintf('  [T+%4.1fs] Coord TX  3 intercept solutions sent\n',t);
    end

    if t>=T_LAUNCH && ~phase_done(7)
        phase_done(7)=true;
        for k=1:N_THREATS
            set(h_sahms(k),'Visible','on');
            sahm_vis(k)=true;
        end
        activate_phase(ph_dots,ph_lbls,7,h_ph_stat, ...
            'PHASE 7 - 3x SAHAM INTERCEPTORS LAUNCHED  500 km/h  INBOUND',h_stat3d,'3x SAHAM INBOUND');
        fprintf('  [T+%4.1fs] 3x SAHAM LAUNCHED  500km/h\n',t);
    end

    % SAHAM flight per threat
    for k=1:N_THREATS
        if sahm_vis(k) && ~intercept_ok(k)
            v2h=hit_pts(k,:)-sahm_pos(k,:);
            d2h=norm(v2h);
            if d2h>20
                sahm_pos(k,:)=sahm_pos(k,:)+(v2h/d2h)*sahm_spd*dt;
            else
                intercept_ok(k)=true;
                intercepts_done=intercepts_done+1;
            end
            set(tl_dh(22),'String',sprintf('%.1f s',max(0,d2h/sahm_spd)));
        end

        % Blast effect on first frame of intercept
        if intercept_ok(k) && ~blast_drawn(k)
            blast_drawn(k)=true;
            draw_blast(ax,hit_pts(k,:));
            set(h_threats(k),'Visible','off');
            set(h_sahms(k),'Visible','off');
            fprintf('  [T+%4.1fs] *** THREAT %d NEUTRALIZED ***\n',t,k);
        end
    end

    % Phase 8 when all intercepted
    if all(intercept_ok) && ~phase_done(8)
        phase_done(8)=true;
        activate_phase(ph_dots,ph_lbls,8,h_ph_stat, ...
            '*** PHASE 8 - ALL 3 INTERCEPTS SUCCESSFUL - SWARM NEUTRALIZED ***', ...
            h_stat3d,'SWARM NEUTRALIZED');
        set(h_stat3d,'Color',[0.5 1.0 0.6]);
        set(h_lwirs(1),'Visible','off'); set(h_lasers(1),'Visible','off');
        fprintf('\n  *** MISSION COMPLETE: ALL THREATS NEUTRALIZED ***\n\n');
    end

    % Update beams to primary threat
    if t>=T_LWIR && ~intercept_ok(1)
        set(h_lwirs(1),'XData',[pod_pos(1) tpos(1,1)], ...
                       'YData',[pod_pos(2) tpos(1,2)], ...
                       'ZData',[97         tpos(1,3)]);
    end
    if t>=T_LASER && ~intercept_ok(1)
        set(h_lasers(1),'XData',[pod_pos(1) tpos(1,1)], ...
                        'YData',[pod_pos(2) tpos(1,2)], ...
                        'ZData',[99         tpos(1,3)]);
    end

    % Acoustic cone toward primary
    if t>=T_MIC && ~intercept_ok(1)
        dir=tpos(1,:)-pod_pos; dir=dir/norm(dir);
        perp=[-dir(2) dir(1) 0]; perp=perp/max(norm(perp),0.01);
        cone_r=400; cone_l=800;
        tip=[pod_pos(1) pod_pos(2) 97];
        base_c=tip+dir*cone_l;
        p1=base_c+perp*cone_r; p2=base_c-perp*cone_r;
        set(h_cone,'XData',[tip(1) p1(1) p2(1)]', ...
                   'YData',[tip(2) p1(2) p2(2)]', ...
                   'ZData',[tip(3) p1(3) p2(3)]');
    end

    % Move threat meshes
    for k=1:N_THREATS
        if ~intercept_ok(k)
            set(h_threats(k),'Vertices',sh_v+tpos(k,:));
        end
        if sahm_vis(k) && ~intercept_ok(k)
            set(h_sahms(k),'Vertices',sahm_v+sahm_pos(k,:));
        end
    end

    % Trails
    for k=1:N_THREATS
        if ~intercept_ok(k)
            tx_h{k}(end+1)=tpos(k,1); ty_h{k}(end+1)=tpos(k,2); tz_h{k}(end+1)=tpos(k,3);
            if length(tx_h{k})>120
                tx_h{k}=tx_h{k}(end-119:end);
                ty_h{k}=ty_h{k}(end-119:end);
                tz_h{k}=tz_h{k}(end-119:end);
            end
            set(h_tt(k),'XData',tx_h{k},'YData',ty_h{k},'ZData',tz_h{k});
        end
        if sahm_vis(k) && ~intercept_ok(k)
            sx_h{k}(end+1)=sahm_pos(k,1);
            sy_h{k}(end+1)=sahm_pos(k,2);
            sz_h{k}(end+1)=sahm_pos(k,3);
            set(h_st(k),'XData',sx_h{k},'YData',sy_h{k},'ZData',sz_h{k});
        end
    end

    % Telemetry update
    if t>=T_LWIR
        set(tl_dh(5), 'String',sprintf('%.1f km/h',  spd_kmh));
        set(tl_dh(6), 'String',sprintf('%.0f m AGL', tpos(1,3)));
        set(tl_dh(7), 'String',sprintf('%.6f N',     lat));
        set(tl_dh(8), 'String',sprintf('%.6f E',     lon));
        set(tl_dh(9), 'String',sprintf('%.1f deg',   bearing));
        set(tl_dh(10),'String',sprintf('%.0f m',     range_m));
        set(tl_dh(11),'String',sprintf('%.1f m/s',   clos_ms));
        set(tl_dh(23),'String',sprintf('%.0f %%',    conf_val));
    end
    if t>=T_LASER
        set(tl_dh(16),'String',sprintf('%.0f m',range_m));
    end
    set(tl_dh(20),'String',sprintf('%d / %d',intercepts_done,N_THREATS));
    threats_active=N_THREATS-intercepts_done;
    set(tl_dh(21),'String',sprintf('%d',threats_active));

    mm=floor(t/60); ss=mod(t,60);
    set(h_timer,'String',sprintf('T +  %d:%04.1f',mm,ss));

    % Mini plots
    t_arr(end+1)   = t;
    spd_arr(end+1) = spd_kmh;
    rng_arr(end+1) = range_m/1000;
    cnf_arr(end+1) = conf_val;
    alt_arr(end+1) = tpos(1,3);
    set(h_spd_plt,'XData',t_arr,'YData',spd_arr);
    set(h_rng_plt,'XData',t_arr,'YData',rng_arr);
    set(h_cnf_plt,'XData',t_arr,'YData',cnf_arr);
    set(h_alt_plt,'XData',t_arr,'YData',alt_arr);

    drawnow;
    pause(0.012);

end

fprintf('=================================================\n');
fprintf('  MISSION COMPLETE  ALL THREATS NEUTRALIZED     \n');
fprintf('=================================================\n\n');

end % function SAHAM_Enhanced


% =========================================================================
%  HELPER — activate phase dot
% =========================================================================
function activate_phase(ph_dots,ph_lbls,idx,h_ph_stat,msg,h_stat3d,short)
set(ph_dots(idx),'MarkerFaceColor',[0.14 0.60 0.22],'MarkerEdgeColor',[0.35 1.0 0.55]);
set(ph_lbls(idx),'Color',[0.35 1.0 0.55]);
set(h_ph_stat,'String',msg);
set(h_stat3d,'String',short);
end


% =========================================================================
%  HELPER — blast explosion effect
% =========================================================================
function draw_blast(ax,pos)
th=linspace(0,2*pi,60);
radii=[60 120 180 240];
alphas=[0.35 0.22 0.14 0.07];
cols={[1.0 0.85 0.20],[1.0 0.60 0.10],[1.0 0.35 0.10],[0.80 0.20 0.05]};
for i=1:length(radii)
    r=radii(i);
    fill3(ax,pos(1)+r*cos(th),pos(2)+r*sin(th),repmat(pos(3),1,60), ...
        cols{i},'FaceAlpha',alphas(i),'EdgeColor','none');
end
plot3(ax,pos(1),pos(2),pos(3),'p', ...
    'MarkerSize',32,'MarkerFaceColor',[1 0.88 0.2], ...
    'MarkerEdgeColor',[1 0.5 0.1],'LineWidth',2.5);
% Shockwave ring
for ri=[80 160 250]
    plot3(ax,pos(1)+ri*cos(th),pos(2)+ri*sin(th),repmat(pos(3),1,60),'-', ...
        'Color',[1.0 0.75 0.20],'LineWidth',0.8);
end
end


% =========================================================================
%  HELPER — 3D box
% =========================================================================
function draw_box3(ax,cx,cy,cz,lx,ly,lz,fc,ec)
dx=lx/2; dy=ly/2;
X=[cx-dx cx+dx cx+dx cx-dx cx-dx cx+dx cx+dx cx-dx];
Y=[cy-dy cy-dy cy+dy cy+dy cy-dy cy-dy cy+dy cy+dy];
Z=[cz    cz    cz    cz    cz+lz cz+lz cz+lz cz+lz];
F=[1 2 3 4;5 6 7 8;1 2 6 5;2 3 7 6;3 4 8 7;4 1 5 8];
patch(ax,'Vertices',[X;Y;Z]','Faces',F, ...
    'FaceColor',fc,'EdgeColor',ec,'LineWidth',0.7,'FaceAlpha',0.85);
end


% =========================================================================
%  HELPER — Shahed-136 mesh
% =========================================================================
function [faces,verts] = build_shahed_mesh()
V=[]; F=[];
n=size(V,1); dx=110; dy=14; dz=11; cx=0; cy=0; cz=0;
v=[cx-dx cy-dy cz-dz;cx+dx cy-dy cz-dz;cx+dx cy+dy cz-dz;cx-dx cy+dy cz-dz; ...
   cx-dx cy-dy cz+dz;cx+dx cy-dy cz+dz;cx+dx cy+dy cz+dz;cx-dx cy+dy cz+dz];
f=[1 2 3;1 3 4;5 8 7;5 7 6;1 5 6;1 6 2;2 6 7;2 7 3;3 7 8;3 8 4;4 8 5;4 5 1]+n;
V=[V;v]; F=[F;f];
n=size(V,1); dx=15; dy=11; dz=9; cx=115; cy=0; cz=0;
v=[cx-dx cy-dy cz-dz;cx+dx cy-dy cz-dz;cx+dx cy+dy cz-dz;cx-dx cy+dy cz-dz; ...
   cx-dx cy-dy cz+dz;cx+dx cy-dy cz+dz;cx+dx cy+dy cz+dz;cx-dx cy+dy cz+dz];
f=[1 2 3;1 3 4;5 8 7;5 7 6;1 5 6;1 6 2;2 6 7;2 7 3;3 7 8;3 8 4;4 8 5;4 5 1]+n;
V=[V;v]; F=[F;f];
n=size(V,1);
V=[V;-30 0 6;-30 0 -6;-120 -200 0;70 0 6;70 0 -6];
F=[F;n+1 n+3 n+2;n+1 n+4 n+3;n+2 n+3 n+5;n+3 n+4 n+5];
n=size(V,1);
V=[V;-30 0 6;-30 0 -6;-120 200 0;70 0 6;70 0 -6];
F=[F;n+1 n+2 n+3;n+1 n+3 n+4;n+2 n+5 n+3;n+3 n+5 n+4];
n=size(V,1); V=[V;-90 0 0;-90 0 12;-120 -50 35]; F=[F;n+1 n+2 n+3];
n=size(V,1); V=[V;-90 0 0;-90 0 12;-120 50 35];  F=[F;n+1 n+3 n+2];
n=size(V,1); dx=16; dy=10; dz=8; cx=-95; cy=0; cz=6;
v=[cx-dx cy-dy cz-dz;cx+dx cy-dy cz-dz;cx+dx cy+dy cz-dz;cx-dx cy+dy cz-dz; ...
   cx-dx cy-dy cz+dz;cx+dx cy-dy cz+dz;cx+dx cy+dy cz+dz;cx-dx cy+dy cz+dz];
f=[1 2 3;1 3 4;5 8 7;5 7 6;1 5 6;1 6 2;2 6 7;2 7 3;3 7 8;3 8 4;4 8 5;4 5 1]+n;
V=[V;v]; F=[F;f];
verts=V; faces=F;
end


% =========================================================================
%  HELPER — SAHAM interceptor mesh
% =========================================================================
function [faces,verts] = build_saham_mesh()
V=[]; F=[];
n=size(V,1); dx=90; dy=8; dz=7; cx=0; cy=0; cz=0;
v=[cx-dx cy-dy cz-dz;cx+dx cy-dy cz-dz;cx+dx cy+dy cz-dz;cx-dx cy+dy cz-dz; ...
   cx-dx cy-dy cz+dz;cx+dx cy-dy cz+dz;cx+dx cy+dy cz+dz;cx-dx cy+dy cz+dz];
f=[1 2 3;1 3 4;5 8 7;5 7 6;1 5 6;1 6 2;2 6 7;2 7 3;3 7 8;3 8 4;4 8 5;4 5 1]+n;
V=[V;v]; F=[F;f];
n=size(V,1); dx=18; dy=7; dz=6; cx=100; cy=0; cz=0;
v=[cx-dx cy-dy cz-dz;cx+dx cy-dy cz-dz;cx+dx cy+dy cz-dz;cx-dx cy+dy cz-dz; ...
   cx-dx cy-dy cz+dz;cx+dx cy-dy cz+dz;cx+dx cy+dy cz+dz;cx-dx cy+dy cz+dz];
f=[1 2 3;1 3 4;5 8 7;5 7 6;1 5 6;1 6 2;2 6 7;2 7 3;3 7 8;3 8 4;4 8 5;4 5 1]+n;
V=[V;v]; F=[F;f];
n=size(V,1);
V=[V;20 0 5;20 0 -5;-40 -120 0;60 0 5;60 0 -5];
F=[F;n+1 n+3 n+2;n+1 n+4 n+3;n+2 n+3 n+5;n+3 n+4 n+5];
n=size(V,1);
V=[V;20 0 5;20 0 -5;-40 120 0;60 0 5;60 0 -5];
F=[F;n+1 n+2 n+3;n+1 n+3 n+4;n+2 n+5 n+3;n+3 n+5 n+4];
n=size(V,1);
V=[V;-80 0 0;-80 0 14;-95 -25 28;-95 25 28];
F=[F;n+1 n+2 n+3;n+1 n+4 n+2];
n=size(V,1); dx=14; dy=7; dz=6; cx=-75; cy=0; cz=8;
v=[cx-dx cy-dy cz-dz;cx+dx cy-dy cz-dz;cx+dx cy+dy cz-dz;cx-dx cy+dy cz-dz; ...
   cx-dx cy-dy cz+dz;cx+dx cy-dy cz+dz;cx+dx cy+dy cz+dz;cx-dx cy+dy cz+dz];
f=[1 2 3;1 3 4;5 8 7;5 7 6;1 5 6;1 6 2;2 6 7;2 7 3;3 7 8;3 8 4;4 8 5;4 5 1]+n;
V=[V;v]; F=[F;f];
verts=V; faces=F;
end
