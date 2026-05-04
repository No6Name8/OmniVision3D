% =========================================================================
%  SAHAM — Counter-Drone Defense System
%  ENHANCED 3D SIMULATION — Full Mission Sequence
%  Version 3.1 — Competition Grade — MATLAB Online Compatible
% =========================================================================
%  UPLOAD TO MATLAB ONLINE (matlab.mathworks.com):
%    1. SAHAM_Enhanced.m   (this file)
%    2. Defensthon-Full-Drone.stl  (optional — built-in mesh used if missing)
%  Then type:  SAHAM_Enhanced   and press Enter
% =========================================================================

function SAHAM_Enhanced()

clc; clear; close all;

fprintf('\n');
fprintf('=========================================\n');
fprintf('  SAHAM  COUNTER-DRONE DEFENSE SYSTEM   \n');
fprintf('  Enhanced 3D Mission Simulation v3.1   \n');
fprintf('=========================================\n\n');

% ── LOAD DRONE MODEL ─────────────────────────────────────────────
fprintf('  Loading SAHAM drone model...\n');
script_dir = fileparts(mfilename('fullpath'));
stl_file = fullfile(script_dir, 'Defensthon-Full-Drone.stl');
if isfile(stl_file)
    sahm_mesh = stlread(stl_file);
    raw_pts   = sahm_mesh.Points;
    cx = mean(raw_pts(:,1)); cy = mean(raw_pts(:,2)); cz = mean(raw_pts(:,3));
    raw_pts(:,1) = raw_pts(:,1) - cx;
    raw_pts(:,2) = raw_pts(:,2) - cy;
    raw_pts(:,3) = raw_pts(:,3) - cz;
    span_x    = max(raw_pts(:,1)) - min(raw_pts(:,1));
    sc        = 40 / span_x;
    sahm_base = raw_pts * sc;
    sahm_faces = sahm_mesh.ConnectivityList;
    fprintf('  Model: %d triangles  |  Scale: %.4f  |  Span: ~40m\n\n', ...
        size(sahm_faces,1), sc);
else
    fprintf('  STL not found — using built-in interceptor mesh.\n\n');
    [sahm_faces, sahm_base] = build_saham_mesh();
end

% ── FIGURE ───────────────────────────────────────────────────────
fig = figure('Name','SAHAM — Enhanced 3D Mission Simulation', ...
    'Color',[0.03 0.06 0.03],'NumberTitle','off', ...
    'Units','normalized','OuterPosition',[0.01 0.02 0.98 0.95]);

% ── TITLE BAR ────────────────────────────────────────────────────
ax_title = axes('Parent',fig,'Position',[0 0.93 1 0.07], ...
    'Color',[0.01 0.04 0.01],'XTick',[],'YTick',[], ...
    'XColor',[0.08 0.28 0.10],'YColor',[0.08 0.28 0.10],'Box','on');
text(0.5,0.62,'SAHAM   |   COUNTER-DRONE DEFENSE SYSTEM', ...
    'Parent',ax_title,'HorizontalAlignment','center', ...
    'FontName','Courier New','FontSize',14,'FontWeight','bold', ...
    'Color',[0.3 1.0 0.5]);
text(0.5,0.14,'ENHANCED 3D MISSION SIMULATION  |  AUTONOMOUS INTERCEPT  |  CONCEPT DEMO', ...
    'Parent',ax_title,'HorizontalAlignment','center', ...
    'FontName','Courier New','FontSize',7,'Color',[0.15 0.55 0.2]);

% ── 3D MAIN AXES ─────────────────────────────────────────────────
ax = axes('Parent',fig,'Position',[0.01 0.22 0.68 0.70]);
hold(ax,'on'); grid(ax,'on'); view(ax,-38,24);
ax.Color=[0.02 0.05 0.02];
ax.XColor=[0.15 0.5 0.2]; ax.YColor=[0.15 0.5 0.2]; ax.ZColor=[0.15 0.5 0.2];
ax.GridColor=[0.08 0.25 0.10]; ax.GridAlpha=0.35;
ax.FontName='Courier New'; ax.FontSize=8;
ax.XLim=[-200 5200]; ax.YLim=[-200 5200]; ax.ZLim=[0 1100];
xlabel(ax,'East  X (m)','Color',[0.2 0.7 0.3],'FontSize',9,'FontName','Courier New');
ylabel(ax,'North Y (m)','Color',[0.2 0.7 0.3],'FontSize',9,'FontName','Courier New');
zlabel(ax,'Altitude Z (m)','Color',[0.2 0.7 0.3],'FontSize',9,'FontName','Courier New');
title(ax,'SAHAM — LIVE 3D ENGAGEMENT VIEW', ...
    'Color',[0.35 1.0 0.55],'FontSize',11,'FontWeight','bold','FontName','Courier New');

% ── TELEMETRY PANEL ──────────────────────────────────────────────
ax_tl = axes('Parent',fig,'Position',[0.70 0.22 0.29 0.70]);
ax_tl.Color=[0.01 0.04 0.01]; ax_tl.XTick=[]; ax_tl.YTick=[];
ax_tl.XLim=[0 1]; ax_tl.YLim=[0 1]; ax_tl.Box='on';
ax_tl.XColor=[0.06 0.22 0.08]; ax_tl.YColor=[0.06 0.22 0.08];
hold(ax_tl,'on');
text(0.5,0.975,'MISSION TELEMETRY','Parent',ax_tl, ...
    'HorizontalAlignment','center','FontName','Courier New', ...
    'FontSize',10,'FontWeight','bold','Color',[0.3 1.0 0.5]);
line([0.04 0.96],[0.955 0.955],'Parent',ax_tl, ...
    'Color',[0.1 0.35 0.12],'LineWidth',0.8);

% ── PHASE BAR ────────────────────────────────────────────────────
ax_ph = axes('Parent',fig,'Position',[0.01 0.14 0.98 0.07]);
ax_ph.Color=[0.01 0.04 0.01]; ax_ph.XTick=[]; ax_ph.YTick=[];
ax_ph.XLim=[0 1]; ax_ph.YLim=[0 1]; ax_ph.Box='on';
ax_ph.XColor=[0.06 0.22 0.08]; ax_ph.YColor=[0.06 0.22 0.08];
hold(ax_ph,'on');

% ── MINI PLOTS ───────────────────────────────────────────────────
ax_spd  = axes('Parent',fig,'Position',[0.01  0.02 0.23 0.11]);
ax_rng  = axes('Parent',fig,'Position',[0.26  0.02 0.23 0.11]);
ax_conf = axes('Parent',fig,'Position',[0.51  0.02 0.23 0.11]);
ax_alt  = axes('Parent',fig,'Position',[0.76  0.02 0.23 0.11]);
mini_ax = [ax_spd ax_rng ax_conf ax_alt];
mini_ttl = {'THREAT SPEED (km/h)','RANGE TO POD (km)', ...
            'SENSOR CONFIDENCE (%)','THREAT ALTITUDE (m)'};
for i=1:4
    mini_ax(i).Color=[0.01 0.04 0.01];
    mini_ax(i).XColor=[0.12 0.40 0.15];
    mini_ax(i).YColor=[0.12 0.40 0.15];
    mini_ax(i).GridColor=[0.07 0.22 0.09];
    mini_ax(i).GridAlpha=0.4;
    mini_ax(i).FontName='Courier New';
    mini_ax(i).FontSize=7;
    grid(mini_ax(i),'on'); hold(mini_ax(i),'on');
    title(mini_ax(i),mini_ttl{i},'Color',[0.25 0.9 0.4], ...
        'FontSize',7,'FontWeight','bold','FontName','Courier New');
    xlabel(mini_ax(i),'Time (s)','Color',[0.15 0.55 0.2], ...
        'FontSize',6,'FontName','Courier New');
end

% ── SCENE: GROUND ────────────────────────────────────────────────
[gx,gy] = meshgrid(linspace(0,5000,10),linspace(0,5000,10));
surf(ax,gx,gy,zeros(size(gx)), ...
    'FaceColor',[0.03 0.08 0.03],'EdgeColor',[0.06 0.18 0.07], ...
    'FaceAlpha',0.7,'EdgeAlpha',0.25);

for alt_r=[200 400 600 800 1000]
    th=linspace(0,2*pi,80);
    plot3(ax,2500+2500*cos(th),2500+2500*sin(th), ...
        repmat(alt_r,1,80),'--','Color',[0.07 0.22 0.09],'LineWidth',0.4);
    text(ax,5200,2500,alt_r,sprintf('%dm',alt_r), ...
        'Color',[0.12 0.40 0.15],'FontSize',7,'FontName','Courier New');
end

% ── DETECTION POD ────────────────────────────────────────────────
pod_pos    = [2500 2500 0];
launch_pos = [800  800  0];

plot3(ax,[pod_pos(1) pod_pos(1)],[pod_pos(2) pod_pos(2)],[0 90], ...
    '-','Color',[0.25 0.9 0.4],'LineWidth',3);
draw_box3(ax,pod_pos(1),pod_pos(2),90,60,40,30, ...
    [0.05 0.18 0.07],[0.3 1.0 0.5]);
th2=linspace(0,2*pi,30);
fill3(ax,pod_pos(1)+15*cos(th2),pod_pos(2)+8+8*sin(th2), ...
    repmat(100,1,30),[0.04 0.12 0.05], ...
    'EdgeColor',[0.3 1.0 0.5],'LineWidth',0.8);
fill3(ax,pod_pos(1)+8*cos(th2),pod_pos(2)+8+8*sin(th2), ...
    repmat(100,1,30),[0.08 0.25 0.10], ...
    'EdgeColor',[0.25 0.8 0.4],'LineWidth',0.5);
for mi=-1:1
    plot3(ax,pod_pos(1)-10,pod_pos(2)+mi*10,95,'o', ...
        'MarkerSize',4,'MarkerFaceColor',[0.25 0.9 0.4], ...
        'MarkerEdgeColor',[0.3 1.0 0.5]);
end
text(ax,pod_pos(1)-300,pod_pos(2),145,'SAHAM DETECTION POD', ...
    'Color',[0.3 1.0 0.5],'FontWeight','bold','FontSize',8,'FontName','Courier New');
text(ax,pod_pos(1)-300,pod_pos(2),115,'LWIR + MIC ARRAY + LASER', ...
    'Color',[0.18 0.65 0.25],'FontSize',7,'FontName','Courier New');
text(ax,pod_pos(1)+20,pod_pos(2),90,'GPS: 24.6880N / 46.7210E', ...
    'Color',[0.15 0.55 0.2],'FontSize',6.5,'FontName','Courier New');

% ── LAUNCHER ─────────────────────────────────────────────────────
draw_box3(ax,launch_pos(1),launch_pos(2),0,80,50,20, ...
    [0.04 0.15 0.06],[0.3 1.0 0.5]);
plot3(ax,[launch_pos(1)-20 launch_pos(1)+40], ...
        [launch_pos(2)     launch_pos(2)], ...
        [20 80],'-','Color',[0.3 1.0 0.5],'LineWidth',5);
text(ax,launch_pos(1)-400,launch_pos(2),110,'TUBE LAUNCHER', ...
    'Color',[0.3 1.0 0.5],'FontWeight','bold','FontSize',8,'FontName','Courier New');
text(ax,launch_pos(1)-400,launch_pos(2),80,'FIXED-WING INTERCEPTOR', ...
    'Color',[0.18 0.65 0.25],'FontSize',7,'FontName','Courier New');

% ── FLIGHT MATH ──────────────────────────────────────────────────
threat_start = [4900 4900 820];
threat_end   = [100  100  280];
threat_spd   = 51.4;
sahm_spd     = 138.9;
tvec_dir     = threat_end - threat_start;
t_flight     = norm(tvec_dir) / threat_spd;

T_SCAN   = 0;
T_LWIR   = 5;
T_MIC    = 10;
T_LASER  = 15;
T_LOCK   = 22;
T_DATATX = 27;
T_LAUNCH = 32;
T_END    = 75;

frac_h  = min((T_LAUNCH + norm(tvec_dir)*0.55/sahm_spd)/t_flight, 0.86);
frac_h  = 0.5*(1-cos(pi*frac_h));
hit_pt  = threat_start + frac_h * tvec_dir;

plot3(ax,[threat_start(1) threat_end(1)], ...
        [threat_start(2) threat_end(2)], ...
        [threat_start(3) threat_end(3)], ...
    '--','Color',[0.45 0.10 0.10],'LineWidth',0.8);
plot3(ax,[launch_pos(1) hit_pt(1)], ...
        [launch_pos(2) hit_pt(2)], ...
        [launch_pos(3) hit_pt(3)], ...
    ':','Color',[0.12 0.45 0.18],'LineWidth',0.8);

plot3(ax,hit_pt(1),hit_pt(2),hit_pt(3),'x', ...
    'MarkerSize',24,'LineWidth',3,'Color',[0.35 1.0 0.55]);
th3=linspace(0,2*pi,50);
plot3(ax,hit_pt(1)+100*cos(th3),hit_pt(2)+100*sin(th3), ...
    repmat(hit_pt(3),1,50),'-','Color',[0.3 1.0 0.5],'LineWidth',1);
text(ax,hit_pt(1)+130,hit_pt(2),hit_pt(3)+60,'INTERCEPT POINT', ...
    'Color',[0.35 1.0 0.55],'FontWeight','bold','FontSize',8,'FontName','Courier New');

% ── BUILD MESHES ─────────────────────────────────────────────────
[sh_f,sh_v] = build_shahed_mesh();

h_threat = patch(ax,'Faces',sh_f,'Vertices',sh_v+threat_start, ...
    'FaceColor',[0.80 0.18 0.18],'EdgeColor','none', ...
    'FaceLighting','gouraud','FaceAlpha',0.92);
h_sahm = patch(ax,'Faces',sahm_faces, ...
    'Vertices',sahm_base+launch_pos, ...
    'FaceColor',[0.15 0.80 0.40],'EdgeColor','none', ...
    'FaceLighting','gouraud','FaceAlpha',0.92,'Visible','off');

light(ax,'Position',[2 2 5],'Style','infinite','Color',[0.85 1.0 0.88]);
light(ax,'Position',[-1 0 2],'Style','infinite','Color',[0.2 0.3 0.22]);
material(ax,'dull');

% ── SENSOR BEAMS ─────────────────────────────────────────────────
h_lwir  = plot3(ax,[0 0],[0 0],[0 0],'--', ...
    'Color',[1.0 0.60 0.10],'LineWidth',1.8,'Visible','off');
h_laser = plot3(ax,[0 0],[0 0],[0 0],'-', ...
    'Color',[1.0 0.95 0.20],'LineWidth',2.0,'Visible','off');
h_dlink = plot3(ax,[pod_pos(1) launch_pos(1)], ...
                   [pod_pos(2) launch_pos(2)],[90 20], ...
    ':','Color',[0.3 1.0 0.5],'LineWidth',1.2,'Visible','off');

% ── TRAILS ───────────────────────────────────────────────────────
h_tt = plot3(ax,nan,nan,nan,'-','Color',[0.7 0.18 0.18],'LineWidth',1.5);
h_st = plot3(ax,nan,nan,nan,'-','Color',[0.18 0.80 0.40],'LineWidth',1.5);

h_stat3d = text(ax,2500,100,1070,'INITIALIZING SYSTEMS...', ...
    'HorizontalAlignment','center','FontName','Courier New', ...
    'FontSize',10,'FontWeight','bold','Color',[0.35 1.0 0.55]);

% ── TELEMETRY FIELDS ─────────────────────────────────────────────
tl_keys = { ...
  'THREAT TYPE','THREAT SPEED','POSITION X (East)', ...
  'POSITION Y (North)','POSITION Z (Alt)','GPS LATITUDE', ...
  'GPS LONGITUDE','BEARING TO POD','RANGE TO POD', ...
  'CLOSING SPEED','SEP1', ...
  'LASER TYPE','LASER WAVELENGTH','LASER RANGE', ...
  'LASER ACCURACY','SEP2', ...
  'SAHAM SPEED','INTERCEPT X','INTERCEPT Y', ...
  'INTERCEPT Z','T-INTERCEPT','CONFIDENCE'};

tl_vals = { ...
  'SHAHED-136 UAV','- km/h','- m', ...
  '- m','- m AGL','-', ...
  '-','-deg','- m', ...
  '- m/s','', ...
  'Nd:YAG pulsed','1064 nm (IR)','- m', ...
  '+/-8 cm CEP','', ...
  '500 km/h', ...
  sprintf('%.0f m',hit_pt(1)), ...
  sprintf('%.0f m',hit_pt(2)), ...
  sprintf('%.0f m',hit_pt(3)), ...
  '- s','- %'};

n_tl  = length(tl_keys);
tl_y  = linspace(0.935, 0.025, n_tl);
tl_dh = gobjects(n_tl,1);

for i=1:n_tl
    k = tl_keys{i};
    v = tl_vals{i};
    if strcmp(k,'SEP1') || strcmp(k,'SEP2')
        line([0.03 0.97],[tl_y(i) tl_y(i)],'Parent',ax_tl, ...
            'Color',[0.08 0.28 0.10],'LineWidth',0.6);
        tl_dh(i) = text(0.5,tl_y(i),'','Parent',ax_tl, ...
            'FontSize',1,'Color',[0 0 0],'FontName','Courier New');
        continue;
    end
    rectangle('Parent',ax_tl, ...
        'Position',[0.03 tl_y(i)-0.017 0.94 0.034], ...
        'EdgeColor',[0.06 0.22 0.08],'FaceColor',[0.015 0.045 0.015], ...
        'LineWidth',0.4,'Curvature',0.05);
    text(0.06,tl_y(i)+0.006,k,'Parent',ax_tl, ...
        'FontName','Courier New','FontSize',6.2,'Color',[0.14 0.50 0.18], ...
        'FontWeight','bold');
    col = [0.22 0.90 0.42];
    if contains(k,'LASER') || contains(k,'WAVELENGTH'), col = [1.0 0.88 0.22]; end
    if contains(k,'INTERCEPT') || contains(k,'CONFIDENCE'), col = [0.28 1.0 0.52]; end
    tl_dh(i) = text(0.97,tl_y(i)-0.004,v,'Parent',ax_tl, ...
        'HorizontalAlignment','right','FontName','Courier New', ...
        'FontSize',8.5,'FontWeight','bold','Color',col);
end

h_timer = text(0.5,0.004,'T +  0:00.0','Parent',ax_tl, ...
    'HorizontalAlignment','center','FontName','Courier New', ...
    'FontSize',8,'Color',[0.18 0.65 0.25]);

% ── PHASE BAR ────────────────────────────────────────────────────
phases  = {'POWER ON','LWIR DETECT','ACOUSTIC', ...
           'LASER RANGE','TARGET LOCK','DATA TX','LAUNCH','INTERCEPT'};
ph_x    = linspace(0.05,0.95,length(phases));
ph_dots = gobjects(length(phases),1);
ph_lbls = gobjects(length(phases),1);

for i=1:length(phases)
    if i < length(phases)
        line([ph_x(i)+0.03 ph_x(i+1)-0.03],[0.40 0.40], ...
            'Parent',ax_ph,'Color',[0.06 0.22 0.08],'LineWidth',1);
    end
    ph_dots(i) = plot(ph_x(i),0.40,'o','Parent',ax_ph, ...
        'MarkerSize',11,'MarkerFaceColor',[0.03 0.10 0.04], ...
        'MarkerEdgeColor',[0.10 0.35 0.14],'LineWidth',1.5);
    ph_lbls(i) = text(ph_x(i),0.10,phases{i},'Parent',ax_ph, ...
        'HorizontalAlignment','center','FontName','Courier New', ...
        'FontSize',7,'Color',[0.12 0.42 0.16],'FontWeight','bold');
end

h_ph_stat = text(0.5,0.80,'SYSTEM STANDBY', ...
    'Parent',ax_ph,'HorizontalAlignment','center', ...
    'FontName','Courier New','FontSize',9,'FontWeight','bold', ...
    'Color',[0.3 1.0 0.5]);

% ── MINI PLOT LINES ──────────────────────────────────────────────
h_spd_plt = plot(ax_spd, nan,nan,'-','Color',[1.0 0.35 0.35],'LineWidth',1.4);
h_rng_plt = plot(ax_rng, nan,nan,'-','Color',[0.25 0.90 0.42],'LineWidth',1.4);
h_cnf_plt = plot(ax_conf,nan,nan,'-','Color',[0.25 0.90 0.42],'LineWidth',1.4);
h_alt_plt = plot(ax_alt, nan,nan,'-','Color',[1.0 0.60 0.10],'LineWidth',1.4);

yline(ax_spd, 185,'--','Color',[0.25 0.9 0.42],'LineWidth',0.8);
text(ax_spd,1,188,'185 km/h nominal','FontSize',5.5,'Color',[0.25 0.9 0.42],'FontName','Courier New');
yline(ax_conf,90,'--','Color',[1.0 0.88 0.22],'LineWidth',0.8);
text(ax_conf,1,92,'lock threshold','FontSize',5.5,'Color',[1.0 0.88 0.22],'FontName','Courier New');

ax_spd.YLim=[0 540];  ax_rng.YLim=[0 8];
ax_conf.YLim=[0 105]; ax_alt.YLim=[0 1000];
set([ax_spd ax_rng ax_conf ax_alt],'XLim',[0 T_END]);

% ── SIMULATION LOOP ──────────────────────────────────────────────
dt       = 0.07;
t_arr=[]; spd_arr=[]; rng_arr=[]; cnf_arr=[]; alt_arr=[];
tx_h=[]; ty_h=[]; tz_h=[];
sx_h=[]; sy_h=[]; sz_h=[];
sahm_pos     = launch_pos;
sahm_vis     = false;
intercept_ok = false;
conf_val     = 0;
phase_done   = false(1,8);
seed_n       = 0;

fprintf('  Animation running...\n\n');

for t = 0:dt:T_END

    if ~ishandle(fig), break; end

    seed_n  = seed_n + 1;
    noise_s = sin(seed_n*7.3)*0.4 + sin(seed_n*13.1)*0.3;

    frac_t = min(t/t_flight,1.0);
    frac_t = 0.5*(1-cos(pi*frac_t));
    tpos   = threat_start + frac_t * tvec_dir;

    spd_kmh  = 185 + noise_s*3.5;
    range_m  = norm(tpos - pod_pos);
    bearing  = mod(atan2d(tpos(2)-pod_pos(2),tpos(1)-pod_pos(1)),360);
    clos_ms  = (spd_kmh/3.6) * cosd(abs(bearing-225));
    lat      = 24.600 + tpos(2)*0.000009;
    lon      = 46.600 + tpos(1)*0.0000114;

    if     t < T_LWIR,  conf_val = 0;
    elseif t < T_MIC,   conf_val = (t-T_LWIR)/(T_MIC-T_LWIR)*42;
    elseif t < T_LASER, conf_val = 42+(t-T_MIC)/(T_LASER-T_MIC)*26;
    elseif t < T_LOCK,  conf_val = 68+(t-T_LASER)/(T_LOCK-T_LASER)*22;
    else,               conf_val = min(97,90+(t-T_LOCK)*0.3);
    end

    if t >= T_SCAN && ~phase_done(1)
        phase_done(1)=true;
        set(ph_dots(1),'MarkerFaceColor',[0.14 0.60 0.22],'MarkerEdgeColor',[0.35 1.0 0.55]);
        set(ph_lbls(1),'Color',[0.35 1.0 0.55]);
        set(h_ph_stat,'String','PHASE 1 - SYSTEM ACTIVE: SCANNING AIRSPACE');
        set(h_stat3d,'String','SCANNING AIRSPACE');
        fprintf('  [T+%4.1fs] System active\n',t);
    end

    if t >= T_LWIR && ~phase_done(2)
        phase_done(2)=true;
        set(ph_dots(2),'MarkerFaceColor',[0.14 0.60 0.22],'MarkerEdgeColor',[0.35 1.0 0.55]);
        set(ph_lbls(2),'Color',[0.35 1.0 0.55]);
        set(h_lwir,'Visible','on');
        set(h_ph_stat,'String','PHASE 2 - LWIR CAMERA: THERMAL SIGNATURE DETECTED');
        set(h_stat3d,'String','LWIR: THERMAL DETECTED');
        set(tl_dh(1),'String','SHAHED-136 UAV');
        fprintf('  [T+%4.1fs] LWIR detection  Bearing:%.1fdeg  Range:%.0fm\n',t,bearing,range_m);
    end

    if t >= T_MIC && ~phase_done(3)
        phase_done(3)=true;
        set(ph_dots(3),'MarkerFaceColor',[0.14 0.60 0.22],'MarkerEdgeColor',[0.35 1.0 0.55]);
        set(ph_lbls(3),'Color',[0.35 1.0 0.55]);
        set(h_ph_stat,'String','PHASE 3 - MIC ARRAY: PROPELLER ACOUSTIC CONFIRMED  63 dB');
        set(h_stat3d,'String','ACOUSTIC LOCK CONFIRMED');
        fprintf('  [T+%4.1fs] Acoustic lock  63 dB\n',t);
    end

    if t >= T_LASER && ~phase_done(4)
        phase_done(4)=true;
        set(ph_dots(4),'MarkerFaceColor',[0.14 0.60 0.22],'MarkerEdgeColor',[0.35 1.0 0.55]);
        set(ph_lbls(4),'Color',[0.35 1.0 0.55]);
        set(h_laser,'Visible','on');
        set(h_ph_stat,'String','PHASE 4 - Nd:YAG LASER 1064nm RANGING ACTIVE  +/-8cm CEP');
        set(h_stat3d,'String','LASER RANGING 1064nm');
        fprintf('  [T+%4.1fs] Laser ranging  1064nm  +/-8cm CEP\n',t);
    end

    if t >= T_LOCK && ~phase_done(5)
        phase_done(5)=true;
        set(ph_dots(5),'MarkerFaceColor',[0.14 0.60 0.22],'MarkerEdgeColor',[0.35 1.0 0.55]);
        set(ph_lbls(5),'Color',[0.35 1.0 0.55]);
        set(h_ph_stat,'String','PHASE 5 - TARGET LOCKED  CONFIDENCE 97%  ALL SENSORS FUSED');
        set(h_stat3d,'String','TARGET LOCKED - 97%');
        fprintf('  [T+%4.1fs] TARGET LOCKED  Confidence:97%%\n',t);
    end

    if t >= T_DATATX && ~phase_done(6)
        phase_done(6)=true;
        set(ph_dots(6),'MarkerFaceColor',[0.14 0.60 0.22],'MarkerEdgeColor',[0.35 1.0 0.55]);
        set(ph_lbls(6),'Color',[0.35 1.0 0.55]);
        set(h_dlink,'Visible','on');
        set(h_ph_stat,'String',sprintf('PHASE 6 - COORD TX: %.4fN / %.4fE  ALT:%.0fm', ...
            24.600+hit_pt(2)*0.000009, 46.600+hit_pt(1)*0.0000114, hit_pt(3)));
        set(h_stat3d,'String','COORD TX TO LAUNCHER');
        fprintf('  [T+%4.1fs] Coord TX  %.4fN/%.4fE  Alt:%.0fm\n', ...
            t,24.600+hit_pt(2)*0.000009,46.600+hit_pt(1)*0.0000114,hit_pt(3));
    end

    if t >= T_LAUNCH && ~phase_done(7)
        phase_done(7)=true;
        set(ph_dots(7),'MarkerFaceColor',[0.14 0.60 0.22],'MarkerEdgeColor',[0.35 1.0 0.55]);
        set(ph_lbls(7),'Color',[0.35 1.0 0.55]);
        sahm_vis=true;
        set(h_sahm,'Visible','on');
        set(h_ph_stat,'String','PHASE 7 - SAHAM INTERCEPTOR LAUNCHED  500 km/h  INBOUND');
        set(h_stat3d,'String','SAHAM LAUNCHED - INBOUND');
        fprintf('  [T+%4.1fs] SAHAM LAUNCHED  500km/h\n',t);
    end

    if sahm_vis && ~intercept_ok
        v2h  = hit_pt - sahm_pos;
        d2h  = norm(v2h);
        if d2h > 25
            sahm_pos = sahm_pos + (v2h/d2h)*sahm_spd*dt;
        else
            intercept_ok=true;
        end
        set(tl_dh(21),'String',sprintf('%.1f s',max(0,d2h/sahm_spd)));
    end

    if intercept_ok && ~phase_done(8)
        phase_done(8)=true;
        set(ph_dots(8),'MarkerFaceColor',[0.35 1.0 0.55],'MarkerEdgeColor',[0.50 1.0 0.60]);
        set(ph_lbls(8),'Color',[0.35 1.0 0.55]);
        set(h_ph_stat,'String','*** PHASE 8 - INTERCEPT SUCCESSFUL - THREAT NEUTRALIZED ***');
        set(h_stat3d,'String','TARGET NEUTRALIZED','Color',[0.5 1.0 0.6]);
        plot3(ax,hit_pt(1),hit_pt(2),hit_pt(3),'p', ...
            'MarkerSize',35,'MarkerFaceColor',[1 0.9 0.25], ...
            'MarkerEdgeColor',[1 0.5 0.1],'LineWidth',2.5);
        set(h_sahm,'Visible','off');
        set(h_lwir,'Visible','off');
        set(h_laser,'Visible','off');
        set(tl_dh(21),'String','0.0 s');
        set(tl_dh(22),'String','97 %');
        fprintf('  [T+%4.1fs] *** TARGET NEUTRALIZED ***\n\n',t);
    end

    if t >= T_LWIR
        set(h_lwir,'XData',[pod_pos(1) tpos(1)],'YData',[pod_pos(2) tpos(2)],'ZData',[92 tpos(3)]);
    end
    if t >= T_LASER
        set(h_laser,'XData',[pod_pos(1) tpos(1)],'YData',[pod_pos(2) tpos(2)],'ZData',[94 tpos(3)]);
    end

    if ~intercept_ok
        set(h_threat,'Vertices',sh_v+tpos);
    end
    if sahm_vis && ~intercept_ok
        set(h_sahm,'Vertices',sahm_base+sahm_pos);
    end

    tx_h(end+1)=tpos(1); ty_h(end+1)=tpos(2); tz_h(end+1)=tpos(3);
    if length(tx_h)>150
        tx_h=tx_h(end-149:end); ty_h=ty_h(end-149:end); tz_h=tz_h(end-149:end);
    end
    set(h_tt,'XData',tx_h,'YData',ty_h,'ZData',tz_h);

    if sahm_vis && ~intercept_ok
        sx_h(end+1)=sahm_pos(1); sy_h(end+1)=sahm_pos(2); sz_h(end+1)=sahm_pos(3);
        set(h_st,'XData',sx_h,'YData',sy_h,'ZData',sz_h);
    end

    if t >= T_LWIR
        set(tl_dh(2), 'String',sprintf('%.1f km/h',  spd_kmh));
        set(tl_dh(3), 'String',sprintf('%.1f m',     tpos(1)));
        set(tl_dh(4), 'String',sprintf('%.1f m',     tpos(2)));
        set(tl_dh(5), 'String',sprintf('%.0f m AGL', tpos(3)));
        set(tl_dh(6), 'String',sprintf('%.6f N',     lat));
        set(tl_dh(7), 'String',sprintf('%.6f E',     lon));
        set(tl_dh(8), 'String',sprintf('%.1f deg',   bearing));
        set(tl_dh(9), 'String',sprintf('%.0f m',     range_m));
        set(tl_dh(10),'String',sprintf('%.1f m/s',   clos_ms));
        set(tl_dh(22),'String',sprintf('%.0f %%',    conf_val));
    end
    if t >= T_LASER
        set(tl_dh(14),'String',sprintf('%.0f m',range_m));
    end

    mm=floor(t/60); ss=mod(t,60);
    set(h_timer,'String',sprintf('T +  %d:%04.1f',mm,ss));

    t_arr(end+1)=t; spd_arr(end+1)=spd_kmh; rng_arr(end+1)=range_m/1000;
    cnf_arr(end+1)=conf_val; alt_arr(end+1)=tpos(3);
    set(h_spd_plt,'XData',t_arr,'YData',spd_arr);
    set(h_rng_plt,'XData',t_arr,'YData',rng_arr);
    set(h_cnf_plt,'XData',t_arr,'YData',cnf_arr);
    set(h_alt_plt,'XData',t_arr,'YData',alt_arr);

    drawnow;
    pause(0.015);

end

fprintf('=========================================\n');
fprintf('  MISSION COMPLETE  THREAT NEUTRALIZED  \n');
fprintf('=========================================\n\n');

end % function SAHAM_Enhanced


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
%  HELPER — SAHAM interceptor mesh (fallback when STL missing)
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
