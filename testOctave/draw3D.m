total = dlmread("time.txt");

%card	prec	time	epoch_time	epoch_avg	avg_pwr	avg_gtemp	avg_pclk
COL_CARD = 1; COL_PREC = 2; COL_TIME = 3; COL_EPOC = 4;
COL_EAVG = 5; COL_GPWR = 6; COL_GTMP = 7; COL_PCLK = 8;

% time column does not used(-1), so we use this item to adjust
% same color with same machine.

% determine whether we use per-GPU data or summed data.
pwr_scale = ones(size(total, 1),1);%total(:,COL_CARD); % 

z_time = total(:, COL_EPOC);
y_pwr  = total(:, COL_GPWR).*pwr_scale; % we calculate total power
x_clk  = total(:, COL_PCLK);
point_color = [];

note_str = [];
note_z_tme = [];
note_y_pwr = [];
note_x_clk = [];


for i = 1:size(total, 1)
  % color : 0 to 1. we have (1,16) (2,16) (1,32) (2,32)
  % mapped to [0.4 1.4 0.8 1.8]
  % Print the label only for default values.
  if abs(total(i, COL_PCLK) + 5*total(i, COL_PREC) - 1130) <= 40 && total(i, COL_TIME) == -1
    note_str = [note_str; sprintf("%d GPU + FP%d", total(i, COL_CARD), total(i, COL_PREC))];
    note_x_clk = [note_x_clk; total(i, COL_PCLK)];
    note_y_pwr = [note_y_pwr; total(i, COL_GPWR).*pwr_scale(i)];
    note_z_tme = [note_z_tme; total(i, COL_EPOC)];
  elseif total(i, COL_TIME) != -1
    note_str = [note_str; sprintf("maxq - %d GPU + FP%d", total(i, COL_CARD), total(i, COL_PREC))];
    note_x_clk = [note_x_clk; total(i, COL_PCLK)];
    note_y_pwr = [note_y_pwr; total(i, COL_GPWR).*pwr_scale(i)];
    note_z_tme = [note_z_tme; total(i, COL_EPOC)];
  endif
  

  if total(i, COL_TIME) == -1 % color is generated
    color_val = total(i,COL_CARD)+total(i,COL_PREC)/40+total(i, COL_TIME);
  else % color is fixed for others.
    color_val = 1.1;
  endif
  point_color = [point_color; color_val];
endfor

% Normalize color in 0-1 range
point_color = (point_color - min(point_color))/(max(point_color) - min(point_color));

freq_list = [900 960 1020:90:1480 1530];
time_list = [120:30:450];
powr_list = [100:50:max(y_pwr(:))];

figHandle = figure();
subplot(2,2,1);
  scatter3(x_clk, y_pwr, z_time, 10, point_color, "filled");
  text(note_x_clk-30, note_y_pwr-20, note_z_tme, note_str,'fontsize',1);
  xlabel("Processor clock");
  set(gca, 'xtick', freq_list);
  set(gca, 'ztick', time_list);
  ylabel("Power consumed");
  zlabel("5 Epochs time");

  
offset_x = 0;
offset_y = 1;
subplot(2,2,2);
  scatter(y_pwr, z_time, 10, point_color, "filled");
  text(note_y_pwr+offset_x, note_z_tme+offset_y, note_str,'fontsize',12);
  set(gca, 'ytick', time_list);
  xlabel("Power per GPU");
  ylabel("5 Epochs time");
  grid on;
  drawFittingLine(y_pwr, z_time, point_color);
  
subplot(2,2,3);
  scatter(x_clk, y_pwr, 10, point_color, "filled");
  text(note_x_clk+offset_x, note_y_pwr+offset_y, note_str,'fontsize',12);
  xlabel("Processor clock");
  set(gca, 'xtick', freq_list);
  ylabel("Power per GPU");
  set(gca, 'ytick', powr_list);
  grid on;
  drawFittingLine(x_clk, y_pwr, point_color);
  
subplot(2,2,4);
  scatter(x_clk, z_time, 10, point_color, "filled");
  text(note_x_clk+offset_x, note_z_tme+offset_y, note_str,'fontsize',12);
  xlabel("Processor clock");
  set(gca, 'xtick', freq_list);
  set(gca, 'ytick', time_list);
  ylabel("5 Epochs time");
  grid on;
  drawFittingLine(x_clk, z_time, point_color);
  
% https://stackoverflow.com/questions/39746234/octave-failing-to-include-parts-of-a-plot-when-saved-to-file
% gl2ps 1.3.8 contains this bug, it cannot display all parts.
print(figHandle, "figure.png", "-dpng", "-S1920,1080");