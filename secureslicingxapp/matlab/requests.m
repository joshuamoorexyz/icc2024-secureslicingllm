
clc
clear all
close all
% The data you provided for the first dataset
bitrate_data1 = [

];



% The data for the second dataset (assuming you have another set of values)
bitrate_data2 = [

];

bitrate_data3 = [
    
];

% Time intervals from 0.00 to 8.00 seconds for the first dataset
time_intervals = 0:1:175;


% % Plot the first dataset
% plot(time_intervals, bitrate_data1, 'o-', 'DisplayName', 'bitrate_data1');
% hold on; % Hold the current plot
% 
% % Plot the second dataset
% plot(time_intervals, bitrate_data2, 'x-', 'DisplayName', 'bitrate_data2');
% 
% xlabel('Time (seconds)');
% ylabel('Bitrate (Mbits/sec)');
% title('Bitrate vs. Time');
% grid on;
% legend;


% Define the window size for moving average smoothing
window_size = 4; % Adjust this value as per your preference
window_size1 = 20;
% Smooth the first dataset using moving average
smoothened_data1 = movmean(bitrate_data1, window_size);

% Smooth the second dataset using moving average
smoothened_data2 = movmean(bitrate_data2, window_size1);



smoothened_data3 = movmean(bitrate_data3, window_size1);

% % Plot the first dataset
% plot(time_intervals, bitrate_data1, 'o-', 'DisplayName', 'bitrate_data1');
% hold on; % Hold the current plot

% Plot the second dataset
% plot(time_intervals, bitrate_data2, 'x-', 'DisplayName', 'bitrate_data2');

% Plot the smoothed data for the first dataset
plot(time_intervals, smoothened_data1, 'r-', 'DisplayName', 'UE1');
hold on;
% Plot the smoothed data for the second dataset
plot(time_intervals, smoothened_data2, 'g-', 'DisplayName', 'UE2');

plot(time_intervals, smoothened_data3, 'g-', 'DisplayName', 'UE3');


xlabel('Time (seconds)');
ylabel('Requests');
title('Requests vs. Time');
grid on;
legend;



