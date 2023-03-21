%% Estimation of localized and distributed electrical activity from a labor EHG
% In this example, we use a sample from the Icelandic EHG database. This
% database can be found at: https://physionet.org/content/ehgdb/1.0.0/

clear all;close all;clc
rng('default')
addpath(genpath('tensor_toolbox'));
addpath(genpath('utils'));
addpath(genpath('example'));

% This sample has been preprocessed and arranged into 4x4xT tensors as
% described in the manuscript.

tensor_original_data = load('tensor_ice001_l_1of1m.mat');

% We reorder Y so that the temporal dimension is the third dimension

Y = tensor_original_data.EHG_tensor_ordered;
fs=10;
Y_reorder = permute(Y,[2 3 1]);

output = tensor_decomposition(Y_reorder, 'max_iterations', 40,'tolerance',1e-2, 'initial_sparse_variance', 0.5, 'verbose',1);

X = output.X;
S = output.S;

Y_reorder = Y_reorder/131*1000; % Convert to uV based on the information provided in the file header.
Y = Y/131*1000;
S = S/131*1000;
X = X/131*1000;

Y_reshaped = reshape(Y_reorder,16,[]);
X_reshaped = reshape(X,16,[]);
S_reshaped = reshape(S,16,[]);

%% plot the results for the upper left electrodes

max_value(1) = max(abs(Y_reorder(:)));
max_value(2) = max(abs(S(:)));
max_value(3) = max(abs(X(:)));

ts = 0:1/fs:length(squeeze(Y_reorder(1,1,:)))/fs-1/fs;
figure(1)
figure(2)
figure(3)

axis_font_size = 15;
title_size = 18;
linewidth = 1;
count = 1;
for j =1:2
    for i = 1:2
        
        figure(1)
        subplot(2,2,count)
        plot(ts,squeeze(Y_reorder(i,j,:)),'LineWidth',linewidth);
        ylim([-max_value(2) max_value(2) ])
        if count == 1 || count ==3
            ylabel('Amplitude [uV]','FontSize',axis_font_size);
        end
        if count == 3 || count == 4
            xlabel('Time [s]','FontSize',axis_font_size);
        end
        str = ['Electrode ', num2str(count)];
        title(str,'FontSize',axis_font_size)
        sgtitle('Original measurements','FontSize',title_size)
        
        figure(2)
        subplot(2,2,count)
        plot(ts,squeeze(S(i,j,:)),'LineWidth',linewidth);
        ylim([-max_value(2) max_value(2) ])
        if count == 1 || count ==3
            ylabel('Amplitude [uV]','FontSize',axis_font_size);
        end
        if count == 3 || count == 4
            xlabel('Time [s]','FontSize',axis_font_size);
        end
        title(str,'FontSize',axis_font_size)
        sgtitle('Localized activity','FontSize',title_size)

                
        figure(3)
        subplot(2,2,count)
        plot(ts,squeeze(X(i,j,:)),'LineWidth',linewidth);
        ylim([-max_value(2) max_value(2) ])
        if count == 1 || count ==3
            ylabel('Amplitude [uV]','FontSize',axis_font_size);
        end
        if count == 3 || count == 4
            xlabel('Time [s]','FontSize',axis_font_size);
        end
        title(str,'FontSize',axis_font_size)
        sgtitle('Distributed activity','FontSize',title_size)
        
        count = count+1;
    end
end










