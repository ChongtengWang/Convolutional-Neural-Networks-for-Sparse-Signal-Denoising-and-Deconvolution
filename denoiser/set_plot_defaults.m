function set_plot_defaults(opt)
% function set_plot_defaults(opt)
%
% Modify MATLAB plot settings
%
% my_plotting_defaults('on')
% my_plotting_defaults('off')

switch opt

    case 'on'

        set(0, 'DefaultAxesTitleFontWeight', 'normal')

        set(0,'DefaultAxesFontSizeMode','manual') % this is needed so that DefaultAxesFontSize works...
        set(0, 'DefaultAxesFontSize', 8);   % does not work when font size is small without preceding line

        set(0, 'DefaultTextFontSize', 10);  % for text function


        % Does not work in Matlab ver 15
        %         set(0, 'DefaultAxesFontSize', 6);
        %         set(0, 'DefaultTextFontSize', 6);

        line_width = 1;
        %         line_width = 0.5;
        set(0, 'DefaultLineLineWidth', line_width)


        %     set(0, 'DefaultFigureColor', 'White')

        % figure color
        %     set(0, 'DefaultLineMarkerSize', 10);

        % tick direction (default 'in' , 'out')
        %     set(0, 'DefaultAxesTickDir', 'out')

%         set(0, 'DefaultAxesColorOrder', [0 0 0])  % Force all plots to be black.


    otherwise

        set(0, 'DefaultAxesFontSize', 'remove');
        set(0, 'DefaultTextFontSize', 'remove');
        set(0, 'DefaultLineLineWidth', 'remove');

        set(0, 'DefaultAxesTickDir', 'remove');

        colors = [
            0         0    1.0000
            1.0000         0         0
            0    1.0000         0
            0         0    0.1724
            1.0000    0.1034    0.7241
            ];

%         set(0, 'DefaultAxesColorOrder', colors)
        set(0, 'DefaultAxesColorOrder', 'remove')

        %         set(0, 'DefaultLineMarkerSize', 'remove');
        % set(0,'DefaultFigureColor', 'remove')
        set(0, 'DefaultAxesTickDir', 'remove')

end

