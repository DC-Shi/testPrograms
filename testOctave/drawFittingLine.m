## Copyright (C) 2019 Daochen Shi

## usage: coeffs = drawFittingLine (X, Y, color, order=2)
##
## Draw fitting lines of given X and Y for each given
## color values. Return values is coefficients for
## each fitted lines.
##
## Example:
##
##  X = rand(10,1); Y = rand(10,1);
##  color = [1;1;1;1;1; 0;0;0;0;0];
##  drawFittingLine (X, Y, point_color);

## Author: daochens
## Keywords: draw fitting line
## Maintainer: daochens

function coeffs = drawFittingLine (X, Y, color, order=2)
  coeffs = [];
  % Draw fitting line for each color.
  % array on for loop require row vector
  % for loop will iterate on each column
  for i = unique(color,"rows")'
  %printf("i=%f\n", i);
    line_x = X(color == i);
    line_y = Y(color == i);
    coeff = polyfit(line_x, line_y, order);
    coeffs = [coeffs; coeff];
    % Get fitted values
    fittedX = linspace(min(line_x), max(line_x), 200);
    fittedY = polyval(coeff, fittedX);
    % Plot the fitted line
    hold on;
    plot(fittedX, fittedY, '-', 'LineWidth', 2, 'Color', hsv2rgb([i, 1, 1]));
    hold off;
  endfor
endfunction
