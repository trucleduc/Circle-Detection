function [circles, L] = detectCircles(varargin)
% 
%     Input:
%         I: input image
%         angleCoverage: default is 180
%         Tmin: default is 0.6
%         distance_tolerance: default is 0.5% of image minimum dimension
%         normal_tolerance: default is 20
%     Output:
%         circles: array of size K x 3 where the first two columns are
%         circle's centers and the last column represents radii
%         L: labeling matrix with the same size to the input image
% 
    narginchk(1, 5);
    nargoutchk(0, 2);
    I = varargin{1};
    params = [180, 0.6, max([2, 0.005 * min([size(I, 1), size(I, 2)])]), 20];
    if nargin >= 2
        for i = 2 : nargin
            if size(varargin{i}, 1) > 0
                params(i - 1) = varargin{i};
            end
        end
    end
    angleCoverage = params(1);
    Tmin = params(2);
    distance_tolerance = params(3);
    normal_tolerance = params(4);
    if (size(I, 3) > 1)
        [~, LS] = LSD(rgb2gray(I));
        E = edge(rgb2gray(I), 'canny');
    else
        [~, LS] = LSD(I);
        E = edge(I, 'canny');
    end
    [y, x] = find(E);
    lines = zeros(size(LS, 2), 4);
    lineLabels = zeros(length(y), 1);
    for i = 1 : size(LS, 2)
        a = LS(4, i) - LS(2, i);
        b = LS(1, i) - LS(3, i);
        c = -a * LS(1, i) - b * LS(2, i);
        d = sqrt(a * a + b * b);
        a = a / d; b = b / d; c = c / d;
        idx = abs(a * x + b * y + c) <= 2 * distance_tolerance & y >= min(LS([2; 4], i)) & y <= max(LS([2; 4], i)) & x >= min(LS([1; 3], i)) & x <= max(LS([1; 3], i));
        lineLabels(idx) = i;
        lines(i, :) = [-b, a, mean(LS([1; 3], i)), mean(LS([2; 4], i))];
    end
    warning('off', 'all');
    [labels, circles] = circleDetection([x, y], lineLabels, lines, distance_tolerance, normal_tolerance, Tmin, angleCoverage);
    warning('on', 'all');
    L = zeros(size(I, 1), size(I, 2));
    L(sub2ind(size(L), y, x)) = labels;
    if (nargout == 0)
        sampleColors = uint8(round(255 * distinguishable_colors(size(circles, 1), [0 0 0; 1 1 1])));
        figure(); axis equal off; hold on;
        imshow(I);
        for i = 1 : size(circles, 1)
            viscircles(circles(i, 1 : 2), circles(i, 3), 'EdgeColor', 'g', 'DrawBackgroundCircle', false, 'LineWidth', 4);
        end
    end
end

function [labels, circles] = circleDetection(points, lineLabels, lines, distance_tolerance, normal_tolerance, Tmin, angleCoverage)
    labels = zeros(size(points, 1), 1);
    circles = zeros(0, 3);
    maxRadius = min(max(points) - min(points));
    blockSize = 1000;
    K = min([26, size(points, 1)]);
    neighbors = knnsearch(points, points, 'K', K + 1);
    normals = estimateNormals(points, neighbors);
    if (size(lines, 1) < 2)
        return;
    end
    list = nchoosek((1 : size(lines, 1)), 2);
    list = [list, zeros(size(list, 1), 3)];
    list = list(radtodeg(real(acos(abs(dot(lines(list(:, 1), 1 : 2), lines(list(:, 2), 1 : 2), 2))))) >= 15, :);
    if (size(list, 1) < 2)
        return;
    end
    applyToGivenRow = @(func, matrix) @(row) func(matrix(row, :));
    applyToRows = @(func, matrix) arrayfun(applyToGivenRow(func, matrix), 1 : size(matrix, 1), 'UniformOutput', false)';
    list(:, 3 : 4) = cell2mat(applyToRows(@(L) initialEstimateCircleCenter(L), [lines(list(:, 1), :), lines(list(:, 2), :)]));
    r1 = sqrt(sum((list(:, 3 : 4) - lines(list(:, 1), 3 : 4)) .^ 2, 2));
    r2 = sqrt(sum((list(:, 3 : 4) - lines(list(:, 2), 3 : 4)) .^ 2, 2));
    idx = abs(r1 - r2) <= 2 * distance_tolerance & max([r1, r2], [], 2) < maxRadius & min([r1, r2], [], 2) > 3 * distance_tolerance & list(:, 3) > min(points(:, 1)) & list(:, 3) < max(points(:, 1)) & list(:, 4) > min(points(:, 2)) & list(:, 4) < max(points(:, 2));
    list = list(idx, :);
    if (size(list, 1) < 2)
        return;
    end
    list(:, 5) = (r1(idx) + r2(idx)) / 2;
    clear idx r1 r2
    idx = true(size(list, 1), 1);
    for i = 1 : size(list, 1)
        inliers = lineLabels == list(i, 1);
        circle_normals = points(inliers, :) - repmat(list(i, 3 : 4), sum(inliers), 1);
        circle_normals = circle_normals ./ repmat(sqrt(sum(circle_normals .^ 2, 2)), 1, 2);
        if (sum(abs(sqrt((points(inliers, 1) - list(i, 3)) .^ 2 + (points(inliers, 2) - list(i, 4)) .^ 2) - list(i, 5)) <= 3 * distance_tolerance & radtodeg(real(acos(abs(dot(normals(inliers, :), circle_normals, 2))))) <= normal_tolerance) / sum(inliers) < 0.8)
            idx(i) = false;
        else
            inliers = lineLabels == list(i, 2);
            circle_normals = points(inliers, :) - repmat(list(i, 3 : 4), sum(inliers), 1);
            circle_normals = circle_normals ./ repmat(sqrt(sum(circle_normals .^ 2, 2)), 1, 2);
            if (sum(abs(sqrt((points(inliers, 1) - list(i, 3)) .^ 2 + (points(inliers, 2) - list(i, 4)) .^ 2) - list(i, 5)) <= 3 * distance_tolerance & radtodeg(real(acos(abs(dot(normals(inliers, :), circle_normals, 2))))) <= normal_tolerance) / sum(inliers) < 0.8)
                idx(i) = false;
            end
        end
    end
    list = list(idx, :);
    clear idx
    if (size(list, 1) < 2)
        return;
    end
    candidates = zeros(blockSize, 3);
    nCandidates = 0;
    xmin = min(list(:, 3)) - 0.05 * (max(list(:, 3)) - min(list(:, 3))); xmax = max(list(:, 3)) + 0.05 * (max(list(:, 3)) - min(list(:, 3)));
    ymin = min(list(:, 4)) - 0.05 * (max(list(:, 4)) - min(list(:, 4))); ymax = max(list(:, 4)) + 0.05 * (max(list(:, 4)) - min(list(:, 4)));
    nbinsxy = min([2 * ceil((max([ymax - ymin, xmax - xmin])) / distance_tolerance), 2 * size(list, 1)]);
    xx = round((list(:, 3) - xmin) / (xmax - xmin) * nbinsxy + 0.5);
    xx(xx < 1) = 1; xx(xx > nbinsxy) = nbinsxy;
    yy = round((list(:, 4) - ymin) / (ymax - ymin) * nbinsxy + 0.5);
    yy(yy < 1) = 1; yy(yy > nbinsxy) = nbinsxy;
    [h, ia, ic] = unique([xx, yy], 'rows', 'stable');
    h = [h, histc(ic, 1 : numel(ia))];
    positionInitials = grpstats(list(:, 3 : 4), ic);
    [~, order] = sort(h(:, end), 1, 'descend');
    positionInitials = positionInitials(order, :);
    clear xx yy h nH ia ic order
    positionCandidates = meanShift(list(:, 3 : 4), positionInitials, 1, distance_tolerance, 1e-6, 50);
    clear xmin xmax ymin ymax xx yy nbinsxy positionInitials
    positionCandidates = clusterByDistance(positionCandidates, knnsearch(positionCandidates, positionCandidates, 'K', min([20, size(positionCandidates, 1)])), distance_tolerance);
    positionIndices = rangesearch(list(:, 3 : 4), positionCandidates, distance_tolerance);
    for i = 1 : size(positionCandidates, 1);
        if (length(positionIndices{i}) > 1 || (length(positionIndices{i}) == 1 && all(sqrt(sum((repmat(positionCandidates(i, :), size(positionCandidates, 1) - 1, 1) - positionCandidates([1 : i - 1, i + 1 : end], :)) .^ 2, 2)) > distance_tolerance)))
            % histogram circle's radii
            rmin = min(list(positionIndices{i}, 5)) - 0.05 * (max(list(positionIndices{i}, 5)) - min(list(positionIndices{i}, 5)));
            rmax = max(list(positionIndices{i}, 5)) + 0.05 * (max(list(positionIndices{i}, 5)) - min(list(positionIndices{i}, 5)));
            nbinsr = min([2 * ceil((rmax - rmin) / distance_tolerance), length(positionIndices{i})]);
            rr = round((list(positionIndices{i}, 5) - rmin) / (rmax - rmin) * nbinsr + 0.5);
            rr(rr < 1) = 1; rr(rr > nbinsr) = nbinsr;
            [h, ia, ic] = unique(rr, 'rows', 'stable');
            h = [h, histc(ic, 1 : numel(ia))];
            radiusInitials = grpstats(list(positionIndices{i}, 5), ic);
            [~, order] = sort(h(:, end), 1, 'descend');
            radiusInitials = radiusInitials(order);
            clear h ia ic rr order
            radiusCandidates = meanShift(list(positionIndices{i}, 5), radiusInitials, 1, distance_tolerance / 2, 1e-6, 50);
            clear radiusInitials
            radiusCandidates = clusterByDistance(radiusCandidates, knnsearch(radiusCandidates, radiusCandidates, 'K', min([10, size(radiusCandidates, 1)])), distance_tolerance);
            for j = 1 : size(radiusCandidates, 1)
                nCandidates = nCandidates + 1;
                candidates(nCandidates, :) = [positionCandidates(i, :), radiusCandidates(j)];
                if (nCandidates == size(candidates, 1))
                    candidates = [candidates; zeros(blockSize, 3)];
                end
            end
        end
    end
    clear positionCandidates radiusCandidates list
    candidates(nCandidates + 1 : end, :) = [];
    angles = [340; 250; 160; 70];
    angles(angles < angleCoverage) = [];
    if (isempty(angles) || angles(end) ~= angleCoverage)
        angles = [angles; angleCoverage];
    end
    for angleLoop = 1 : length(angles)
        idx = find(labels == 0);
        if (length(idx) < 2 * pi * (3 * distance_tolerance) * Tmin)
            break;
        end
        [L, C, validCandidates] = subCircleDetection(points(idx, :), normals(idx, :), candidates, distance_tolerance, normal_tolerance, Tmin, angles(angleLoop));
        candidates = candidates(validCandidates, :);
        if (size(C, 1) > 0)
            for i = 1 : size(C, 1)
                flag = false;
                for j = 1 : size(circles, 1)
                    if (sqrt((C(i, 1) - circles(j, 1)) .^ 2 + (C(i, 2) - circles(j, 2)) .^ 2) <= distance_tolerance && abs(C(i, 3) - circles(j, 3)) <= distance_tolerance)
                        flag = true;
                        labels(idx(L == i)) = j;
                        break;
                    end
                end
                if (~flag)
                    labels(idx(L == i)) = size(circles, 1) + 1;
                    circles = [circles; C(i, :)];
                end
            end
        end
    end
end

function [labels, circles, validCandidates] = subCircleDetection(points, normals, list, distance_tolerance, normal_tolerance, Tmin, angleCoverage)
    labels = zeros(size(points, 1), 1);
    circles = zeros(0, 3);
    maxRadius = min(max(points) - min(points));
    validCandidates = true(size(list, 1), 1);
    convergence = list;
    for i = 1 : size(list, 1)
        circleCenter = list(i, 1 : 2);
        circleRadius = list(i, 3);
        tbins = min([180, floor(2 * pi * circleRadius * Tmin)]);
        circle_normals = points - repmat(circleCenter, size(points, 1), 1);
        circle_normals = circle_normals ./ repmat(sqrt(sum(circle_normals .^ 2, 2)), 1, 2);
        inliers = find(labels == 0 & abs(sqrt((points(:, 1) - circleCenter(1)) .^ 2 + (points(:, 2) - circleCenter(2)) .^ 2) - circleRadius) <= 2 * distance_tolerance & radtodeg(real(acos(abs(dot(normals, circle_normals, 2))))) <= normal_tolerance);
        inliers = inliers(takeInliers(points(inliers, :), circleCenter, tbins));
        a = circleCenter(1); b = circleCenter(2); r = circleRadius; cnd = 0;
        [newa, newb, newr, newcnd] = fitCircle(points(inliers, :));
        if (newcnd == 0)
            circle_normals = points - repmat([newa, newb], size(points, 1), 1);
            circle_normals = circle_normals ./ repmat(sqrt(sum(circle_normals .^ 2, 2)), 1, 2);
            newinliers = find(labels == 0 & abs(sqrt((points(:, 1) - newa) .^ 2 + (points(:, 2) - newb) .^ 2) - newr) <= 2 * distance_tolerance & radtodeg(real(acos(abs(dot(normals, circle_normals, 2))))) <= normal_tolerance);
            newinliers = newinliers(takeInliers(points(newinliers, :), [newa, newb], tbins));
            if (sqrt((newa - a) .^ 2 + (newb - b) .^ 2) <= 4 * distance_tolerance && abs(newr - r) <= 4 * distance_tolerance)
                if (length(newinliers) >= length(inliers))
                    a = newa; b = newb; r = newr; cnd = newcnd;
                    inliers = newinliers;
                end
            end
        end
        if (length(inliers) >= floor(2 * pi * circleRadius * Tmin))
            convergence(i, :) = [a, b, r];
            if (any(sqrt(sum((convergence(1 : i - 1, 1 : 2) - repmat([a, b], i - 1, 1)) .^ 2, 2)) <= distance_tolerance & abs(convergence(1 : i - 1, 3) - repmat(r, i - 1, 1)) <= distance_tolerance))
                validCandidates(i) = false;
            end
            if (cnd == 0 && r < maxRadius && length(inliers) >= floor(2 * pi * r * Tmin) && isComplete(points(inliers, :), [a, b], tbins, angleCoverage))
                if (all(sqrt(sum((circles(:, 1 : 2) - repmat([a, b], size(circles, 1), 1)) .^ 2, 2)) > distance_tolerance | abs(circles(:, 3) - repmat(r, size(circles, 1), 1)) > distance_tolerance))
                    line_normal = pca(points(inliers, :));
                    line_normal = line_normal(:, 2)';
                    line_point = mean(points(inliers, :));
                    if (sum(abs(dot(points(inliers, :) - repmat(line_point, length(inliers), 1), repmat(line_normal, length(inliers), 1), 2)) <= distance_tolerance & radtodeg(real(acos(abs(dot(normals(inliers, :), repmat(line_normal, length(inliers), 1), 2))))) <= normal_tolerance) / length(inliers) < 0.8)
                        labels(inliers) = size(circles, 1) + 1;
                        circles = [circles; [a, b, r]];
                        validCandidates(i) = false;
                    end
                end
            end
        else
            validCandidates(i) = false;
        end
    end
end

function center = initialEstimateCircleCenter(L)
    center = ([L(1 : 2); L(5 : 6)] \ dot([L(3 : 4); L(7 : 8)], [L(1 : 2); L(5 : 6)], 2))';
end

function [a, b, r, cnd] = fitCircle(points)
    A = [sum(points(:, 1)), sum(points(:, 2)), size(points, 1); sum(points(:, 1) .* points(:, 2)), sum(points(:, 2) .* points(:, 2)), sum(points(:, 2)); sum(points(:, 1) .* points(:, 1)), sum(points(:, 1) .* points(:, 2)), sum(points(:, 1))];
    if (abs(det(A)) < 1e-9)
        cnd = 1;
        a = mean(points(:, 1));
        b = mean(points(:, 2));
        r = min(max(points) - min(points));
        return;
    end
    cnd = 0;
    B = [-sum(points(:, 1) .* points(:, 1) + points(:, 2) .* points(:, 2)); -sum(points(:, 1) .* points(:, 1) .* points(:, 2) + points(:, 2) .* points(:, 2) .* points(:, 2)); -sum(points(:, 1) .* points(:, 1) .* points(:, 1) + points(:, 1) .* points(:, 2) .* points(:, 2))];
    t = A \ B;
    a = -0.5 * t(1);
    b = -0.5 * t(2);
    r = sqrt((t(1) .^ 2 + t(2) .^ 2) / 4 - t(3));
end

function [result, longest_inliers] = isComplete(x, center, tbins, angleCoverage)
    [theta, ~] = cart2pol(x(:, 1) - center(1), x(:, 2) - center(2));
    tmin = -pi; tmax = pi;
    tt = round((theta - tmin) / (tmax - tmin) * tbins + 0.5);
    tt(tt < 1) = 1; tt(tt > tbins) = tbins;
    h = histc(tt, 1 : tbins);
    longest_run = 0;
    start_idx = 1;
    end_idx = 1;
    while (start_idx <= tbins)
        if (h(start_idx) > 0)
            end_idx = start_idx;
            while (start_idx <= tbins && h(start_idx) > 0)
                start_idx = start_idx + 1;
            end
            inliers = [end_idx, start_idx - 1];
            inliers = find(tt >= inliers(1) & tt <= inliers(2));
            run = max(theta(inliers)) - min(theta(inliers));
            if (longest_run < run)
                longest_run = run;
                longest_inliers = inliers;
            end
        end
        start_idx = start_idx + 1;
    end
    if (h(1) > 0 && h(tbins) > 0)
        start_idx = 1;
        while (start_idx < tbins && h(start_idx) > 0)
            start_idx = start_idx + 1;
        end
        end_idx = tbins;
        while (end_idx > 1 && end_idx > start_idx && h(end_idx) > 0)
            end_idx = end_idx - 1;
        end
        inliers = [start_idx - 1, end_idx + 1];
        run = max(theta(tt <= inliers(1)) + 2 * pi) - min(theta(tt >= inliers(2)));
        inliers = find(tt <= inliers(1) | tt >= inliers(2));
        if (longest_run < run)
            longest_run = run;
            longest_inliers = inliers;
        end
    end
    result = radtodeg(longest_run) >= angleCoverage || sum(h > 0) * (360 / tbins) >= min([360, 1.2 * angleCoverage]);
end

function idx = takeInliers(x, center, tbins)
    [theta, ~] = cart2pol(x(:, 1) - center(1), x(:, 2) - center(2));
    tmin = -pi; tmax = pi;
    tt = round((theta - tmin) / (tmax - tmin) * tbins + 0.5);
    tt(tt < 1) = 1; tt(tt > tbins) = tbins;
    h = histc(tt, 1 : tbins);
    mark = zeros(tbins, 1);
    compSize = zeros(tbins, 1);
    nComps = 0;
    queue = zeros(tbins, 1);
    du = [-1, 1];
    for i = 1 : tbins
        if (h(i) > 0 && mark(i) == 0)
            nComps = nComps + 1;
            mark(i) = nComps;
            front = 1; rear = 1;
            queue(front) = i;
            while (front <= rear)
                u = queue(front);
                front = front + 1;
                for j = 1 : 2
                    v = u + du(j);
                    if (v == 0)
                        v = tbins;
                    end
                    if (v > tbins)
                        v = 1;
                    end
                    if (mark(v) == 0 && h(v) > 0)
                        rear = rear + 1;
                        queue(rear) = v;
                        mark(v) = nComps;
                    end
                end
            end
            compSize(nComps) = sum(ismember(tt, find(mark == nComps)));
        end
    end
    compSize(nComps + 1 : end) = [];
    maxCompSize = max(compSize);
    validComps = find(compSize >= maxCompSize * 0.1 & compSize > 10);
    validBins = find(ismember(mark, validComps));
    idx = ismember(tt, validBins);
end
