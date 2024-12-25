function image_processing_gui
    % 创建 figure
    hFig = figure('Name', '图像处理GUI', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    
    % 添加菜单
    uimenu(hFig, 'Label', '&文件', 'Callback', @file_menu_callback);
    uimenu(hFig, 'Label', '&编辑', 'Callback', @edit_menu_callback);
    
    % 添加按钮
    openBtn = uicontrol('Style', 'pushbutton', 'String', '打开图像', 'Position', [20, 400, 100, 30], 'Callback', @open_image_callback);
    histBtn = uicontrol('Style', 'pushbutton', 'String', '显示直方图', 'Position', [130, 400, 100, 30], 'Callback', @show_histogram_callback);
    enhanceBtn = uicontrol('Style', 'pushbutton', 'String', '对比度增强', 'Position', [240, 400, 100, 30], 'Callback', @enhance_contrast_callback);
    transformBtn = uicontrol('Style', 'pushbutton', 'String', '图像变换', 'Position', [350, 400, 100, 30], 'Callback', @image_transform_callback);
    noiseFilterBtn = uicontrol('Style', 'pushbutton', 'String', '加噪与滤波', 'Position', [20, 370, 100, 30], 'Callback', @noise_filter_callback);
    edgeExtractBtn = uicontrol('Style', 'pushbutton', 'String', '边缘提取', 'Position', [130, 370, 100, 30], 'Callback', @edge_extraction_callback);
    targetExtractBtn = uicontrol('Style', 'pushbutton', 'String', '目标提取', 'Position', [240, 370, 100, 30], 'Callback', @target_extraction_callback);
    featureExtractBtn = uicontrol('Style', 'pushbutton', 'String', '特征提取', 'Position', [350, 370, 100, 30], 'Callback', @feature_extraction_callback);
    
   % 添加坐标轴用于显示图像
    ax = axes('Parent', hFig, 'Units', 'Pixels', 'Position', [20, 20, 350, 350]);
    % 初始化显示一个空白图像
    im = uint8(zeros(100, 100, 3));
    imshow(im);
    % 将图像数据存储在UserData中
    set(ax, 'UserData', im);
end


% 打开图像回调函数
function open_image_callback(hObject, eventdata)
    [filename, pathname] = uigetfile({'*.jpg;*.png;*.tif', 'All Image Files'; ...
                                       '*.*',           'All Files'}, ...
                                      'Select an Image');
    if isequal(filename, 0)
        disp('User selected Cancel');
        return;
    else
        fullpath = fullfile(pathname, filename);
        im = imread(fullpath);
        axesHandle = findobj(gcf, 'Type', 'axes');
        imshow(im);
        set(axesHandle, 'UserData', im); % 更新UserData属性
    end
end


% 显示直方图回调函数
function show_histogram_callback(hObject, eventdata)
    axesHandle = findobj(gcf, 'Type', 'axes');
    original_image = getimage(axesHandle);
    gray_image = rgb2gray(original_image);
    
    % 显示原始直方图
    figure;
    counts = imhist(gray_image);
    bar(counts);
    title('原始灰度直方图');
    
    % 直方图均衡化
    equalized_image = histeq(gray_image);
    figure;
    counts_equalized = imhist(equalized_image);
    bar(counts_equalized);
    title('均衡化直方图');
    
    % 显示均衡化后的图像
    axesHandle = findobj(gcf, 'Type', 'axes');
    imshow(equalized_image, 'Parent', axesHandle);
end


% 对比度增强回调函数
function enhance_contrast_callback(hObject, eventdata)
    % 获取当前图像
    axesHandle = findobj(gcf, 'Type', 'axes');
    originalImage = getimage(axesHandle);
    grayImage = rgb2gray(originalImage); % 转换为灰度图像

    % 线性变换
    enhancedImageLinear = imadjust(grayImage); % 默认参数，也可以自定义参数

    % 对数变换
    logImage = log(1 + double(grayImage));
    enhancedImageLog = im2uint8(mat2gray(logImage));

    % 指数变换
    gamma = 2.2; % 调整gamma值以增强对比度
    % 确保像素值在0到255范围内
    enhancedImageExp = im2uint8(255 * (double(grayImage) / 255).^gamma);

    % 显示增强后的图像
    figure;
    subplot(1, 3, 1);
    imshow(enhancedImageLinear);
    title('线性变换增强');

    subplot(1, 3, 2);
    imshow(enhancedImageLog);
    title('对数变换增强');

    subplot(1, 3, 3);
    imshow(enhancedImageExp);
    title(['指数变换增强 (gamma=' num2str(gamma) ')']);
end


% 图像变换回调函数
function image_transform_callback(hObject, eventdata)
    axesHandle = findobj(gcf, 'Type', 'axes');
    original_image = getimage(axesHandle);
    
    % 缩放变换
    scale_factor = 0.5;
    scaled_image = imresize(original_image, scale_factor);
    
    % 旋转变换
    angle = 45;
    rotated_image = imrotate(original_image, angle);
    
    % 显示变换后的图像
    figure;
    subplot(1, 2, 1);
    imshow(scaled_image);
    title('缩放变换');
    
    subplot(1, 2, 2);
    imshow(rotated_image);
    title('旋转变换');
end


% 加噪与滤波回调函数
function noise_filter_callback(hObject, eventdata)
    % 获取当前 figure 的坐标轴句柄
    axesHandle = findobj(gcf, 'Type', 'axes');
    
    % 获取坐标轴下的图像对象
    imageHandle = findobj(axesHandle, 'Type', 'image');
    
    if ~isempty(imageHandle)
        % 获取图像数据
        img = imageHandle.CData;
        
        % 示例：添加高斯噪声
        noisy_img = imnoise(img, 'gaussian');
        
        % 示例：使用均值滤波器
        filtered_img = imgaussfilt(noisy_img, 2);
        
        % 显示加噪和滤波后的图像
        figure;
        subplot(1, 2, 1);
        imshow(noisy_img);
        title('加噪图像');
        
        subplot(1, 2, 2);
        imshow(filtered_img);
        title('滤波后的图像');
    else
        disp('没有找到图像数据。');
    end
end


 % 边缘提取回调函数
function edge_extraction_callback(hObject, eventdata)
    % 获取当前 figure 的坐标轴句柄
    axesHandle = findobj(gcf, 'Type', 'axes');
    
    % 获取坐标轴下的图像对象
    imageHandle = findobj(axesHandle, 'Type', 'image');
    
    if ~isempty(imageHandle)
        % 获取图像数据
        img = imageHandle.CData;
        
        % 转换为灰度图像
        grayImg = rgb2gray(img);
        
        % 使用Robert算子进行边缘提取
        edge_robert = edge(grayImg, 'roberts');
        
        % 使用Prewitt算子进行边缘提取
        edge_prewitt = edge(grayImg, 'prewitt');

        % 使用Sobel算子进行边缘提取
        edge_sobel = edge(grayImg, 'sobel');      
        
        % 显示Robert边缘提取结果
        figure;
        subplot(1, 3, 1);
        imshow(edge_robert);
        title('Robert边缘提取');
        
        % 显示Prewitt边缘提取结果
        subplot(1, 3, 2);
        imshow(edge_prewitt);
        title('Prewitt边缘提取');

        % 显示Sobel边缘提取结果
        subplot(1, 3, 3);
        imshow(edge_sobel);
        title('Sobel边缘提取');
    else
        disp('没有找到图像数据。');
    end
end


% 目标提取回调函数
function target_extraction_callback(hObject, eventdata)
    % 获取当前坐标轴
    ax = findobj(gcf, 'Type', 'axes');
    
    if ~isempty(ax)
        % 获取坐标轴中的图像数据
        im = get(ax, 'UserData');
        
        if ~isempty(im)
            % 转换为灰度图像
            grayImg = rgb2gray(im);
            
            % 计算图像的直方图
            [counts, binLocations] = imhist(grayImg);
            
            % 使用Otsu方法找到最佳阈值
            level = graythresh(grayImg);
            
            % 使用找到的阈值进行二值化
            binaryImg = imbinarize(grayImg, level);
            
            % 可以使用形态学操作来清理二值图像
            se = strel('disk', 3); % 定义结构元素
            binaryImg = imopen(binaryImg, se); % 开运算
            binaryImg = imclose(binaryImg, se); % 闭运算
            
            % 显示二值化后的图像
            figure;
            imshow(binaryImg);
            title('提取的目标');
            
            % 更新坐标轴的UserData以保存二值化图像
            set(ax, 'UserData', binaryImg);
        else
            error('没有图像数据可以处理。');
        end
    else
        error('找不到坐标轴对象。');
    end
end


% 特征提取回调函数
function feature_extraction_callback(hObject, eventdata)
    axesHandle = findobj(gcf, 'Type', 'axes');
    
    if ~isempty(axesHandle)
        im = get(axesHandle, 'UserData');
        
        if ~isempty(im)
            if size(im, 3) == 3
                grayImage = rgb2gray(im);
            else
                grayImage = im;
            end
            
            lbpFeatures = extractLBPFeatures(grayImage);
            [hogFeatures, hogVisualization] = extractHOGFeatures(grayImage);
            
            figure;
            imshow(lbpFeatures, []);
            title('LBP Features Image');
            
            figure;
            imshow(hogVisualization, []);
            title('HOG Features Image');
        else
            error('没有图像数据可以提取特征。');
        end
    else
        error('找不到坐标轴对象。');
    end
end

function lbpFeatures = extractLBPFeatures(image)
    % 自定义实现 LBP 特征提取
    lbpFeatures = customLBP(image);
end

function [hogFeatures, hogVisualization] = extractHOGFeatures(image)
    % 自定义实现 HOG 特征提取
    [hogFeatures, hogVisualization] = customHOG(image);
end

function lbpFeatures = customLBP(image)
    % 这里是自定义 LBP 特征提取算法的实现
    % 以下代码仅为示例，需要根据实际情况编写
    P = 8; % LBP 算法的邻域点数
    R = 1; % LBP 算法的半径
    lbpFeatures = zeros(size(image));
    
    for i = 1:size(image, 1)
        for j = 1:size(image, 2)
            % 计算 LBP 值
            lbpValue = 0;
            for k = 0:P-1
                x = i + R * cos(k * 2 * pi / P);
                y = j - R * sin(k * 2 * pi / P);
                % 双线性插值
                lbpValue = lbpValue + 2^k * (interp2(image, x, y) >= image(i, j));
            end
            lbpFeatures(i, j) = lbpValue;
        end
    end
end

function [hogFeatures, hogVisualization] = customHOG(image)
    % 参数设置
    cellSize = [8 8]; % 细胞单元大小
    blockSize = [2 2]; % 块大小
    numBins = 9; % 直方图的方向数量
    
    % 计算图像的梯度
    [Gx, Gy] = gradient(image);
    Mag = sqrt(Gx.^2 + Gy.^2); % 梯度幅度
    Angle = atan2(Gy, Gx) * (180/pi); % 梯度方向，转换为度
    
    % 将角度范围限制在0到180度
    Angle(Angle < 0) = Angle(Angle < 0) + 180;
    
    % 初始化HOG特征和可视化图像
    hogFeatures = zeros(numel(Angle), 1);
    hogVisualization = zeros(size(image));
    
    % 计算每个细胞单元的直方图
    for i = 1:cellSize(1):size(image, 1)-cellSize(1)+1
        for j = 1:cellSize(2):size(image, 2)-cellSize(2)+1
            % 当前细胞单元的梯度幅度和角度
            cellMag = Mag(i:i+cellSize(1)-1, j:j+cellSize(2)-1);
            cellAngle = Angle(i:i+cellSize(1)-1, j:j+cellSize(2)-1);
            
            % 计算直方图
            binEdges = linspace(0, 180, numBins+1);
            [histCounts, ~] = histcounts(cellAngle, binEdges);
            histCounts = histCounts / sum(histCounts); % 归一化直方图
            
            % 将直方图添加到HOG特征中
            hogFeatures((i-1)*cellSize(1)+(j-1)*cellSize(2)+1:i*cellSize(1)+j*cellSize(2)) = histCounts;
            
            % 可视化直方图
            hogVisualization(i:i+cellSize(1)-1, j:j+cellSize(2)-1) = max(histCounts) * ones(cellSize);
        end
    end
    
    % 可视化图像归一化
    hogVisualization = hogVisualization / max(hogVisualization(:));
end

