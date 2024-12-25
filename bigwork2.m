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
    
    % 创建按钮选择直方图处理方式
    hEqButton = uicontrol('Style', 'pushbutton', ...
                          'String', '直方图均衡化', ...
                          'Position', [100, 300, 150, 50], ...
                          'Callback', @equalizeHistogramCallback);
    
    hMatchButton = uicontrol('Style', 'pushbutton', ...
                             'String', '直方图匹配（规定化）', ...
                             'Position', [300, 300, 150, 50], ...
                             'Callback', @matchHistogramCallback);
    
    % 直方图均衡化回调函数
    function equalizeHistogramCallback(hObject, eventdata)
        equalized_image = histeq(gray_image);
        figure;
        counts_equalized = imhist(equalized_image);
        bar(counts_equalized);
        title('均衡化直方图');
        
        % 显示均衡化后的图像
        axesHandle = findobj(gcf, 'Type', 'axes');
        imshow(equalized_image, 'Parent', axesHandle);
    end
    
    % 直方图匹配（规定化）回调函数
    function matchHistogramCallback(hObject, eventdata)
        % 这里需要一个参考图像的直方图，为了示例，我们使用原始图像的直方图
        reference_counts = imhist(gray_image);
        matched_image = histeq(gray_image, reference_counts);
        figure;
        counts_matched = imhist(matched_image);
        bar(counts_matched);
        title('匹配直方图');
        
        % 显示匹配后的图像
        axesHandle = findobj(gcf, 'Type', 'axes');
        imshow(matched_image, 'Parent', axesHandle);
    end
end



% 对比度增强回调函数
function enhance_contrast_callback(hObject, eventdata)
    % 获取主GUI的坐标轴句柄
    ax = findobj(gcf, 'Type', 'axes');
    if ~isempty(ax)
        originalImage = get(ax, 'UserData');
        if ~isempty(originalImage)
            % 创建一个新的窗口用于对比度增强
            hEnhanceFig = figure('Name', '对比度增强', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
            
            % 在新窗口中添加坐标轴显示原始图像
            enhanceAx = axes('Parent', hEnhanceFig);
            imshow(originalImage, 'Parent', enhanceAx);
            set(enhanceAx, 'UserData', originalImage); % 存储图像数据供后续使用
            
            % 在新窗口中添加按钮
            uicontrol('Style', 'pushbutton', 'String', '线性变换', ...
                      'Position', [20 100 100 30], 'Callback', @linearTransformCallback, 'Parent', hEnhanceFig);
            uicontrol('Style', 'pushbutton', 'String', '对数变换', ...
                      'Position', [20 70 100 30], 'Callback', @logTransformCallback, 'Parent', hEnhanceFig);
            uicontrol('Style', 'pushbutton', 'String', '指数变换', ...
                      'Position', [20 40 100 30], 'Callback', @expTransformCallback, 'Parent', hEnhanceFig);
        end
    end
end

% 线性变换回调函数
function linearTransformCallback(hObject, eventdata)
    % 获取新窗口的坐标轴句柄
    ax = findobj(gcf, 'Type', 'axes');
    if ~isempty(ax)
        originalImage = get(ax, 'UserData');
        if ~isempty(originalImage)
            grayImage = rgb2gray(originalImage);
            enhancedImage = imadjust(grayImage);
            figure;
            imshow(enhancedImage);
            title('线性变换');
        end
    end
end

% 对数变换回调函数
function logTransformCallback(hObject, eventdata)
    % 获取新窗口的坐标轴句柄
    ax = findobj(gcf, 'Type', 'axes');
    if ~isempty(ax)
        originalImage = get(ax, 'UserData');
        if ~isempty(originalImage)
            grayImage = rgb2gray(originalImage);
            logImage = log(1 + double(grayImage));
            enhancedImage = im2uint8(mat2gray(logImage));
            figure;
            imshow(enhancedImage);
            title('对数变换');
        end
    end
end

% 指数变换回调函数
function expTransformCallback(hObject, eventdata)
    % 获取新窗口的坐标轴句柄
    ax = findobj(gcf, 'Type', 'axes');
    if ~isempty(ax)
        originalImage = get(ax, 'UserData');
        if ~isempty(originalImage)
            grayImage = rgb2gray(originalImage);
            gamma = 2.2;
            enhancedImage = im2uint8(255 * (double(grayImage) / 255).^gamma);
            figure;
            imshow(enhancedImage);
            title('指数变换(gmma=2.2)');
        end
    end
end



% 图像变换回调函数
function image_transform_callback(hObject, eventdata)
    axesHandle = findobj(gcf, 'Type', 'axes');
    original_image = getimage(axesHandle);
    
    % 创建一个新窗口
    hTransformFig = figure('Name', '图像变换', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    
    % 添加缩放变换按钮
    uicontrol('Style', 'pushbutton', 'String', '缩放变换', ...
              'Position', [20 120 100 30], 'Callback', @scaleTransformCallback, 'Parent', hTransformFig);
    
    % 添加旋转变换按钮
    uicontrol('Style', 'pushbutton', 'String', '旋转变换', ...
              'Position', [20 80 100 30], 'Callback', @rotateTransformCallback, 'Parent', hTransformFig);
    
    % 保存原始图像以便在回调中使用
    setappdata(hTransformFig, 'original_image', original_image);
end

% 缩放变换回调函数
function scaleTransformCallback(hObject, eventdata)
    hTransformFig = hObject.Parent;
    original_image = getappdata(hTransformFig, 'original_image');
    
    % 弹出对话框让用户输入缩放倍数
    prompt = {'请输入缩放倍数:'};
    dlgtitle = '输入缩放倍数';
    numlines = 1;
    definput = {'1'}; % 默认值
    scale_factor_str = inputdlg(prompt, dlgtitle, numlines, definput);
    
    % 检查用户是否输入了值
    if isempty(scale_factor_str)
        return; % 如果用户取消或没有输入，则返回
    end
    
    % 转换输入为数字
    scale_factor = str2double(scale_factor_str{1});
    
    % 检查转换是否成功
    if isnan(scale_factor)
        errordlg('请输入有效的缩放倍数。', '错误');
        return;
    end
    
    % 应用缩放变换
    scaled_image = imresize(original_image, scale_factor);
    figure;
    imshow(scaled_image);
    title('缩放变换');
end

% 旋转变换回调函数
function rotateTransformCallback(hObject, eventdata)
    hTransformFig = hObject.Parent;
    original_image = getappdata(hTransformFig, 'original_image');
    
    % 弹出对话框让用户输入旋转角度
    prompt = {'请输入旋转角度 (度):'};
    dlgtitle = '输入旋转角度';
    numlines = 1;
    definput = {'0'}; % 默认值
    angle_str = inputdlg(prompt, dlgtitle, numlines, definput);
    
    % 检查用户是否输入了值
    if isempty(angle_str)
        return; % 如果用户取消或没有输入，则返回
    end
    
    % 转换输入为数字
    angle = str2double(angle_str{1});
    
    % 检查转换是否成功
    if isnan(angle)
        errordlg('请输入有效的旋转角度。', '错误');
        return;
    end
    
    % 应用旋转变换
    rotated_image = imrotate(original_image, angle);
    figure;
    imshow(rotated_image);
    title(['旋转变换 (角度: ' num2str(angle) '°)']);
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
        
        % 创建新窗口以选择噪声类型和参数
        hNoiseFig = figure('Name', '选择噪声类型和参数', 'NumberTitle', 'off', 'MenuBar', 'none');
        uicontrol('Style', 'popupmenu', 'String', {'高斯噪声', '椒盐噪声'}, 'Position', [100 200 100 25], 'Callback', @selectNoiseType);
        uicontrol('Style', 'edit', 'Position', [220 200 100 25], 'String', '0.01', 'Callback', @getNoiseParam);
        
        % 用于存储噪声参数
        setappdata(hNoiseFig, 'originalImage', img);
    else
        disp('没有找到图像数据。');
    end
end

function selectNoiseType(hObject, eventdata)
    hNoiseFig = hObject.Parent;
    noiseType = hObject.String{get(hObject, 'Value')};
    noiseParam = str2double(getappdata(hNoiseFig, 'noiseParam'));
    
    % 检查噪声参数是否有效
    if isnan(noiseParam) || noiseParam < 0
        noiseParam = 0.01; % 如果转换失败或小于0，使用默认值
    end
    
    img = getappdata(hNoiseFig, 'originalImage');
    
    % 根据选择的噪声类型添加噪声
    switch noiseType
        case '高斯噪声'
            % 确保方差是非负的
            variance = max(noiseParam^2, 0);
            noisy_img = imnoise(img, 'gaussian', 0, variance);
        case '椒盐噪声'
            % 确保噪声密度小于等于1
            density = min(noiseParam, 1);
            noisy_img = imnoise(img, 'salt & pepper', density);
    end
    
    % 显示加噪后的图像
    hNoisyFig = figure('Name', '加噪后的图像', 'NumberTitle', 'off', 'MenuBar', 'none');
    imshow(noisy_img);
    title('加噪后的图像');
    
    % 提供滤波选项
    uicontrol('Style', 'pushbutton', 'String', '空域滤波', 'Position', [50 50 100 25], 'Callback', @spatialFilter);
    uicontrol('Style', 'pushbutton', 'String', '频域滤波', 'Position', [200 50 100 25], 'Callback', @frequencyFilter);
    
    % 存储加噪后的图像
    setappdata(hNoisyFig, 'noisyImage', noisy_img);
end

function getNoiseParam(hObject, eventdata)
    hNoiseFig = hObject.Parent;
    noiseParam = str2double(get(hObject, 'String')); % 确保 'String' 转换为数值
    if isnan(noiseParam)
        noiseParam = 0.01; % 如果转换失败，使用默认值
    end
    setappdata(hNoiseFig, 'noiseParam', noiseParam);
end

function spatialFilter(hObject, eventdata)
    hNoisyFig = hObject.Parent;
    noisy_img = getappdata(hNoisyFig, 'noisyImage');
    
    % 示例：使用均值滤波器
    filtered_img = imgaussfilt(noisy_img, 2);
    
    % 显示滤波后的图像
    figure;
    imshow(filtered_img);
    title('空域滤波后的图像');
end

function frequencyFilter(hObject, eventdata)
    hNoisyFig = hObject.Parent;
    noisy_img = getappdata(hNoisyFig, 'noisyImage');
    
    % 转换图像到频域
    noisy_img_freq = fft2(double(noisy_img));
    noisy_img_freq = fftshift(noisy_img_freq); % 将零频率分量移到中心
    
    [M, N] = size(noisy_img_freq);
    
    % 设计滤波器遮罩 H，这里假设你有一个函数来设计滤波器
    H = designFilter(M, N); % 你需要确保这个函数返回正确大小的滤波器
    
    % 检查 H 是否为二维数组
    if ndims(H) ~= 2
        error('滤波器 H 必须是二维数组。');
    end
    
    % 确保H的大小与noisy_img_freq相匹配
    [rows_H, cols_H] = size(H);
    if rows_H ~= M || cols_H ~= N
        % 如果 H 的大小不匹配，使用 padarray 来调整大小
        % 假设使用边缘值复制来填充
        H = padarray(H, [(M-rows_H) (N-cols_H)], 'replicate');
    end
    
    % 应用滤波器
    filtered_img_freq = noisy_img_freq .* H;
    
    % 将滤波后的图像转换回空间域
    filtered_img = ifftshift(filtered_img_freq);
    filtered_img = ifft2(filtered_img);
    filtered_img = real(filtered_img); % 取实部，因为图像通常是实数
    
    % 显示滤波后的图像
    figure;
    imshow(filtered_img, []);
    title('频域滤波后的图像');
end

% 假设的设计滤波器函数
function H = designFilter(M, N)
    % 这里应该返回一个大小为 MxN 的滤波器遮罩
    % 例如，一个简单的低通滤波器
    H = ones(M, N); % 这里只是一个示例，你应该根据需要设计滤波器
end



% 定义 displayEdge 函数，用于显示边缘提取结果
function displayEdge(edgeType, grayImg, titleStr)
    switch edgeType
        case 'roberts'
            edgeFunc = @edge;
        case 'prewitt'
            edgeFunc = @(img) edge(img, 'prewitt');
        case 'sobel'
            edgeFunc = @(img) edge(img, 'sobel');
        otherwise
            error('未知的边缘提取算子');
    end
    
    % 使用指定的边缘提取算子
    edgeImg = edgeFunc(grayImg);
    
    % 显示边缘提取结果
    figure;
    imshow(edgeImg);
    title(titleStr);
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
        
        % 创建新窗口
        edgeFig = figure('Name', '边缘提取选择', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
        
        % 创建按钮，并设置回调函数
        uicontrol('Style', 'pushbutton', 'String', 'Robert边缘提取', ...
          'Position', [20 120 100 30], ...
          'Callback', @(src, event) displayEdge('roberts', grayImg, 'Robert边缘提取'));

        uicontrol('Style', 'pushbutton', 'String', 'Prewitt边缘提取', ...
          'Position', [20 80 100 30], ...
          'Callback', @(src, event) displayEdge('prewitt', grayImg, 'Prewitt边缘提取'));

        uicontrol('Style', 'pushbutton', 'String', 'Sobel边缘提取', ...
          'Position', [20 40 100 30], ...
          'Callback', @(src, event) displayEdge('sobel', grayImg, 'Sobel边缘提取'));
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
                   
            % 显示特征
            figure;
            subplot(1, 2, 1); bar(lbpFeatures); title('LBP 特征'); % 显示 LBP 特征
            subplot(1, 2, 2); imagesc(hogVisualization); colormap gray; title('HOG 特征'); % 显示 HOG 特征可视化
        else
            error('没有图像数据可以提取特征。');
        end
    else
        error('找不到坐标轴对象。');
    end
end


