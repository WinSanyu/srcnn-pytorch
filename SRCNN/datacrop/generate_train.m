clear;close all;

folder = 'Train';
saveInputPath = 'Train_sub_bic';
saveLabelPath = 'Train_sub';
size_input = 33;
size_label = 21;
stride = 14;

%% scale factors
scale = 3;

%% initialization
padding = abs(size_input - size_label)/2;
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];
if ~exist(saveInputPath, 'dir')
    mkdir(saveInputPath);
end
if ~exist(saveLabelPath, 'dir')
    mkdir(saveLabelPath);
end
%% generate data
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));
    
    im_label = modcrop(image, scale);
    [hei,wid] = size(im_label);
    im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic');
    count = 0;
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
            subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);
            
            imwrite(subim_input, fullfile(saveInputPath, [filepaths(i).name(1:end-4)  '_' num2str(count) '.bmp']));  
            imwrite(subim_label, fullfile(saveLabelPath, [filepaths(i).name(1:end-4)  '_' num2str(count) '.bmp']));  
            
            count=count+1;

        end
    end
end
