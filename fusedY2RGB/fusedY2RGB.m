clear all
clc

Format='*.jpg';

folder1='.../ue';
folder2='.../oe';
folderf='.../fused_Y';

filepath1=dir(fullfile(folder1,Format));
filepath2=dir(fullfile(folder2,Format));
filepathf=dir(fullfile(folderf,Format));

L=length(filepath1);
for pic=1:L
    img1=imread(fullfile(folder1,filepath1(pic).name));
    img2=imread(fullfile(folder2,filepath2(pic).name));
    imgf_y=double(imread(fullfile(folderf,filepathf(pic).name)));
    
    [height,width,channel]=size(img1);
    if channel==3
        ycbcr1 = double(rgb2ycbcr(img1));
        cb1=ycbcr1(:,:,2);
        cr1=ycbcr1(:,:,3);
    end
    
    [height,width,channel]=size(img2);
    if channel==3
        ycbcr2 = double(rgb2ycbcr(img2));
        cb2=ycbcr2(:,:,2);
        cr2=ycbcr2(:,:,3);
    end
    
    %% fuse Cb/Cr
    if exist('cb1') && exist('cb2')
        for i=1:height
            for j=1:width
                x1=cb1(i,j); x2=cb2(i,j);
                if (x1==128)&&(x2==128)
                    cbf(i,j)=128;
                else
                    cbf(i,j)=(x1*abs(x1-128)+x2*abs(x2-128))/(abs(x1-128)+abs(x2-128));
                end
            end
        end
    elseif exist('cb1')
        cbf=cb1;
    elseif exist('cb2')
        cbf=cb2;
        
    end
    
    if exist('cr1') && exist('cr2')
        for i=1:height
            for j=1:width
                x1=cr1(i,j); x2=cr2(i,j);
                if (x1==128)&&(x2==128)
                    crf(i,j)=128;
                else
                    crf(i,j)=(x1*abs(x1-128)+x2*abs(x2-128))/(abs(x1-128)+abs(x2-128));
                end
            end
        end
    elseif exist('cb1')
        crf=cr1;
    elseif exist('cb2')
        crf=cr2;
    end
    
    if exist('cbf') && exist('crf')
        imgf_ycbcr(:,:,1)=imgf_y;
        imgf_ycbcr(:,:,2)=cbf;
        imgf_ycbcr(:,:,3)=crf;
        imgf=uint8(YCbCr2rgb(imgf_ycbcr));
    end
    
%     figure(1)
%     subplot(131),imshow(img1);
%     subplot(132),imshow(img2);
%     subplot(133),imshow(imgf);
    imwrite(imgf,[num2str(pic),'.jpg']);
    clear imgf_ycbcr cb1 cb2 cf1 cf2 cbf crf
    
end
