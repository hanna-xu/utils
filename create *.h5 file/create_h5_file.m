clear;
close all;


savepath='****.h5';
size_input=192;
stride=200;
Format='*.jpg';
img_ch=1; % image channel: 1 for gray, 3 for RGB
ratio=1; % >1: if you want to resize the source image to expand the number of training data 

% initialization
data=zeros(size_input,size_input,2*img_ch,1);

folder1='...';
filepath1=dir(fullfile(folder1,Format));
folder2='...';
filepath2=dir(fullfile(folder2,Format));

count=0;
L1=length(filepath1);
L2=length(filepath2);
L=min([L1,L2]);

tic
for i=1:L
    img1=imread(dir(fullfile(folder1,filepath1(i).name,Format)));
    img2=imread(dir(fullfile(folder2,filepath2(i).name,Format)));
    
    [height,width,channel]=size(img1);
    for c=1:channel
        img1_double(:,:,c)=im2double(img1(:,:,c));
        img2_double(:,:,c)=im2double(img2(:,:,c));    
        i1(:,:,c)=imresize(img1_double(:,:,c),[fix(height*ratio),fix(width*ratio)]);
        i2(:,:,c)=imresize(img2_double(:,:,c),[fix(height*ratio),fix(width*ratio)]);      
    end
    clear img1 img2


    [height,width,channel]=size(i1);
   
    for x=1:stride:(height-size_input+1)
        for y=1:stride:(width-size_input+1)    
            sub_img1(:,:,:)=i1(x:x+size_input-1,y:y+size_input-1,:);
            sub_img2(:,:,:)=i2(x:x+size_input-1,y:y+size_input-1,:);
            count=count+1;
            data(:,:,1:1+img_ch-1,count)=sub_img1(:,:,:);
            data(:,:,1+img_ch:1+2*img_ch-1,count)=sub_img2(:,:,:);
           
%%  If you want to create more training data by flipping in vertical and horizantal direction
%             count=count+1;
%             for c=1:channel
%                 sub_img11(:,:,c)=flipud(sub_img1(:,:,c));
%                 sub_img21(:,:,c)=flipud(sub_img2(:,:,c));
%             end
%             data(:,:,1:1+img_ch-1,count)=sub_img11(:,:,:);
%             data(:,:,1+img_ch:1+2*img_ch-1,count)=sub_img21(:,:,:);
%             
%             count=count+1;
%             for c=1:channel
%                 sub_img12(:,:,c)=fliplr(sub_img1(:,:,c));
%                 sub_img22(:,:,c)=fliplr(sub_img2(:,:,c));
%             end
%             data(:,:,1:1+img_ch-1,count)=sub_img12(:,:,:);
%             data(:,:,1+img_ch:1+2*img_ch-1,count)=sub_img22(:,:,:);      

            figure(1)
            A=data(:,:,1:1+img_ch-1,count);
            B=data(:,:,1+img_ch:1+2*img_ch-1,count);
            subplot(221),imshow(A);
            subplot(222),imshow(B);
            subplot(223),imshow(sub_img1);
            subplot(224),imshow(sub_img2);
        end
    end
      
    if mod(i,10)==0
        i
        toc
    end
    clear i1 i2 img1_double img2_double
end

order=randperm(count);
data=data(:,:,:,order);

chunksz=12;
totalct=0;
create_flag=true;

for batch_num=1:floor(count/chunksz)
    lastread=(batch_num-1)*chunksz;
    batch_data=data(:,:,:,lastread+1:lastread+chunksz);
    startloc=struct('dat',[1,1,1,totalct+1]);
    if batch_num~=1
        create_flag=false;
    end
    curr_dat_sz=store2hdf5(savepath,batch_data,create_flag,startloc,chunksz);
    totalct=curr_dat_sz(end);
end
