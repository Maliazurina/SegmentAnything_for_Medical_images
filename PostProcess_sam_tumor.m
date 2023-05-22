clear
close all
clc;

data_path = [pwd '\SAM_data'];
ct_path = 'C:\T3_Pickens_Malia\SABR_CT\CT';

[ipath_list,scanid_list]=fn_scanid(data_path);

for i = 67:size(scanid_list,1)
    ipath = ipath_list{i};
    scanid = scanid_list(i);
    fprintf('Processing patient %04d\n', scanid);

    SA_Mask = niftiread(ipath_list{i});
    seg_info = niftiinfo(ipath_list{i});
    info = niftiinfo([ct_path '\CT_' num2str(scanid,'%04d') '.nii.gz']);

    filename_nifti =[data_path '\SAMR_' num2str(scanid,'%04d')];

    %lungMask(lungMask>0)=1;


    %-----------
    %for k=1:size(lungMask,3)
    %img=lungMask(:,:,k);
    %img=imfill(img,'holes');
    %lungMask(:,:,k)=img;
    %end

    seg_info.PixelDimensions = info.PixelDimensions;
    seg_info.raw = info.raw;
    seg_info.TransformName =  info.TransformName;
    seg_info.Transform = info.Transform;
    seg_info.Datatype = 'uint8';
    niftiwrite(uint8(SA_Mask),filename_nifti,seg_info,'Compressed',true);
    


end