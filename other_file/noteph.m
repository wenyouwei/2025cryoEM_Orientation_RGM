
%% Panhuan 
% save note

%% imshow 2D images
noisy_projs = cryo_addnoise(projs,SNR,'gaussian');
return;
figure;viewstack(noisy_projs,5,5); % Show some noisy projections
 p=noisy_projs(:,:,85);
 % Scale image to be between 0 and 1
 pmin=min(p(:));
 pmax=max(p(:));
 p=(p-pmin)./(pmax-pmin);
 figure; imshow(p);  % Show some noisy projections
 print('-djpeg',strcat('./figsph/image',fname,num2str(1./SNR),'graynoise.jpg'));
 figure; imagesc(noisy_projs(:,:,85));axis off;
 print('-djpeg',strcat('./figsph/image',fname,num2str(1./SNR),'noise.jpg'));
