%requires running the "NS_spectral_solver" first

%this is just for compairison with the fourier modes
w=permute(u,[2 3 1 4]);
w_hat=fft2(w);
absw_hat_vec=sqrt(mean(abs(w_hat).^2,[3 4]));
absw_hat_vec=absw_hat_vec(:);
absk=abs(sqrt(kx.^2+ky.^2));
wsort=flipud(sort(absw_hat_vec));
indfourier=[1:length(wsort)+1];

sz=size(w);
w=reshape(w,sz(1)*sz(2),[]);
sz=size(w);
%The meat of it. re-shape tensor into a matrix (columns are spatial data,
%rows are temporal data (and possibly later parameter data)
[Modes,S,~]=svd(w-mean(w,[2:length(sz)]),'econ');

%energy is just the squared singular values
Energy=diag(S).^2;
%produce indicies for the plot
indsvd=[1:length(Energy)+1];

%plot residual mode energy
figure(1)
loglog(indfourier,[1; 1-cumsum(wsort.^2)/sum(wsort.^2)],'k',indsvd,[1; 1-cumsum(Energy)/sum(Energy)],'--k','LineWidth',2)
%loglog(indsvd,[1; 1-cumsum(Energy)/sum(Energy)],'k','LineWidth',2)
axis([1 10^4 10^-6 1])
set(0,'defaultAxesFontSize',30)
xlabel('Mode # (Energy Sorted)')
ylabel('Residual Energy')
legend('Fourier Decomposition','POD')
%legend('POD')

%view various interesting modes
figure(2)
m1=Modes(:,1);
M1=reshape(m1,sqrt(sz(1)),sqrt(sz(1))); %re-shapes the long vector representing the mode into a matrix for which the mode actually looks acceptable
imagesc(M1)
set(gca,'visible','off')
figure(3)
m2=Modes(:,2);
M2=reshape(m2,sqrt(sz(1)),sqrt(sz(1)));
imagesc(M2)
set(gca,'visible','off')
figure(4)
m10=Modes(:,10);
M10=reshape(m10,sqrt(sz(1)),sqrt(sz(1)));
imagesc(M10)
set(gca,'visible','off')