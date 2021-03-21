
%define fluid parameters
Re=3000;
tic
%define grid
xevalpts=150; 
yevalpts=150;
xlim=1;
ylim=1;
time=10;
ntimes=30;
tspan=linspace(0,time,ntimes);
x=linspace(-xlim,xlim,xevalpts);
y=linspace(-ylim,ylim,yevalpts);

[X,Y]=meshgrid(x,y);

pgrid=4;
%pgrid=4;

sigma1=logspace(-1,log10(1/2),pgrid);
sigma2=logspace(-1,log10(1/2),pgrid);
beta=logspace(-1,log10(1),pgrid);
alpha=logspace(-1,log10(1),pgrid);
%{
sigma1=10.^((log10(1/2)+1)*rand(pgrid,1)-1);
sigma2=10.^((log10(1/2)+1)*rand(pgrid,1)-1);
beta=10.^((log10(1)+1)*rand(pgrid,1)-1);
alpha=10.^((log10(1)+1)*rand(pgrid,1)-1);
%}


%define basis wavefunctions
for j=1:yevalpts
kx(j,:)=1i*2*pi/xlim/2*[0:xevalpts/2-1, 0, -xevalpts/2+1:-1];
end
for j=1:xevalpts
ky(:,j)=1i*2*pi/ylim/2*[0:yevalpts/2-1, 0, -yevalpts/2+1:-1];
end
options=odeset('Reltol',1e-13);
w_out_vec=zeros(ntimes,xevalpts*yevalpts,pgrid,pgrid,pgrid,pgrid);
parfor i=1:pgrid
    for j=1:pgrid
        for k=1:pgrid
            for m=1:pgrid
sigma1=10.^((log10(1/2)+1)*rand(pgrid,1)-1);
sigma2=10.^((log10(1/2)+1)*rand(pgrid,1)-1);
beta=10.^((log10(1)+1)*rand(pgrid,1)-1);
alpha=10.^((log10(1)+1)*rand(pgrid,1)-1);
w_0=1/(pi*sigma1(i)^2)*1/(1+beta(j))*exp(-(((X-alpha(m)/2)/sigma1(i)).^2+(Y/sigma1(i)).^2))+1/(pi*sigma2(k)^2)*beta(j)/(1+beta(j))*exp(-(((X+alpha(m)/2)/sigma2(k)).^2+(Y/sigma2(k)).^2)); %define Lamb-Oseen Vortex (work-in progress)

%solve for stream function
tic
[t(:,i,j,k,m),w_out_vec(:,:,i,j,k,m)]=ode113(@(t,w) NS(t,w,kx,ky,Re,xevalpts,yevalpts),tspan,w_0,options);
toc
fprintf('Grid: %i %i %i %i \n',i,j,k,m)
            end
        end
    end
end
%generate streamfunction (in case I want to plot that instead of the
%vorticity
w_out=reshape(w_out_vec,ntimes,yevalpts,xevalpts,[]);
xp=xevalpts;
yp=yevalpts;
w_hat=fft2(single(w_out));
wpic=real(ifft2(w_hat,yp,xp));
Laplacian_fourier=kx.^2+ky.^2;
partial_x_fourier=kx;
partial_y_fourier=ky;
Laplacian_fourier_regularized=Laplacian_fourier;
Laplacian_fourier_regularized(1,1)=1;
Laplacian_fourier_regularized(yevalpts/2+1,1)=100;
Laplacian_fourier_regularized(1,xevalpts/2+1)=100;
Laplacian_fourier_regularized(yevalpts/2+1,xevalpts/2+1)=100;
psi_fourier=-w_hat./Laplacian_fourier_regularized;
%v=real(ifft2(-partial_x_fourier.*psi_fourier)); 
%u=real(ifft2(partial_y_fourier.*psi_fourier));
psi=real(ifft2(psi_fourier));
toc
videogen(log(wpic+0.01))%calls video generation function and plots vorticity in log space

function wdot_vec=NS(t,w,kx,ky,Re,xevalpts,yevalpts)
%send data to the GPU
kx=gpuArray(single(kx));
ky=gpuArray(single(ky));
Re=gpuArray(single(Re));
xevalpts=gpuArray(single(xevalpts));
yevalpts=gpuArray(single(yevalpts));
w=gpuArray(single(w));
w=reshape(w,[yevalpts,xevalpts]);
w_fourier=fft2(w);

%define functional operators in fourier space
Laplacian_fourier=kx.^2+ky.^2;
partial_x_fourier=kx; %partial x derivative
partial_y_fourier=ky; %partial y derivative

%Laplacian in fourier space
Laplacian_fourier_regularized=Laplacian_fourier;

%regularizing the Laplacian (to eliminate divide-by-zero errors at troublesome wave-numbers)
Laplacian_fourier_regularized(1,1)=1;
Laplacian_fourier_regularized(yevalpts/2+1,1)=1;
Laplacian_fourier_regularized(1,xevalpts/2+1)=1;
Laplacian_fourier_regularized(yevalpts/2+1,xevalpts/2+1)=1;


psi_fourier=-w_fourier./Laplacian_fourier_regularized; %

%define derivatives in real space (use 2/3 rule for cutting out unphysical
%growth of waves from non-linearity of equation)
%{
v=real(ifft2(two_thirds_rule(-partial_x_fourier.*psi_fourier))); 
u=real(ifft2(two_thirds_rule(partial_y_fourier.*psi_fourier)));
wx=real(ifft2(two_thirds_rule(kx.*w_fourier)));
wy=real(ifft2(two_thirds_rule(ky.*w_fourier)));
laplace_w=real(ifft2(Laplacian_fourier.*w_fourier));
%}

%define derivatives in real space without cutting off any frequencies
v=real(ifft2(-partial_x_fourier.*psi_fourier)); %y velocity
u=real(ifft2(partial_y_fourier.*psi_fourier)); %x velocity
wx=real(ifft2(kx.*w_fourier)); %vorticity x-derivative
wy=real(ifft2(ky.*w_fourier)); %vorticity y-derivative
laplace_w=real(ifft2(Laplacian_fourier.*w_fourier)); %laplacian of vorticity

%solve NS equation for time variable
wdot=(1/Re*laplace_w-(u.*wx+v.*wy)); %This is the Navier-Stokes equations!

%convert back to vector and send data back to the cpu
wdot_vec=double(gather(wdot(:)));
end
function var_hat=two_thirds_rule(var_hat)
%gets rid of the highest 2/3 frequency data because these modes tend to go
%unstable due to non-linear interactions with other un-modeled modes
sz=size(var_hat);
yi=1+round((sz(1)-1)/2-((sz(1)-1)/2-1)/3);
ye=1+round((sz(1)-1)/2+((sz(1)-1)/2-1)/3);
xi=1+round((sz(2)-1)/2-((sz(2)-1)/2-1)/3);
xe=1+round((sz(2)-1)/2+((sz(2)-1)/2-1)/3);
var_hat(yi:ye,xi:xe)=zeros(ye-yi+1,xe-xi+1);
end

