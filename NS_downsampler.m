newsize = 64;
sz=size(w_out_vec);
oldsize = sqrt(sz(2));
times = sz(1);
batchsize = prod(sz(3:end));
u = reshape(w_out_vec,times,oldsize,oldsize,[]);
u=permute(u,[4 2 3 1]);

grid = linspace(1,oldsize,newsize);
%{
dsModetensor = zeros(newsize,newsize,newsize^2);
Modetensormat = reshape(Modetensor,oldsize,oldsize,[]);
for i=1:newsize^2
dsModetensor(:,:,i) = interp2(Modetensormat(:,:,i),grid',grid,'cubic');
end
Modetensor=dsModetensor;
%}
dsu = zeros(batchsize,newsize,newsize,times);
for i=1:batchsize
 for j=1:times
        dsu(i,:,:,j) = interp2(squeeze(u(i,:,:,j)),grid',grid,'cubic');
     
 end
end
u=dsu;

