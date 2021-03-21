figure(5)
loglog(indfourier,[1; 1-cumsum(wsort.^2)/sum(wsort.^2)],'--k',indsvd2,[1; 1-cumsum(Energy2)/sum(Energy2)],'r','LineWidth',2)
hold on
loglog(indsvd3,[1; 1-cumsum(Energy3)/sum(Energy3)],'b','LineWidth',2)
loglog(indsvd4,[1; 1-cumsum(Energy4)/sum(Energy4)],'g','LineWidth',2)
loglog(indsvd5,[1; 1-cumsum(Energy5)/sum(Energy5)],'y','LineWidth',2)
loglog(indsvd6,[1; 1-cumsum(Energy6)/sum(Energy6)],'k','LineWidth',2)
axis([1 10^4 10^-6 1])
set(0,'defaultAxesFontSize',30)
xlabel('Mode # (Energy Sorted)')
ylabel('Residual Energy')
legend('Fourier Decomposition','POD Batch size: 16','POD Batch size: 81','POD Batch size: 256','POD Batch size: 625','POD Batch size: 1296')


