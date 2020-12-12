function [U1,S1]=Incsvd(U,S,A)
%ע����Ҫ��������ֵ�ֽ⣬����������ֵ�ֽ�
%���㷨����Ҫ������ʵ��U1*S1*U1'=U*S*U'+AA'
%���U=0,ֻ��Ҫ��AA'��������ֵ�ֽ⣬���ǵ�AA'�γɵľ���ϴ����Զ�A��������ֵ�ֽ⣺A=U2*S2*U2',AA'=U2*S2*S2'*U2',S1=S2*S2'����߰�0�ȥ���ˣ���ʡ�洢�ռ䣬U2=U2(:,size(S2*S2'))
%���U!=0,����IncSVD��˼�룬�ȼ���U'*A,��Ϊ�е�ʱ��س���UU'I���������������ڽ�������ʱI-UU'�������0����ʱ����ɽϴ���鷳����Ҫ����������ų���
%�����UU'=I,��ʱI-UU'=0,���ʱ���������RA
%�����UU'!=I,����A-U(U'A)���õ�P,Ȼ�����[U'A RA]'*[U'A RA]

flag=0;
%����P��RA
if U==0
    [U1 S2 V1]=svd(full(A),'econ');
    S1=S2'*S2;
    %U1=U1(:,1:size(S1,1));
    return;
else
    UA=U'*A; %֮�����������Ƕ����˷����ٵ�ԭ��
end

Ra=A-U*UA;
if(Ra<0.0000001)  %�е�ʱ��UU'=Iʱ������������Ľ��tmp������0����߾���Ϊ���ų��������
    if sum(sum(U'*U-eye(size(U,2))))<0.0000001
        flag=1;
    end
end
if flag==0
    if issparse(Ra)~=0
        Ra=full(Ra);
    end
    P=orth(Ra);

    RA=P'*Ra;
    Ktmp=[UA;RA]*[UA;RA]';
else
    Ktmp=UA*UA';
end

%�˴����ù�ʽ(6)����K

[rowlen,collen]=size(Ktmp);
[Srowlen,Scollen]=size(S);
tmpS=[[S zeros(Srowlen,collen-Scollen)];zeros(rowlen-Srowlen,Scollen),zeros(collen-Scollen)];
K=tmpS+Ktmp;

%��ߵ�U2,S2,V2�ֱ��������е�U',S'��V'
[U2,S2]=eig(K);
if flag==0
    U1=[U P]*U2;
else
    U1=U*U2;
end
S1=S2;
end
