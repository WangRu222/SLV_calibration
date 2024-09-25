function E= stackvector(Y,para)
%para determines the direction of stack. If S direction, para=1; if z
%direction, para=2
if para==1
    E=repmat(Y,1,length(Y));
elseif para==2
    E=repmat(Y,length(Y),1);
end
 E=E';