% THIS FUNCTION CALCULATES THE INVERSE OF F
function F_inverse = F_inverse_question_2(u, beta, mu)
    F_inverse = mu + beta*log(1./(log(1./u)));
end