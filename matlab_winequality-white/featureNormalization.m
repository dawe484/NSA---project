function [X, ni, sigma] = featureNormalization(X)

    ni = mean(X);
    sigma = std(X);
    
    m=size(X,1);
    
    A=repmat(ni,m,1); %zkopiruje radek matice pod sebe
    B=repmat(sigma,m,1);
    
    X_norm=(X-A)./B;
    
    X = X_norm;
    
    %min(X_norm)
    %max(X_norm)

end