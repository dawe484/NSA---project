clear,clc,close all 
% MATLAB neural network back propagation code
% by AliReza KashaniPour & Phil Brierley
% first version 29 March 2006
% 2's edition 14 agust 2007
%
% This code implements the basic backpropagation of
% error learning algorithm. The network has tanh hidden  
% neurons and a linear output neuron.
%
% Upraveno pro n� p��klad - grafy, v�pisy, ...


%% Initializing
hidden_neurons = 10;
iterations = 100;
learn_rate = 0.1;

% nacteni trenovacich dat
load('data.mat'); %vytvori X Y

figure('units','normalized','outerposition',[0 0 1 1]);
set(0,'DefaultFigureWindowStyle','docked')

% vykresleni rozdeleni podle alkoholu a kvality
fixed_acidity = X(:,1);
alcohol = X(:,11); % 11. (posledni) sloupec z X
quality = Y(:,1); % 1. sloupec z Y

bad = quality < 5;
medium = quality == 5;
ok = quality == 6;
good = quality > 6;

hold on
figure(1)
set(1,'WindowStyle','docked');
plot( alcohol(bad), fixed_acidity(bad), 'bo' )
plot( alcohol(medium), fixed_acidity(medium), 'ko')
plot( alcohol(ok), fixed_acidity(ok), 'ro' )
plot( alcohol(good), fixed_acidity(good), 'go' )
% plot(alcohol, fixed_acidity, '.')
set(gca,'Color',[0.3 0.3 0.3]);
hold off

xlabel('alcohol')
ylabel('fixed acidity')
title('Alcohol � Fixed acidity = Quality')
lf1 = legend('bad < 5','medium = 5', 'ok = 6', 'good > 6');
vf1 = get(lf1,'title');
set(vf1,'string','Quality Legend');

% fprintf('Program paused. Press enter to continue.\n');
% pause;

% vykreslen� X-ov�ch dat v grafu
figure(2)
set(2,'WindowStyle','docked');
plot(X)
title('Vykreslen� dat')

% fprintf('Program paused. Press enter to continue.\n');
% pause;

% vykreslen� po�tu v�n dle kvality
count_3 = sum(sum(quality == 3));
count_4 = sum(sum(quality == 4));
count_5 = sum(sum(quality == 5));
count_6 = sum(sum(quality == 6));
count_7 = sum(sum(quality == 7));
count_8 = sum(sum(quality == 8));
count_9 = sum(sum(quality == 9));

count_quality = [3;4;5;6;7;8;9];
counts = [count_3, count_4, count_5, count_6, count_7, count_8, count_9];

figure(3)
set(3,'WindowStyle','docked');
bar(count_quality, counts, 'FaceColor',[0 .9 .9])
xlabel('quality');
ylabel('count');

% vykreslen� mno�stv� v�n dle obsahu alkoholu
% fprintf('Program paused. Press enter to continue.\n');
% pause;

count_a01 = sum(sum(alcohol < 8.5));
count_a02 = sum(sum(alcohol < 9) - sum(alcohol < 8.5));
count_a03 = sum(sum(alcohol < 9.5) - sum(alcohol < 9));
count_a04 = sum(sum(alcohol < 10) - sum(alcohol < 9.5));
count_a05 = sum(sum(alcohol < 10.5) - sum(alcohol < 10));
count_a06 = sum(sum(alcohol < 11) - sum(alcohol < 10.5));
count_a07 = sum(sum(alcohol < 11.5) - sum(alcohol < 11));
count_a08 = sum(sum(alcohol < 12) - sum(alcohol < 11.5));
count_a09 = sum(sum(alcohol < 12.5) - sum(alcohol < 12));
count_a10 = sum(sum(alcohol < 13) - sum(alcohol < 12.5));
count_a11 = sum(sum(alcohol < 13.5) - sum(alcohol < 13));
count_a12 = sum(sum(alcohol < 14) - sum(alcohol < 13.5));
count_a13 = sum(sum(alcohol < 14.5) - sum(alcohol < 14));

a_a = [8.5;9;9.5;10;10.5;11;11.5;12;12.5;13;13.5;14;14.5];
c_a = [count_a01, count_a02, count_a03, count_a04, count_a05, ...
    count_a06, count_a07, count_a08, count_a09, count_a10, count_a11, ...
    count_a12, count_a13];

figure(4);
set(4,'WindowStyle','docked');
bar(a_a, c_a, 'FaceColor',[0 .9 .9])
xlabel('alcohol');
ylabel('count');

% fprintf('Program paused. Press enter to continue.\n');
% pause;

% figure(5); % mus� b�t zakomentov�no, jinak nefunguje dock na corrplot
% set(5,'WindowStyle','docked');
corrplot(X) % chvilku trv� vykreslen�
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;

[X, ~, ~]=featureNormalization(X);

% rozd�len� dat na tr�novac� a testovac�
train_datax = X(1:floor(0.7*size(X, 1)), :); % 70 % tr�novac� data
test_datax = X(floor(0.7*size(X, 1))+1:end, :); % 30 % testovac� data

train_datay = Y(1:floor(0.7*size(Y, 1)), :);
test_datay = Y(floor(0.7*size(Y, 1))+1:end, :);

train_inp = train_datax; % train_inp = X;
train_out = train_datay; % train_out = Y;

% % vstupy
% ni_inp = mean(train_inp);
% sigma_inp = std(train_inp);
% train_inp = (train_inp(:,:) - ni_inp(:,1)) / sigma_inp(:,1);

% v�stupy
train_out = train_out';
ni_out = mean(train_out);
sigma_out = std(train_out);
train_out = (train_out(:,:) - ni_out(:,1)) / sigma_out(:,1);
train_out = train_out';

% na�ten� po�tu vstup� (kolik jich je celkem) - 4898 celkem vzork� v X
num_input = size(train_inp,1); % u� nen� 4898, ale pouze 70 % z toho = 3428
num_test_input = size(test_datax,1); % 1470

% p�id�n� bias ke vstup�m (nakonec)
bias = ones(num_input,1);
train_inp = [train_inp bias];

% na�ten� kolik je atribut� u vstupn�ch prom�nn�ch (+ bias = 12)
inputs = size(train_inp,2);

% p�id�n� tla��tka pokud bychom cht�li u�en� p�ed�asn� ukon�it
btnstop = uicontrol('Style','PushButton','String','Stop', 'Position', [5 5 70 20],'callback','earlystop = 1;'); 
earlystop = 0;

% nastaven� po��te�n�ch vah na random hodnoty
weight_input_hidden = (randn(inputs,hidden_neurons) - 0.5)/10; % v�hy ...
% mezi vstupn� vrtvou a skrytou vrtsvou
weight_hidden_output = (randn(1,hidden_neurons) - 0.5)/10; % v�hy ...
% mezi skrytou vrstvou a v�stupn� vrstvou

%% Learning

% cyklus doby u�en� (jak dlouho se bude u�it)
for iter = 1:iterations 
    
    reg_par = learn_rate / 10; % regulacni parametr - v�choz� 0.01
    
    % cyklus tolikr�t kolik je vstup�
    for j = 1:num_input
        
        % v�b�r n�hodn�ho vstupu
        inpnum = round((rand * num_input) + 0.5);
        if inpnum > num_input
            inpnum = num_input;
        elseif inpnum < 1
            inpnum = 1;    
        end
       
        % nastaven� dan�ho vstupu
        this_inp = train_inp(inpnum,:);
        act = train_out(inpnum,1);
        
        % v�po�et chyby na�eho n�hodn�ho vzorku
        hval = (tanh(this_inp*weight_input_hidden))'; % sigmoid fce ...
        % nahrazena tanh fc� -> d�v� lep�� v�sledky
        pred = hval'*weight_hidden_output';
        error = pred - act;
        
        % nastaven� v�hy mezi skrytou a v�stupn� vrstvou
        delta_HO = error.*reg_par .*hval;
        weight_hidden_output = weight_hidden_output - delta_HO';

        % nastaven� vah mezi vstupn� a skrytou vrstvou
        delta_IH = learn_rate.*error.*weight_hidden_output'.*(1-(hval.^2))*this_inp;
        weight_input_hidden = weight_input_hidden - delta_IH';
        
    end
    
    % v�pis chyby s�t� po ka�d� iteraci
    pred = weight_hidden_output*tanh(train_inp*weight_input_hidden)';
    error = pred' - train_out;
    err(iter) =  (sum(error.^2))^0.5;
    
    figure(6);
    set(6,'WindowStyle','docked');
    plot(err)
    
    % zastaven� u�en� po stisknut� tla��tka STOP
    if earlystop
        fprintf('n�siln� ukon�eno v iteraci: %d\n',iter); 
        break 
    end 

    %zastaven� u�en� pokud je chyba dostate�n� mal�
    if err(iter) < 0.001
        fprintf('konvergovalo v iteraci: %d\n',iter);
        break 
    end
       
end

%% Finish
fprintf('u�en� ukon�eno po %d iterac�ch\n',iter);
count_quality = (train_out* sigma_out(:,1)) + ni_out(:,1);
b = (pred'* sigma_out(:,1)) + ni_out(:,1);
procent = (b*100)./count_quality;
err_proc = abs(100-procent)

% zobrazen� kvality, predikce, chyby a procento �sp�chu
act_pred_err_proc = [count_quality b b-count_quality err_proc]

error_sum = sum(err_proc)/num_input % misclassification error (mse)
train_succ = 100-error_sum

%% Testing

train_test = test_datax;

ni_test = mean(train_test);
sigma_test = std(train_test);
train_test = (train_test(:,:) - ni_test(:,1)) / sigma_test(:,1);
bias = ones(num_test_input,1);
train_test = [train_test bias];

pred = weight_hidden_output*tanh(train_test*weight_input_hidden)';

figure(7)
plot3(train_test(:,1),train_test(:,11),act_pred_err_proc((1:(size(train_test,1))),2),'sb','linewidth',1)
grid on,title('P�ibli�n� v�sledek (pomoc� neuronov� s�t�)');
figure(8)
plot3(test_datax(:,1),test_datax(:,11),test_datay, 'sg','linewidth',1)
grid on,title('Origin�ln� v�sledek');

