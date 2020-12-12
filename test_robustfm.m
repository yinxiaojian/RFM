noise = 0;

% add noise to test the robust
if noise > 0
    [out_train_X, out_train_Y] = add_noise(train_X, train_Y, noise);
else
    out_train_X = train_X;
    out_train_Y = train_Y;
end

% load data
training.train_X = out_train_X;
training.train_Y = out_train_Y;

validation.test_X = test_X;
validation.test_Y = test_Y;

% pack paras
% pars.task = 'regression';
pars.task = 'binary-classification';
% pars.task = 'multi-classification';
pars.iter_num = 1;
pars.epoch = 50;
pars.minibatch = 10;

% initial model
[~, p] = size(train_X);
class_num = max(train_Y);
pars.p = p;

%% fm
rng('default');

pars.reg = 5e-2;
pars.factors_num = 10;
pars.epoch = 20;

pars.w0 = 0;
pars.W = zeros(1,p);
pars.V = 0.1*randn(p,pars.factors_num);

pars.learning_rate = 1e4;
pars.t0 = 1e5;

% for robust fm
pars.rob_alpha = 0.3;
pars.rob_beta = 0.3;

% for capped fm
pars.Z = zeros(p,p);
pars.alpha = 1e-2;
pars.beta = 1e-4;
pars.epsilon1 = 0.1;
pars.epsilon2 = 3;
pars.epsilon3 = 0.01;
pars.truncated_k = 10;
pars.minibatch = 1000;

disp('Training FM...')
[model_fm, metric_fm] = capped_fm(training, validation, pars);



