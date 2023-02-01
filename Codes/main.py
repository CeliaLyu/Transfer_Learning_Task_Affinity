from utils import *

data_type = 'synthetic'
# data_type = 'weather'

'''
    Simulate data
'''

if data_type == 'synthetic':

    t = np.linspace(0, 100, int(1e+4), endpoint=False)  # time
    freq = np.linspace(1, 0.1, 10)  # cosine frequency
    w = 2 * np.pi * freq

    dat = []  # list of ten datasets
    for i in range(len(freq)):
        np.random.seed(i)
        cos = np.cos(w[i] * t)
        noise = np.random.normal(scale=.1*(i+1), size=len(t))
        trend = .05 * t * i**2
        dat.append(cos + noise + trend)  # cosine with noise and trend

    # plot data in the first 500 periods
    fig, axes = plt.subplots(2, 5,  figsize=(18, 9))
    for i in range(len(dat)):
        ax = axes.flat[i]
        ax.plot(dat[i][:500])
        ax.set_ylim(-2, 25)
        ax.set_xlabel('time period')
        ax.set_title('Dataset {}'.format(i+1))
    # plt.suptitle("Synthetic data in the first 500 periods", fontsize=16)
    plt.subplots_adjust(hspace=.3)
    plt.tight_layout(pad=2)
    plt.savefig('Tex/figures/syn_data.png', dpi=600)
    plt.show(block=True)


'''
    Weather data
'''
if data_type == 'weather':

    dat = []
    weather = pd.read_csv('Data/dew_point_temp.csv')
    states = weather.columns[2:]
    for state in states:
        dat.append(weather[state][:10000].to_numpy())  # keep the first 10000 observations

    # plot data in the first 1500 periods
    fig, axes = plt.subplots(5, 2, figsize=(18, 8))
    for i in range(len(dat)):
        ax = axes.flat[i]
        ax.plot(dat[i][:1500], label=states[i], )
        ax.set_ylim(-20, 90)
        ax.set_ylabel('dew point')
        ax.legend(fontsize=20)
    plt.subplots_adjust(hspace=.3)
    plt.tight_layout(pad=2)
    plt.savefig('Tex/figures/weather_data.png', dpi=600)
    plt.show(block=True)


'''
    Train base model
'''

rnn_module = "srnn"
# rnn_module = "lstm"

optimizer_name = 'Adam'
scheduler_name = 'MultiStepLR'

if data_type == 'synthetic':
    window_size = 20
if data_type == 'weather':
    window_size = 20

train_base_idx = 0

batch_size = 32
input_dim = 1
hidden_dim = 40
out_dim = 1
num_layers = 2
num_dir = 1  # prediction direction, one-sided or two-sided
if data_type == 'synthetic':
    num_epochs = 101
if data_type == 'weather':
    num_epochs = 101
learning_rate = 0.0001


# train test split
split_time = int(.9 * len(dat[0]))
x_train = dat[0][:split_time]
x_test = dat[0][split_time:]

train_loader = Windowed_Dataset(x_train, window_size=window_size, stride=1, batch_size=batch_size, shuffle=True)
test_loader = Windowed_Dataset(x_test, window_size=window_size, stride=1, batch_size=batch_size, shuffle=False)


torch.manual_seed(2022)
model = RNN(input_dim, hidden_dim, num_layers, out_dim, num_dir, rnn_module)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = make_optimizer(optimizer_name, model, lr=learning_rate)
scheduler = make_scheduler(scheduler_name, optimizer, milestones=[50], factor=0.1)
best_loss = 1e+100

last_exp2_loss_train = []
last_exp2_loss_test = []
for epoch in range(num_epochs):
    running_train_loss = train(model, train_loader, batch_size, optimizer, criterion, scheduler)
    running_test_loss = test(model, test_loader, batch_size, criterion)

    if epoch % 5 == 0:
        print('Epoch {} : Training loss is {:.4f}'.format(epoch, running_train_loss))
        print('Epoch {} : Test loss is {:.4f}'.format(epoch, running_test_loss))

    if epoch in [20, 50, 100]:
        last_exp2_loss_train.append(running_train_loss)
        last_exp2_loss_test.append(running_test_loss)

    if best_loss > running_test_loss:
        torch.save(model, 'Codes/ckpt/{}_ckpt_time_series_{}_trainw_{}.pth'.format(data_type, rnn_module, train_base_idx))
        best_loss = running_test_loss


# prediction the input series (dataset 0)
predict_idx = 0  # predict dataset k using model trained with dataset 0
model_path = 'Codes/ckpt/{}_ckpt_time_series_{}_trainw_{}.pth'.format(data_type, rnn_module, train_base_idx)
rnn_result = prediction(model_path, dat[predict_idx], window_size)

# plot the first 5000 predicted and true series value
fig, axes = plt.subplots(2, 1, figsize=(12, 3))
axes[0].plot(dat[predict_idx][window_size:window_size+5000], label='observed')
axes[1].plot(rnn_result[:5000], label='predicted')
axes[0].set_title('observed')
axes[1].set_title('predicted')
plt.subplots_adjust(hspace=.3)
plt.tight_layout(pad=2)
plt.savefig('Tex/figures/{}_{}_trainw_{}_pred_{}.png'.format(data_type, rnn_module, train_base_idx, predict_idx), dpi=600)
plt.show(block=True)


'''
    Fisher distance
'''

# https://github.com/pytorch/captum/issues/564
torch.backends.cudnn.enabled = False

# fisher distance between dataset 0 and other datasets
model = torch.load(model_path)
model = model.to(device)

total_time = int(len(dat[train_base_idx]))
batch_size_test = total_time - window_size

x_dat = []  # dataloaders of all datasets
for i in range(len(dat)):
    x = dat[i][:total_time]  # all data
    x = Windowed_Dataset(x, window_size=window_size, stride=1, batch_size=batch_size_test, shuffle=False)  # only 1 batch
    x_dat.append(x)

# distance between dataset 0 and other datasets
lis_dis = []
for i in range(len(x_dat)):
    lis_dis.append(compute_distance(model, x_dat[0], x_dat[i], batch_size_test, batch_size_test))

# plot fisher distance between dataset 0 and other datasets
plt.figure(figsize=(8, 5))
plt.plot(lis_dis)
plt.ylabel('Distance from dataset {}'.format(train_base_idx))
if data_type == 'weather':
    plt.xticks(range(len(x_dat)), states)
    plt.xlabel("States")
if data_type == 'synthetic':
    plt.xticks(range(len(x_dat)), list(range(len(x_dat))))
    plt.xlabel('Initialization order of generated datasets')
plt.savefig('Tex/figures/fisher_dist_{}_{}_trainw_{}_.png'.format(rnn_module, data_type, train_base_idx, rnn_module), dpi=600)
plt.show(block=True)


'''
    Experiment 1
'''

batch_size_train = split_time - window_size
batch_size_test = (total_time - split_time) - window_size

# train loaders of each dataset, only 1 batch in each loader
x_dat_train = []
for i in range(len(dat)):
    x = dat[i][:split_time]
    x = Windowed_Dataset(x, window_size=window_size, stride=1, batch_size=batch_size_train, shuffle=False)
    x_dat_train.append(x)
# test loaders of each dataset, only 1 batch in each loader
x_dat_test = []
for i in range(len(dat)):
    x = dat[i][split_time:]
    x = Windowed_Dataset(x, window_size=window_size, stride=1, batch_size=batch_size_test, shuffle=False)
    x_dat_test.append(x)


lis_num_epochs = [0, 1, 2, 3, 4]  # number of  gradient update, one epoch is one batch
lis_loss_all_grad_train = []
lis_loss_all_grad_test = []

model = torch.load(model_path)
model = model.to(device)

for i in range(len(x_dat_train)):

    model_original = model  # model trained with dateset 0, before update
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_original.parameters(), lr=learning_rate)
    scheduler = make_scheduler(scheduler_name, optimizer, milestones=[50], factor=0.1)

    iter = 0
    for e in lis_num_epochs:  # i.e., x order of gradient
        while iter <= e:
            running_train_loss = train(model_original, x_dat_train[i], batch_size_train, optimizer, criterion, scheduler)
            running_test_loss = test(model_original, x_dat_test[i], batch_size_test, criterion)

            iter += 1
            lis_loss_all_grad_train.append(running_train_loss)
            lis_loss_all_grad_test.append(running_test_loss)

lis_loss_all_grad_train = np.reshape(lis_loss_all_grad_train, (len(dat), len(lis_num_epochs)))
lis_loss_all_grad_train = lis_loss_all_grad_train.T
lis_loss_all_grad_test = np.reshape(lis_loss_all_grad_test, (len(dat), len(lis_num_epochs)))
lis_loss_all_grad_test = lis_loss_all_grad_test.T


fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for i in range(len(lis_loss_all_grad_train)):
    axes[0].plot(lis_loss_all_grad_train[i], label='The {}th order gradient'.format(i))
    axes[1].plot(lis_loss_all_grad_test[i], label='The {}th order gradient'.format(i))
    axes[0].set_title('Training loss')
    axes[1].set_title('Testing loss')
    axes[0].legend()
    axes[1].legend()
    if data_type == 'weather':
        for i in range(2):
            axes[i].set_xticks(range(len(x_dat)), states)
            axes[i].set_xlabel("States")
    if data_type == 'synthetic':
        for i in range(2):
            axes[i].set_xticks(range(len(x_dat)), list(range(len(x_dat))))
            axes[i].set_xlabel('Initialization order of generated datasets')
plt.tight_layout(pad=2)
plt.savefig('Tex/figures/grad_update_{}_{}_trainw_{}.png'.format(data_type, rnn_module, train_base_idx), dpi=600)
plt.show(block=True)


'''
Experiment2
'''

'''
Derive the converged models
'''

num_epochs = 200
learning_rate = 1e-3

for m in range(1):  # if take average: 10, if not: 1
    for train_base_idx in range(len(dat)):  # train with different dataset

        x_train = dat[train_base_idx][:split_time]
        x_test = dat[train_base_idx][split_time:]

        train_loader = Windowed_Dataset(x_train, window_size=window_size, stride=1, batch_size=batch_size, shuffle=True)
        test_loader = Windowed_Dataset(x_test, window_size=window_size, stride=1, batch_size=batch_size, shuffle=False)

        torch.manual_seed(2022)
        model = RNN(input_dim, hidden_dim, num_layers, out_dim, num_dir, rnn_module)
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = make_optimizer(optimizer_name, model, lr=learning_rate)
        scheduler = make_scheduler(scheduler_name, optimizer, milestones=[50], factor=0.1)
        best_loss = 1e+100
        
        for epoch in range(num_epochs):
            running_train_loss = train(model, train_loader, batch_size_train, optimizer, criterion, scheduler)
            running_test_loss = test(model, test_loader, batch_size_test, criterion)

            if epoch % 10 == 0:
                print('Epoch {} : Training loss is {:.4f}'.format(epoch, running_train_loss))
                print('Epoch {} : Test loss is {:.4f}'.format(epoch, running_test_loss))

            if best_loss > running_test_loss:
                torch.save(model, 'Codes/ckpt/exp2_{}_ckpt_time_series_{}_trainw_{}.pth'.format(data_type, rnn_module, train_base_idx))
                # torch.save(model, 'ckpt/{}_ckpt_time_series_{}_train{}_round{}.pth'.format(data_type, rnn_module, k, m))  # if take average
                best_loss = running_test_loss


'''
    Performance on original dataset
'''

test_loss = []
for dat_idx in range(len(dat)):
    # test_loss_i = []  # if take average
    for k in range(1):  # if take average: 10, if not: 1
        # model = torch.load('ckpt/{}_ckpt_time_series_{}_train{}_round{}.pth'.format(data_type, rnn_module, i, k))  # if take average
        model_path = 'Codes/ckpt/exp2_{}_ckpt_time_series_{}_trainw_{}.pth'.format(data_type, rnn_module, dat_idx)
        model = torch.load(model_path)
        model = model.to(device)

        x_test = dat[0][split_time:]  # dataset 0, target dataset
        test_loader = Windowed_Dataset(x_test, window_size=window_size, stride=1, batch_size=batch_size, shuffle=False)
        criterion = nn.MSELoss()
        running_test_loss = test(model, test_loader, batch_size, criterion)
    #     test_loss_i.append(running_test_loss)  # if take average
    # test_loss.append(np.mean(test_loss_i))  # if take average
    test_loss.append(running_test_loss)


plt.plot(test_loss)
plt.ylabel('Test loss on the target dataset')
if data_type == 'weather':
    plt.xticks(range(len(x_dat)), states)
    plt.xlabel("States")
if data_type == 'synthetic':
    plt.xticks(range(len(x_dat)), list(range(len(x_dat))))
    plt.xlabel('Initialization order of generated dataset that the model is trained on')
plt.savefig('Tex/figures/exp2_performance_10datasets_{}_{}_{}.png'.format(rnn_module, data_type, train_base_idx), dpi=600)
plt.show(block=True)


'''
    Fisher distances
'''

'''one round: using result from round 0'''
lis_loss_dat0 = []
fig, axes = plt.subplots(5, 2, figsize=(8, 14))
for k in range(len(dat)):
    model_path = 'Codes/ckpt/exp2_{}_ckpt_time_series_{}_trainw_{}.pth'.format(data_type, rnn_module, k)
    # model = torch.load('ckpt/{}_ckpt_time_series_{}_train{}_round2.pth'.format(data_type, rnn_module, k))  # if take average
    model = torch.load(model_path)
    model = model.to(device)

    # distance between dataset 0 and other datasets
    lis_dis = []
    for i in range(len(x_dat)):
        lis_dis.append(compute_distance(model, x_dat[k], x_dat[i], batch_size_test, batch_size_test))
    lis_loss_dat0.append(lis_dis[0])

    x = list(range(len(x_dat)))
    ax = axes.flat[k]
    ax.plot(x, lis_dis)
    if data_type == 'synthetic':
        plt.xticks(range(len(x_dat)), list(range(len(x_dat))))
        ax.set_title('Train with ds {}'.format(k), fontsize=10)
    if data_type == 'weather':
        ax.set_xticks(range(len(x_dat)), states)
        ax.set_title('Train with ds {}'.format(states[k]), fontsize=10)
    ax.ticklabel_format(scilimits=(0, 3), style='sci', axis='y')
plt.subplots_adjust(wspace=.3, hspace=.5)
plt.savefig('Tex/figures/exp2_{}_{}_fisher_distance_10datasets.png'.format(data_type, rnn_module), dpi=600)
plt.show(block=True)



'''average for 10 rounds (ckpt/Old)'''
'''
for i in range(10):’

    # distance between dataset i and other datasets
    lis_dis = []
    for j in range(len(x_dat)):
        dis = []
        for k in range(10):
            model = torch.load('Codes/ckpt/Old/ckpt_time_series_srnn_train{}_round{}.pth'.format(i,k))
            model = model.to(device)
            dis.append(compute_distance(model, x_dat[i], x_dat[j],batch_size_test))
        lis_dis.append(np.mean(dis))
    lis_dis

    x = list(range(len(x_dat)))

    plt.figure(figsize=(10,6))
    # plotting the points
    plt.plot(x, lis_dis)
    # naming the x axis
    plt.xlabel('x - Initialization order of generated datasets')
    # naming the y axis
    plt.ylabel('y - Average Distance from the ith dataset used to generate the model')
    # function to show the plot
    plt.suptitle('model trained on dataset: {}'.format(i))
    plt.savefig('Tex/figures/{}_fisherdistance_average_dataset{}.png'.format{data_type,i}, dpi=600)
    plt.show(block=True)
'''

'''
    Down sampling dataset 0
'''

stride_list = [200, 100, 20, 1]

# down sampled dataset 0, train
x_dat_train_2 = []
for stride in stride_list:
    x = dat[0][:split_time]
    batch_size_train_2 = int(split_time/stride)
    x = Windowed_Dataset(x, window_size=window_size, stride=stride, batch_size=batch_size_train_2, shuffle=False)
    x_dat_train_2.append(x)

# down sampled dataset 0, test
x_dat_test_2 = []
for stride in stride_list:
    x = dat[0][split_time:]
    batch_size_test_2 = int((total_time - split_time) / stride)
    x = Windowed_Dataset(x, window_size=window_size, stride=stride, batch_size=batch_size_test_2, shuffle=False)
    x_dat_test_2.append(x)


'''
use test loss or fisher distance to select the best model
'''

model_selection = 'fisher_distance'  # use fisher distance to select the best model
model_selection = 'loss'  # use test loss to select the best model
window_number_list = [int(len(dat[0])/stride) for stride in stride_list]

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
best_models = []

for i in range(len(x_dat_test_2)):  # down sampled dataset 0
    best_loss = 1e+100  # loss or fisher distance
    best_model = 0
    loss = []  # loss or fisher distance
    for j in range(1, 10):  # models trained with 10 original datasets
        model_path = 'Codes/ckpt/exp2_{}_ckpt_time_series_{}_trainw_{}.pth'.format(data_type, rnn_module, j)
        model = torch.load(model_path)
        model = model.to(device)
        criterion = nn.MSELoss()

        batch_size_train_2 = int(split_time/stride_list[i])
        if model_selection == 'loss':
            running_test_loss = test(model, x_dat_train_2[i], batch_size_train_2, criterion)  # 'test' the training data
            loss.append(running_test_loss)
        if model_selection == 'fisher_distance':
            running_test_loss = compute_distance(model, x_dat_train_2[i], x_dat[j], batch_size_test, batch_size_train_2)
            loss.append(running_test_loss)
        if best_loss > running_test_loss:
            best_loss = running_test_loss
            best_model = j
            torch.save(model, 'Codes/ckpt/exp2_best_model_{}_{}_ckpt_time_series_{}_stride_{}.pth'.format(
                                                            model_selection, data_type, rnn_module, stride_list[i]))
    best_models.append(best_model)

    ax = axes.flat[i]
    ax.plot(loss, label='best model: {}'.format(best_model))
    if data_type == 'weather':
        ax.set_xticks(range(len(x_dat)-1), states[1:])
        ax.set_xlabel("States")
    if data_type == 'synthetic':
        ax.set_xticks(range(len(x_dat)-1), list(range(1, len(x_dat))))
        ax.set_xlabel('Order of generated datasets')
    ax.legend()
    ax.set_title('Window number: {}'.format(window_number_list[i]))
    ax.ticklabel_format(scilimits=(0, 3), style='sci', axis='y')
plt.subplots_adjust(hspace=.3)
plt.tight_layout(pad=2)
plt.savefig('Tex/figures/exp_down_sample_model_select_{}_{}_{}.png'.format(model_selection, rnn_module, data_type), dpi=600)
plt.show(block=True)    
print(best_models)



'''
Generate dataframe(row is dataset 0 size, column is epoch) with the best model found
'''

lis_num_epochs_2 = [20, 50, 100]
train_loss = []
test_loss = []

for i in range(len(x_dat_train_2)):  # 4 down sampled dataset 0

    # best_model = best_models_f[i]
    # model = torch.load('ckpt/exp2_best_model_f_{}_ckpt_time_series_{}_stride_{}.pth'.format(data_type, rnn_module, best_model),map_location=torch.device('cpu') if (torch.cuda.is_available()==False)else None)
    # model = torch.load('ckpt/{}_ckpt_time_series_{}_dat{}.pth'.format(data_type, rnn_module, i),map_location=torch.device('cpu') if (torch.cuda.is_available()==False)else None)
    for e in lis_num_epochs_2:
        model_path = 'Codes/ckpt/exp2_best_model_{}_{}_ckpt_time_series_{}_stride_{}.pth'.format(
                                                                model_selection, data_type, rnn_module, stride_list[i])
        model = torch.load(model_path)
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = make_optimizer(optimizer_name, model, lr=learning_rate)
        scheduler = make_scheduler(scheduler_name, optimizer, milestones=[50], factor=0.1)
        running_train_loss_m = 0
        best_loss = 1e+10

        iter = 0

        while iter <= e:
            batch_size_train_2 = int(split_time / stride_list[i])
            running_train_loss_m = train(model, x_dat_train_2[i], batch_size_train_2, optimizer, criterion, scheduler)

            batch_size_test_2 = int((total_time - split_time) / stride_list[i])
            running_test_loss_m = test(model, x_dat_test_2[i], batch_size_test_2, criterion)

            iter += 1

        train_loss.append(running_train_loss_m)
        test_loss.append(running_test_loss_m)

train_loss = np.reshape(train_loss, (len(stride_list), len(lis_num_epochs_2)))
test_loss = np.reshape(test_loss, (len(stride_list), len(lis_num_epochs_2)))
train_loss[len(stride_list)-1, :] = last_exp2_loss_train  # full target data with no transfer learning
test_loss[len(stride_list)-1, :] = last_exp2_loss_test

## row is dataset 0 size (total number of windows), column is epoch
window_number_list = [int(len(dat[0])/stride) for stride in stride_list]
train_loss_dt = pd.DataFrame(train_loss, columns=lis_num_epochs_2, index=window_number_list)
test_loss_dt = pd.DataFrame(test_loss, columns=lis_num_epochs_2, index=window_number_list)
# train_loss_dt.to_csv('train_loss_f.csv')
# test_loss_dt.to_csv('test_loss_f.csv')

print(train_loss_dt.to_latex())
print(test_loss_dt.to_latex())

dfi.export(train_loss_dt, 'Tex/figures/exp2_down_sample_{}_{}_{}_train.png'.format(model_selection, data_type, rnn_module))
dfi.export(test_loss_dt, 'Tex/figures/exp2_down_sample_{}_{}_{}_test.png'.format(model_selection, data_type, rnn_module))
