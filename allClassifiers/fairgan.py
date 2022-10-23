from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
from tqdm.notebook import tqdm as tqdm
from torch.autograd import Variable
import numpy as np
from torch.distributions.one_hot_categorical import OneHotCategorical
torch.backends.cudnn.benchmark = True

# https://github.com/rcamino/multi-categorical-gans/tree/master/multi_categorical_gans/methods/medgan

class Encoder(nn.Module):
    def __init__(self, input_size, latent_size, layer_sizes=[128]):
        super().__init__()
        self.MLP = nn.Sequential()

        layer_sizes = [input_size] + layer_sizes + [latent_size]

        for i in range(len(layer_sizes)):
            if i not in [len(layer_sizes) - 1]:
                self.MLP.add_module(str(i), nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                self.MLP.add_module(str(i)+"act", nn.Tanh())
        #self.MLP.add_module("last_layer", nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    def forward(self, x):        
        return self.MLP(x)

class SingleOutput(nn.Module):
    def __init__(self, previous_layer_size, output_size, activation=None):
        super(SingleOutput, self).__init__()
        if activation is None:
            self.model = nn.Linear(previous_layer_size, output_size)
        else:
            self.model = nn.Sequential(nn.Linear(previous_layer_size, output_size), activation)

    def forward(self, hidden, training=False, temperature=None):
        return self.model(hidden)

class MultiCategorical(nn.Module):
    def __init__(self, input_size, variable_sizes):
        super(MultiCategorical, self).__init__()

        self.output_layers = nn.ModuleList()
        self.output_activations = nn.ModuleList()

        for i, variable_size in enumerate(variable_sizes):
            self.output_layers.append(nn.Linear(input_size, variable_size))
            self.output_activations.append(CategoricalActivation())

    def forward(self, inputs, training=True, temperature=None, concat=True):
        outputs = []
        for output_layer, output_activation in zip(self.output_layers, self.output_activations):
            logits = output_layer(inputs)
            output = output_activation(logits, training=training, temperature=temperature)
            outputs.append(output)
        if concat:
            return torch.cat(outputs, dim=1)
        else:
            return outputs

class CategoricalActivation(nn.Module):
    def __init__(self):
        super(CategoricalActivation, self).__init__()

    def forward(self, logits, training=True, temperature=None):
        # gumbel-softmax (training and evaluation)
        if temperature is not None:
            return F.gumbel_softmax(logits, hard=not training, tau=temperature)
        # softmax training
        elif training:
            return F.softmax(logits, dim=1)
        # softmax evaluation
        else:
            return OneHotCategorical(logits=logits).sample()

class Decoder(nn.Module):
    def __init__(self, latent_size, output_size, layer_sizes=[128]):
        super().__init__()
        self.MLP = nn.Sequential()
        #input_size = latent_size
        #self.MLP.add_module("input", nn.Linear(input_size, layer_sizes[0]))
        layer_sizes = [latent_size] + layer_sizes
        for i in range(len(layer_sizes)):
            if i not in [len(layer_sizes) - 1]:
                self.MLP.add_module(str(i), nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                self.MLP.add_module(str(i)+"act", nn.Tanh())
                #self.MLP.add_module(str(i)+"bn", nn.BatchNorm1d(layer_sizes[i + 1], 0.8))
        self.MLP.add_module("last_layer", nn.Linear(layer_sizes[len(layer_sizes) - 1], output_size))
        # Inputs normalized between 0 and 1
        self.MLP.add_module("last_act", nn.Sigmoid())
        # if type(output_size) is int:
        #     self.output_layer = SingleOutput(layer_sizes[-1], output_size, activation=nn.Sigmoid())
        # elif type(output_size) is list:
        #     self.output_layer = MultiCategorical(layer_sizes[-1], output_size)
        
    def forward(self, x):
        return self.MLP(x)

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

class SourceDiscriminator(nn.Module):
    def __init__(self, input_size, layer_sizes=[256,128]):
        super().__init__()
        self.MLP = nn.Sequential()
        layer_sizes = [input_size] + layer_sizes

        for i in range(len(layer_sizes)):
            if i not in [len(layer_sizes) - 1]:
                self.MLP.add_module(str(i), nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                self.MLP.add_module(str(i)+"act", nn.LeakyReLU(0.1))
                #self.MLP.add_module(str(i)+"bn", nn.BatchNorm1d(layer_sizes[i+1], 0.8))
        self.MLP.add_module("last_layer", nn.Linear(layer_sizes[-1], 1))
        self.MLP.add_module("last_act", nn.Sigmoid())
    
    def minibatch_averaging(self, inputs):
        """
        This method is explained in the MedGAN paper.
        """
        mean_per_feature = torch.mean(inputs, 0)
        mean_per_feature_repeated = mean_per_feature.repeat(len(inputs), 1)
        return torch.cat((inputs, mean_per_feature_repeated), 1)
        
    def forward(self, x):
        #x = self.minibatch_averaging(x)
        return self.MLP(x)

class SensitiveDiscriminator(nn.Module):
    def __init__(self, input_size, layer_sizes=[256,128]):
        super().__init__()
        self.MLP = nn.Sequential()
        layer_sizes = [input_size] + layer_sizes

        for i in range(len(layer_sizes)):
            if i not in [len(layer_sizes) - 1]:
                self.MLP.add_module(str(i), nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                self.MLP.add_module(str(i)+"act", nn.LeakyReLU(0.1))
                #self.MLP.add_module(str(i)+"bn", nn.BatchNorm1d(layer_sizes[i+1], 0.8))
        self.MLP.add_module("last_layer", nn.Linear(layer_sizes[-1], 1))
        self.MLP.add_module("last_act", nn.Sigmoid())
        
    def forward(self, x):
        return self.MLP(x)

class Generator(nn.Module):
    def __init__(self, latent_size, layer_sizes=[128,128]):
        super().__init__()
        layer_sizes = [latent_size] + layer_sizes
        self.modules = []
        self.batch_norms = []

        for i in range(len(layer_sizes)):
            if i not in [len(layer_sizes) - 1, len(layer_sizes) - 2]:
                self.add_generator_module("hidden_{:d}".format(i), layer_sizes[i], layer_sizes[i + 1], nn.LeakyReLU(0.1), 0.01)
        self.add_generator_module("output", layer_sizes[-2], layer_sizes[-1], nn.Tanh(), 0.01)
    
    def add_generator_module(self, name, in_size, out_size, activation, bn_decay):
        batch_norm = nn.BatchNorm1d(out_size, momentum=(1 - bn_decay))
        module = nn.Sequential(
            nn.Linear(in_size, out_size, bias=False),  # bias is not necessary because of the batch normalization
            batch_norm,
            activation
        )
        self.modules.append(module)
        self.add_module(name, module)
        self.batch_norms.append(batch_norm)
    
    def batch_norm_train(self, mode=True):
        for batch_norm in self.batch_norms:
            batch_norm.train(mode=mode)
        
    def forward(self, x):
        for module in self.modules:
            # Cannot write "outputs += module(outputs)" because it is an inplace operation (no differentiable)
            x = module(x) + x  # shortcut connection
        return x

def discriminator_loss(real_output, fake_output, model_type):
    cross_entropy = nn.BCEWithLogitsLoss()
    assert (model_type =='D1') or (model_type=='D2'), \
     "model_type argument must be (str) with value 'D1' or 'D2'"
    if model_type == 'D1':
        real_loss = cross_entropy(real_output, torch.ones_like(real_output, requires_grad=False))
        fake_loss = cross_entropy(fake_output, torch.zeros_like(fake_output, requires_grad=False))
        loss = real_loss + fake_loss
    elif model_type == 'D2':
        # Real output here means the correct sensitive attribute value, while fake_output means the prediction by D2
        loss = cross_entropy(fake_output, real_output)
    return loss

def generator_loss(fake_ones, fake_output, adversary):
    '''
       D1 adv loss =   cross_entropy( tf.ones_like(D1_preds), D1_preds )
       D2 adv loss = - cross_entropy( sensitive_label, D2_preds )
    '''
    cross_entropy = nn.BCEWithLogitsLoss()
    assert (adversary =='D1') or (adversary=='D2'), \
     "adversary argument must be (str) with value 'D1' or 'D2'"
    if adversary == 'D1':
        adverserial_loss = cross_entropy(fake_output, fake_ones)
    elif adversary == 'D2':
        # Fake ones here means the correct sensitive attribute value, while fake_output means the prediction by D2 (Follwing Algorithm 1, Line 12 in the paper)
        # Negative because we want to minimize (Compare with Algorithm 1 and generator D1 condition)
        # Can be negative Cross Entropy
        adverserial_loss = -cross_entropy(fake_output, fake_ones)
    return adverserial_loss

def get_gaussian_latent_vectors(batch_size, latent_dim):
    return torch.randn((batch_size, latent_dim))

def sampleFairGan(generator, decoder,  num_samples=50, latent_dim=128):
    with torch.no_grad():
        noise =  Variable(Tensor(np.random.normal(0, 1, (num_samples, latent_dim - 1)))).cuda().float()
        #sensitiveAttrs = torch.vstack((torch.ones(int(num_samples/2)), torch.zeros(num_samples - int(num_samples/2))))
        sensitiveAttrs = torch.cat((torch.ones(int(num_samples/2)), torch.zeros(num_samples - int(num_samples/2))))
        sensitiveAttrs = sensitiveAttrs.cuda().float().view(-1,1)
        finalNoiseVector = torch.cat((noise, sensitiveAttrs), axis=-1)
        ## Threshold Values for Y and Z
        samples = decoder(generator(finalNoiseVector)).cpu().numpy()
        for i in samples:
            if i[-1] <= 0.5:
                i[-1] = 0.0
            else:
                i[-1] = 1.0
        # Don't Need to threshold for sensitive attribute, since we already know them (Since we condition on them)
        sensitiveAttrs = sensitiveAttrs.cpu().numpy().reshape((-1,))
        return samples[:, :-1], samples[:, -1], sensitiveAttrs

def trainFairGan(dataloader, d1_input_size, d2_input_size, autoencoder_input_size, output_size, autoencoder_latent_size=128, generator_latent_size=128, epochs=500, normalEpochs=250, lam=1, verbose=False):
    # Initialize models
    autoencoder = AutoEncoder(Encoder(autoencoder_input_size,autoencoder_latent_size), Decoder(autoencoder_latent_size,output_size))
    autoencoder.cuda()
    generator = Generator(generator_latent_size, layer_sizes=[128, autoencoder_latent_size])
    generator.cuda()
    D1 = SourceDiscriminator(d1_input_size)
    D1.cuda()
    D2 = SensitiveDiscriminator(d2_input_size)
    D2.cuda()

    # Pretraining Autoencoder
    autoencoder = trainAutoEncoder(autoencoder, dataloader)
    decoder = autoencoder.decoder

    optimizer_D1 = torch.optim.Adam(D1.parameters(), 0.0001, betas=(0.5,0.999))
    optimizer_D2 = torch.optim.Adam(D2.parameters(), 0.0001, betas=(0.5,0.999))
    optimizer_G = torch.optim.Adam(generator.parameters(), 0.0001, betas=(0.5,0.999))

    print('\nTraining FairGan')
    global_D1_loss = []
    global_D2_loss = []
    global_gen_D1_loss = []
    global_gen_D2_loss = []
    for epoch in range(epochs):
        if verbose == True:
            D1_losses = []
            gen_D1_losses = []
            D2_losses = []
            gen_D2_losses = []
        for i, (imgs, labels, sensitives) in enumerate(dataloader):
            # Train D1
            D1.zero_grad()
            ''' Gdec input of noise P(z) and sensitive condition P(s)'''
            #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], generator_latent_size - 1)))).cuda().float()
            z = Variable(Tensor(get_gaussian_latent_vectors(imgs.shape[0], generator_latent_size - 1).cuda())).float()
            # Fix for toy experiments .view(-1,1) om imgs
            #imgs = Variable(imgs).view(-1,1).cuda().float()
            imgs = Variable(imgs).cuda().float()
            imgs = Tensor(imgs)
            sensitives = Variable(sensitives).view(-1,1).cuda().float()
            sensitives = Tensor(sensitives)
            labels = Variable(labels).cuda().view(-1,1).cuda().float()
            labels = Tensor(labels)
            Gdec_input = torch.cat([z, sensitives], axis=-1) # [Z, S]
            #Gdec_input = Gdec_input.reshape((-1,generator_latent_size,))
            fake_output = decoder(generator(Gdec_input)) # [X, Y | Z, S]

            ''' D1 outputs for Real P(x, y, s) and Generated P( x', y', s')'''
            disc_1_output_real = D1(torch.cat([imgs, labels, sensitives], axis=-1))
            disc_1_output_generated = D1(torch.cat([fake_output.detach(), sensitives], axis=-1))

            # Adversarial Step for D1
            D1_loss = discriminator_loss(disc_1_output_real, disc_1_output_generated, model_type='D1')
            D1_loss.backward()
            if verbose == True:
                D1_losses.append(D1_loss.item())
            optimizer_D1.step()

            # Generator Update
            generator.zero_grad()
            disc_1_output_generated = D1(torch.cat([fake_output, sensitives], axis=-1))
            # Sending Ones to do the 'Flip to One' Log Trick
            gen_loss_D1 = generator_loss(torch.ones_like(disc_1_output_generated, requires_grad=False),
                                                          disc_1_output_generated, adversary='D1')
            gen_loss_D1.backward()
            if verbose == True:
                gen_D1_losses.append(gen_loss_D1.item())
            optimizer_G.step()

            if (epoch + 1) > normalEpochs:
                ''' D2 predicting sensitive attribute on generated  P( x', y' | s ) '''
                D2.zero_grad()
                #z_0 = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], generator_latent_size - 1)))).cuda().float()
                #z_1 = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], generator_latent_size - 1)))).cuda().float()
                z_0 = Variable(Tensor(get_gaussian_latent_vectors(imgs.shape[0], generator_latent_size - 1).cuda())).float()
                z_1 = Variable(Tensor(get_gaussian_latent_vectors(imgs.shape[0], generator_latent_size - 1).cuda())).float()
                Gdec_input_0 = torch.cat([z_0, torch.zeros_like(sensitives)], axis=-1) # [Z, S=0]
                fake_output_0 = decoder(generator(Gdec_input_0)) # [X, Y | Z, S=0]
                Gdec_input_1 = torch.cat([z_1, torch.ones_like(sensitives)], axis=-1) # [Z, S=1]
                fake_output_1 = decoder(generator(Gdec_input_1)) # [X, Y | Z, S=1]
                disc_2_output = D2(torch.vstack((fake_output_0.detach(), fake_output_1.detach())))

                # Adversarial Step for D2
                D2_loss = lam*discriminator_loss(torch.vstack((torch.zeros_like(sensitives), torch.ones_like(sensitives))), disc_2_output, model_type='D2')
                D2_loss.backward()
                if verbose == True:
                    D2_losses.append(D2_loss.item())
                optimizer_D2.step()

                # Generator Update
                generator.zero_grad()
                disc_2_output = D2(torch.vstack((fake_output_0, fake_output_1)))
                gen_loss_D2 = lam*generator_loss(torch.vstack((torch.zeros_like(sensitives), torch.ones_like(sensitives))), disc_2_output, adversary='D2')
                gen_loss_D2.backward()
                if verbose == True:
                    gen_D2_losses.append(gen_loss_D2.item())
                optimizer_G.step()
                
        if verbose == True:
            if (epoch + 1) > normalEpochs:
                print(f'Fairness + Accuracy Maximizing Phase -> Epoch: {epoch}, D1 Loss: {np.mean(D1_losses)}, D2 Loss: {np.mean(D2_losses)}, Generator (with D1): {np.mean(gen_D1_losses)}, Generator (with D2): {np.mean(gen_D2_losses)}')
                global_D1_loss.append(np.mean(D1_losses))
                global_gen_D1_loss.append(np.mean(gen_D1_losses))
                global_D2_loss.append(np.mean(D2_losses))
                global_gen_D2_loss.append(np.mean(gen_D2_losses))
            else:
                print(f'Accuracy Maximizing Phase -> Epoch: {epoch}, D1 Loss: {np.mean(D1_losses)}, Generator (with D1): {np.mean(gen_D1_losses)}')
                global_D1_loss.append(np.mean(D1_losses))
                global_gen_D1_loss.append(np.mean(gen_D1_losses))
            print('-----------------------')
    return generator, decoder, global_D1_loss, global_gen_D1_loss, global_D2_loss, global_gen_D2_loss

def trainAutoEncoder(autoencoder, dataloader, epochs=50, verbose=False):
    #print('\nAutoencoder Pretraining\n')
    aeLoss = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), 0.0001, betas=(0.5,0.999))
    for epoch in range(epochs):
        if verbose == True:
            losses = []
        for i, (imgs, labels, _) in enumerate(dataloader):
            # Temporary .view() adjustment for toy experiments
            #imgs = Variable(imgs).cuda().view(-1,1).float()
            imgs = Variable(imgs).cuda().float()
            imgs = Tensor(imgs)
            #sensitives = Variable(sensitives).view(-1,1).cuda().float()
            #sensitives = Tensor(sensitives)
            labels = Variable(labels).cuda().view(-1,1).float()
            labels = Tensor(labels)
            newInput = torch.cat((imgs, labels), -1)
            output = autoencoder(newInput)
            loss = aeLoss(output, newInput)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if verbose == True:
                losses.append(loss.item())
        if verbose == True:
            print(sum(losses)/len(losses))
    return autoencoder

class TabularDataset(Dataset):
    def __init__(self, x, y, z, transform=None):
        self.x = x
        self.y = y
        self.a = z
        self.min = np.min(self.x, axis=0)
        self.max = np.max(self.x, axis=0)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(np.asarray([self.x[index]])), self.y[index], self.a[index]
        else:
            sample = (self.x[index] - self.min)/(self.max - self.min)
            return sample, self.y[index], self.a[index]

def train_fairgan_data_classifier(train_dataset, base_classifier='lr'):
    dataset = TabularDataset(train_dataset.features[:,:-1], train_dataset.labels, train_dataset.protected_attributes.squeeze())
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    train_data_shape = train_dataset.features[:,:-1].shape[1]
    fairGenerator, fairDecoder, d1_loss, gen_d1_loss, d2_loss, gen_d2_loss = trainFairGan(data_loader, d1_input_size=train_data_shape + 2, d2_input_size=train_data_shape + 1, autoencoder_input_size=train_data_shape + 1, output_size=train_data_shape + 1, verbose=False)
    new_x_train, new_y_train, new_z_train = sampleFairGan(fairGenerator, fairDecoder, len(train_dataset.features))
    # Inverse transform
    new_x_train = new_x_train*(dataset.max - dataset.min) + dataset.min
    if base_classifier == 'lr':
        model = LogisticRegression()
    elif base_classifier == 'svm':
        model = SVC()
    model.fit(new_x_train, new_y_train)
    return model
