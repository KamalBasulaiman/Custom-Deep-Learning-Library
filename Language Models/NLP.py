# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn.functional as F
from tests import test_prediction, test_generation
from torch.nn.utils.rnn import pad_sequence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load all that we need

dataset = np.load('../dataset/wiki.train.npy', allow_pickle=True)
fixtures_pred = np.load('../fixtures/prediction.npz')  # dev
fixtures_gen = np.load('../fixtures/generation.npy')  # dev (32,20)
fixtures_pred_test = np.load('../fixtures/prediction_test.npz')  # test
fixtures_gen_test = np.load('../fixtures/generation_test.npy')  # test (128,30)
vocab = np.load('../dataset/vocab.npy')

# data loader

class LanguageModelDataLoader(DataLoader):
    
    def __init__(self, dataset, batch_size, time_steps=200, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.time_steps = time_steps
        random.shuffle(self.dataset)


    def __iter__(self): 
        stack = np.concatenate(self.dataset)
        counter = 0
        list_of_batches = []
        max_len = len(stack)

        while(counter < max_len):
            self.time_steps = int(random.normalvariate(70,5))
        
            if counter + (self.time_steps*self.batch_size) + 1 <= max_len:
                input_batch_size = []; output_batch_size = []
        
                for i in range(counter,counter+(self.time_steps*self.batch_size),self.time_steps): # iterate through batch and slice by seqs
                    predictor = []; response = []
        
                    if i+self.time_steps <= counter+(self.time_steps*self.batch_size):
        
                        for j in stack[i:i+self.time_steps]:
                              predictor.append(j)
        
                        for j in stack[i+1:i+self.time_steps+1]:
                              response.append(j)
                        input_batch_size.append(predictor)
                        output_batch_size.append(response)
        
                list_of_batches.append((input_batch_size,output_batch_size))
                counter = counter + (self.time_steps * self.batch_size)
                yield (input_batch_size,output_batch_size)
        
            else:    
                break

# model
def emb_dropout(embed, words, dropout, scale=None):
    if dropout:
      mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
      W_emb = Variable(mask) * embed.weight
    else:
      W_emb = embed.weight
    pad_idx = -1
    t = F.embedding(words, W_emb, pad_idx, embed.max_norm, embed.norm_type, embed.scale_grad_by_freq, embed.sparse)
    
    return t


class LanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = 1150
        self.embedding_size = 400
        self.embedd_drop_rate = 0.1
        
        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(input_size = self.embedding_size,hidden_size=self.hidden_size,num_layers=3, dropout=0.2, batch_first=True, bidirectional=True)
        self.decoder = nn.Linear(self.hidden_size*2, vocab_size)

        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-0.1, 0.1)


    def forward(self, x):
        batch_size = x.shape[0]
        if self.training:
            embeddings = emb_dropout(self.embedding, x, self.embedd_drop_rate)
        else:
            embeddings = emb_dropout(self.embedding, x, 0)
        predict, hidden = self.lstm(embeddings, None)
        predictions = self.decoder(predict)
        out = predictions.view(batch_size,-1,self.vocab_size)
        return out

# Commented out IPython magic to ensure Python compatibility.
# model trainer
class LanguageModelTrainer:
    def __init__(self, model, loader, max_epochs=1, run_id='exp'):
        """
            Use this class to train your model
        """
        self.model = model
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id
        
        # TODO: Define your optimizer and criterion here
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001,weight_decay=0.00001)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 1, gamma=0.8)

    def train(self):
        self.model.train() # set to training mode
        epoch_loss = 0
        num_batches = 0
        
        for batch_num, (inputs, targets) in enumerate(self.loader):
            epoch_loss += self.train_batch(torch.LongTensor(np.array(inputs)), torch.LongTensor(np.array(targets)))

        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
#                       % (self.epochs, self.max_epochs, epoch_loss))
        self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets):
        
        prediction = model(inputs.to(DEVICE))
        loss = self.criterion(output.view(-1,output.size(2)),targets.view(-1).to(DEVICE))
        self.optimizer.zero_grad() 
        loss_item = loss.item()
        loss.backward()
        self.optimizer.step()

        return loss_item


    
    def test(self):
        
        self.model.eval() # set to eval mode
        predictions = TestLanguageModel.prediction(fixtures_pred['inp'], self.model) # get predictions
        self.predictions.append(predictions)
        generated_logits = TestLanguageModel.generation(fixtures_gen, 10, self.model) # generated predictions for 10 words
        generated_logits_test = TestLanguageModel.generation(fixtures_gen_test, 10, self.model)
        nll = test_prediction(predictions, fixtures_pred['out'])
        generated = test_generation(fixtures_gen, generated_logits, vocab)
        generated_test = test_generation(fixtures_gen_test, generated_logits_test, vocab)
        self.val_losses.append(nll)
        
        self.generated.append(generated)
        self.generated_test.append(generated_test)
        self.generated_logits.append(generated_logits)
        self.generated_logits_test.append(generated_logits_test)
        
        # generate predictions for test data
        predictions_test = TestLanguageModel.prediction(fixtures_pred_test['inp'], self.model) # get predictions
        self.predictions_test.append(predictions_test)
            
        print('[VAL]  Epoch [%d/%d]   Loss: %.4f'
#                       % (self.epochs, self.max_epochs, nll))
        return nll

    def save(self):
                
        model_path = os.path.join('experiments', self.run_id, 'model-{}.pt'.format(self.epochs))
        torch.save(self.model.state_dict(),
            model_path)
        
        np.save(os.path.join('experiments', self.run_id, 'predictions-{}.npy'.format(self.epochs)), self.predictions[-1])
        np.save(os.path.join('experiments', self.run_id, 'predictions-test-{}.npy'.format(self.epochs)), self.predictions_test[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-{}.npy'.format(self.epochs)), self.generated_logits[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-test-{}.npy'.format(self.epochs)), self.generated_logits_test[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}-test.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated_test[-1])

class TestLanguageModel:
    def prediction(inp, model):
        """            
            :param inp:
            :return: a np.ndarray of logits
        """
        out = model(torch.LongTensor(inp).to(DEVICE))

        out = out.permute(1,0,2)
        y = out[-1][:][:].detach().cpu().numpy()

        return y


    def generation(inp, forward, model):
        """
            Generate a sequence of words given a starting sequence.
            :param inp: Initial sequence of words (batch size, length)
            :param forward: number of additional words to generate
            :return: generated words (batch size, forward)
        """        
        generate_sequence = []
        out = model(torch.LongTensor(inp).to(DEVICE))
        out = out.permute(1,0,2)
        _,F = torch.max(out[-1][:][:],dim=1) 
        F = F.unsqueeze(1).detach().cpu().numpy()
        generate_sequence.append(F)

        if forward > 1:
            for i in range(forward-1):
                out = model(F)
                out = out.permute(1,0,2)[-1][:][:]
                _,F = torch.max(out,dim=1).unsqueeze(1).detach().cpu().numpy()
                generate_sequence.append(F)

        x = []
        y = []
        
        for i in range(inp.shape[0]):
        
          y = []
        
          for j in range(forward):
        
            y.append(generate_sequence[j][i][0])

          x.append(y)

        return x

# Other hyperparameters

NUM_EPOCHS = 55
BATCH_SIZE = 128

run_id = str(int(time.time()))
if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
os.mkdir('./experiments/%s' % run_id)
print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)

model = LanguageModel(len(vocab)).to(DEVICE)
loader = LanguageModelDataLoader(dataset=dataset, batch_size=BATCH_SIZE,shuffle=True)
trainer = LanguageModelTrainer(model=model, loader=loader, max_epochs=NUM_EPOCHS, run_id=run_id)

best_nll = 1e30 
for epoch in range(NUM_EPOCHS):
    trainer.train()
    nll = trainer.test()
    if nll < best_nll:
        best_nll = nll
        print("Saving model, predictions and generated output for epoch "+str(epoch+1)+" with NLL: "+ str(best_nll))
        trainer.save()

# plot training curves
plt.figure()
plt.plot(range(1, trainer.epochs + 1), trainer.train_losses, label='Training losses')
plt.plot(range(1, trainer.epochs + 1), trainer.val_losses, label='Validation losses')
plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.legend()
plt.show()

# see generated output
print (trainer.generated[-1]) # get last generated output