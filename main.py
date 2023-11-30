import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import TCRModel, clf_loss_func

import torchdata.datapipes as dp
import torchtext.transforms as T
from torchtext.vocab import build_vocab_from_iterator

from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def tokenize(text):
    """
    Tokenize an English text and return a list of tokens
    """
    return [token for token in str(text)]

def getTokens(data_iter):
    """
    Function to yield tokens from an iterator. 
    """
    for antigen, TCR, _ in data_iter:
        yield tokenize(antigen) + tokenize(TCR)

def getTransform(vocab, start_token):
    """
    Create transforms based on given vocabulary. The returned transform is applied to sequence
    of tokens.
    """
    text_tranform = T.Sequential(
        # Converts the sentences to indices based on given vocabulary
        T.VocabTransform(vocab=vocab),
        # Add start_token at beginning. 
        T.AddToken(start_token, begin=True),
    )
    return text_tranform

def applyTransform(sequence_pair, vocab):
    """
    Apply transforms to sequence of tokens in a sequence pair
    """
    return (
        getTransform(vocab, 1)(tokenize(sequence_pair[0])),
        getTransform(vocab, 2)(tokenize(sequence_pair[1])),
        int(sequence_pair[2])
    )

def applyPadding(sequence):
    return T.ToTensor(0)(list(sequence))

'''
:param file_path: path to csv file
:return: tuple of tensors (data, label)
'''
def process_data(file_path):

    data_pipe = dp.iter.IterableWrapper([file_path])
    data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
    data_pipe = data_pipe.parse_csv(skip_lines=1, delimiter=',', as_tuple=True)

    vocab = build_vocab_from_iterator(
        getTokens(data_pipe),
        specials= ['<pad>', '<cls>', '<seq>', '<unk>'],
        special_first=True
    )
    vocab.set_default_index(vocab['<unk>'])

    data_pipe = data_pipe.map(lambda x: applyTransform(x, vocab))
    antigens, TCRs, interactions = zip(*data_pipe)
    antigens, TCRs, labels = applyPadding(antigens), applyPadding(TCRs), applyPadding(interactions)
    
    data = torch.concat((antigens, TCRs), 1)
    return data, labels


class Classifier():
    def __init__(self):
        self.learning_rate = 0.001

        self.model = TCRModel()#.to(device)
        self.model.to(device)
        # self.cos = nn.CosineSimilarity(dim=2)
        # self.pretrain_loss = nn.BCELoss().to(device)

        # self.loss_func = clf_loss_func.to(device)
        # self.optimizer = optim.RAdam(self.model.parameters(), lr=self.learning_rate)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)

    
    #   TODO:  Use broadcasting to construct a batch_size x batch_size matrix of cosine similarities.
    #   Loss should take difference of this sim matrix with identity matrix (1s along diag zeros elsewhere).
    #
    #   What should final vector be?  Last state contains an embedding for each token.  Do I take the mean?
    def contrastive_loss(self, base_embed, aug_embed):
        base_embed = torch.broadcast_to(base_embed, (base_embed.shape[0], base_embed.shape[0], base_embed.shape[1]))
        aug_embed = torch.broadcast_to(aug_embed, (aug_embed.shape[0], aug_embed.shape[0], aug_embed.shape[1]))

        sims = self.cos(base_embed, torch.transpose(aug_embed, 0, 1))
        loss = self.pretrain_loss(nn.Sigmoid()(sims), torch.eye(sims.shape[0]).to(device))

        return loss


    def pretrain(self, x, permute_num=8):
        base_x, _ = x
        base_x = base_x.to(device)
        aug_x = torch.clone(base_x)#.to(device)

        self.optimizer.zero_grad()

        for i in range(base_x.shape[0]):
            valid_tokens = torch.nonzero(torch.logical_and(torch.logical_and(base_x[i] != 0, base_x[i] != 1), base_x[i] != 2), as_tuple=True)
            aug_x[i, valid_tokens[0][torch.randperm(valid_tokens[0].shape[0])[:permute_num]]] = aug_x[i, valid_tokens[0][torch.randperm(valid_tokens[0].shape[0])[:permute_num]]]

        mask = torch.where(base_x != 0, torch.tensor(1), torch.tensor(0))

        base_embed = self.model(base_x, mask, classification=False)
        aug_embed = self.model(aug_x, mask, classification=False)

        # Compute the loss
        loss = self.contrastive_loss(base_embed[:, 0, :], aug_embed[:, 0, :])

        #   NOTE:  why is this ones_like here?
        # grad_output = torch.ones_like(loss)  
        
        # Backpropagation and optimization
        # self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train_classifier(self, x):
        '''
        Train the model on one batch of data
        :param x: train data
        :return: (mean) loss of the model on the batch
        '''
        X, y = x
        X, y = X.to(device), y.to(device)
        mask = torch.where(X != 0, torch.tensor(1), torch.tensor(0))

        self.optimizer.zero_grad()

        output = self.model(X, mask)

        # Compute the loss
        loss = self.loss_func(output, y)
        loss = loss.mean()
        # grad_output = torch.ones_like(loss)  
        
        # Backpropagation and optimization
        # self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    
    def eval_classifier(self, x):
        X, y = x
        X, y = X.to(device), y.to(device)
        mask = torch.where(X != 0, torch.tensor(1), torch.tensor(0))

        output = self.model(X, mask)

        y_hat = torch.argmax(output, 1)

        perc_correct = torch.nonzero(y_hat == y).shape[0] / y.shape[0]

        return perc_correct


if __name__ == "__main__":

    epoch = 16
    bsz = 256

    print("Processing data...")
    
    data, labels = process_data("./data.csv")

    dataset = TensorDataset(data, labels)
    
    #   TODO:  Split three ways and perform 3-fold cross-validation (train on 2, test on 1).
    # Define the sizes of the training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=bsz)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=bsz)
    
    print("Processing complete")

    classifier = Classifier()

    # print("Pretraining...")
    # loss = 0

    classifier.model.train()
    
    # for i in tqdm(range(epoch)):
    #     for batch_ndx, sample in enumerate(tqdm(train_loader, leave=False)):
    #         loss += classifier.pretrain(sample)
        
    #     print(f"Loss per epoch: {loss / i+1}")

    print("Fine tuning with classification training...")

    loss_func = clf_loss_func.to(device)
    optimizer = optim.RAdam(classifier.model.parameters(), lr=0.001)
    
    for i in tqdm(range(epoch)):
        loss = 0

        for batch_ndx, sample in enumerate(tqdm(train_loader, leave=False)):

            # loss += classifier.train_classifier(sample)

            X, y = sample
            X, y = X.to(device), y.to(device)
            mask = torch.where(X != 0, torch.tensor(1), torch.tensor(0))

            optimizer.zero_grad()

            output = classifier.model(X, mask)

            # Compute the loss
            loss = loss_func(output, y).mean()
            # loss = loss.mean()
            # grad_output = torch.ones_like(loss)  
            
            # Backpropagation and optimization
            # self.optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch loss: {loss / batch_ndx+1}")

    classifier.model.eval()
    perc_correct = []

    print("Evaluate classification...")
    for batch_ndx, sample in enumerate(tqdm(test_loader, leave=False)):
        perc_correct.append(classifier.eval_classifier(sample))
        
    print(f"Percent Correct: {torch.mean(torch.tensor(perc_correct))}")
    