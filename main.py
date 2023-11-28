import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import TCRModel, clf_loss_func

import torchdata.datapipes as dp
import torchtext.transforms as T
from torchtext.vocab import build_vocab_from_iterator

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

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

        self.model = TCRModel()
        self.loss_func = clf_loss_func
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)

    def train(self, x):
        '''
        Train the model on one batch of data
        :param x: train data
        :return: (mean) loss of the model on the batch
        '''
        self.model.to(device)
        self.model.train()

        X, y = x
        mask = torch.where(X != 0, torch.tensor(1), torch.tensor(0))

        output = self.model(X, mask)

        # Compute the loss
        loss = self.loss_func(output, y)
        grad_output = torch.ones_like(loss)  
        
        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward(grad_output)
        self.optimizer.step()

        return torch.mean(loss)


if __name__ == "__main__":

    epoch = 1
    bsz = 32

    print("Processing data...")
    
    data, labels = process_data("./data.csv")

    dataset = TensorDataset(data, labels)
    
    # Define the sizes of the training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=bsz, num_workers=4)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=bsz, num_workers=4)
    
    print("Processing complete")

    classifier = Classifier()

    print("Training...")

    loss = 0
    
    for i in tqdm(range(epoch)):
        for batch_ndx, sample in enumerate(tqdm(train_loader, leave=False)):
            loss += classifier.train(sample)
        
        print(f"Softmax probabilities: {loss / i+1}")
    