# Baseline model: https://drive.google.com/file/d/1Popb7jTjCuk4GITBDDWMQ1Kz0cQ2tbMJ/view?usp=drive_link
# Pre-trained model: https://drive.google.com/file/d/1Gzj0H_UZw7j4lLHKSXbNPSkRvWGdRdT0/view?usp=drive_link

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split

from model import TCRModel, clf_loss_func, BERT_CONFIG

import torchdata.datapipes as dp
import torchtext.transforms as T
from torchtext.vocab import build_vocab_from_iterator

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# START -- Data processing functions

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

def antigenTransform(vocab, start_token):
    """
    Create transforms based on given vocabulary. The returned transform is applied to sequence
    of tokens.
    """
    text_tranform = T.Sequential(
        # Converts the sentences to indices based on given vocabulary
        T.VocabTransform(vocab=vocab),
        # Add start_token at the beginning.
        T.AddToken(start_token, begin=True),
    )
    return text_tranform

def TCRTransform(vocab, start_token, end_token):
    """
    Create transforms based on given vocabulary. The returned transform is applied to sequence
    of tokens.
    """
    text_tranform = T.Sequential(
        # Converts the sentences to indices based on given vocabulary
        T.VocabTransform(vocab=vocab),
        # Add start_token at the beginning.
        T.AddToken(start_token, begin=True),
        # Add end_token at the end.
        T.AddToken(end_token, begin=False),
    )
    return text_tranform

def applyTransform(sequence_pair, vocab):
    """
    Apply transforms to sequence of tokens in a sequence pair
    """
    return (
        antigenTransform(vocab, 1)(tokenize(sequence_pair[0])) + TCRTransform(vocab, 2, 2)(tokenize(sequence_pair[1])),
        int(sequence_pair[2])
    )

def applyPadding(sequence):
    return T.ToTensor(0, dtype=torch.long)(list(sequence))

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
        specials= ['<pad>', '<cls>', '<sep>', '<mask>', '<unk>'],
        special_first=True
    )
    vocab.set_default_index(vocab['<unk>'])

    data_pipe = data_pipe.map(lambda x: applyTransform(x, vocab))
    input_ids, labels = zip(*data_pipe)
    input_ids, labels = applyPadding(input_ids), applyPadding(labels)

    return input_ids, labels

# END -- Data processing functions


# START -- Contrastive learning

def device_as(t1, t2):
   """
   Moves t1 to the device of t2
   """
   return t1.to(t2.device)

class ContrastiveLoss(nn.Module):
   """
   Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
   """
   def __init__(self, temperature=0.5):
       super().__init__()
       self.temperature = temperature


   def calc_similarity_batch(self, a, b):
       representations = torch.cat([a, b], dim=0)
       return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)


   def forward(self, proj_1, proj_2):
       """
       proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
       where corresponding indices are pairs
       z_i, z_j in the SimCLR paper
       """
       batch_size = proj_1.shape[0]
       z_i = F.normalize(proj_1, p=2, dim=1)
       z_j = F.normalize(proj_2, p=2, dim=1)

       similarity_matrix = self.calc_similarity_batch(z_i, z_j)

       sim_ij = torch.diag(similarity_matrix, batch_size)
       sim_ji = torch.diag(similarity_matrix, -batch_size)

       positives = torch.cat([sim_ij, sim_ji], dim=0)

       nominator = torch.exp(positives / self.temperature)

       mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

       denominator = device_as(mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

       all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
       loss = torch.sum(all_losses) / (2 * batch_size)
       return loss

# END -- Contrastive learning

class Classifier():
    def __init__(self):
        self.learning_rate = 1e-5

        self.model = TCRModel()
        self.model.to(device)

        self.pretrain_loss_func = ContrastiveLoss(temperature=0.5)
        self.pretrain_optimizer = optim.SGD(self.model.parameters(), lr=1e-3)

        self.loss_func = clf_loss_func.to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self.accum_iter = 4  


    def save(self, filename):
        self.model.save(filename)


    def load(self, filename):
        self.model.load(filename)


    def augment(self, base_X, num_tokens_to_replace):
        projection = torch.clone(base_X)

        indices = (base_X > 2).nonzero()

        for row_idx in range(projection.size(0)):
            row_tokens = indices[indices[:, 0] == row_idx][:, 1]
            if row_tokens.numel() > 0:
                random_token_idx = row_tokens[torch.randint(0, row_tokens.numel(), (num_tokens_to_replace,))]
                projection[row_idx, random_token_idx] = 3
        
        return projection


    def pretrain_step(self, batch_ndx, x, max):

        base_X, _ = x
        base_X = base_X.to(device)

        proj_1 = self.augment(base_X, num_tokens_to_replace=1)
        proj_2 = self.augment(base_X, num_tokens_to_replace=1)

        proj_1_mask = torch.where(proj_1 != 0, torch.tensor(1), torch.tensor(0))
        proj_2_mask = torch.where(proj_2 != 0, torch.tensor(1), torch.tensor(0))

        proj_1_output = self.model(proj_1, proj_1_mask, classification=False)
        proj_2_output = self.model(proj_2, proj_2_mask, classification=False)

        proj_1_output = proj_1_output[:,0,:]
        proj_2_output = proj_2_output[:,0,:]

        loss = self.pretrain_loss_func(proj_1_output, proj_2_output)

        # Backpropagation and optimization
        loss.backward()

        if ((batch_ndx + 1) % self.accum_iter == 0) or (batch_ndx + 1 == max):
            self.pretrain_optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()


    def finetune_step(self, x):
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

        # Backpropagation and optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def eval_step(self, x):
        X, y = x
        X, y = X.to(device), y.to(device)
        mask = torch.where(X != 0, torch.tensor(1), torch.tensor(0))

        output = self.model(X, mask)

        y_hat = torch.argmax(output, 1)

        perc_correct = torch.nonzero(y_hat == y).shape[0] / y.shape[0]

        return perc_correct
    
    def pretrain(self, train_loader, epochs):
        for j in tqdm(range(epochs)):
            ep_loss = 0.0
            for batch_ndx, sample in enumerate(tqdm(train_loader, leave=False)):
                ep_loss += self.pretrain_step(batch_ndx, sample, len(train_loader))
            
            print(f"Epoch loss: {ep_loss / batch_ndx+1}")

    
    def finetune(self, train_loader, epochs):
        for j in tqdm(range(epochs)):
            ep_loss = 0.0
            for batch_ndx, sample in enumerate(tqdm(train_loader, leave=False)):
                ep_loss += self.finetune_step(sample)
            
            print(f"Epoch loss: {ep_loss / batch_ndx+1}")


    def evaluate(self, test_loader):
        perc_correct = []

        for batch_ndx, sample in enumerate(tqdm(test_loader, leave=False)):
            perc_correct.append(self.eval_step(sample))
            
        print(f"Percent Correct: {torch.mean(torch.tensor(perc_correct))}")


if __name__ == "__main__":

    pre_train = True
    epoch = 3
    bsz = 1024

    print("Processing data...")

    input_ids, labels = process_data("./data.csv")

    dataset = TensorDataset(input_ids, labels)

    data_loader = DataLoader(dataset, shuffle=True, batch_size=bsz)

    print("Processing complete")

    classifier = Classifier()
    classifier.model.train()

    if pre_train:

        print("Pre-training...")
        classifier.pretrain(data_loader, epoch)
    
    epoch = 3
    bsz = 1024

    # k-fold cross validation
    # The model is trained and evaluated k times, each time using a different fold as the test set and the remaining folds as the training set.
    k = 3 # Number of folds
    total_size = len(dataset)
    fold_sizes = [total_size // k, total_size // k, total_size - ((k-1) * (total_size // 3))]

    fold_datasets = random_split(dataset, fold_sizes)

    for fold_idx, test_fold in enumerate(fold_datasets):
        train_folds = [fold_datasets[i] for i in range(k) if i != fold_idx]
        train_dataset = torch.utils.data.ConcatDataset(train_folds)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=bsz)
        test_loader = DataLoader(test_fold, shuffle=False, batch_size=bsz)

        print("Training classifier...")
        classifier.finetune(train_loader, epoch)

        classifier.model.eval()
        perc_correct = []

        print("Evaluate classification...")

        classifier.evaluate(test_loader)

    if pre_train:
        classifier.save("pretrained.pt")
    else:
        classifier.save("model.pt")

