import torch
import torch.nn.functional as F 
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split

from model import TCRModel, clf_loss_func, BERT_CONFIG

import torchdata.datapipes as dp
import torchtext.transforms as T
from torchtext.vocab import build_vocab_from_iterator

from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    
    return antigens, TCRs, labels

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
   def __init__(self, batch_size, temperature=0.5):
       super().__init__()
       self.batch_size = batch_size
       self.temperature = temperature
       self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()


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

       denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

       all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
       loss = torch.sum(all_losses) / (2 * self.batch_size)
       return loss
      
# END -- Contrastive learning


class ClassifierModel(nn.Module):

    def __init__(self, mlp_dim=3):
        super(ClassifierModel, self).__init__()
        self.classifier = TCRModel()
        self.mlp = nn.Linear(BERT_CONFIG.hidden_size, mlp_dim)


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        classification: bool = True
    ):
        output = self.classifier(input_ids, attention_mask, classification)
        
        # Is this a good idea?
        if not classification:
            output = self.mlp(output)
            output = torch.sum(output, 2)

        return output


class Classifier():
    def __init__(self, bsz):
        self.learning_rate = 0.001

        self.model = ClassifierModel()
        self.model.to(device)

        self.loss_func = clf_loss_func.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.pretrain_loss_func = ContrastiveLoss(bsz)
        self.pretrain_optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)


    def save(self, filename):
        self.model.save(filename)


    def load(self, filename):
        self.model.load(filename)


    def pretrain(self, x):
        
        X1, X2, _ = x

        base_X = torch.concat((X1,X2), 1)
        aug_X = torch.concat((X2,X1), 1)
        
        base_X = base_X.to(device)
        aug_X = aug_X.to(device)

        base_mask = torch.where(base_X != 0, torch.tensor(1), torch.tensor(0))
        aug_mask = torch.where(aug_X != 0, torch.tensor(1), torch.tensor(0))

        self.optimizer.zero_grad()

        base_output = self.model(base_X, base_mask, classification=False)
        aug_output = self.model(aug_X, aug_mask, classification=False)

        loss = self.pretrain_loss_func(base_output, aug_output)

        # Backpropagation and optimization
        loss.backward()
        self.pretrain_optimizer.step()

        return loss.item()
    

    def train_classifier(self, x):
        '''
        Train the model on one batch of data
        :param x: train data
        :return: (mean) loss of the model on the batch
        '''
        X1, X2, y = x
        X = torch.concat((X1,X2), 1)
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
    
    
    def eval_classifier(self, x):
        X1, X2, y = x
        X = torch.concat((X1,X2), 1)
        X, y = X.to(device), y.to(device)
        mask = torch.where(X != 0, torch.tensor(1), torch.tensor(0))

        output = self.model(X, mask)

        y_hat = torch.argmax(output, 1)

        perc_correct = torch.nonzero(y_hat == y).shape[0] / y.shape[0]

        return perc_correct


if __name__ == "__main__":

    pre_train = True
    epoch = 16
    bsz = 256

    print("Processing data...")
    
    anitgens, TCRs, labels = process_data("./data.csv")

    dataset = TensorDataset(anitgens, TCRs, labels)
    indices = [i for i, label in enumerate(labels) if label == 1]
    
    subset = Subset(dataset, indices)
    subset_data_loader = DataLoader(subset, shuffle=True, batch_size=bsz)

    print("Processing complete")

    classifier = Classifier(bsz)
    classifier.model.train()

    if pre_train:
        
        print("Pre-training...")

        for i in tqdm(range(epoch)):
            pre_train_loss = 0

            for batch_ndx, sample in enumerate(tqdm(subset_data_loader, leave=False)):
                pre_train_loss += classifier.pretrain(sample)

            print(f"Epoch loss (pre-training): {pre_train_loss / batch_ndx+1}")
        

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

        for i in tqdm(range(epoch)):
            loss = 0

            for batch_ndx, sample in enumerate(tqdm(train_loader, leave=False)):
                loss += classifier.train_classifier(sample)

            print(f"Epoch loss: {loss / batch_ndx+1}")

        classifier.model.eval()
        perc_correct = []


        print("Evaluate classification...")
        
        for batch_ndx, sample in enumerate(tqdm(test_loader, leave=False)):
            perc_correct.append(classifier.eval_classifier(sample))
            
        print(f"Percent Correct: {torch.mean(torch.tensor(perc_correct))}")
    
    