import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import sys
import matplotlib.pyplot as plt

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerEncoder, SpeechClassifier, TransformerDecoder, WindowedTransformerDecoder
from utilities import Utilities

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 100 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels


def compute_classifier_accuracy(encoder, classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    encoder.eval()
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            out, _ = encoder(X)            
            # Take the mean across the sequence dimension (dim=1)
            mean_embeds = out.mean(dim=1)
            # Pass the averaged embeddings to the classifier
            outputs = classifier(mean_embeds)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
            
    accuracy = (100 * total_correct / total_samples)
    encoder.train()
    classifier.train()
    return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader."""
    decoderLMmodel.eval()
    losses = []
    with torch.no_grad():
        for i, (X, Y) in enumerate(data_loader):
            if i >= eval_iters: break
            X, Y = X.to(device), Y.to(device)
            _, loss, _ = decoderLMmodel(X, targets=Y)
            losses.append(loss.item())
    
    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()
    decoderLMmodel.train()
    return perplexity


# PART 1
def main_part1():
    print("\n=== Running Part 1: Encoder & Classifier ===")
    
    # Create the folder for Part 1 results
    os.makedirs("part1", exist_ok=True)

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) 
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)

    print("Initializing Encoder and Classifier...")
    encoder = TransformerEncoder(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    classifier = SpeechClassifier(n_input, n_hidden, n_output).to(device)
    
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Part 1.5: Report Parameters
    params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\n[Part 1.5] Total parameters in Encoder: {params}")

    # Part 1.4: Sanity Check
    print("\n[Part 1.4] Running Sanity Check...")
    utils = Utilities(tokenizer, encoder)
    sample_sentence = "The American people expect us to work together."
    utils.sanity_check(sample_sentence, block_size, folder="part1")
    
    print("\n[Part 1.3] Starting Training Loop...")
    
    # List to store accuracy per epoch for plotting
    epoch_accuracies = [] 
    
    for epoch in range(epochs_CLS):
        encoder.train()
        classifier.train()
        total_loss = 0
        
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            out, _ = encoder(xb)
            mean_embeds = out.mean(dim=1)
            logits = classifier(mean_embeds)
            
            loss = criterion(logits, yb)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Part 1.5: Accuracy per epoch
        test_accuracy = compute_classifier_accuracy(encoder, classifier, test_CLS_loader)
        epoch_accuracies.append(test_accuracy) # Store the accuracy
        
        print(f"Epoch [{epoch+1}/{epochs_CLS}] | Loss: {total_loss/len(train_CLS_loader):.4f} | Test Accuracy: {test_accuracy:.2f}%")
        
    # plot accuracy
    print("\nGenerating Accuracy Plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs_CLS + 1), epoch_accuracies, marker='o', linestyle='-', color='b')
    plt.title('Part 1: Classifier Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.xticks(range(1, epochs_CLS + 1)) # Ensure integer ticks for epochs
    
    # Save the plot
    plot_path = os.path.join("part1", "accuracy_plot.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Part 1 Finished. Plots saved to '{plot_path}' and 'part1/' directory.")


# PART 2
def main_part2():
    print("\n=== Running Part 2: Decoder Language Model ===")
    
    os.makedirs("part2", exist_ok=True)

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) 
    print("Vocabulary size is", tokenizer.vocab_size)

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    print("Initializing Decoder...")
    decoder = TransformerDecoder(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    optimizer_lm = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    # Part 2.4: Parameter count
    params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\n[Part 2.4] Total parameters in Decoder: {params}")

    print("\n[Part 2.2] Starting Language Model Training Loop...")
    decoder.train()
    
    # Train for a maximum of 500 iterations
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer_lm.zero_grad()
        logits, loss, _ = decoder(xb, targets=yb)
        loss.backward()
        optimizer_lm.step()
        
        # Print Perplexity every 100 iterations (and at iteration 0)
        if i % 100 == 0 or i == max_iters - 1:
            current_ppl = torch.exp(loss).item()
            print(f"Iter {i}: Loss {loss.item():.4f} | Batch Perplexity: {current_ppl:.2f}")

    # Evaluation
    print("\n[Part 2.4] Evaluating Perplexity...")
    
    # Evaluate Training Perplexity first
    print("Evaluating on Training Set (subset)...")
    # We use a subset of batches (eval_iters) so it doesn't take forever
    train_ppl = compute_perplexity(decoder, train_LM_loader, eval_iters=eval_iters)
    print(f"Final Perplexity for Training Set: {train_ppl:.2f}")
    
    # Evaluate test sets
    print("\nEvaluating on Test Sets...")
    for name in ['obama', 'wbush', 'hbush']:
        filename = f"speechesdataset/test_LM_{name}.txt"
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
            ds = LanguageModelingDataset(tokenizer, text, block_size)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
            ppl = compute_perplexity(decoder, dl)
            print(f"Perplexity for {name}: {ppl:.2f}")
        else:
            print(f"Warning: Could not find {filename}")

    # Sanity Check for Decoder
    print("\n[Part 2.3] Running Decoder Sanity Check...")
    utils_lm = Utilities(tokenizer, decoder)
    utils_lm.sanity_check("The American people expect us to work together.", block_size, folder="part2")
    
    print("\nPart 2 Finished. Plots saved to 'part2/' directory.")


# PART 3
def main_part3():
    print("\n=== Running Part 3: Local Window Attention Decoder ===")
    
    os.makedirs("part3", exist_ok=True)

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) 
    print("Vocabulary size is", tokenizer.vocab_size)

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    print("Initializing Windowed Decoder (Window Size = 5)...")
    # Using the new Windowed architecture
    decoder = WindowedTransformerDecoder(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size, window_size=5).to(device)
    optimizer_lm = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\nTotal parameters in Windowed Decoder: {params}")

    print("\nStarting Windowed Language Model Training Loop...")
    decoder.train()
    
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer_lm.zero_grad()
        logits, loss, _ = decoder(xb, targets=yb)
        loss.backward()
        optimizer_lm.step()
        
        if i % 100 == 0 or i == max_iters - 1:
            current_ppl = torch.exp(loss).item()
            print(f"Iter {i}: Loss {loss.item():.4f} | Batch Perplexity: {current_ppl:.2f}")

    # Evaluation
    print("\nEvaluating Perplexity with Local Window Attention...")
    print("Evaluating on Training Set (subset)...")
    train_ppl = compute_perplexity(decoder, train_LM_loader, eval_iters=eval_iters)
    print(f"Final Perplexity for Training Set: {train_ppl:.2f}")
    
    print("\nEvaluating on Test Sets...")
    for name in ['obama', 'wbush', 'hbush']:
        filename = f"speechesdataset/test_LM_{name}.txt"
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
            ds = LanguageModelingDataset(tokenizer, text, block_size)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
            ppl = compute_perplexity(decoder, dl)
            print(f"Perplexity for {name}: {ppl:.2f}")
        else:
            print(f"Warning: Could not find {filename}")

    # Sanity Check for Part 3
    print("\nRunning Windowed Decoder Sanity Check...")
    utils_lm = Utilities(tokenizer, decoder)
    utils_lm.sanity_check("The American people expect us to work together.", block_size, folder="part3")
    
    print("\nPart 3 Finished. Plots saved to 'part3/' directory.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [part1 | part2 | part3]")
        print("No arguments provided. Running ALL parts sequentially...\n")
        main_part1()
        main_part2()
        main_part3()
        return

    mode = sys.argv[1]
    
    if mode == "part1":
        main_part1()
    elif mode == "part2":
        main_part2()
    elif mode == "part3":
        main_part3()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main.py [part1 | part2 | part3]")

if __name__ == "__main__":
    main()