
import matplotlib.pyplot as plt
import torch

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size, folder=None):
        # Prepare the folder if provided
        import os
        if folder is not None:
            os.makedirs(folder, exist_ok=True)
            
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the encoder model
        with torch.no_grad():
            outputs = self.model(input_tensor)
            attn_maps = outputs[-1]

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            att_map = attn_map[0, 0] # Get Head 0 of Batch 0

            # Check if the attention probabilities sum to 1 over rows
            total_prob_over_rows = torch.sum(att_map, dim=1)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                print(f"Layer {j+1}: Failed normalization test: probabilities do not sum to 1.0 over rows")

            # Convert to numpy for plotting
            att_map_np = att_map.detach().cpu().numpy()

            # Create a heatmap of the attention map
            fig, ax = plt.subplots()
            cax = ax.imshow(att_map_np, cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()  
            fig.colorbar(cax, ax=ax)  
            plt.title(f"Attention Map {j + 1} (Head 0)")
            
            # Save the plot to the specific folder
            filename = f"attention_map_{j + 1}.png"
            if folder:
                filename = os.path.join(folder, filename)
            
            plt.savefig(filename)
            print(f"Saved attention map to {filename}")
            
            plt.close(fig)