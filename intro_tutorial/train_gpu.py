import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import copy
import time
start_time = time.time()
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load a small subset of the dataset
    dataset = load_dataset('ag_news', split='train[:10]')

    # Load smaller model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=4).to(device)
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')

    # Tokenize inputs
    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Prepare data loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Store initial weights
    initial_state = copy.deepcopy(model.state_dict())

    # Training loop
    model.train()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['label'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Compare initial and final weights
    weights_changed = False
    for key in initial_state:
        if not torch.equal(initial_state[key], model.state_dict()[key]):
            weights_changed = True
            break

    if weights_changed:
        print("✅Model weights updated successfully on GPU.")
    else:
        print("❌Model weights did not update on GPU.")
except Exception as e:
    print(f"❌Error during GPU training: {e}")

finally:
    print(f"Total time taken: {time.time() - start_time}.4f")
