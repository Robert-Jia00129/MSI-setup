import copy
import time
start_time = time.time()

def test_imports():
    print("Running Import Test...")
    try:
        import torch
        import transformers
        import datasets
        print("✅ All imports are successful.\n")
    except ImportError as e:
        print(f"❌ Import error: {e}\n")

def test_dataset_loading():
    print("Running Dataset Loading Test...")
    try:
        from datasets import load_dataset
        dataset = load_dataset('ag_news', split='train[:3]')
        print(f"✅ Dataset loaded successfully with {len(dataset)} examples.\n")
        return dataset
    except Exception as e:
        print(f"❌ Error loading dataset: {e}\n")
        return None

def test_model_loading():
    print("Running Model Loading Test...")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model = AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=4)
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        print("✅ Small model and tokenizer loaded successfully.\n")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}\n")
        return None, None

def test_training_cpu(dataset, model, tokenizer):
    print("Running CPU Training Test...")
    try:
        import torch

        # Tokenize inputs
        def tokenize(batch):
            return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        # Prepare data loader
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        # Set up optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        # Store initial weights
        initial_state = copy.deepcopy(model.state_dict())

        # Training loop
        model.train()
        print("Training started:")
        for batch in loader:
            print("-", end="")
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
            print("✅ Training on CPU completed successfully and model weights updated.\n")
        else:
            print("❌ Training on CPU completed but model weights did not update.\n")
    except Exception as e:
        print(f"❌ Error during CPU training: {e}\n")

if __name__ == "__main__":
    test_imports()
    dataset = test_dataset_loading()
    model, tokenizer = test_model_loading()
    if dataset and model and tokenizer:
        test_training_cpu(dataset, model, tokenizer)
    else:
        print("⚠️ Skipping training test due to previous errors.\n")

    print(f"Total time taken: {time.time() - start_time}.4f")
