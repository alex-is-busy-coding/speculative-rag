from datasets import load_dataset

class BioASQDataFetcher:
    """
    A class to fetch and manage the rag-mini-bioasq dataset from Hugging Face.
    """
    
    def __init__(self, dataset_path="rag-datasets/rag-mini-bioasq", 
                 config_name="question-answer-passages", 
                 split="test"):
        self.dataset_path = dataset_path
        self.config_name = config_name
        self.split = split
        self.dataset = None

    def load_data(self):
        """
        Loads the dataset split into memory from Hugging Face.
        """
        print(f"Loading dataset: {self.dataset_path} | Config: {self.config_name} | Split: {self.split}...")
        try:
            # load_dataset takes the path, configuration (subset), and the split
            self.dataset = load_dataset(self.dataset_path, self.config_name, split=self.split)
            print(f"Successfully loaded {len(self.dataset)} rows.")
        except Exception as e:
            print(f"Failed to load dataset: {e}")

    def get_row(self, index: int):
        """
        Fetches a specific row by its index.
        """
        if self.dataset is None:
            self.load_data()
            
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Row index {index} is out of bounds for dataset of size {len(self.dataset)}.")
            
        return self.dataset[index]

    def iterate_batches(self, batch_size=10):
        """
        A generator that yields batches of data, useful for LLM training loops.
        """
        if self.dataset is None:
            self.load_data()
            
        for i in range(0, len(self.dataset), batch_size):
            yield self.dataset[i : i + batch_size]

"""
# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Initialize the fetcher for the full 'test' split
    fetcher = BioASQDataFetcher(
        dataset_path="rag-datasets/rag-mini-bioasq",
        config_name="question-answer-passages",
        split="test"
    )
    
    # Load the data explicitly
    fetcher.load_data()
    
    # 1. Check the total number of examples
    total_rows = len(fetcher.dataset)
    print(f"\nTotal rows available in the test split: {total_rows}")
    
    # 2. Inspect the first row to understand the schema
    print("\n--- Schema of First Row ---")
    first_row = fetcher.get_row(0)
    for key, value in first_row.items():
        if isinstance(value, list):
            print(f"-> {key}: List of {len(value)} items")
        else:
            print(f"-> {key}: {type(value).__name__} (Preview: {str(value)[:50]}...)")

    # 3. Simulate grabbing batches (useful before moving to PyTorch)
    print("\n--- Testing Batch Iteration ---")
    for batch_idx, batch in enumerate(fetcher.iterate_batches(batch_size=4)):
        # 'batch' returns a dictionary where each key contains a list of 4 items
        print(f"Batch {batch_idx + 1}: Contains {len(batch['id'])} items.")
        if batch_idx == 2: # Stop after 3 batches just for testing
            break
    """