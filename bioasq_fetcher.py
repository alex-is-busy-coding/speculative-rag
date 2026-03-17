from datasets import load_dataset


class BioASQDataFetcher:
    """
    A class to fetch and manage the rag-mini-bioasq dataset from Hugging Face.
    """
    
    def __init__(self, dataset_path="rag-datasets/rag-mini-bioasq", 
                 qa_config_name="question-answer-passages", 
                 corpus_config_name = "text-corpus",
                 split="test"):
        self.dataset_path = dataset_path
        self.qa_config_name = qa_config_name
        self.corpus_config_name = corpus_config_name
        self.split = split
        self.qa_dataset = None
        self.corpus = None

    def load_data(self):
        """
        Loads the dataset split into memory from Hugging Face.
        """
        print(f"Loading QA pairs from '{self.dataset_path}'...")
        self.qa_dataset = load_dataset(
            self.dataset_path,
            self.qa_config_name,
            split = self.split
        )
        print(f"  {len(self.qa_dataset)} QA rows loaded.")

        print(f"Loading text corpus from '{self.dataset_path}' ...")
        self.corpus = load_dataset(
            self.dataset_path,
            self.corpus_config_name,
            split = 'passages'
        )
        print(f"  {len(self.corpus)} passages loaded.")

    def ensure_loaded(self):
        if self.qa_dataset is None or self.corpus is None:
            self.load_data()
        
       

    def get_row(self, index: int):
        """
        Fetches a specific row by its index.
        """
        self.ensure_loaded()
            
        if index < 0 or index >= len(self.qa_dataset):
            raise IndexError(f"Row index {index} is out of bounds for dataset of size {len(self.qa_dataset)}.")
            
        return self.qa_dataset[index]

    def iterate_batches(self, batch_size=10):
        """
        A generator that yields batches of data, useful for LLM training loops.
        """
        self.ensure_loaded()
            
        for i in range(0, len(self.qa_dataset), batch_size):
            yield self.qa_dataset[i : i + batch_size]

    def get_all_passages(self):
        '''
        Returns every passage from the text-corpus config.
        This gets indexed in FAISS
        '''
        self.ensure_loaded()
        return [row['passage'] for row in self.corpus]

    def get_qa_pairs(self):
        'Returns (questions, gold_answers) for evaluation.'
        self.ensure_loaded()
        questions = [row['question'] for row in self.qa_dataset]
        answers = [row['answer'] for row in self.qa_dataset]
        return questions, answers

    def inspect(self, n = 2):
        'Prints n sample rows from each config (for verification)'
        self.ensure_loaded()
        print('\n QA sample rows')
        for i in range(min(n, len(self.qa_dataset))):
            row = self.qa_dataset[i]
            print(f" Q : {row['question']}")
            print(f" A : {row['answer'][:120]}...")
            print(f" IDs : {row['relevant_passage_ids'][:60]}")
            
        for i in range(min(n, len(self.corpus))):
            row = self.corpus[i]
            print(f" id : {row['id']}")
            print(f" passage : {row['passage'][:120]}...")
        

