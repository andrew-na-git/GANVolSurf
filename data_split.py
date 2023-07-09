from libraries import *

class OptionPricingDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, df, target_col):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        self.target = target_col
        self.option_data = df

        # Save target(y) and predictors(X)
        self.X = self.option_data.drop([self.target], axis=1)
        self.y = self.option_data[self.target]

    def __len__(self):
        return self.option_data.shape[0]

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]


raw_data = pd.read_csv("./your_path_here") # import your data here using path to file
dataset = OptionPricingDataset(raw_data, "local volatility")

# Split into training and test
train_size = int(0.85 * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = random_split(dataset, [train_size, test_size])

# Dataloaders
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)