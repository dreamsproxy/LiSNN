import pandas as pd

class sliding_window:
    def batch(df:pd.DataFrame, batch_size:int = 3, seq_len:int = 4):
        batches = []
        for i in range(0, len(df) - batch_size + 1, seq_len):
            batch = df.iloc[i:i+batch_size]
            batches.append(batch)
        return batches

    def example():
        # Example usage:
        # Assuming df is your pandas DataFrame
        print("Example usage:")
        print("\tAssuming df is your pandas DataFrame")
        print("\tYour DataFrame with shape (row, cols)")
        print()
        print("df = pd.DataFrame(...)")
        print("Define the size of the sliding window")
        print("window_size = 10")
        print()
        print("Define the stride of the sliding window")
        print("stride = 5")
        print()
        print("Get batches using sliding window")
        print("batches = sliding_window(df, window_size, stride)")