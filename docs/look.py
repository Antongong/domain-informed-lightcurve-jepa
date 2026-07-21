from datasets import load_from_disk

# 1. Load one of your folders (replace with your actual path)
dataset = load_from_disk("./path/to/your/downloaded/dataset/train")

# 2. Print out the names of all columns/features available
print("Columns:", dataset.column_names)

# 3. Print out the very first row of data to see what it looks like
print("First row:", dataset[0])