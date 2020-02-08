from data.Dataset import Dataset

if __name__ == "__main__":
    print('hello world!')
    dataset = Dataset()
    #dataset.read_dataset("", "6:2:2")
    #dataset.save_example()
    dataset.load_examples()
    train_set, dev_set , test_set = dataset.get_split()
    print(len(train_set), len(dev_set), len(test_set))
    for dev_example in dev_set:
        print(dev_example)