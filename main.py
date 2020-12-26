from libv1 import DataLoader, default_collate


def demo_test_1():
    from simplev1_datatset import SimpleV1Dataset
    simple_dataset = SimpleV1Dataset()
    dataloader = DataLoader(simple_dataset, batch_size=2, collate_fn=default_collate)
    for data in dataloader:
        print(data)


if __name__ == '__main__':
    demo_test_1()
