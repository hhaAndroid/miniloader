class _MapDatasetFetcher(object):
    def __init__(self, dataset, collate_fn):
        self.dataset = dataset
        self.collate_fn = collate_fn

    def fetch(self, possibly_batched_index):
        data = [self.dataset[idx] for idx in possibly_batched_index]
        return self.collate_fn(data)
